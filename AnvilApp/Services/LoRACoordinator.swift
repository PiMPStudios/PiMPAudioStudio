// AnvilApp/Services/LoRACoordinator.swift
import Foundation
import AnvilCore
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "LoRACoordinator")

/// Manages LoRA adapter injection and removal on MusicPipeline.
/// Defensively detects base DiT model changes and clears adapters automatically.
/// Non-throwing — LoRA failures are silent so generation still proceeds.
@MainActor
final class LoRACoordinator {
    private let pipeline: MusicPipeline
    private let registry: ModelRegistry
    private var currentlyInjectedAdapterID: String?
    private var lastKnownBaseModelID: String?

    init(pipeline: MusicPipeline, registry: ModelRegistry) {
        self.pipeline = pipeline
        self.registry = registry
    }

    /// Diffs the desired adapter against the current one and injects/removes as needed.
    /// If the base DiT model changed since last call, clears any existing adapter first.
    /// Non-throwing: failures are logged and generation proceeds without adapter.
    func ensureAdapter(selectedID: String?) async {
        // Defensive: clear if base model changed
        let activeBaseID = await registry.activeModelID(for: .aceStepDiT)
        let activeSpec = await registry.activeSpec(for: .aceStepDiT)
        if activeBaseID != lastKnownBaseModelID {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
            }
            lastKnownBaseModelID = activeBaseID
        }

        let desired = selectedID
        let current = currentlyInjectedAdapterID

        guard desired != current else { return }

        // Remove the currently injected adapter if present
        if current != nil {
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }

        // Inject the new adapter if one was selected
        guard let adapterID = desired else { return }

        do {
            let adapters = AdapterRegistry.listAdapters()
            guard let adapter = adapters.first(where: { $0.id == adapterID }) else {
                logger.warning("Adapter '\(adapterID)' not found in registry — skipping injection")
                return
            }
            if !isCompatible(adapter: adapter, activeBaseID: activeBaseID, activeSpec: activeSpec) {
                logger.warning("Adapter '\(adapterID)' was trained for '\(adapter.baseModelID)', active DiT is '\(activeBaseID ?? "unknown")' — skipping injection")
                return
            }
            let _ = try await pipeline.injectLoRA(config: adapter.config)
            let weights = try await AdapterRegistry.loadWeights(adapterID: adapterID)
            try await pipeline.loadLoRAWeights(weights, adapterID: adapterID)
            currentlyInjectedAdapterID = adapterID
            logger.info("Injected LoRA adapter '\(adapterID)'")
        } catch {
            logger.error("LoRA injection failed for '\(adapterID)': \(error.localizedDescription)")
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }
    }

    /// Apply a full adapter stack to the DiT.  Each entry is `(adapterID, strength)`.
    ///
    /// - Empty stack → removes any current adapter.
    /// - Single entry → equivalent to single-adapter injection using the
    ///   contribution API (strength 1.0 for the primary adapter).
    /// - Multiple entries → multi-adapter stacking: each adapter's weights are
    ///   loaded as a `LoRAContribution` on every LoRALinear layer; the forward
    ///   pass sums `strength_i * loraA_i^T @ loraB_i^T` for every entry.
    ///
    /// Non-throwing — LoRA failures are logged and generation proceeds without adapter.
    func ensureStack(stack: [(id: String, weight: Float)]) async {
        // Defensive: clear if base model changed
        let activeBaseID = await registry.activeModelID(for: .aceStepDiT)
        let activeSpec = await registry.activeSpec(for: .aceStepDiT)
        if activeBaseID != lastKnownBaseModelID {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
            }
            lastKnownBaseModelID = activeBaseID
        }

        // Empty stack — remove any existing injection
        if stack.isEmpty {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
            }
            return
        }

        // Validate entries: must exist in registry and match the active base model
        let allAdapters = AdapterRegistry.listAdapters()
        let validEntries: [(adapter: LoRAAdapter, strength: Float)] = stack.compactMap { entry in
            guard let adapter = allAdapters.first(where: { $0.id == entry.id }) else {
                logger.warning("Adapter '\(entry.id)' not found in registry — skipping")
                return nil
            }
            if !isCompatible(adapter: adapter, activeBaseID: activeBaseID, activeSpec: activeSpec) {
                logger.warning("Adapter '\(entry.id)' trained for '\(adapter.baseModelID)', active DiT is '\(activeBaseID ?? "unknown")' — skipping")
                return nil
            }
            return (adapter, entry.weight)
        }

        guard !validEntries.isEmpty else {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
            }
            return
        }

        // Inject LoRALinear layers if the primary adapter changed or nothing is injected
        let primaryAdapter = validEntries[0].adapter
        if currentlyInjectedAdapterID != primaryAdapter.id {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
            }
            do {
                let _ = try await pipeline.injectLoRA(config: primaryAdapter.config)
                currentlyInjectedAdapterID = primaryAdapter.id
                logger.info("Injected LoRALinear layers for '\(primaryAdapter.id)'")
            } catch {
                logger.error("LoRA injection failed for '\(primaryAdapter.id)': \(error.localizedDescription)")
                currentlyInjectedAdapterID = nil
                return
            }
        }

        // Clear any contributions from a previous generate call, then re-apply
        // the current stack (handles strength changes and stack edits between runs)
        await pipeline.clearLoRAContributions()

        for (adapter, strength) in validEntries {
            do {
                let weights = try await AdapterRegistry.loadWeights(adapterID: adapter.id)
                await pipeline.addLoRAContribution(
                    adapterID: adapter.id,
                    displayName: adapter.name,
                    source: adapter.source,
                    weights: weights,
                    strength: strength
                )
                logger.info("Added LoRA contribution '\(adapter.id)' strength=\(strength)")
            } catch {
                logger.error("Failed to load weights for '\(adapter.id)': \(error.localizedDescription)")
            }
        }
    }

    /// Removes any injected adapter and resets all coordinator state.
    /// Call on unloadAllWeights() and training start.
    func clearAll() async {
        if currentlyInjectedAdapterID != nil {
            await pipeline.clearLoRAContributions()
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }
        lastKnownBaseModelID = nil
    }

    private func isCompatible(
        adapter: LoRAAdapter,
        activeBaseID: String?,
        activeSpec: ModelSpec?
    ) -> Bool {
        guard let activeBaseID else { return true }
        guard adapter.baseModelCategory == .aceStepDiT else {
            return adapter.baseModelID == activeBaseID
        }
        if adapter.baseModelID == activeBaseID { return true }

        guard let adapterVariant = canonicalACEVariant(adapter.baseModelID),
              let activeVariant = canonicalACEVariant(
                activeBaseID,
                spec: activeSpec
              ) else {
            return false
        }
        return adapterVariant == activeVariant
    }

    private func canonicalACEVariant(_ id: String, spec: ModelSpec? = nil) -> String? {
        let values = [
            id,
            spec?.id,
            spec?.displayName,
            spec?.huggingFaceRepo,
            spec?.variantFamily,
        ]
        let haystack = values
            .compactMap { $0?.lowercased() }
            .joined(separator: " ")

        if haystack.contains("xl") || haystack.contains("ace-step-1.5-xl") {
            return "ace-step-dit-xl"
        }
        if haystack.contains("turbo") {
            return "ace-step-dit-turbo"
        }
        if haystack.contains("sft") {
            return "ace-step-dit-sft"
        }
        if haystack.contains("base") || haystack.contains("full") {
            return "ace-step-dit-base"
        }
        return nil
    }
}
