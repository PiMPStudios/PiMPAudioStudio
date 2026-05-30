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
        AudioDiagnostics.memory("lora/ensureAdapter/begin")
        AudioDiagnostics.note(
            "lora/ensureAdapter/request",
            "selected=\(selectedID ?? "nil") current=\(currentlyInjectedAdapterID ?? "nil") activeBase=\(activeBaseID ?? "nil") lastBase=\(lastKnownBaseModelID ?? "nil")"
        )
        if activeBaseID != lastKnownBaseModelID {
            if currentlyInjectedAdapterID != nil {
                AudioDiagnostics.note("lora/ensureAdapter/baseChanged", "clearing injected=\(currentlyInjectedAdapterID ?? "nil")")
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
            }
            lastKnownBaseModelID = activeBaseID
        }

        let desired = selectedID
        let current = currentlyInjectedAdapterID

        guard desired != current else {
            AudioDiagnostics.note("lora/ensureAdapter/noop", "desired already active")
            AudioDiagnostics.memory("lora/ensureAdapter/end")
            return
        }

        // Remove the currently injected adapter if present
        if current != nil {
            AudioDiagnostics.note("lora/ensureAdapter/removeCurrent", "current=\(current ?? "nil")")
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }

        // Inject the new adapter if one was selected
        guard let adapterID = desired else {
            AudioDiagnostics.note("lora/ensureAdapter/clear", "no selected adapter")
            AudioDiagnostics.memory("lora/ensureAdapter/end")
            return
        }

        do {
            let adapters = AdapterRegistry.listAdapters()
            guard let adapter = adapters.first(where: { $0.id == adapterID }) else {
                logger.warning("Adapter '\(adapterID)' not found in registry — skipping injection")
                AudioDiagnostics.note("lora/ensureAdapter/missing", "adapter=\(adapterID)")
                AudioDiagnostics.memory("lora/ensureAdapter/end")
                return
            }
            AudioDiagnostics.note("lora/ensureAdapter/adapter", diagnosticsAdapterDescription(adapter, strength: 1.0))
            if !isCompatible(adapter: adapter, activeBaseID: activeBaseID, activeSpec: activeSpec) {
                logger.warning("Adapter '\(adapterID)' was trained for '\(adapter.baseModelID)', active DiT is '\(activeBaseID ?? "unknown")' — skipping injection")
                AudioDiagnostics.note(
                    "lora/ensureAdapter/incompatible",
                    "adapter=\(adapterID) adapterBase=\(adapter.baseModelID) activeBase=\(activeBaseID ?? "nil")"
                )
                AudioDiagnostics.memory("lora/ensureAdapter/end")
                return
            }
            AudioDiagnostics.memory("lora/ensureAdapter/beforeInject")
            let weights = try await AdapterRegistry.loadWeights(adapterID: adapterID)
            AudioDiagnostics.note("lora/ensureAdapter/weights", "adapter=\(adapterID) tensors=\(weights.count)")
            switch adapter.kind {
            case .lora:
                let _ = try await pipeline.injectLoRA(config: adapter.config)
                try await pipeline.loadLoRAWeights(weights, adapterID: adapterID)
            case .lokr:
                let _ = try await pipeline.injectLoKR(config: lokrConfig(for: adapter))
                try await pipeline.loadLoKRWeights(weights, adapterID: adapterID)
            }
            currentlyInjectedAdapterID = adapterID
            logger.info("Injected \(adapter.kind.rawValue.uppercased()) adapter '\(adapterID)'")
            AudioDiagnostics.memory("lora/ensureAdapter/end")
        } catch {
            logger.error("LoRA injection failed for '\(adapterID)': \(error.localizedDescription)")
            AudioDiagnostics.note("lora/ensureAdapter/error", "adapter=\(adapterID) error=\(error.localizedDescription)")
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
            AudioDiagnostics.memory("lora/ensureAdapter/end")
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
        AudioDiagnostics.memory("lora/ensureStack/begin")
        AudioDiagnostics.note(
            "lora/ensureStack/request",
            "stack=[\(stack.map { "\($0.id)@\(String(format: "%.3f", $0.weight))" }.joined(separator: ","))] current=\(currentlyInjectedAdapterID ?? "nil") activeBase=\(activeBaseID ?? "nil") lastBase=\(lastKnownBaseModelID ?? "nil")"
        )
        if activeBaseID != lastKnownBaseModelID {
            if currentlyInjectedAdapterID != nil {
                AudioDiagnostics.note("lora/ensureStack/baseChanged", "clearing injected=\(currentlyInjectedAdapterID ?? "nil")")
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
            }
            lastKnownBaseModelID = activeBaseID
        }

        // Empty stack — remove any existing injection
        if stack.isEmpty {
            if currentlyInjectedAdapterID != nil {
                AudioDiagnostics.note("lora/ensureStack/empty", "removing current=\(currentlyInjectedAdapterID ?? "nil")")
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
            }
            AudioDiagnostics.memory("lora/ensureStack/end")
            return
        }

        // Validate entries: must exist in registry and match the active base model
        let allAdapters = AdapterRegistry.listAdapters()
        let validEntries: [(adapter: LoRAAdapter, strength: Float)] = stack.compactMap { entry in
            guard let adapter = allAdapters.first(where: { $0.id == entry.id }) else {
                logger.warning("Adapter '\(entry.id)' not found in registry — skipping")
                AudioDiagnostics.note("lora/ensureStack/missing", "adapter=\(entry.id)")
                return nil
            }
            if !isCompatible(adapter: adapter, activeBaseID: activeBaseID, activeSpec: activeSpec) {
                logger.warning("Adapter '\(entry.id)' trained for '\(adapter.baseModelID)', active DiT is '\(activeBaseID ?? "unknown")' — skipping")
                AudioDiagnostics.note(
                    "lora/ensureStack/incompatible",
                    "adapter=\(entry.id) adapterBase=\(adapter.baseModelID) activeBase=\(activeBaseID ?? "nil")"
                )
                return nil
            }
            return (adapter, entry.weight)
        }
        let sameKindEntries: [(adapter: LoRAAdapter, strength: Float)]
        if let primaryKind = validEntries.first?.adapter.kind {
            sameKindEntries = validEntries.filter { $0.adapter.kind == primaryKind }
            let dropped = validEntries.count - sameKindEntries.count
            if dropped > 0 {
                AudioDiagnostics.note(
                    "lora/ensureStack/kindFilter",
                    "primaryKind=\(primaryKind.rawValue) droppedMixedKind=\(dropped)"
                )
            }
        } else {
            sameKindEntries = []
        }

        AudioDiagnostics.note(
            "lora/ensureStack/valid",
            "validCount=\(sameKindEntries.count) valid=[\(sameKindEntries.map { diagnosticsAdapterDescription($0.adapter, strength: $0.strength) }.joined(separator: ","))]"
        )

        guard !sameKindEntries.isEmpty else {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
            }
            AudioDiagnostics.note("lora/ensureStack/noValidEntries", "cleared=\(currentlyInjectedAdapterID == nil)")
            AudioDiagnostics.memory("lora/ensureStack/end")
            return
        }

        // Inject LoRALinear layers if the primary adapter changed or nothing is injected
        let primaryAdapter = sameKindEntries[0].adapter
        if currentlyInjectedAdapterID != primaryAdapter.id {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
            }
            do {
                AudioDiagnostics.memory("lora/ensureStack/beforeInject")
                switch primaryAdapter.kind {
                case .lora:
                    let _ = try await pipeline.injectLoRA(config: primaryAdapter.config)
                case .lokr:
                    let _ = try await pipeline.injectLoKR(config: lokrConfig(for: primaryAdapter))
                }
                currentlyInjectedAdapterID = primaryAdapter.id
                logger.info("Injected \(primaryAdapter.kind.rawValue.uppercased()) layers for '\(primaryAdapter.id)'")
                AudioDiagnostics.memory("lora/ensureStack/afterInject")
            } catch {
                logger.error("LoRA injection failed for '\(primaryAdapter.id)': \(error.localizedDescription)")
                AudioDiagnostics.note("lora/ensureStack/injectError", "adapter=\(primaryAdapter.id) error=\(error.localizedDescription)")
                currentlyInjectedAdapterID = nil
                AudioDiagnostics.memory("lora/ensureStack/end")
                return
            }
        }

        // Clear any contributions from a previous generate call, then re-apply
        // the current stack (handles strength changes and stack edits between runs)
        switch primaryAdapter.kind {
        case .lora:
            await pipeline.clearLoRAContributions()
        case .lokr:
            await pipeline.clearLoKRContributions()
        }

        for (adapter, strength) in sameKindEntries {
            do {
                let weights = try await AdapterRegistry.loadWeights(adapterID: adapter.id)
                AudioDiagnostics.note("lora/ensureStack/weights", "adapter=\(adapter.id) tensors=\(weights.count) strength=\(String(format: "%.3f", strength))")
                switch adapter.kind {
                case .lora:
                    await pipeline.addLoRAContribution(
                        adapterID: adapter.id,
                        displayName: adapter.name,
                        source: adapter.source,
                        weights: weights,
                        strength: strength
                    )
                case .lokr:
                    await pipeline.addLoKRContribution(
                        adapterID: adapter.id,
                        displayName: adapter.name,
                        source: adapter.source,
                        weights: weights,
                        strength: strength
                    )
                }
                logger.info("Added \(adapter.kind.rawValue.uppercased()) contribution '\(adapter.id)' strength=\(strength)")
            } catch {
                logger.error("Failed to load weights for '\(adapter.id)': \(error.localizedDescription)")
                AudioDiagnostics.note("lora/ensureStack/loadError", "adapter=\(adapter.id) error=\(error.localizedDescription)")
            }
        }
        AudioDiagnostics.memory("lora/ensureStack/end")
    }

    /// Removes any injected adapter and resets all coordinator state.
    /// Call on unloadAllWeights() and training start.
    func clearAll() async {
        AudioDiagnostics.note("lora/clearAll", "current=\(currentlyInjectedAdapterID ?? "nil") lastBase=\(lastKnownBaseModelID ?? "nil")")
        if currentlyInjectedAdapterID != nil {
            await pipeline.clearLoRAContributions()
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }
        lastKnownBaseModelID = nil
        AudioDiagnostics.memory("lora/clearAll/end")
    }

    private func diagnosticsAdapterDescription(_ adapter: LoRAAdapter, strength: Float) -> String {
        "id=\(adapter.id):name=\(adapter.name):kind=\(adapter.kind.rawValue):strength=\(String(format: "%.3f", strength)):source=\(adapter.source.rawValue):rank=\(adapter.config.rank):alpha=\(String(format: "%.3f", adapter.config.alpha)):dropout=\(String(format: "%.3f", adapter.config.dropout)):base=\(adapter.baseModelID):patterns=\(adapter.config.injectionPatterns)"
    }

    private func lokrConfig(for adapter: LoRAAdapter) -> LoKRConfig {
        adapter.lokrConfig ?? LoKRConfig(
            linearDim: adapter.config.rank,
            alpha: adapter.config.alpha,
            dropout: adapter.config.dropout,
            targetLayers: adapter.config.targetLayers,
            exactInjectionPatterns: adapter.config.exactInjectionPatterns,
            learningRate: adapter.config.learningRate,
            maxEpochs: adapter.config.maxEpochs,
            batchSize: adapter.config.batchSize,
            warmupFraction: adapter.config.warmupFraction,
            gradientAccumulationSteps: adapter.config.gradientAccumulationSteps
        )
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
