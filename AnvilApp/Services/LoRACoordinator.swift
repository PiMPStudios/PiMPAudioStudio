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

    /// Removes any injected adapter and resets all coordinator state.
    /// Call on unloadAllWeights() and training start.
    func clearAll() async {
        if currentlyInjectedAdapterID != nil {
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }
        lastKnownBaseModelID = nil
    }
}
