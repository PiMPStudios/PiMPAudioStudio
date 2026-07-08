// PiMPAudioStudio/Services/LoRACoordinator.swift
import Foundation
import MLX
import PiMPAudioCore
import OSLog

private let logger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "LoRACoordinator")

/// Manages LoRA adapter injection and removal on MusicPipeline.
/// Defensively detects base DiT model changes and clears adapters automatically.
/// Non-throwing — LoRA failures are silent so generation still proceeds.
@MainActor
final class LoRACoordinator {
    private let pipeline: MusicPipeline
    private let registry: ModelRegistry
    private var currentlyInjectedAdapterID: String?
    private var currentlyInjectedKind: AdapterKind?
    private var currentlyInjectedLayerIndices: Set<Int>?
    private var lastKnownBaseModelID: String?

    private struct LoadedAdapterEntry {
        let adapter: LoRAAdapter
        let strength: Float
        let weights: [(String, MLXArray)]
        let trainedPaths: [String]?
    }

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
                currentlyInjectedKind = nil
                currentlyInjectedLayerIndices = nil
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
            currentlyInjectedKind = nil
            currentlyInjectedLayerIndices = nil
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
            let trainedPaths = trainedPaths(for: adapter, weightKeys: weights.map(\.0))
            let layerIndices = layerIndices(for: adapter, trainedPaths: trainedPaths)
            AudioDiagnostics.note(
                "lora/ensureAdapter/weights",
                "adapter=\(adapterID) tensors=\(weights.count) trainedPaths=\(trainedPaths?.count ?? 0) layers=\(layerIndices?.sorted() ?? [])"
            )
            switch adapter.kind {
            case .lora:
                let injectedPaths = try await pipeline.injectLoRA(
                    config: adapter.config,
                    layerIndices: layerIndices
                )
                try await pipeline.loadLoRAWeights(
                    weights,
                    adapterID: adapterID,
                    paths: trainedPaths ?? injectedPaths
                )
            case .lokr:
                let injectedPaths = try await pipeline.injectLoKR(
                    config: lokrConfig(for: adapter),
                    layerIndices: layerIndices
                )
                try await pipeline.loadLoKRWeights(
                    weights,
                    adapterID: adapterID,
                    paths: trainedPaths ?? injectedPaths
                )
            }
            currentlyInjectedAdapterID = adapterID
            currentlyInjectedKind = adapter.kind
            currentlyInjectedLayerIndices = layerIndices
            logger.info("Injected \(adapter.kind.rawValue.uppercased()) adapter '\(adapterID)'")
            AudioDiagnostics.memory("lora/ensureAdapter/end")
        } catch {
            logger.error("LoRA injection failed for '\(adapterID)': \(error.localizedDescription)")
            AudioDiagnostics.note("lora/ensureAdapter/error", "adapter=\(adapterID) error=\(error.localizedDescription)")
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
            currentlyInjectedKind = nil
            currentlyInjectedLayerIndices = nil
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
    ///   pass sums each adapter's own alpha-scaled contribution.
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
                currentlyInjectedKind = nil
                currentlyInjectedLayerIndices = nil
            }
            lastKnownBaseModelID = activeBaseID
        }

        // Empty stack — remove any existing injection
        if stack.isEmpty {
            if currentlyInjectedAdapterID != nil {
                AudioDiagnostics.note("lora/ensureStack/empty", "removing current=\(currentlyInjectedAdapterID ?? "nil")")
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
                currentlyInjectedKind = nil
                currentlyInjectedLayerIndices = nil
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
                currentlyInjectedKind = nil
                currentlyInjectedLayerIndices = nil
            }
            AudioDiagnostics.note("lora/ensureStack/noValidEntries", "cleared=\(currentlyInjectedAdapterID == nil)")
            AudioDiagnostics.memory("lora/ensureStack/end")
            return
        }

        var loadedEntries: [LoadedAdapterEntry] = []
        for (adapter, strength) in sameKindEntries {
            do {
                let weights = try await AdapterRegistry.loadWeights(adapterID: adapter.id)
                let paths = trainedPaths(for: adapter, weightKeys: weights.map(\.0))
                loadedEntries.append(LoadedAdapterEntry(
                    adapter: adapter,
                    strength: strength,
                    weights: weights,
                    trainedPaths: paths
                ))
                AudioDiagnostics.note(
                    "lora/ensureStack/weights",
                    "adapter=\(adapter.id) tensors=\(weights.count) strength=\(String(format: "%.3f", strength)) trainedPaths=\(paths?.count ?? 0)"
                )
            } catch {
                logger.error("Failed to load weights for '\(adapter.id)': \(error.localizedDescription)")
                AudioDiagnostics.note("lora/ensureStack/loadError", "adapter=\(adapter.id) error=\(error.localizedDescription)")
            }
        }

        guard !loadedEntries.isEmpty else {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
                currentlyInjectedKind = nil
                currentlyInjectedLayerIndices = nil
            }
            AudioDiagnostics.note("lora/ensureStack/noLoadedEntries", "cleared=\(currentlyInjectedAdapterID == nil)")
            AudioDiagnostics.memory("lora/ensureStack/end")
            return
        }

        // Inject LoRA/LoKR layers if the primary adapter or layer footprint changed.
        let primaryAdapter = loadedEntries[0].adapter
        let injectionLayerIndices = unionLayerIndices(for: loadedEntries)
        let needsInjection = currentlyInjectedAdapterID != primaryAdapter.id
            || currentlyInjectedKind != primaryAdapter.kind
            || currentlyInjectedLayerIndices != injectionLayerIndices
        if needsInjection {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
            }
            do {
                AudioDiagnostics.memory("lora/ensureStack/beforeInject")
                switch primaryAdapter.kind {
                case .lora:
                    let _ = try await pipeline.injectLoRA(
                        config: primaryAdapter.config,
                        layerIndices: injectionLayerIndices
                    )
                case .lokr:
                    let _ = try await pipeline.injectLoKR(
                        config: lokrConfig(for: primaryAdapter),
                        layerIndices: injectionLayerIndices
                    )
                }
                currentlyInjectedAdapterID = primaryAdapter.id
                currentlyInjectedKind = primaryAdapter.kind
                currentlyInjectedLayerIndices = injectionLayerIndices
                logger.info("Injected \(primaryAdapter.kind.rawValue.uppercased()) layers for '\(primaryAdapter.id)'")
                AudioDiagnostics.memory("lora/ensureStack/afterInject")
            } catch {
                logger.error("LoRA injection failed for '\(primaryAdapter.id)': \(error.localizedDescription)")
                AudioDiagnostics.note("lora/ensureStack/injectError", "adapter=\(primaryAdapter.id) error=\(error.localizedDescription)")
                currentlyInjectedAdapterID = nil
                currentlyInjectedKind = nil
                currentlyInjectedLayerIndices = nil
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

        for entry in loadedEntries {
            switch entry.adapter.kind {
            case .lora:
                await pipeline.addLoRAContribution(
                    adapterID: entry.adapter.id,
                    displayName: entry.adapter.name,
                    source: entry.adapter.source,
                    weights: entry.weights,
                    strength: entry.strength,
                    scaling: entry.adapter.config.scaling,
                    paths: entry.trainedPaths
                )
            case .lokr:
                let config = lokrConfig(for: entry.adapter)
                await pipeline.addLoKRContribution(
                    adapterID: entry.adapter.id,
                    displayName: entry.adapter.name,
                    source: entry.adapter.source,
                    weights: entry.weights,
                    strength: entry.strength,
                    scaling: config.scaling,
                    paths: entry.trainedPaths
                )
            }
            logger.info("Added \(entry.adapter.kind.rawValue.uppercased()) contribution '\(entry.adapter.id)' strength=\(entry.strength)")
        }
        AudioDiagnostics.memory("lora/ensureStack/end")
    }

    /// Removes any injected adapter and resets all coordinator state.
    /// Call on unloadAllWeights() and training start.
    func clearAll() async {
        AudioDiagnostics.note("lora/clearAll", "current=\(currentlyInjectedAdapterID ?? "nil") lastBase=\(lastKnownBaseModelID ?? "nil") kind=\(currentlyInjectedKind?.rawValue ?? "nil")")
        // Always clear both contribution lists. removeLoRA restores base linears;
        // clearing only LoRA contributions left LoKR contribution arrays orphaned
        // if remove ever failed partway.
        await pipeline.clearLoRAContributions()
        await pipeline.clearLoKRContributions()
        if currentlyInjectedAdapterID != nil || currentlyInjectedKind != nil {
            await pipeline.removeLoRA()
        }
        currentlyInjectedAdapterID = nil
        currentlyInjectedKind = nil
        currentlyInjectedLayerIndices = nil
        lastKnownBaseModelID = nil
        AudioDiagnostics.memory("lora/clearAll/end")
    }

    private func diagnosticsAdapterDescription(_ adapter: LoRAAdapter, strength: Float) -> String {
        let weightDecompose = adapter.kind == .lokr ? ":weightDecompose=\(lokrConfig(for: adapter).weightDecompose)" : ""
        return "id=\(adapter.id):name=\(adapter.name):kind=\(adapter.kind.rawValue):strength=\(String(format: "%.3f", strength)):source=\(adapter.source.rawValue):rank=\(adapter.config.rank):alpha=\(String(format: "%.3f", adapter.config.alpha)):dropout=\(String(format: "%.3f", adapter.config.dropout))\(weightDecompose):base=\(adapter.baseModelID):patterns=\(adapter.config.injectionPatterns):trainedLayers=\(adapter.effectiveTrainedLayerIndices ?? []):trainedPaths=\(adapter.effectiveTrainedPaths?.count ?? 0)"
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
            gradientAccumulationSteps: adapter.config.gradientAccumulationSteps,
            trainingLayerCount: adapter.config.trainingLayerCount,
            weightDecompose: false
        )
    }

    private func isCompatible(adapter: LoRAAdapter, activeBaseID: String?, activeSpec: ModelSpec?) -> Bool {
        adapter.isCompatible(with: activeBaseID, activeSpec: activeSpec)
    }

    private func trainedPaths(for adapter: LoRAAdapter, weightKeys: [String]) -> [String]? {
        if let paths = adapter.effectiveTrainedPaths {
            return paths
        }
        let derived = LoRAAdapter.trainedPaths(fromWeightKeys: weightKeys, kind: adapter.kind)
        return derived.isEmpty ? nil : derived
    }

    private func layerIndices(for adapter: LoRAAdapter, trainedPaths: [String]?) -> Set<Int>? {
        if let trainedPaths {
            let parsed = LoRAAdapter.layerIndices(from: trainedPaths)
            if !parsed.isEmpty {
                return Set(parsed)
            }
        }
        if let indices = adapter.effectiveTrainedLayerIndices {
            return Set(indices)
        }
        return nil
    }

    private func unionLayerIndices(for entries: [LoadedAdapterEntry]) -> Set<Int>? {
        var union = Set<Int>()
        for entry in entries {
            guard let indices = layerIndices(
                for: entry.adapter,
                trainedPaths: entry.trainedPaths
            ) else {
                return nil
            }
            union.formUnion(indices)
        }
        return union.isEmpty ? nil : union
    }
}
