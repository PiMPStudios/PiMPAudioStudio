import Foundation
import AnvilCore
import MLX

struct GenerationResult: Sendable {
    let fileURL: URL
    let entry: HistoryEntry
}

@MainActor
final class GenerationService: ObservableObject {
    @Published var state: GenerationState = .idle
    @Published var currentResult: GenerationResult?

    private var generationTask: Task<Void, Never>?
    private let pipeline: MusicPipeline
    private let conditioning: ConditioningActor
    private let weightConverter: WeightConverter
    private let historyStore: HistoryStore
    private let outputDirectoryManager: OutputDirectoryManager
    private let sfxModelManager: SFXModelManager
    private let registry: ModelRegistry
    private var currentlyLoadedDiTModelID: String?
    private var currentlyLoadedVAEModelID: String?
    private var currentlyInjectedAdapterID: String?
    var selectedAdapterID: String?

    init(
        pipeline: MusicPipeline,
        conditioning: ConditioningActor,
        weightConverter: WeightConverter,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager,
        sfxModelManager: SFXModelManager,
        registry: ModelRegistry
    ) {
        self.pipeline = pipeline
        self.conditioning = conditioning
        self.weightConverter = weightConverter
        self.historyStore = historyStore
        self.outputDirectoryManager = outputDirectoryManager
        self.sfxModelManager = sfxModelManager
        self.registry = registry
    }

    func generate(request: GenerationRequest) {
        generationTask?.cancel()

        generationTask = Task { [weak self] in
            guard let self else { return }

            guard !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                self.state = .error("Prompt cannot be empty")
                return
            }

            guard let outputDir = self.outputDirectoryManager.directory else {
                self.state = .error("No output directory selected")
                return
            }

            let useStableAudio = self.shouldUseStableAudio(request: request)

            do {
                let result: AudioResult
                if useStableAudio {
                    result = try await self.generateWithStableAudio(request: request)
                } else {
                    result = try await self.generateWithACEStep(request: request)
                }

                try Task.checkCancellation()

                let fileName = HistoryEntry.fileName(prompt: request.text)
                let fileURL = outputDir.appending(path: fileName)

                try WAVWriter.write(
                    samples: result.samples,
                    sampleRate: result.sampleRate,
                    to: fileURL
                )

                let entry = HistoryEntry(
                    id: UUID(),
                    createdAt: Date(),
                    prompt: request.text,
                    lyrics: request.lyrics,
                    duration: request.duration,
                    seed: request.seed ?? 0,
                    method: request.method.rawValue,
                    guidanceScale: request.guidanceScale,
                    precision: request.precision.rawValue,
                    fileName: fileName,
                    pipeline: nil,
                    originalPrompt: nil
                )

                self.historyStore.add(entry)
                self.currentResult = GenerationResult(fileURL: fileURL, entry: entry)
                self.state = .complete
            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .error(error.localizedDescription)
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        state = .cancelled
    }

    private func shouldUseStableAudio(request: GenerationRequest) -> Bool {
        switch request.pipelineOverride {
        case .aceStep: return false
        case .stableAudio: return true
        case .auto:
            return request.resolvedMode == .soundFX || request.resolvedMode == .ambient
        }
    }

    private func generateWithStableAudio(request: GenerationRequest) async throws -> AudioResult {
        // Check for active model swap before readiness check
        try await sfxModelManager.ensureActiveModel()

        if !sfxModelManager.isReady {
            state = .downloadingModel(progress: 0)
            try await sfxModelManager.downloadIfNeeded()
            try await sfxModelManager.loadPipeline()
        }

        guard let sfxPipeline = sfxModelManager.pipeline else {
            throw GenerationError.pipelineNotReady
        }

        let totalSteps = 100
        state = .generating(step: 0, totalSteps: totalSteps, eta: 0)

        let progressStream = await sfxPipeline.progress
        let progressTask = Task { [weak self] in
            guard let self else { return }
            for await progress in progressStream {
                if case .diffusion(let dp) = progress {
                    let perStep = dp.currentStep > 0
                        ? dp.elapsedTime / Double(dp.currentStep)
                        : 0
                    let eta = perStep * Double(dp.totalSteps - dp.currentStep)
                    self.state = .generating(
                        step: dp.currentStep,
                        totalSteps: dp.totalSteps,
                        eta: eta
                    )
                }
            }
        }
        defer { progressTask.cancel() }

        let clampedDuration = min(request.duration, 47.0)
        let tokenizer = CLAPTokenizer.forTesting()
        let config = SFXDiffusionConfig(
            steps: totalSteps,
            guidanceScale: request.guidanceScale > 1.0 ? request.guidanceScale : 7.0,
            seed: request.seed,
            latentShape: [1, Int(ceil(clampedDuration * 44100.0 / 1024.0)), 64]
        )

        state = .generating(step: 0, totalSteps: config.steps, eta: 0)

        let result = try await sfxPipeline.generate(
            text: request.text,
            duration: clampedDuration,
            config: config,
            tokenizer: tokenizer
        )

        state = .decoding
        return result
    }

    private func generateWithACEStep(request: GenerationRequest) async throws -> AudioResult {
        // Check DiT model swap
        let activeDiTID = await registry.activeModelID(for: .aceStepDiT)
        guard let activeDiTID else {
            throw GenerationError.modelNotConfigured("ACE-Step DiT")
        }
        if activeDiTID != currentlyLoadedDiTModelID {
            state = .loadingWeights(progress: 0)
            try await loadDiTWeights()
        }

        // Check VAE model swap
        let activeVAEID = await registry.activeModelID(for: .aceStepVAE)
        guard let activeVAEID else {
            throw GenerationError.modelNotConfigured("ACE-Step VAE")
        }
        if activeVAEID != currentlyLoadedVAEModelID {
            try await loadVAEWeights()
        }

        await ensureAdapter()

        try Task.checkCancellation()
        state = .conditioning

        let config = DiffusionConfig(
            seed: request.seed,
            method: request.method,
            guidanceScale: request.guidanceScale,
            precision: request.precision
        )

        let totalSteps = config.steps ?? 8
        state = .generating(step: 0, totalSteps: totalSteps, eta: 0)

        let progressStream = await pipeline.progress
        let progressTask = Task { [weak self] in
            guard let self else { return }
            for await progress in progressStream {
                if case .diffusion(let dp) = progress {
                    let perStep = dp.currentStep > 0
                        ? dp.elapsedTime / Double(dp.currentStep)
                        : 0
                    let eta = perStep * Double(dp.totalSteps - dp.currentStep)
                    self.state = .generating(
                        step: dp.currentStep,
                        totalSteps: dp.totalSteps,
                        eta: eta
                    )
                }
            }
        }
        defer { progressTask.cancel() }

        let result = try await pipeline.generate(
            text: request.text,
            lyrics: request.lyrics,
            referenceAudio: nil,
            duration: request.duration,
            config: config,
            conditioning: conditioning
        )

        state = .decoding
        return result
    }

    private func loadDiTWeights() async throws {
        guard let spec = await registry.activeSpec(for: .aceStepDiT) else {
            throw GenerationError.modelNotConfigured("ACE-Step DiT")
        }
        let baseDir = await registry.baseDirectory
        let previousModelID = currentlyLoadedDiTModelID
        state = .loadingWeights(progress: 0)

        await pipeline.unloadDiTWeights()
        currentlyLoadedDiTModelID = nil

        // Remove adapter when base model changes
        if currentlyInjectedAdapterID != nil {
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }

        do {
            let weightsPath = spec.localPath(for: "model.safetensors", base: baseDir)
            let rawWeights = try await weightConverter.loadSafetensors(from: weightsPath)
            let weights = await weightConverter.convertDiTWeights(stateDict: rawWeights)
            await pipeline.loadDiTWeights(weights, modelID: spec.id)
            currentlyLoadedDiTModelID = spec.id
        } catch {
            // Revert active model to previous working model
            if let previousModelID {
                try? await registry.setActiveModel(id: previousModelID, for: .aceStepDiT)
            }
            throw error
        }
    }

    // MARK: - LoRA Adapter Injection

    private func ensureAdapter() async {
        let desired = selectedAdapterID
        let current = currentlyInjectedAdapterID

        if desired == current { return }

        // Remove current if injected
        if current != nil {
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }

        // Inject new if selected
        guard let adapterID = desired else { return }

        do {
            let adapters = AdapterRegistry.listAdapters()
            guard let adapter = adapters.first(where: { $0.id == adapterID }) else {
                selectedAdapterID = nil
                return
            }
            let _ = await pipeline.injectLoRA(config: adapter.config)
            let weights = try AdapterRegistry.loadWeights(adapterID: adapterID)
            await pipeline.loadLoRAWeights(weights, adapterID: adapterID)
            currentlyInjectedAdapterID = adapterID
        } catch {
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
            selectedAdapterID = nil
        }
    }

    private func loadVAEWeights() async throws {
        guard let spec = await registry.activeSpec(for: .aceStepVAE) else {
            throw GenerationError.modelNotConfigured("ACE-Step VAE")
        }
        let baseDir = await registry.baseDirectory
        let previousModelID = currentlyLoadedVAEModelID

        await pipeline.unloadVAEWeights()
        currentlyLoadedVAEModelID = nil

        do {
            let vaeFile = spec.files.first(where: { $0.contains("vae") }) ?? "vae.safetensors"
            let weightsPath = spec.localPath(for: vaeFile, base: baseDir)
            let rawWeights = try await weightConverter.loadSafetensors(from: weightsPath)
            let weights = await weightConverter.convertVAEWeights(stateDict: rawWeights)
            await pipeline.loadVAEWeights(weights, modelID: spec.id)
            currentlyLoadedVAEModelID = spec.id
        } catch {
            // Revert active model to previous working model
            if let previousModelID {
                try? await registry.setActiveModel(id: previousModelID, for: .aceStepVAE)
            }
            throw error
        }
    }
}

enum GenerationError: Error, LocalizedError {
    case pipelineNotReady
    case modelNotConfigured(String)

    var errorDescription: String? {
        switch self {
        case .pipelineNotReady: return "Sound FX pipeline is not ready"
        case .modelNotConfigured(let name): return "\(name) model is not configured"
        }
    }
}
