import Foundation
import MLX

public actor MusicPipeline {
    private let vae: VAEActor
    private let diffusion: DiffusionActor
    private var _state: PipelineState = .ready
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?

    /// Audio sample rate of the ACE-Step v1.5 VAE (read from
    /// `vae/config.json`'s `sampling_rate` field — 48000 Hz).
    public static let defaultSampleRate = 48000
    /// Total VAE downsampling = product of `downsampling_ratios`
    /// `[2, 4, 4, 6, 10]` from the v1.5 VAE config = 1920. Latent frames
    /// per second = sampleRate / hopLength = 48000 / 1920 = 25.
    public static let hopLength = 1920

    public init(vae: VAEActor, diffusion: DiffusionActor) {
        self.vae = vae
        self.diffusion = diffusion
    }

    public var state: PipelineState { _state }

    public func makeProgressStream() -> AsyncStream<GenerationProgress> {
        let (stream, continuation) = AsyncStream<GenerationProgress>.makeStream()
        self.progressContinuation = continuation
        return stream
    }

    public static func latentFrames(
        forDuration duration: TimeInterval,
        sampleRate: Int = defaultSampleRate,
        hopLength: Int = hopLength
    ) -> Int {
        Int(ceil(Double(sampleRate) * duration / Double(hopLength)))
    }

    public func loadDiTWeights(_ weights: [(String, MLXArray)], modelID: String) async {
        await diffusion.loadDiTWeights(weights, modelID: modelID)
    }

    public func configureDiT(config: DiTConfig) async {
        await diffusion.configureDiT(config: config)
    }

    public func unloadDiTWeights() async {
        await diffusion.unloadDiTWeights()
    }

    public func loadVAEWeights(_ weights: [(String, MLXArray)], modelID: String) async {
        await vae.loadWeights(weights, modelID: modelID)
    }

    public func unloadVAEWeights() async {
        await vae.unloadWeights()
    }

    // MARK: - LoRA

    public func injectLoRA(config: LoRAConfig, layerIndices: Set<Int>? = nil) async throws -> [String] {
        try await diffusion.injectLoRA(config: config, layerIndices: layerIndices)
    }

    public func removeLoRA() async {
        await diffusion.removeLoRA()
    }

    public func loadLoRAWeights(_ weights: [(String, MLXArray)], adapterID: String) async throws {
        try await diffusion.loadLoRAWeights(weights, adapterID: adapterID)
    }

    public func loadLoRAWeights(
        _ weights: [(String, MLXArray)],
        adapterID: String,
        paths: [String]
    ) async throws {
        try await diffusion.loadLoRAWeights(weights, adapterID: adapterID, paths: paths)
    }

    @discardableResult
    public func addLoRAContribution(
        adapterID: String,
        displayName: String?,
        source: AdapterSource,
        weights: [(String, MLXArray)],
        strength: Float,
        scaling: Float,
        paths: [String]? = nil
    ) async -> Int {
        await diffusion.addLoRAContribution(
            adapterID: adapterID,
            displayName: displayName,
            source: source,
            weights: weights,
            strength: strength,
            scaling: scaling,
            paths: paths
        )
    }

    public func clearLoRAContributions() async {
        await diffusion.clearLoRAContributions()
    }

    public func injectLoKR(config: LoKRConfig, layerIndices: Set<Int>? = nil) async throws -> [String] {
        try await diffusion.injectLoKR(config: config, layerIndices: layerIndices)
    }

    public func loadLoKRWeights(_ weights: [(String, MLXArray)], adapterID: String) async throws {
        try await diffusion.loadLoKRWeights(weights, adapterID: adapterID)
    }

    public func loadLoKRWeights(
        _ weights: [(String, MLXArray)],
        adapterID: String,
        paths: [String]
    ) async throws {
        try await diffusion.loadLoKRWeights(weights, adapterID: adapterID, paths: paths)
    }

    @discardableResult
    public func addLoKRContribution(
        adapterID: String,
        displayName: String?,
        source: AdapterSource,
        weights: [(String, MLXArray)],
        strength: Float,
        scaling: Float,
        paths: [String]? = nil
    ) async -> Int {
        await diffusion.addLoKRContribution(
            adapterID: adapterID,
            displayName: displayName,
            source: source,
            weights: weights,
            strength: strength,
            scaling: scaling,
            paths: paths
        )
    }

    public func clearLoKRContributions() async {
        await diffusion.clearLoKRContributions()
    }

    public var injectedAdapterID: String? {
        get async { await diffusion.injectedAdapterID }
    }

    /// Encode audio to latent mean (used for training dataset preparation).
    public func encodeMean(audio: MLXArray, precision: Precision) async throws -> MLXArray {
        try await vae.encodeMean(audio: audio, precision: precision)
    }

    /// Run LoRA training through the actor chain.
    public func trainLoRA(
        samples: [TrainingSample],
        config: LoRAConfig,
        resumeFrom checkpoint: TrainingCheckpoint? = nil,
        initialWeights: [(String, MLXArray)]? = nil,
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) async throws -> AdapterTrainingResult {
        try await diffusion.trainLoRA(
            samples: samples,
            config: config,
            resumeFrom: checkpoint,
            initialWeights: initialWeights,
            onProgress: onProgress
        )
    }

    public func trainLoKR(
        samples: [TrainingSample],
        config: LoKRConfig,
        resumeFrom checkpoint: TrainingCheckpoint? = nil,
        initialWeights: [(String, MLXArray)]? = nil,
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) async throws -> AdapterTrainingResult {
        try await diffusion.trainLoKR(
            samples: samples,
            config: config,
            resumeFrom: checkpoint,
            initialWeights: initialWeights,
            onProgress: onProgress
        )
    }

    /// Cancel ongoing training cooperatively.
    public func cancelTraining() async {
        await diffusion.cancelTraining()
    }

    public func generate(
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        encoderMask: MLXArray?,
        duration: TimeInterval,
        config: DiffusionConfig
    ) async throws -> AudioResult {
        let totalStart = Date()
        _state = .generating(step: 0, totalSteps: config.steps ?? 8)

        let frames = Self.latentFrames(forDuration: duration)
        let batchSize = encoderStates.shape[0]

        progressContinuation?.yield(.diffusion(DiffusionProgress(
            currentStep: 0, totalSteps: config.steps ?? 8, elapsedTime: 0
        )))

        // Wire up per-step progress from DiffusionActor before calling generate().
        // DiffusionActor.generate() yields to its own internal progressContinuation
        // after each step, but ONLY if makeProgressStream() has been called first to
        // initialise that continuation. Without this relay the upstream progress task
        // in ACEStepGenerationService only ever sees step-0 and then .decoding,
        // so the UI progress bar never advances past "0/N".
        let diffProgressStream = await diffusion.makeProgressStream()
        let outerContinuation = self.progressContinuation   // Sendable capture
        let diffRelayTask = Task {
            for await dp in diffProgressStream {
                outerContinuation?.yield(.diffusion(dp))
            }
        }
        defer { diffRelayTask.cancel() }

        let diffResult = try await diffusion.generate(
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            encoderMask: encoderMask,
            shape: (batch: batchSize, frames: frames, channels: 64),
            config: config
        )

        progressContinuation?.yield(.decoding)
        let audio = try await vae.decode(latents: diffResult.latents, precision: config.precision)

        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffResult.timeCosts.diffusionTime,
            perStepTime: diffResult.timeCosts.perStepTime,
            totalTime: totalTime
        )

        _state = .ready
        progressContinuation?.yield(.complete)
        progressContinuation?.finish()
        progressContinuation = nil

        return AudioResult(
            samples: audio,
            sampleRate: Self.defaultSampleRate,
            duration: duration,
            timeCosts: costs
        )
    }

    public func generate(
        text: String,
        lyrics: String,
        referenceAudio: MLXArray?,
        duration: TimeInterval,
        config: DiffusionConfig,
        conditioning: ConditioningActor
    ) async throws -> AudioResult {
        let cond = try await conditioning.encode(
            text: text,
            lyrics: lyrics,
            referenceAudio: referenceAudio,
            duration: duration,
            precision: config.precision
        )
        return try await generate(
            encoderStates: cond.encoderStates,
            contextLatents: cond.contextLatents,
            encoderMask: cond.encoderMask,
            duration: duration,
            config: config
        )
    }
}
