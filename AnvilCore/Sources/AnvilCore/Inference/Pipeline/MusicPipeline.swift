import Foundation
import MLX

public actor MusicPipeline {
    private let vae: VAEActor
    private let diffusion: DiffusionActor
    private var _state: PipelineState = .ready
    private var progressContinuation: AsyncStream<GenerationProgress>.Continuation?
    private var _progress: AsyncStream<GenerationProgress>?

    public static let defaultSampleRate = 44100
    public static let hopLength = 2048

    public init(vae: VAEActor, diffusion: DiffusionActor) {
        self.vae = vae
        self.diffusion = diffusion
    }

    public var state: PipelineState { _state }

    public var progress: AsyncStream<GenerationProgress> {
        if let existing = _progress { return existing }
        let (stream, continuation) = AsyncStream<GenerationProgress>.makeStream()
        self.progressContinuation = continuation
        self._progress = stream
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

    public func injectLoRA(config: LoRAConfig) async throws -> [String] {
        try await diffusion.injectLoRA(config: config)
    }

    public func removeLoRA() async {
        await diffusion.removeLoRA()
    }

    public func loadLoRAWeights(_ weights: [(String, MLXArray)], adapterID: String) async throws {
        try await diffusion.loadLoRAWeights(weights, adapterID: adapterID)
    }

    public var injectedAdapterID: String? {
        get async { await diffusion.injectedAdapterID }
    }

    /// Encode audio to latent mean (used for training dataset preparation).
    public func encodeMean(audio: MLXArray, precision: Precision) async -> MLXArray {
        await vae.encodeMean(audio: audio, precision: precision)
    }

    /// Run LoRA training through the actor chain.
    public func trainLoRA(
        samples: [TrainingSample],
        config: LoRAConfig,
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) async throws -> (weights: [(String, MLXArray)], finalLoss: Float?) {
        try await diffusion.trainLoRA(samples: samples, config: config, onProgress: onProgress)
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

        let diffResult = try await diffusion.generate(
            encoderStates: encoderStates,
            contextLatents: contextLatents,
            encoderMask: encoderMask,
            shape: (batch: batchSize, frames: frames, channels: 64),
            config: config
        )

        progressContinuation?.yield(.decoding)
        let audio = await vae.decode(latents: diffResult.latents, precision: config.precision)

        let totalTime = Date().timeIntervalSince(totalStart)
        let costs = TimeCosts(
            diffusionTime: diffResult.timeCosts.diffusionTime,
            perStepTime: diffResult.timeCosts.perStepTime,
            totalTime: totalTime
        )

        _state = .ready
        progressContinuation?.yield(.complete)

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
        let cond = await conditioning.encode(
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
