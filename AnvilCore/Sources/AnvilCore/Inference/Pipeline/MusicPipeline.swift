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
