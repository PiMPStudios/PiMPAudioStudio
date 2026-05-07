import Foundation
import MLX

public struct AudioLoadResult: Sendable {
    public let duration: TimeInterval
    public let sampleRate: Int
    public let latentFrames: Int
}

public actor InpaintCoordinator {
    private let vaeActor: VAEActor
    private let sfxVAEActor: SFXVAEActor?
    private let diffusionActor: DiffusionActor?
    private let sfxDiffusionActor: SFXDiffusionActor?
    private let conditioningActor: ConditioningActor?
    private let clapEncoder: CLAPEncoder?
    private let historyManager: EditHistoryManager

    private var currentLatents: MLXArray?
    private var currentAudio: MLXArray?
    private var totalDuration: TimeInterval = 0
    private var sampleRate: Int = 44100
    private var latentFrames: Int = 0

    public var hasLoadedAudio: Bool { currentLatents != nil }
    public var canUndo: Bool { historyManager.canUndo }
    public var canRedo: Bool { historyManager.canRedo }
    public var recentEdits: [EditSnapshot] { historyManager.recentEdits }

    public static let aceStepHopSize = 2048
    public static let sfxHopSize = 2048

    public init(
        vaeActor: VAEActor,
        sfxVAEActor: SFXVAEActor? = nil,
        diffusionActor: DiffusionActor? = nil,
        sfxDiffusionActor: SFXDiffusionActor? = nil,
        conditioningActor: ConditioningActor? = nil,
        clapEncoder: CLAPEncoder? = nil
    ) {
        self.vaeActor = vaeActor
        self.sfxVAEActor = sfxVAEActor
        self.diffusionActor = diffusionActor
        self.sfxDiffusionActor = sfxDiffusionActor
        self.conditioningActor = conditioningActor
        self.clapEncoder = clapEncoder
        self.historyManager = EditHistoryManager()
    }

    public static func hopSize(for pipeline: PipelineChoice) -> Int {
        switch pipeline {
        case .stableAudio: return sfxHopSize
        case .aceStep, .auto: return aceStepHopSize
        }
    }

    // MARK: - Load

    public func loadAudio(
        samples: MLXArray,
        sampleRate: Int,
        precision: Precision = .float32
    ) async throws -> AudioLoadResult {
        self.sampleRate = sampleRate
        let numSamples = samples.shape[1]
        self.totalDuration = Double(numSamples) / Double(sampleRate)

        let latents = try await vaeActor.encodeMean(audio: samples, precision: precision)
        self.currentLatents = latents
        self.currentAudio = samples
        self.latentFrames = latents.shape[1]

        let initialSnapshot = EditSnapshot(
            description: "Original",
            audioData: Data(),
            latents: nil
        )
        historyManager.push(initialSnapshot)

        return AudioLoadResult(
            duration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames
        )
    }

    // MARK: - Edit

    public func edit(request: InpaintRequest) -> AsyncStream<InpaintProgress> {
        AsyncStream { continuation in
            Task {
                do {
                    try await self.runEdit(request: request, continuation: continuation)
                } catch {
                    continuation.yield(.failed(error.localizedDescription))
                    continuation.finish()
                }
            }
        }
    }

    // MARK: - Undo / Redo

    public func undo() -> EditSnapshot? {
        historyManager.undo()
    }

    public func redo() -> EditSnapshot? {
        historyManager.redo()
    }

    // MARK: - State access

    public func getCurrentAudio() -> MLXArray? {
        currentAudio
    }

    // MARK: - Private isolated pipeline

    private func runEdit(
        request: InpaintRequest,
        continuation: AsyncStream<InpaintProgress>.Continuation
    ) async throws {
        guard let latents = currentLatents else {
            continuation.yield(.failed("No audio loaded"))
            continuation.finish()
            return
        }

        continuation.yield(.buildingMask)

        let hop = Self.hopSize(for: request.pipeline)
        let mask = MaskBuilder.buildMask(
            selection: request.selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            latentDim: latents.shape[2],
            hopSize: hop
        )

        let noisedLatents = LatentBlender.injectNoise(
            latents: latents,
            mask: mask,
            strength: request.strength,
            seed: request.seed
        )

        continuation.yield(.encoding)

        let denoised: MLXArray

        switch request.pipeline {
        case .aceStep, .auto:
            guard let diffActor = diffusionActor,
                  let condActor = conditioningActor else {
                continuation.yield(.failed("ACE-Step pipeline not available"))
                continuation.finish()
                return
            }

            let schedule = TimestepMapper.truncatedSchedule(
                strength: request.strength,
                totalSteps: 100,
                shift: 3.0
            )

            let conditioning = try await condActor.encode(
                text: request.regionPrompt,
                lyrics: "",
                referenceAudio: nil,
                duration: totalDuration,
                precision: .float32
            )

            let config = DiffusionConfig(
                method: .ode,
                guidanceScale: request.guidanceScale
            )

            let result = try await diffActor.inpaint(
                startLatents: noisedLatents,
                schedule: schedule,
                encoderStates: conditioning.encoderStates,
                contextLatents: conditioning.contextLatents,
                encoderMask: conditioning.encoderMask,
                config: config
            )
            denoised = result.latents

        case .stableAudio:
            guard let sfxDiff = sfxDiffusionActor,
                  let clap = clapEncoder else {
                continuation.yield(.failed("Stable Audio pipeline not available"))
                continuation.finish()
                return
            }

            let schedule = TimestepMapper.truncatedSchedule(
                strength: request.strength,
                totalSteps: 100,
                shift: 1.0
            )

            let clapEmbedding = try await clap.encode(
                text: request.regionPrompt,
                tokenizer: CLAPTokenizer.forTesting()
            )

            let sfxConfig = SFXDiffusionConfig(
                steps: schedule.count,
                guidanceScale: request.guidanceScale,
                seed: request.seed,
                latentShape: latents.shape
            )

            let result = try await sfxDiff.inpaint(
                startLatents: noisedLatents,
                schedule: schedule,
                conditioning: clapEmbedding,
                config: sfxConfig
            )
            denoised = result.latents
        }

        let composited = LatentBlender.composite(
            original: latents,
            denoised: denoised,
            mask: mask
        )

        continuation.yield(.decoding)

        let audio: MLXArray
        switch request.pipeline {
        case .aceStep, .auto:
            audio = try await vaeActor.decode(latents: composited, precision: .float32)
        case .stableAudio:
            guard let sfxVAE = sfxVAEActor else {
                continuation.yield(.failed("SFX VAE not available"))
                continuation.finish()
                return
            }
            audio = await sfxVAE.decode(latents: composited)
        }

        updateState(latents: composited, audio: audio, request: request)

        continuation.yield(.complete)
        continuation.finish()
    }

    private func updateState(latents: MLXArray, audio: MLXArray, request: InpaintRequest) {
        currentLatents = latents
        currentAudio = audio
        let snapshot = EditSnapshot(
            description: descriptionFor(request),
            audioData: Data(),
            latents: nil,
            request: request
        )
        historyManager.push(snapshot)
    }

    private func descriptionFor(_ request: InpaintRequest) -> String {
        switch request.editMode {
        case .inpaint:
            let start = String(format: "%.1f", request.selection.startTime)
            let end = String(format: "%.1f", request.selection.endTime)
            let prompt = String(request.regionPrompt.prefix(30))
            return "Inpaint \(start)s\u{2013}\(end)s \u{2014} \(prompt)"
        case .repaint:
            let prompt = String(request.regionPrompt.prefix(40))
            return "Repaint \u{2014} \(prompt)"
        }
    }
}
