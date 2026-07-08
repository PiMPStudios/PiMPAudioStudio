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
    private let historyManager: EditHistoryManager

    // ACE-Step latents (48 kHz, stride 1920)
    private var currentLatents: MLXArray?
    // SAO latents (44.1 kHz, stride 2048) — populated lazily on first SAO inpaint
    private var sfxCurrentLatents: MLXArray?
    private var sfxLatentFrames: Int = 0

    private var currentAudio: MLXArray?
    private var totalDuration: TimeInterval = 0
    private var sampleRate: Int = 44100
    private var latentFrames: Int = 0

    /// Stable Audio Open conditioning pipeline — set at init or via `setSFXPipeline`.
    private var sfxPipeline: SoundFXPipeline?
    /// T5 tokenizer for SAO conditioning.
    private var sfxTokenizer: PiMPAudioTokenizer?

    public var hasLoadedAudio: Bool {
        currentLatents != nil || sfxCurrentLatents != nil || currentAudio != nil
    }
    public var canUndo: Bool { historyManager.canUndo }
    public var canRedo: Bool { historyManager.canRedo }
    public var recentEdits: [EditSnapshot] { historyManager.recentEdits }

    public static let aceStepHopSize = MusicPipeline.hopLength
    public static let sfxHopSize = SoundFXPipeline.hopLength

    public init(
        vaeActor: VAEActor,
        sfxVAEActor: SFXVAEActor? = nil,
        diffusionActor: DiffusionActor? = nil,
        sfxDiffusionActor: SFXDiffusionActor? = nil,
        conditioningActor: ConditioningActor? = nil,
        sfxPipeline: SoundFXPipeline? = nil,
        sfxTokenizer: PiMPAudioTokenizer? = nil
    ) {
        self.vaeActor = vaeActor
        self.sfxVAEActor = sfxVAEActor
        self.diffusionActor = diffusionActor
        self.sfxDiffusionActor = sfxDiffusionActor
        self.conditioningActor = conditioningActor
        self.sfxPipeline = sfxPipeline
        self.sfxTokenizer = sfxTokenizer
        self.historyManager = EditHistoryManager()
    }

    /// Update the SAO conditioning pipeline at runtime (called when SAO weights load).
    public func setSFXPipeline(_ pipeline: SoundFXPipeline?, tokenizer: PiMPAudioTokenizer?) {
        sfxPipeline = pipeline
        sfxTokenizer = tokenizer
        // Invalidate cached SAO latents so they're re-encoded on next SAO inpaint
        sfxCurrentLatents = nil
        sfxLatentFrames = 0
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
        AudioDiagnostics.memory("editor/loadAudio/begin")
        AudioDiagnostics.note(
            "editor/loadAudio/config",
            "samplesShape=\(samples.shape) sampleRate=\(sampleRate) duration=\(totalDuration) precision=\(precision)"
        )
        AudioDiagnostics.tensor("editor/loadAudio/samples", samples)

        let latents = try await vaeActor.encodeMean(audio: samples, precision: precision)
        self.currentLatents = latents
        self.currentAudio = samples
        self.latentFrames = latents.shape[1]
        AudioDiagnostics.tensor("editor/loadAudio/aceLatents", latents)

        // Also eagerly encode SAO latents if the SFX VAE is already loaded.
        // If it isn't, SAO inpainting will encode lazily on first request.
        self.sfxCurrentLatents = nil
        self.sfxLatentFrames   = 0
        if let sfxVAE = sfxVAEActor, await sfxVAE.isLoaded {
            if let sfxLat = try? await sfxVAE.encodeMean(audio: samples) {
                self.sfxCurrentLatents = sfxLat
                self.sfxLatentFrames   = sfxLat.shape[1]
                AudioDiagnostics.tensor("editor/loadAudio/sfxLatents", sfxLat)
            } else {
                AudioDiagnostics.note("editor/loadAudio/sfxLatents", "encode failed")
            }
        } else {
            AudioDiagnostics.note("editor/loadAudio/sfxLatents", "sfxVAE not loaded")
        }

        let initialSnapshot = EditSnapshot(
            description: "Original",
            audioData: Data(),
            latents: nil
        )
        historyManager.push(initialSnapshot)

        let result = AudioLoadResult(
            duration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames
        )
        AudioDiagnostics.memory("editor/loadAudio/end")
        return result
    }

    /// Replace the currently loaded audio with an edited buffer and refresh the
    /// editable latent cache so subsequent AI edits operate on the new audio.
    public func replaceAudio(
        samples: MLXArray,
        sampleRate: Int,
        description: String,
        precision: Precision = .float32
    ) async throws -> AudioLoadResult {
        self.sampleRate = sampleRate
        let numSamples = samples.shape[1]
        self.totalDuration = Double(numSamples) / Double(sampleRate)
        AudioDiagnostics.memory("editor/replaceAudio/begin")
        AudioDiagnostics.note(
            "editor/replaceAudio/config",
            "description=\(description) samplesShape=\(samples.shape) sampleRate=\(sampleRate) duration=\(totalDuration) precision=\(precision)"
        )
        AudioDiagnostics.tensor("editor/replaceAudio/samples", samples)

        let latents = try await vaeActor.encodeMean(audio: samples, precision: precision)
        self.currentLatents = latents
        self.currentAudio = samples
        self.latentFrames = latents.shape[1]
        AudioDiagnostics.tensor("editor/replaceAudio/aceLatents", latents)

        self.sfxCurrentLatents = nil
        self.sfxLatentFrames = 0
        if let sfxVAE = sfxVAEActor, await sfxVAE.isLoaded {
            if let sfxLat = try? await sfxVAE.encodeMean(audio: samples) {
                self.sfxCurrentLatents = sfxLat
                self.sfxLatentFrames = sfxLat.shape[1]
                AudioDiagnostics.tensor("editor/replaceAudio/sfxLatents", sfxLat)
            } else {
                AudioDiagnostics.note("editor/replaceAudio/sfxLatents", "encode failed")
            }
        } else {
            AudioDiagnostics.note("editor/replaceAudio/sfxLatents", "sfxVAE not loaded")
        }

        let snapshot = EditSnapshot(
            description: description,
            audioData: Data(),
            latents: nil
        )
        historyManager.push(snapshot)

        let result = AudioLoadResult(
            duration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames
        )
        AudioDiagnostics.memory("editor/replaceAudio/end")
        return result
    }

    // MARK: - Edit

    public func edit(request: InpaintRequest) -> AsyncStream<InpaintProgress> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    try await self.runEdit(request: request, continuation: continuation)
                } catch is CancellationError {
                    continuation.yield(.failed("Edit cancelled"))
                    continuation.finish()
                } catch {
                    continuation.yield(.failed(error.localizedDescription))
                    continuation.finish()
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
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
        try Task.checkCancellation()
        guard currentLatents != nil || currentAudio != nil else {
            continuation.yield(.failed("No audio loaded"))
            continuation.finish()
            return
        }

        AudioDiagnostics.memory("editor/edit/begin")
        AudioDiagnostics.note(
            "editor/edit/request",
            "mode=\(request.editMode.rawValue) pipeline=\(request.pipeline.rawValue) selection=\(request.selection.startTime)...\(request.selection.endTime) band=\(diagnosticsBandDescription(request.selection.frequencyBand)) strength=\(request.strength) guidance=\(request.guidanceScale) seed=\(request.seed.map(String.init) ?? "random") totalDuration=\(totalDuration) sampleRate=\(sampleRate) aceFrames=\(latentFrames) sfxFrames=\(sfxLatentFrames) promptChars=\(request.regionPrompt.count) originalPromptChars=\(request.originalPrompt?.count ?? 0)"
        )

        continuation.yield(.buildingMask)
        try Task.checkCancellation()

        // ── Resolve pipeline-specific latents ────────────────────────────────
        // ACE-Step uses 48 kHz latents (stride 1920); SAO uses 44.1 kHz latents
        // (stride 2048).  We resolve the correct latent space here and do all
        // mask / noise / composite work in that space.

        let workingLatents: MLXArray
        let workingFrames: Int
        let workingHop: Int

        switch request.pipeline {
        case .stableAudio:
            guard let sfxVAE = sfxVAEActor else {
                continuation.yield(.failed("Stable Audio VAE not available"))
                continuation.finish()
                return
            }
            // Lazily encode SAO latents on first SAO inpaint (or after an
            // ACE-Step edit invalidated them).
            if let existing = sfxCurrentLatents {
                workingLatents = existing
                workingFrames  = sfxLatentFrames
            } else {
                guard let audio = currentAudio else {
                    continuation.yield(.failed("No source audio to encode"))
                    continuation.finish()
                    return
                }
                try Task.checkCancellation()
                let sfxLat = try await sfxVAE.encodeMean(audio: audio)
                sfxCurrentLatents = sfxLat
                sfxLatentFrames   = sfxLat.shape[1]
                workingLatents    = sfxLat
                workingFrames     = sfxLat.shape[1]
            }
            workingHop = SoundFXPipeline.hopLength

        case .aceStep, .auto:
            guard let aceLatents = currentLatents else {
                continuation.yield(.failed("ACE-Step latents not available"))
                continuation.finish()
                return
            }
            workingLatents = aceLatents
            workingFrames  = latentFrames
            workingHop     = Self.aceStepHopSize
        }
        AudioDiagnostics.note(
            "editor/edit/resolvedPipeline",
            "pipeline=\(request.pipeline.rawValue) workingShape=\(workingLatents.shape) workingFrames=\(workingFrames) hop=\(workingHop)"
        )
        AudioDiagnostics.tensor("editor/edit/workingLatents", workingLatents)

        // ── Mask ──────────────────────────────────────────────────────────────
        let mask = MaskBuilder.buildMask(
            selection: request.selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: workingFrames,
            latentDim: workingLatents.shape[2],
            hopSize: workingHop
        )

        AudioDiagnostics.tensor("editor/edit/mask", mask)

        continuation.yield(.encoding)
        try Task.checkCancellation()

        // ── Diffusion ─────────────────────────────────────────────────────────
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
            AudioDiagnostics.floats("editor/aceInpaint/schedule", schedule)
            let noiseStrength = TimestepMapper.noiseStrength(for: schedule)
            AudioDiagnostics.note(
                "editor/aceInpaint/noise",
                "sliderStrength=\(request.strength) scheduleNoiseStrength=\(noiseStrength) scheduleSteps=\(schedule.count)"
            )
            let noisedLatents = LatentBlender.injectNoise(
                latents: workingLatents,
                mask: mask,
                strength: noiseStrength,
                seed: request.seed
            )
            AudioDiagnostics.tensor("editor/edit/noisedLatents", noisedLatents)
            AudioDiagnostics.memory("editor/edit/afterNoise")
            guard !schedule.isEmpty else {
                // strength == 0 → no denoising; preserve the current latents.
                denoised = noisedLatents
                break
            }

            let conditioning = try await condActor.encode(
                text: request.regionPrompt,
                lyrics: "",
                referenceAudio: nil,
                duration: totalDuration,
                precision: .float32
            )
            AudioDiagnostics.tensor("editor/aceInpaint/encoderStates", conditioning.encoderStates)
            AudioDiagnostics.tensor("editor/aceInpaint/encoderMask", conditioning.encoderMask)
            AudioDiagnostics.tensor("editor/aceInpaint/contextLatents", conditioning.contextLatents)

            let diffProgressStream = await diffActor.makeProgressStream()
            let diffRelayTask = Task {
                for await progress in diffProgressStream {
                    continuation.yield(.diffusion(progress))
                }
            }
            defer { diffRelayTask.cancel() }

            try Task.checkCancellation()
            let result = try await diffActor.inpaint(
                startLatents: noisedLatents,
                schedule: schedule,
                encoderStates: conditioning.encoderStates,
                contextLatents: conditioning.contextLatents,
                encoderMask: conditioning.encoderMask,
                config: DiffusionConfig(method: .ode, guidanceScale: request.guidanceScale)
            )
            denoised = result.latents

        case .stableAudio:
            guard let sfxDiff = sfxDiffusionActor else {
                continuation.yield(.failed("Stable Audio diffusion actor not available"))
                continuation.finish()
                return
            }
            guard let sfxPipe = sfxPipeline, let sfxTok = sfxTokenizer else {
                continuation.yield(.failed(
                    "Stable Audio conditioning not available — load the Stable Audio model first."
                ))
                continuation.finish()
                return
            }

            // Truncated SAO rectified-flow schedule
            let schedule = TimestepMapper.sfxTruncatedSchedule(
                strength: request.strength,
                totalSteps: 100
            )
            AudioDiagnostics.floats("editor/sfxInpaint/schedule", schedule)
            let noiseStrength = TimestepMapper.noiseStrength(for: schedule)
            AudioDiagnostics.note(
                "editor/sfxInpaint/noise",
                "sliderStrength=\(request.strength) scheduleNoiseStrength=\(noiseStrength) scheduleSteps=\(schedule.count)"
            )
            let noisedLatents = LatentBlender.injectNoise(
                latents: workingLatents,
                mask: mask,
                strength: noiseStrength,
                seed: request.seed
            )
            AudioDiagnostics.tensor("editor/edit/noisedLatents", noisedLatents)
            AudioDiagnostics.memory("editor/edit/afterNoise")
            guard !schedule.isEmpty else {
                // strength == 0 → nothing to do, return working latents unchanged
                denoised = noisedLatents
                break
            }

            let cond = await sfxPipe.buildConditioning(
                text: request.regionPrompt,
                duration: totalDuration,
                tokenizer: sfxTok
            )
            AudioDiagnostics.tensor("editor/sfxInpaint/crossCond", cond.crossCond)
            AudioDiagnostics.tensor("editor/sfxInpaint/globalCond", cond.globalCond)
            if let uncross = cond.uncrossCondTokens {
                AudioDiagnostics.tensor("editor/sfxInpaint/uncrossCondTokens", uncross)
            } else {
                AudioDiagnostics.note("editor/sfxInpaint/uncrossCondTokens", "nil")
            }

            let sfxConfig = SFXDiffusionConfig(
                steps: schedule.count,
                guidanceScale: request.guidanceScale,
                seed: request.seed,
                latentShape: workingLatents.shape,
                sigmaMax: schedule.first ?? 1.0
            )

            let sfxProgressStream = await sfxDiff.makeProgressStream()
            let sfxRelayTask = Task {
                for await progress in sfxProgressStream {
                    if case .diffusion(let diffusionProgress) = progress {
                        continuation.yield(.diffusion(diffusionProgress))
                    }
                }
            }
            defer { sfxRelayTask.cancel() }

            try Task.checkCancellation()
            let sfxResult = try await sfxDiff.inpaint(
                startLatents: noisedLatents,
                schedule: schedule,
                crossCond: cond.crossCond,
                globalCond: cond.globalCond,
                config: sfxConfig,
                uncrossCondTokens: cond.uncrossCondTokens
            )
            denoised = sfxResult.latents
        }
        AudioDiagnostics.tensor("editor/edit/denoisedLatents", denoised)
        AudioDiagnostics.memory("editor/edit/afterDiffusion")

        // ── Composite + decode ────────────────────────────────────────────────
        let composited = LatentBlender.composite(
            original: workingLatents,
            denoised: denoised,
            mask: mask
        )
        AudioDiagnostics.tensor("editor/edit/compositedLatents", composited)

        continuation.yield(.decoding)
        try Task.checkCancellation()

        let audio: MLXArray
        switch request.pipeline {
        case .aceStep, .auto:
            audio = try await vaeActor.decode(latents: composited, precision: .float32)
        case .stableAudio:
            guard let sfxVAE = sfxVAEActor else {
                continuation.yield(.failed("SFX VAE not available for decode"))
                continuation.finish()
                return
            }
            audio = try await sfxVAE.decode(latents: composited)
        }
        AudioDiagnostics.tensor("editor/edit/outputAudio", audio)
        AudioDiagnostics.memory("editor/edit/afterDecode")

        // ── State update ─────────────────────────────────────────────────────
        // After a SAO inpaint: update SAO latents, invalidate ACE-Step latents.
        // After an ACE-Step inpaint: update ACE-Step latents, invalidate SAO latents.
        switch request.pipeline {
        case .stableAudio:
            sfxCurrentLatents = composited
            sfxLatentFrames   = composited.shape[1]
            currentLatents    = nil   // stale; re-encode on next ACE-Step inpaint
        case .aceStep, .auto:
            currentLatents  = composited
            latentFrames    = composited.shape[1]
            sfxCurrentLatents = nil   // stale; re-encode on next SAO inpaint
        }
        currentAudio = audio
        let snapshot = EditSnapshot(
            description: descriptionFor(request),
            audioData: Data(),
            latents: nil,
            request: request
        )
        historyManager.push(snapshot)

        continuation.yield(.complete)
        continuation.finish()
        AudioDiagnostics.memory("editor/edit/end")
    }

    private func diagnosticsBandDescription(_ band: FrequencyBand?) -> String {
        guard let band else { return "none" }
        return "\(band.lowCutHz)-\(band.highCutHz)Hz"
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
