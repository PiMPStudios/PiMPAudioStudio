import Foundation
import MLX

public struct AudioLoadResult: Sendable {
    public let duration: TimeInterval
    public let sampleRate: Int
    public let latentFrames: Int
}

public actor InpaintCoordinator {
    private let vaeActor: VAEActor
    private var sfxVAEActor: SFXVAEActor?
    private let diffusionActor: DiffusionActor?
    private var sfxDiffusionActor: SFXDiffusionActor?
    /// Mutable so ACE-Step can replace the launch-time stub with a real
    /// tokenizer-backed actor after weights load (same upgrade path as training).
    private var conditioningActor: ConditioningActor?
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

    /// Replace the ACE-Step conditioning actor (stub → real after model load).
    public func setConditioningActor(_ actor: ConditioningActor?) {
        conditioningActor = actor
    }

    /// Wire Stable Audio VAE + diffusion actors when SAO weights become available.
    public func setSFXActors(vae: SFXVAEActor?, diffusion: SFXDiffusionActor?) {
        sfxVAEActor = vae
        sfxDiffusionActor = diffusion
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

    /// Store PCM for the editor **without** VAE-encoding.
    ///
    /// Full-track ACE encode used to run here and routinely left 40–50 GiB of
    /// MLX GPU cache resident after History → Edit / Open (see Diagnostics
    /// `editor/loadAudio/end`). Latents are built lazily on the first AI edit
    /// via ``ensureACELatents`` / SAO encode in ``runEdit``.
    public func loadAudio(
        samples: MLXArray,
        sampleRate: Int,
        precision: Precision = .float32
    ) async throws -> AudioLoadResult {
        AudioDiagnostics.memory("editor/loadAudio/begin")
        let result = adoptSourceAudio(
            samples: samples,
            sampleRate: sampleRate,
            tag: "editor/loadAudio",
            precision: precision
        )
        let initialSnapshot = EditSnapshot(
            description: "Original",
            audioData: Data(),
            latents: nil
        )
        historyManager.push(initialSnapshot)
        AudioDiagnostics.note(
            "editor/loadAudio/deferredEncode",
            "latentFrames=0 (VAE encode deferred until first Inpaint/Repaint)"
        )
        AudioDiagnostics.memory("editor/loadAudio/end")
        return result
    }

    /// Replace the working PCM buffer after DSP (or a UI buffer sync).
    ///
    /// Does **not** re-encode latents — that was a second 20–50 GiB cache hit
    /// after polish. Existing latents are invalidated; the next AI edit encodes.
    public func replaceAudio(
        samples: MLXArray,
        sampleRate: Int,
        description: String,
        precision: Precision = .float32
    ) async throws -> AudioLoadResult {
        AudioDiagnostics.memory("editor/replaceAudio/begin")
        AudioDiagnostics.note(
            "editor/replaceAudio/config",
            "description=\(description) samplesShape=\(samples.shape) sampleRate=\(sampleRate)"
        )
        let result = adoptSourceAudio(
            samples: samples,
            sampleRate: sampleRate,
            tag: "editor/replaceAudio",
            precision: precision
        )
        let snapshot = EditSnapshot(
            description: description,
            audioData: Data(),
            latents: nil
        )
        historyManager.push(snapshot)
        AudioDiagnostics.note(
            "editor/replaceAudio/deferredEncode",
            "latents invalidated; encode deferred until next AI edit"
        )
        AudioDiagnostics.memory("editor/replaceAudio/end")
        return result
    }

    /// Push the latest UI buffer into the coordinator without recording history
    /// (e.g. before Inpaint after local DSP). Invalidates latents if PCM changed.
    public func syncSourceAudio(
        samples: MLXArray,
        sampleRate: Int
    ) {
        AudioDiagnostics.memory("editor/syncSourceAudio/begin")
        _ = adoptSourceAudio(
            samples: samples,
            sampleRate: sampleRate,
            tag: "editor/syncSourceAudio",
            precision: .float32
        )
        AudioDiagnostics.memory("editor/syncSourceAudio/end")
    }

    /// Store PCM, drop any cached latents, flush MLX free-list.
    @discardableResult
    private func adoptSourceAudio(
        samples: MLXArray,
        sampleRate: Int,
        tag: String,
        precision: Precision
    ) -> AudioLoadResult {
        _ = precision // reserved for future quantized sample paths
        self.sampleRate = sampleRate
        let numSamples = samples.shape[1]
        self.totalDuration = Double(numSamples) / Double(sampleRate)
        AudioDiagnostics.note(
            "\(tag)/config",
            "samplesShape=\(samples.shape) sampleRate=\(sampleRate) duration=\(totalDuration)"
        )
        AudioDiagnostics.tensor("\(tag)/samples", samples)

        self.currentAudio = samples
        // Drop prior latent caches so the next AI edit encodes this buffer.
        self.currentLatents = nil
        self.latentFrames = 0
        self.sfxCurrentLatents = nil
        self.sfxLatentFrames = 0
        Self.flushMLXCache(tag: "\(tag)/afterAdopt")

        return AudioLoadResult(
            duration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: 0
        )
    }

    /// ACE latents for the current PCM, encoding only if missing.
    private func ensureACELatents(precision: Precision) async throws -> MLXArray {
        if let existing = currentLatents {
            return existing
        }
        guard let audio = currentAudio else {
            throw InpaintCoordinatorError.noSourceAudio
        }
        AudioDiagnostics.memory("editor/ensureACELatents/begin")
        let latents = try await vaeActor.encodeMean(audio: audio, precision: precision)
        self.currentLatents = latents
        self.latentFrames = latents.shape[1]
        AudioDiagnostics.tensor("editor/ensureACELatents/aceLatents", latents)
        Self.flushMLXCache(tag: "editor/ensureACELatents/afterEncode")
        AudioDiagnostics.memory("editor/ensureACELatents/end")
        return latents
    }

    /// SAO latents for the current PCM, encoding only if missing.
    private func ensureSFXLatents() async throws -> MLXArray {
        if let existing = sfxCurrentLatents {
            return existing
        }
        guard let sfxVAE = sfxVAEActor else {
            throw InpaintCoordinatorError.sfxVAEUnavailable
        }
        guard let audio = currentAudio else {
            throw InpaintCoordinatorError.noSourceAudio
        }
        AudioDiagnostics.memory("editor/ensureSFXLatents/begin")
        let sfxLat = try await sfxVAE.encodeMean(audio: audio)
        sfxCurrentLatents = sfxLat
        sfxLatentFrames = sfxLat.shape[1]
        AudioDiagnostics.tensor("editor/ensureSFXLatents/sfxLatents", sfxLat)
        Self.flushMLXCache(tag: "editor/ensureSFXLatents/afterEncode")
        AudioDiagnostics.memory("editor/ensureSFXLatents/end")
        return sfxLat
    }

    /// Return MLX's free-list to the OS so one VAE encode does not park 40–50 GiB.
    private static func flushMLXCache(tag: String) {
        let before = Memory.snapshot()
        Memory.clearCache()
        let after = Memory.snapshot()
        AudioDiagnostics.note(
            tag,
            "clearCache active=\(after.activeMemory) cacheBefore=\(before.cacheMemory) cacheAfter=\(after.cacheMemory)"
        )
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
            guard sfxVAEActor != nil else {
                continuation.yield(.failed("Stable Audio VAE not available"))
                continuation.finish()
                return
            }
            // Lazily encode SAO latents on first SAO inpaint (or after PCM change).
            try Task.checkCancellation()
            continuation.yield(.encoding)
            do {
                let sfxLat = try await ensureSFXLatents()
                workingLatents = sfxLat
                workingFrames = sfxLatentFrames
            } catch {
                continuation.yield(.failed(error.localizedDescription))
                continuation.finish()
                return
            }
            workingHop = SoundFXPipeline.hopLength

        case .aceStep, .auto:
            // Lazily encode ACE latents on first AI edit (not on file open).
            try Task.checkCancellation()
            continuation.yield(.encoding)
            do {
                let aceLatents = try await ensureACELatents(precision: .float32)
                workingLatents = aceLatents
                workingFrames = latentFrames
            } catch {
                continuation.yield(.failed(error.localizedDescription))
                continuation.finish()
                return
            }
            workingHop = Self.aceStepHopSize
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

        // Drop encode/decode scratch so Active+Cache does not keep climbing after each edit.
        Self.flushMLXCache(tag: "editor/edit/afterComplete")

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

// MARK: - Errors

public enum InpaintCoordinatorError: Error, LocalizedError, Sendable {
    case noSourceAudio
    case sfxVAEUnavailable

    public var errorDescription: String? {
        switch self {
        case .noSourceAudio:
            return "No source audio to encode for AI edit."
        case .sfxVAEUnavailable:
            return "Stable Audio VAE not available."
        }
    }
}
