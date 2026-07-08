#if canImport(Darwin)
import Darwin
#endif

public enum TimestepMapper {
    /// Returns the rectified-flow noise level that matches a truncated inpaint schedule.
    ///
    /// Inpaint/img2img starts diffusion from the first timestep in the truncated
    /// schedule.  Noise injection must use that same value so the model is not
    /// asked to denoise a latent sampled at a different `t`.
    public static func noiseStrength(for schedule: [Float]) -> Float {
        schedule.first ?? 0.0
    }

    /// Maps a strength value (0.0–1.0) to the start step index into the diffusion schedule.
    ///
    /// - Parameters:
    ///   - strength: Edit strength in the range [0.0, 1.0].
    ///   - totalSteps: Total number of diffusion steps.
    /// - Returns: The index of the first timestep to use.
    public static func startStep(forStrength strength: Float, totalSteps: Int) -> Int {
        Int(Float(totalSteps) * strength)
    }

    /// Returns the tail of the diffusion schedule starting at the step determined by `strength`.
    ///
    /// - Parameters:
    ///   - strength: Edit strength in the range [0.0, 1.0].
    ///   - totalSteps: Total number of diffusion steps.
    ///   - shift: Shift parameter forwarded to `TimestepSchedule.compute`.
    /// - Returns: The truncated timestep schedule.
    public static func truncatedSchedule(
        strength: Float,
        totalSteps: Int,
        shift: Float
    ) -> [Float] {
        let start = startStep(forStrength: strength, totalSteps: totalSteps)

        if start >= totalSteps {
            // strength == 1.0 → start from pure noise, use the full schedule
            return TimestepSchedule.compute(shift: shift, inferSteps: totalSteps)
        }

        if start == 0 {
            // strength == 0.0 → no denoising
            return []
        }

        let fullSchedule = TimestepSchedule.compute(shift: shift, inferSteps: totalSteps)
        return Array(fullSchedule[start...])
    }

    // MARK: - Stable Audio Open (logSNR sigmoid RF schedule)

    /// Full Stable Audio Open rectified-flow schedule (logSNR sigmoid).
    ///
    /// Matches `SFXDiffusionActor.generateRFSchedule` exactly.  Extracted here
    /// so `InpaintCoordinator` can produce a truncated schedule without calling
    /// into the actor.
    ///
    /// - Parameters:
    ///   - steps:    Number of diffusion steps (e.g. 100).
    ///   - sigmaMax: Starting sigma; SAO uses `1.0` (rectified flow, not EDM).
    /// - Returns: `steps` sigma values descending from `sigmaMax` toward 0.
    public static func sfxRFSchedule(steps: Int, sigmaMax: Float = 1.0) -> [Float] {
        let logsnrMax: Float = sigmaMax < 1.0
            ? logf((1.0 - sigmaMax) / sigmaMax + 1e-6)
            : -6.0
        let logsnrMin: Float = 2.0
        return (0..<steps).map { i in
            if i == 0 { return sigmaMax }
            let logsnr = logsnrMax + Float(i) / Float(steps) * (logsnrMin - logsnrMax)
            return 1.0 / (1.0 + expf(logsnr))
        }
    }

    /// Truncated SAO RF schedule — returns the tail of the full schedule starting
    /// at the step index computed from `strength`.
    ///
    /// - Parameters:
    ///   - strength:   Edit strength in [0, 1]; 1.0 → full schedule; 0.0 → empty.
    ///   - totalSteps: Full schedule length (e.g. 100).
    ///   - sigmaMax:   Passed to `sfxRFSchedule`; SAO default is `1.0`.
    /// - Returns: Truncated t-value sequence for `SFXDiffusionActor.inpaint`.
    public static func sfxTruncatedSchedule(
        strength: Float,
        totalSteps: Int,
        sigmaMax: Float = 1.0
    ) -> [Float] {
        let start = startStep(forStrength: strength, totalSteps: totalSteps)
        guard start > 0 else { return [] }
        let full = sfxRFSchedule(steps: totalSteps, sigmaMax: sigmaMax)
        if start >= totalSteps { return full }
        return Array(full[start...])
    }
}
