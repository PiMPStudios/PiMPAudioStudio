public enum TimestepMapper {
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
}
