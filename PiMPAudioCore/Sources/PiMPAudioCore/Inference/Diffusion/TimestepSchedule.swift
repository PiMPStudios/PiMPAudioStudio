public enum TimestepSchedule {
    static let validTimesteps: [Float] = [
        1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
        0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
        0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
        0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
    ]

    static let validShifts: [Float] = [1.0, 2.0, 3.0]

    static let shiftTimesteps: [Float: [Float]] = [
        1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
              0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
        3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
              0.6428571428571429, 0.5, 0.3],
    ]

    public static func compute(
        shift: Float = 3.0,
        timesteps: [Float]? = nil,
        inferSteps: Int? = nil
    ) -> [Float] {
        if let timesteps = timesteps {
            var ts = timesteps
            while let last = ts.last, last == 0 {
                ts.removeLast()
            }
            guard !ts.isEmpty else {
                return computeFallback(shift: shift)
            }
            if ts.count > 20 {
                ts = Array(ts.prefix(20))
            }
            return ts.map { t in
                Self.validTimesteps.min(by: { abs($0 - t) < abs($1 - t) }) ?? t
            }
        }

        if let inferSteps = inferSteps, inferSteps > 0 {
            // Match the checked-in ACE-Step v1.5 MLX/PyTorch generation code:
            // get_timestep_schedule(... infer_steps=N) uses
            // [1 - i/N for i in 0..<N], then applies the shift transform. The
            // final implicit x0 step subtracts vt * current_timestep.
            var raw = (0..<inferSteps).map { i in
                1.0 - Float(i) / Float(inferSteps)
            }
            if shift != 1.0 {
                raw = raw.map { t in
                    shift * t / (1.0 + (shift - 1.0) * t)
                }
            }
            return raw
        }

        return computeFallback(shift: shift)
    }

    private static func computeFallback(shift: Float) -> [Float] {
        guard let nearestShift = Self.validShifts.min(by: { abs($0 - shift) < abs($1 - shift) }),
              let timesteps = Self.shiftTimesteps[nearestShift] else {
            return Self.validTimesteps
        }
        return timesteps
    }
}
