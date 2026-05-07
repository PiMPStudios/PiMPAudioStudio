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
                Self.validTimesteps.min(by: { abs($0 - t) < abs($1 - t) })!
            }
        }

        if let inferSteps = inferSteps, inferSteps > 0 {
            // Match upstream FlowMatchEulerDiscreteScheduler.set_timesteps
            // (acestep/schedulers/scheduling_flow_match_euler_discrete.py).
            //
            // Upstream:
            //   timesteps = np.linspace(num_train_timesteps, 1.0, num_inference_steps)
            //   sigmas = timesteps / num_train_timesteps
            //   sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            //
            // For num_train_timesteps=1000 and N=8 this produces sigmas going
            // from 1.0 down to ~0.001 (sigma_min) — covering the FULL flow-
            // matching trajectory. The previous formula `1 - i/N` stopped at
            // 1/N (e.g., 0.125 for N=8), leaving the last 12.5% of the
            // trajectory uncovered and producing noisy latents that decoded
            // to static. (See finalStep below — it implicitly appends a
            // terminal sigma=0, mirroring upstream's `torch.cat([sigmas,
            // zeros(1)])`.)
            let sigmaMin: Float = 1.0 / 1000.0  // num_train_timesteps = 1000
            var raw: [Float]
            if inferSteps == 1 {
                raw = [1.0]
            } else {
                raw = (0..<inferSteps).map { i in
                    let frac = Float(i) / Float(inferSteps - 1)
                    return 1.0 + frac * (sigmaMin - 1.0)
                }
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
        let nearestShift = Self.validShifts.min(by: { abs($0 - shift) < abs($1 - shift) })!
        return Self.shiftTimesteps[nearestShift]!
    }
}
