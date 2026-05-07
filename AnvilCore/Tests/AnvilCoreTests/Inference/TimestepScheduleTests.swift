import Testing
@testable import AnvilCore

@Suite("TimestepSchedule")
struct TimestepScheduleTests {
    @Test("Lookup table for shift 3.0 returns 8 steps")
    func lookupShift3() {
        let schedule = TimestepSchedule.compute(shift: 3.0)
        #expect(schedule.count == 8)
        #expect(schedule[0] == 1.0)
        #expect(schedule[7] == 0.3)
    }

    @Test("Lookup table for shift 1.0")
    func lookupShift1() {
        let schedule = TimestepSchedule.compute(shift: 1.0)
        #expect(schedule.count == 8)
        #expect(schedule[0] == 1.0)
        #expect(schedule[7] == 0.125)
    }

    @Test("Lookup table for shift 2.0")
    func lookupShift2() {
        let schedule = TimestepSchedule.compute(shift: 2.0)
        #expect(schedule.count == 8)
        #expect(schedule[0] == 1.0)
    }

    @Test("Shift rounds to nearest valid")
    func shiftRounding() {
        let schedule = TimestepSchedule.compute(shift: 2.7)
        #expect(schedule.count == 8)
        #expect(schedule[0] == 1.0)
        #expect(schedule[7] == 0.3)
    }

    @Test("Linspace with inferSteps overrides lookup")
    func linspaceSteps() {
        let schedule = TimestepSchedule.compute(shift: 3.0, inferSteps: 4)
        #expect(schedule.count == 4)
        #expect(schedule[0] == 1.0)
        for i in 1..<schedule.count {
            #expect(schedule[i] < schedule[i - 1])
        }
    }

    @Test("Custom timesteps override everything")
    func customTimesteps() {
        let custom: [Float] = [1.0, 0.75, 0.5, 0.25]
        let schedule = TimestepSchedule.compute(shift: 3.0, timesteps: custom)
        #expect(schedule.count == 4)
        #expect(schedule[0] == 1.0)
    }

    @Test("Custom timesteps trailing zeros removed")
    func trailingZeros() {
        let custom: [Float] = [1.0, 0.5, 0.0, 0.0]
        let schedule = TimestepSchedule.compute(shift: 3.0, timesteps: custom)
        #expect(schedule.count == 2)
    }

    @Test("Linspace with shift 1.0 covers full trajectory to sigma_min")
    func linspaceNoShift() {
        // Mirrors upstream FlowMatchEulerDiscreteScheduler: linspace from 1.0
        // down to sigma_min = 1/num_train_timesteps = 0.001 over N points.
        // For N=4 the spacing is (1.0 - 0.001) / 3 ≈ 0.333.
        let schedule = TimestepSchedule.compute(shift: 1.0, inferSteps: 4)
        let sigmaMin: Float = 0.001
        let step: Float = (1.0 - sigmaMin) / 3.0
        #expect(schedule.count == 4)
        #expect(abs(schedule[0] - 1.0) < 1e-6)
        #expect(abs(schedule[1] - (1.0 - step)) < 1e-6)
        #expect(abs(schedule[2] - (1.0 - 2.0 * step)) < 1e-6)
        #expect(abs(schedule[3] - sigmaMin) < 1e-6)
    }

    @Test("Linspace 8 steps shift 3 matches upstream FlowMatchEuler")
    func linspaceMatchesUpstream() {
        // Reference values computed from upstream's
        //   timesteps = np.linspace(1000, 1, 8); sigmas = timesteps / 1000;
        //   sigmas = 3 * sigmas / (1 + 2 * sigmas)
        // (acestep/schedulers/scheduling_flow_match_euler_discrete.py).
        let schedule = TimestepSchedule.compute(shift: 3.0, inferSteps: 8)
        let expected: [Float] = [
            1.00000, 0.94737, 0.88291, 0.80044, 0.69288, 0.54704, 0.33602, 0.00299
        ]
        #expect(schedule.count == 8)
        for (i, e) in expected.enumerated() {
            #expect(abs(schedule[i] - e) < 1e-3, "step \(i): got \(schedule[i]) expected \(e)")
        }
    }
}
