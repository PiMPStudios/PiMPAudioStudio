import Testing
@testable import PiMPAudioCore

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

    @Test("Linspace with shift 1.0 matches upstream v1.5 generation")
    func linspaceNoShift() {
        // Mirrors the checked-in ACE-Step v1.5 generation code:
        // [1 - i/N for i in 0..<N], excluding the terminal zero.
        let schedule = TimestepSchedule.compute(shift: 1.0, inferSteps: 4)
        #expect(schedule.count == 4)
        #expect(abs(schedule[0] - 1.0) < 1e-6)
        #expect(abs(schedule[1] - 0.75) < 1e-6)
        #expect(abs(schedule[2] - 0.5) < 1e-6)
        #expect(abs(schedule[3] - 0.25) < 1e-6)
    }

    @Test("Linspace 8 steps shift 3 matches upstream v1.5 generation")
    func linspaceMatchesUpstream() {
        // Reference values match ACE-Step v1.5 MLX get_timestep_schedule()
        // with infer_steps=8, and the Turbo fixed SHIFT_TIMESTEPS[3.0].
        let schedule = TimestepSchedule.compute(shift: 3.0, inferSteps: 8)
        let expected: [Float] = [
            1.00000, 0.954545, 0.90000, 0.833333, 0.75000, 0.642857, 0.50000, 0.30000
        ]
        #expect(schedule.count == 8)
        for (i, e) in expected.enumerated() {
            #expect(abs(schedule[i] - e) < 1e-3, "step \(i): got \(schedule[i]) expected \(e)")
        }
    }
}
