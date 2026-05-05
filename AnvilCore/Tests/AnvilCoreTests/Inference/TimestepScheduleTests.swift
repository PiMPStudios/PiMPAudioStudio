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

    @Test("Linspace with shift 1.0 produces evenly spaced")
    func linspaceNoShift() {
        let schedule = TimestepSchedule.compute(shift: 1.0, inferSteps: 4)
        #expect(schedule.count == 4)
        #expect(abs(schedule[0] - 1.0) < 1e-6)
        #expect(abs(schedule[1] - 0.75) < 1e-6)
        #expect(abs(schedule[2] - 0.5) < 1e-6)
        #expect(abs(schedule[3] - 0.25) < 1e-6)
    }
}
