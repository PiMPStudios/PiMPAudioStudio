import Testing
@testable import PiMPAudioCore

@Suite("TimestepMapper")
struct TimestepMapperTests {

    // MARK: - startStep

    @Test("Zero strength returns start step 0")
    func zeroStrengthStartStep() {
        #expect(TimestepMapper.startStep(forStrength: 0.0, totalSteps: 20) == 0)
    }

    @Test("Full strength returns start step equal to totalSteps")
    func fullStrengthStartStep() {
        #expect(TimestepMapper.startStep(forStrength: 1.0, totalSteps: 20) == 20)
    }

    @Test("Half strength returns start step at midpoint")
    func halfStrengthStartStep() {
        #expect(TimestepMapper.startStep(forStrength: 0.5, totalSteps: 20) == 10)
    }

    // MARK: - truncatedSchedule length

    @Test("Truncated schedule has correct length (totalSteps - startStep)")
    func truncatedScheduleLength() {
        let totalSteps = 20
        let strength: Float = 0.25
        let start = TimestepMapper.startStep(forStrength: strength, totalSteps: totalSteps)
        let schedule = TimestepMapper.truncatedSchedule(
            strength: strength,
            totalSteps: totalSteps,
            shift: 3.0
        )
        #expect(schedule.count == totalSteps - start)
    }

    @Test("Truncated schedule starts from correct point in the full schedule")
    func truncatedScheduleStartsCorrectly() {
        let totalSteps = 20
        let strength: Float = 0.25
        let start = TimestepMapper.startStep(forStrength: strength, totalSteps: totalSteps)
        let full = TimestepSchedule.compute(shift: 3.0, inferSteps: totalSteps)
        let truncated = TimestepMapper.truncatedSchedule(
            strength: strength,
            totalSteps: totalSteps,
            shift: 3.0
        )
        #expect(truncated == Array(full[start...]))
    }

    // MARK: - edge cases

    @Test("Zero strength produces empty schedule")
    func zeroStrengthEmptySchedule() {
        let schedule = TimestepMapper.truncatedSchedule(
            strength: 0.0,
            totalSteps: 20,
            shift: 3.0
        )
        #expect(schedule.isEmpty)
    }

    @Test("Full strength produces schedule with length equal to totalSteps")
    func fullStrengthFullSchedule() {
        let totalSteps = 20
        let schedule = TimestepMapper.truncatedSchedule(
            strength: 1.0,
            totalSteps: totalSteps,
            shift: 3.0
        )
        #expect(schedule.count == totalSteps)
    }
}
