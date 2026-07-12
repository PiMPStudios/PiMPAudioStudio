import Testing
@testable import PiMPAudioCore

@Test func validationSeedIsStableAcrossEpochs() {
    let runSeed: UInt64 = 42
    let fixed = AdapterTrainingRandom.validationSeed(runSeed: runSeed)

    // Epoch must not enter the validation seed — otherwise epoch scores are
    // not comparable for best-checkpoint selection.
    for epoch in 1...20 {
        let mixedWithEpoch = AdapterTrainingRandom.mix(runSeed, UInt64(truncatingIfNeeded: epoch))
        #expect(fixed != mixedWithEpoch)
        #expect(AdapterTrainingRandom.validationSeed(runSeed: runSeed) == fixed)
    }
}

@Test func validationSeedDependsOnRunSeed() {
    #expect(
        AdapterTrainingRandom.validationSeed(runSeed: 1)
            != AdapterTrainingRandom.validationSeed(runSeed: 2)
    )
}
