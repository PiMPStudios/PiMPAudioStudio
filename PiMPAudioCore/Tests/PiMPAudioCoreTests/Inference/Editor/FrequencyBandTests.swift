import Testing
@testable import PiMPAudioCore

@Test func frequencyBandAcceptsValidRange() throws {
    let band = try FrequencyBand(lowCutHz: 200, highCutHz: 4_000)
    #expect(band.lowCutHz == 200)
    #expect(band.highCutHz == 4_000)
}

@Test func frequencyBandRejectsEmptyOrInvertedRange() {
    #expect(throws: FrequencyBandError.self) {
        try FrequencyBand(lowCutHz: 4_000, highCutHz: 4_000)
    }
    #expect(throws: FrequencyBandError.self) {
        try FrequencyBand(lowCutHz: 5_000, highCutHz: 1_000)
    }
}

@Test func frequencyBandRejectsOutOfBounds() {
    #expect(throws: FrequencyBandError.self) {
        try FrequencyBand(lowCutHz: -1, highCutHz: 1_000)
    }
    #expect(throws: FrequencyBandError.self) {
        try FrequencyBand(lowCutHz: 100, highCutHz: 24_001)
    }
}
