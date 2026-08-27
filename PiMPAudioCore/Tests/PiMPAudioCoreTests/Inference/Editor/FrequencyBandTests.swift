@testable import PiMPAudioCore
import Testing

@Test func frequencyBandAcceptsValidRange() throws {
    let band = try FrequencyBand(lowCutHz: 200, highCutHz: 4000)
    #expect(band.lowCutHz == 200)
    #expect(band.highCutHz == 4000)
}

@Test func frequencyBandRejectsEmptyOrInvertedRange() {
    #expect(throws: FrequencyBandError.self) {
        try FrequencyBand(lowCutHz: 4000, highCutHz: 4000)
    }
    #expect(throws: FrequencyBandError.self) {
        try FrequencyBand(lowCutHz: 5000, highCutHz: 1000)
    }
}

@Test func frequencyBandRejectsOutOfBounds() {
    #expect(throws: FrequencyBandError.self) {
        try FrequencyBand(lowCutHz: -1, highCutHz: 1000)
    }
    #expect(throws: FrequencyBandError.self) {
        try FrequencyBand(lowCutHz: 100, highCutHz: 24001)
    }
}
