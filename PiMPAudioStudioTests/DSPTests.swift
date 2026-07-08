import Testing
@testable import PiMPAudioStudio

@Test func parametricEQQConvertsToAVAudioBandwidthOctaves() {
    let bandwidth = AVAudioEngineDSP.bandwidthOctaves(forQ: 1.0)

    #expect(abs(bandwidth - 1.388_483_8) < 0.000_001)
}

@Test func parametricEQHigherQProducesNarrowerBandwidth() {
    let wide = AVAudioEngineDSP.bandwidthOctaves(forQ: 0.8)
    let narrow = AVAudioEngineDSP.bandwidthOctaves(forQ: 10.0)

    #expect(narrow < wide)
    #expect(abs(narrow - 0.144_209_9) < 0.000_001)
}

@Test func parametricEQBandwidthIsClampedToAVAudioRange() {
    #expect(AVAudioEngineDSP.bandwidthOctaves(forQ: 0.1) == 5.0)
    #expect(AVAudioEngineDSP.bandwidthOctaves(forQ: 10_000) == 0.05)
}
