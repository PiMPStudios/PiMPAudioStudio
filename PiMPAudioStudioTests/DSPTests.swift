import Testing
@testable import PiMPAudioStudio

@Test func normalizeRMSScalesBufferToTargetDBFS() throws {
    let sourceSamples: [Float] = [0.10, -0.10, 0.20, -0.20, 0.10, -0.10, 0.20, -0.20]
    let source = AudioBufferConverter.fromFloats(sourceSamples, sampleRate: 48_000, channels: 1)

    let processed = try SimpleDSP.normalize(source, mode: .rms(targetDB: -12.0))
    let processedSamples = AudioBufferConverter.toFloats(processed)

    #expect(abs(SimpleDSP.rmsDBFS(processedSamples) - -12.0) < 0.02)
}

@Test func normalizeRMSCapsUpwardGainBeforeClipping() throws {
    let sourceSamples: [Float] = [0.90, -0.20, 0.10, -0.10]
    let source = AudioBufferConverter.fromFloats(sourceSamples, sampleRate: 48_000, channels: 1)

    let processed = try SimpleDSP.normalize(source, mode: .rms(targetDB: -1.0))
    let processedSamples = AudioBufferConverter.toFloats(processed)
    let peak = processedSamples.map(abs).max() ?? 0

    #expect(peak <= 1.0)
    #expect(abs(peak - 1.0) < 0.001)
}

@Test func wsolaTimeStretchKeepsFinalWindowTailForCompressedOutput() {
    let frameCount = 8_192
    let samples = (0..<frameCount).map { index in
        Float((index % 256) - 128) / 128.0
    }

    let stretched = WSOLAProcessor.stretch(
        samples,
        rate: 2.0,
        sampleRate: 48_000,
        channels: 1
    )

    #expect(stretched.count == 4_096)
}

@Test func reverbTailIncludesDecayAndPreDelayWhenWetSignalIsPresent() {
    let parameters = ReverbParameters(decayTime: 0.25, wetDryMix: 0.5, preDelay: 50, damping: 0.5)

    let tailFrames = AVAudioEngineDSP.reverbTailFrameCount(parameters: parameters, sampleRate: 48_000)

    #expect(tailFrames == 14_400)
}

@Test func reverbTailIsSkippedForDryOnlyMix() {
    let parameters = ReverbParameters(decayTime: 2.0, wetDryMix: 0, preDelay: 100, damping: 0.5)

    let tailFrames = AVAudioEngineDSP.reverbTailFrameCount(parameters: parameters, sampleRate: 48_000)

    #expect(tailFrames == 0)
}

@Test func reverbDampingLowersWetPathLowPassCutoff() {
    let open = AVAudioEngineDSP.reverbDampingCutoffHz(forDamping: 0, sampleRate: 48_000)
    let damped = AVAudioEngineDSP.reverbDampingCutoffHz(forDamping: 1, sampleRate: 48_000)

    #expect(open == 20_000)
    #expect(damped == 2_000)
    #expect(damped < open)
}

@Test func reverbRenderKeepsDecayTailForWholeTrack() throws {
    let sampleRate = 48_000
    var samples = [Float](repeating: 0, count: 2_048)
    samples[0] = 0.8
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: sampleRate, channels: 1)
    let parameters = ReverbParameters(decayTime: 0.1, wetDryMix: 0.5, preDelay: 10, damping: 0.5)

    let processed = try AVAudioEngineDSP.applyReverb(to: source, parameters: parameters, target: .wholeTrack)

    #expect(processed.frameCount == source.frameCount + AVAudioEngineDSP.reverbTailFrameCount(
        parameters: parameters,
        sampleRate: sampleRate
    ))
}

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
