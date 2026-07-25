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

@Test func normalizePeakTargetsMinusOneDBFS() throws {
    // Peak 0.5 → target -1 dBFS (~0.891)
    let sourceSamples: [Float] = [0.5, -0.25, 0.1, -0.1]
    let source = AudioBufferConverter.fromFloats(sourceSamples, sampleRate: 48_000, channels: 1)

    let processed = try SimpleDSP.normalize(source, mode: .peak(targetDBFS: -1.0))
    let peak = LoudnessMeter.peakDBFS(AudioBufferConverter.toFloats(processed))

    #expect(abs(peak - (-1.0)) < 0.05)
}

@Test func normalizePeakScalesDownHotFiles() throws {
    let sourceSamples: [Float] = [0.99, -0.99, 0.5, -0.5]
    let source = AudioBufferConverter.fromFloats(sourceSamples, sampleRate: 48_000, channels: 1)

    let processed = try SimpleDSP.normalize(source, mode: .peak(targetDBFS: -3.0))
    let peak = LoudnessMeter.peakDBFS(AudioBufferConverter.toFloats(processed))

    #expect(abs(peak - (-3.0)) < 0.08)
}

@Test func trimSilenceRemovesLeadingAndTrailingPadding() throws {
    let sampleRate = 1_000
    let channels = 1
    // 100 silent, 200 tone, 100 silent
    var samples = [Float](repeating: 0, count: 400)
    for i in 100..<300 {
        samples[i] = 0.5
    }
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: sampleRate, channels: channels)

    let processed = try SimpleDSP.trimSilence(
        source,
        parameters: TrimSilenceParameters(
            thresholdDBFS: -40,
            maxLeadingSeconds: 1,
            maxTrailingSeconds: 1,
            edgeFadeMilliseconds: 0
        )
    )

    #expect(processed.frameCount == 200)
}

@Test func trimSilenceRespectsMaxLeadCap() throws {
    let sampleRate = 100
    var samples = [Float](repeating: 0, count: 300)
    for i in 200..<300 {
        samples[i] = 0.8
    }
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: sampleRate, channels: 1)

    // 2.0s of silence at start but max lead 0.5s → keep 1.5s silence + 1s tone
    let processed = try SimpleDSP.trimSilence(
        source,
        parameters: TrimSilenceParameters(
            thresholdDBFS: -40,
            maxLeadingSeconds: 0.5,
            maxTrailingSeconds: 1,
            edgeFadeMilliseconds: 0
        )
    )

    #expect(processed.frameCount == 250)
}

@Test func highPassAttenuatesNearDCContent() throws {
    let sampleRate = 48_000
    // Constant DC should be killed by HPF
    let samples = [Float](repeating: 0.4, count: sampleRate / 10)
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: sampleRate, channels: 1)

    let processed = try SimpleDSP.highPass(
        source,
        parameters: HighPassParameters(cutoffHz: 80)
    )
    let out = AudioBufferConverter.toFloats(processed)
    // After settling, residual should be near zero
    let tail = out.suffix(2_000)
    let residual = tail.map(abs).reduce(0, +) / Float(tail.count)

    #expect(residual < 0.02)
}

@Test func loudnessMeterReportsHotPeak() {
    let samples: [Float] = [0.0, 0.5, -1.0, 0.25]
    let peak = LoudnessMeter.peakDBFS(samples)
    #expect(abs(peak - 0.0) < 0.01)
}

@Test func dspProcessorAppliesNormalizeParameters() throws {
    let processor = DSPProcessor()
    let samples: [Float] = [0.25, -0.25, 0.1, -0.1]
    let buffer = AudioBufferConverter.fromFloats(samples, sampleRate: 48_000, channels: 1)

    let result = try processor.apply(
        effect: .normalize,
        to: buffer,
        target: .wholeTrack,
        parameters: .normalize(NormalizeParameters(targetPeakDBFS: -1.0))
    )
    let peak = LoudnessMeter.peakDBFS(AudioBufferConverter.toFloats(result))
    #expect(abs(peak - (-1.0)) < 0.08)
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

@Test func stereoFadeInUsesSameGainPerFrame() throws {
    // Interleaved stereo: L0 R0 L1 R1 ... constant 1.0
    let channels = 2
    let sampleRate = 100
    let frames = 10
    let samples = [Float](repeating: 1.0, count: frames * channels)
    let buffer = AudioBufferConverter.fromFloats(samples, sampleRate: sampleRate, channels: channels)
    let faded = try SimpleDSP.fadeIn(buffer, start: 0, end: 0.1) // full 10 frames
    let out = AudioBufferConverter.toFloats(faded)

    // Every frame should apply the same gain to L and R
    for frame in 0..<frames {
        let l = out[frame * channels]
        let r = out[frame * channels + 1]
        #expect(abs(l - r) < 1e-5, "frame \(frame): L=\(l) R=\(r) should match")
    }
    // First frame near 0, last frame near 1
    #expect(out[0] < 0.15)
    #expect(out[(frames - 1) * channels] > 0.85)
}

@Test func dspProcessorRejectsMissingEQParameters() throws {
    let processor = DSPProcessor()
    let samples = [Float](repeating: 0.1, count: 256)
    let buffer = AudioBufferConverter.fromFloats(samples, sampleRate: 48_000, channels: 1)
    #expect(throws: DSPProcessor.ProcessingError.self) {
        try processor.apply(effect: .eq, to: buffer, target: .wholeTrack, parameters: .none)
    }
}
