import Testing
import Foundation
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

@Test func truePeakDetectsIntersamplePeakLinearInterpolationCannotSee() {
    // fs/4 sine sampled half a sample off its crests: every sample sits at
    // sin(45°)·A, so the continuous waveform peaks ~3.01 dB above the sample peak.
    // Linear interpolation can never report this — an intermediate on the segment
    // between two samples is bounded by max(|a|, |b|).
    let samples = (0..<4_000).map { index -> Float in
        0.5 * sin(2 * .pi * (Float(index) + 0.5) * 0.25)
    }
    let samplePeak = LoudnessMeter.peakDBFS(samples)
    let truePeak = LoudnessMeter.truePeakDBFS(samples, channels: 1)

    #expect(truePeak - samplePeak > 2.5)
    #expect(truePeak - samplePeak < 3.2)
}

@Test func truePeakMatchesSamplePeakOnWellSampledSignal() {
    // A slow sine is densely sampled, so there is almost nothing between samples.
    let samples = (0..<4_000).map { index -> Float in
        0.6 * sin(Float(index) * 0.01)
    }
    let overshoot = LoudnessMeter.truePeakDBFS(samples, channels: 1)
        - LoudnessMeter.peakDBFS(samples)

    #expect(overshoot >= 0)
    #expect(overshoot < 0.1)
}

@Test func truePeakScansEveryChannelIndependently() {
    // Intersample content only in the right channel must still be reported.
    var samples = [Float](repeating: 0, count: 4_000 * 2)
    for frame in 0..<4_000 {
        samples[frame * 2] = 0.2 * sin(Float(frame) * 0.01)
        samples[frame * 2 + 1] = 0.5 * sin(2 * .pi * (Float(frame) + 0.5) * 0.25)
    }
    let truePeak = LoudnessMeter.truePeakDBFS(samples, channels: 2)

    #expect(truePeak - LoudnessMeter.peakDBFS(samples) > 2.5)
}

@Test func approximateLUFSSumsChannelPowersLikeBS1770() {
    let mono = (0..<4_800).map { index -> Float in 0.3 * sin(Float(index) * 0.05) }
    var stereo = [Float](repeating: 0, count: mono.count * 2)
    for (index, value) in mono.enumerated() {
        stereo[index * 2] = value
        stereo[index * 2 + 1] = value
    }

    let monoLUFS = LoudnessMeter.approximateLUFS(mono, channels: 1)
    let stereoLUFS = LoudnessMeter.approximateLUFS(stereo, channels: 2)

    // BS.1770 sums per-channel powers (G = 1.0 for L/R). Averaging across interleaved
    // samples instead would read identical L/R the *same* as mono, i.e. 3.01 dB low,
    // which made the loudness stage over-boost every stereo track.
    #expect(abs((stereoLUFS - monoLUFS) - 3.0103) < 0.01)
}

@Test func int16RoundTripReproducesItsInputExactly() {
    let values: [Int16] = [0, 1, -1, 7, -7, 1_000, -1_000, 20_000, -20_000, 32_767, -32_767]
    var data = Data()
    for value in values {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }
    let buffer = AudioBuffer(data: data, sampleRate: 48_000, channels: 1)

    // A DSP stage that changes nothing must really change nothing — otherwise every
    // no-op stage in a chain quietly erodes the signal.
    #expect(FloatAudio(buffer).toAudioBuffer().data == buffer.data)
}

@Test func monoPeakReflectsMidChannel() {
    // Out-of-phase stereo: sample peaks high, mono mid cancels
    let samples: [Float] = [
        0.8, -0.8,
        0.8, -0.8,
        0.8, -0.8,
        0.8, -0.8,
    ]
    let monoPeak = LoudnessMeter.monoPeakDBFS(samples, channels: 2)
    #expect(monoPeak < -40)
}

@Test func foldToMonoAveragesLeftAndRight() {
    let samples: [Float] = [0.5, 0.3, 0.0, 1.0]
    let stereo = AudioBufferConverter.fromFloats(samples, sampleRate: 48_000, channels: 2)
    let mono = LoudnessMeter.foldToMono(stereo)
    #expect(mono.channels == 1)
    #expect(mono.frameCount == 2)
    let out = AudioBufferConverter.toFloats(mono)
    #expect(abs(out[0] - 0.4) < 0.01)
    #expect(abs(out[1] - 0.5) < 0.01)
}

@Test func softGateExpansionReducesBelowThreshold() {
    let gr = DynamicsProcessor.expansionGainReductionDB(
        levelDB: -60,
        thresholdDB: -48,
        ratio: 2,
        rangeDB: 24
    )
    // 12 dB under at 2:1 → GR = -12 * (1 - 0.5) = -6
    #expect(abs(gr - (-6.0)) < 0.01)
}

@Test func softGateDoesNotExpandAboveThreshold() {
    let gr = DynamicsProcessor.expansionGainReductionDB(
        levelDB: -30,
        thresholdDB: -48,
        ratio: 2,
        rangeDB: 12
    )
    #expect(abs(gr) < 0.001)
}

@Test func softGateAttenuatesQuietTail() throws {
    var samples = [Float](repeating: 0.5, count: 1_000)
    samples.append(contentsOf: [Float](repeating: 0.01, count: 2_000))
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: 48_000, channels: 1)

    let gated = try DynamicsProcessor.softGate(
        source,
        parameters: SoftGateParameters(
            thresholdDBFS: -30,
            ratio: 4,
            attackMilliseconds: 1,
            releaseMilliseconds: 20,
            rangeDB: 24
        )
    )
    let out = AudioBufferConverter.toFloats(gated)
    let quietTail = out.suffix(500).map(abs).reduce(0, +) / 500
    let loudHead = out.prefix(200).map(abs).reduce(0, +) / 200
    #expect(quietTail < 0.01)
    #expect(loudHead > 0.3)
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

@Test func compressorGainReductionIsNegativeAboveThreshold() {
    let gr = DynamicsProcessor.gainReductionDB(
        levelDB: -6,
        thresholdDB: -18,
        ratio: 4,
        kneeDB: 0
    )
    // 12 dB over threshold at 4:1 → GR = 12 * (1/4 - 1) = -9
    #expect(abs(gr - (-9.0)) < 0.01)
}

@Test func compressorDoesNotReduceBelowThreshold() {
    let gr = DynamicsProcessor.gainReductionDB(
        levelDB: -24,
        thresholdDB: -18,
        ratio: 4,
        kneeDB: 0
    )
    #expect(abs(gr) < 0.001)
}

@Test func limiterEnforcesCeiling() throws {
    let samples: [Float] = [0.99, -0.99, 0.5, -0.5]
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: 48_000, channels: 1)
    let limited = try DynamicsProcessor.limit(
        source,
        parameters: LimiterParameters(ceilingDBFS: -1.0, releaseMilliseconds: 20)
    )
    let peak = LoudnessMeter.peakDBFS(AudioBufferConverter.toFloats(limited))
    #expect(peak <= -0.95)
}

@Test func loudnessStagingRaisesQuietMaterial() throws {
    // Quiet tone
    let samples = (0..<4_800).map { i -> Float in
        0.02 * sin(Float(i) * 0.05)
    }
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: 48_000, channels: 1)
    let before = LoudnessMeter.approximateLUFS(AudioBufferConverter.toFloats(source))

    let staged = try DynamicsProcessor.aimLoudness(
        source,
        parameters: LoudnessParameters(targetLUFS: -14, ceilingDBFS: -1.0, maxBoostDB: 24)
    )
    let after = LoudnessMeter.approximateLUFS(AudioBufferConverter.toFloats(staged))
    let peak = LoudnessMeter.peakDBFS(AudioBufferConverter.toFloats(staged))

    #expect(after > before + 3)
    #expect(peak <= -0.9)
}

@Test func compressThenLimitPipelineViaProcessor() throws {
    let processor = DSPProcessor()
    let samples = (0..<2_000).map { i -> Float in
        let t = Float(i)
        return 0.7 * sin(t * 0.1) + 0.2 * sin(t * 0.03)
    }
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: 48_000, channels: 1)

    let compressed = try processor.apply(
        effect: .compress,
        to: source,
        target: .wholeTrack,
        parameters: .compress(.light)
    )
    let limited = try processor.apply(
        effect: .limit,
        to: compressed,
        target: .wholeTrack,
        parameters: .limit(.streaming)
    )
    let peak = LoudnessMeter.peakDBFS(AudioBufferConverter.toFloats(limited))
    #expect(peak <= -0.9)
}

@Test func polishForKeepGentleLeavesPeakHeadroom() throws {
    let sampleRate = 48_000
    // Hot signal with silent pads
    var samples = [Float](repeating: 0, count: sampleRate / 2)
    for i in (sampleRate / 10)..<(sampleRate / 2 - sampleRate / 20) {
        samples[i] = 0.95 * sin(Float(i) * 0.08)
    }
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: sampleRate, channels: 1)

    let polished = try PolishForKeepProcessor.apply(
        source,
        recipe: PolishForKeepRecipe(
            target: .gentle,
            tone: .neutral,
            glue: .off,
            space: .off,
            trimSilenceAndFades: true
        )
    )

    #expect(polished.frameCount < source.frameCount)
    let peak = LoudnessMeter.peakDBFS(AudioBufferConverter.toFloats(polished))
    #expect(abs(peak - (-3.0)) < 0.25)
}

@Test func polishForKeepStreamingLimitsPeak() throws {
    let samples = (0..<8_000).map { i -> Float in
        0.6 * sin(Float(i) * 0.07)
    }
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: 48_000, channels: 1)

    let polished = try PolishForKeepProcessor.apply(
        source,
        recipe: PolishForKeepRecipe(
            target: .streaming,
            tone: .brighter,
            glue: .light,
            space: .off,
            trimSilenceAndFades: false
        )
    )
    let peak = LoudnessMeter.peakDBFS(AudioBufferConverter.toFloats(polished))
    #expect(peak <= -0.9)
}

@Test func polishForKeepViaDSPProcessor() throws {
    let processor = DSPProcessor()
    let samples = (0..<4_000).map { i -> Float in
        0.4 * sin(Float(i) * 0.05)
    }
    let source = AudioBufferConverter.fromFloats(samples, sampleRate: 48_000, channels: 1)

    let result = try processor.apply(
        effect: .polishForKeep,
        to: source,
        target: .wholeTrack,
        parameters: .polishForKeep(.default)
    )
    #expect(result.frameCount > 0)
    let peak = LoudnessMeter.peakDBFS(AudioBufferConverter.toFloats(result))
    #expect(peak <= -0.9)
}

@Test func loudnessStagingLimitsBoostedPeaksInFloat() throws {
    let sampleRate = 48_000
    // Quiet bed plus sparse transients: the LUFS boost pushes peaks well past full
    // scale, so this only behaves correctly if the limiter sees un-clipped float.
    let samples = (0..<sampleRate).map { index -> Float in
        let t = Float(index) / Float(sampleRate)
        var value = 0.05 * sin(2 * .pi * 220 * t)
        if index % 4_800 < 120 { value += 0.45 * sin(2 * .pi * 2_000 * t) }
        return value
    }
    let source = FloatAudio(samples: samples, sampleRate: sampleRate, channels: 1)
    let parameters = LoudnessParameters(targetLUFS: -14, ceilingDBFS: -1.0, maxBoostDB: 24)

    // Reference: same gain, then limit, with no Int16 hop in between.
    let current = LoudnessMeter.approximateLUFS(samples, channels: 1)
    let delta = min(-14 - current, 24)
    let gained = source.replacingSamples(samples.map { $0 * pow(10, delta / 20) })
    let reference = try DynamicsProcessor.limit(
        gained,
        parameters: LimiterParameters(ceilingDBFS: -1.0, releaseMilliseconds: 50)
    )

    // Guard the premise: the boost really does overshoot full scale.
    #expect(gained.peakAmplitude() > 1.5)

    let staged = try DynamicsProcessor.aimLoudness(source, parameters: parameters)
    let maxDeviation = zip(staged.samples, reference.samples)
        .map { abs($0 - $1) }
        .max() ?? 0

    #expect(maxDeviation < 1e-5)
    #expect(LoudnessMeter.peakDBFS(staged.samples) <= -0.9)
}

@Test func polishForKeepStreamingLandsOnTargetForStereo() throws {
    let sampleRate = 48_000
    let frames = 24_000
    var samples = [Float](repeating: 0, count: frames * 2)
    for frame in 0..<frames {
        let value = 0.25 * sin(Float(frame) * 0.06)
        samples[frame * 2] = value
        samples[frame * 2 + 1] = value
    }
    let source = FloatAudio(samples: samples, sampleRate: sampleRate, channels: 2)

    let polished = try PolishForKeepProcessor.apply(
        source,
        recipe: PolishForKeepRecipe(
            target: .streaming,
            tone: .neutral,
            glue: .light,
            space: .off,
            trimSilenceAndFades: false
        )
    )

    #expect(polished.channels == 2)
    #expect(LoudnessMeter.peakDBFS(polished.samples) <= -0.9)
    // This material never reaches the ceiling, so the gain stage lands exactly on
    // target. A channel-averaged meter would leave it ~3 dB hot instead.
    let lufs = LoudnessMeter.approximateLUFS(polished.samples, channels: 2)
    #expect(abs(lufs - (-14)) < 0.5)
}

@Test func polishForKeepAppliesDarkToneAndSpaceWithinCeiling() throws {
    let sampleRate = 48_000
    let samples = (0..<12_000).map { index -> Float in 0.4 * sin(Float(index) * 0.05) }
    let source = FloatAudio(samples: samples, sampleRate: sampleRate, channels: 1)

    let polished = try PolishForKeepProcessor.apply(
        source,
        recipe: PolishForKeepRecipe(
            target: .louder,
            tone: .darker,
            glue: .medium,
            space: .room,
            trimSilenceAndFades: false
        )
    )

    // Space runs before the limiter, so the wet tail lengthens the take.
    #expect(polished.frameCount > source.frameCount)
    #expect(LoudnessMeter.peakDBFS(polished.samples) <= -0.4)
}

@Test func polishForKeepToneEQOnlyWhenNotNeutral() {
    #expect(PolishForKeepProcessor.toneEQParameters(for: .neutral) == nil)
    #expect(PolishForKeepProcessor.toneEQParameters(for: .brighter) != nil)
    #expect(PolishForKeepProcessor.compressorParameters(for: .off) == nil)
    #expect(PolishForKeepProcessor.compressorParameters(for: .light) != nil)
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
