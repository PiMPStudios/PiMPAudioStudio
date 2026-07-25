import Foundation
import Accelerate

/// Level readouts for the editor polish path.
///
/// - Sample peak: max |sample| in the buffer
/// - True-peak estimate: max after 4× polyphase windowed-sinc oversampling
///   (BS.1770-style interpolation; not a certified meter, but it does surface
///   real intersample peaks)
/// - Mono peak: peak of (L+R)/2 on stereo (mono-compatibility cue)
/// - Approximate LUFS: ungated mean-square with BS.1770 channel summation and the
///   −0.691 offset, but no K-weighting filter (aiming only)
enum LoudnessMeter {
    struct Metrics: Equatable, Sendable {
        var peakDBFS: Float
        var truePeakDBFS: Float
        var monoPeakDBFS: Float?
        var approximateLUFS: Float
        var duration: TimeInterval
        var sampleRate: Int
        var channels: Int

        var isStereo: Bool { channels >= 2 }

        /// Headroom under a 0 dBFS full-scale sample peak.
        var samplePeakHeadroomDB: Float { -peakDBFS }

        /// True-peak overshoot vs sample peak (dB); positive means TP hotter than sample peak.
        var truePeakOvershootDB: Float { truePeakDBFS - peakDBFS }
    }

    static func measure(_ buffer: AudioBuffer) -> Metrics {
        measure(FloatAudio(buffer))
    }

    static func measure(_ audio: FloatAudio) -> Metrics {
        let samples = audio.samples
        let channels = audio.channels
        return Metrics(
            peakDBFS: peakDBFS(samples),
            truePeakDBFS: truePeakDBFS(samples, channels: channels),
            monoPeakDBFS: channels >= 2 ? monoPeakDBFS(samples, channels: channels) : nil,
            approximateLUFS: approximateLUFS(samples, channels: channels),
            duration: audio.duration,
            sampleRate: audio.sampleRate,
            channels: audio.channels
        )
    }

    static func peakDBFS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return -120 }
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        return linearToDBFS(peak)
    }

    // MARK: - True peak

    /// Oversampling factor of the polyphase interpolator.
    private static let oversampleFactor = 4
    /// Input samples used on each side of an interpolated position.
    private static let kernelHalfSpan = 12
    private static var kernelTaps: Int { kernelHalfSpan * 2 }

    /// Windowed-sinc kernels for fractional phases 1/4, 2/4, 3/4.
    ///
    /// Phase 0 is the identity (the sample itself), so it is not stored. Linear
    /// interpolation is deliberately **not** used here: a linear intermediate always
    /// lies on the segment between its two neighbours, so it can never exceed
    /// `max(|a|, |b|)` and can never reveal an intersample peak.
    private static let interpolationKernels: [[Float]] = makeInterpolationKernels()

    /// L1 norm of the widest kernel — bounds how far an interpolated value can exceed
    /// the largest input sample in its span. Used to skip blocks that provably cannot
    /// contain the true peak.
    private static let kernelL1Norm: Float = {
        interpolationKernels
            .map { $0.reduce(0) { $0 + abs($1) } }
            .max() ?? 1
    }()

    private static func makeInterpolationKernels() -> [[Float]] {
        let taps = kernelHalfSpan * 2
        var kernels: [[Float]] = []
        kernels.reserveCapacity(oversampleFactor - 1)

        for phase in 1..<oversampleFactor {
            let fraction = Double(phase) / Double(oversampleFactor)
            var kernel = [Double](repeating: 0, count: taps)
            var sum = 0.0

            for tap in 0..<taps {
                // Tap `tap` multiplies the input sample at offset (tap - kernelHalfSpan + 1).
                let offset = Double(tap - kernelHalfSpan + 1)
                let x = fraction - offset
                let sinc = abs(x) < 1e-9 ? 1.0 : sin(.pi * x) / (.pi * x)
                let position = Double(tap) / Double(taps - 1)
                // Blackman window keeps sidelobes low enough that the L1 bound stays tight.
                let window = 0.42
                    - 0.5 * cos(2.0 * .pi * position)
                    + 0.08 * cos(4.0 * .pi * position)
                kernel[tap] = sinc * window
                sum += kernel[tap]
            }

            // Unity DC gain so a constant signal interpolates to itself.
            if abs(sum) > 1e-12 {
                for tap in 0..<taps { kernel[tap] /= sum }
            }
            kernels.append(kernel.map { Float($0) })
        }
        return kernels
    }

    /// 4× polyphase windowed-sinc peak estimate.
    ///
    /// Only blocks that can possibly beat the running best are convolved: an
    /// interpolated value is bounded by `max|x| within its span × ‖kernel‖₁`, so once
    /// blocks are visited in descending order of local maximum the scan exits early.
    /// This keeps whole-track metering cheap on multi-minute buffers without
    /// approximating the result.
    static func truePeakDBFS(_ samples: [Float], channels: Int = 1) -> Float {
        guard !samples.isEmpty else { return -120 }
        let channelCount = max(channels, 1)
        let frameCount = samples.count / channelCount
        guard frameCount > 1 else { return peakDBFS(samples) }

        // Phase 0 of the bank is the identity — start from the plain sample peak.
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))

        let taps = kernelTaps
        let blockSize = max(4_096, taps * 4)
        let blockCount = (frameCount + blockSize - 1) / blockSize

        var channelSamples = [Float](repeating: 0, count: frameCount)
        var padded = [Float](repeating: 0, count: blockSize + taps - 1)
        var phaseOutput = [Float](repeating: 0, count: blockSize)
        var blockMaxima = [Float](repeating: 0, count: blockCount)

        for channel in 0..<channelCount {
            for frame in 0..<frameCount {
                channelSamples[frame] = samples[frame * channelCount + channel]
            }

            for block in 0..<blockCount {
                let start = block * blockSize
                let count = min(blockSize, frameCount - start)
                var blockMax: Float = 0
                channelSamples.withUnsafeBufferPointer { pointer in
                    vDSP_maxmgv(pointer.baseAddress! + start, 1, &blockMax, vDSP_Length(count))
                }
                blockMaxima[block] = blockMax
            }

            // A block's outputs also read its neighbours' edges, so bound with the halo.
            let bounds: [(index: Int, bound: Float)] = (0..<blockCount).map { block in
                var span = blockMaxima[block]
                if block > 0 { span = max(span, blockMaxima[block - 1]) }
                if block + 1 < blockCount { span = max(span, blockMaxima[block + 1]) }
                return (block, span * kernelL1Norm)
            }.sorted { $0.bound > $1.bound }

            for entry in bounds {
                // Sorted descending: once one block cannot beat `peak`, none can.
                if entry.bound <= peak { break }

                let start = entry.index * blockSize
                let count = min(blockSize, frameCount - start)
                let paddedCount = count + taps - 1

                for i in 0..<paddedCount {
                    let sourceFrame = start + i - (kernelHalfSpan - 1)
                    padded[i] = (sourceFrame >= 0 && sourceFrame < frameCount)
                        ? channelSamples[sourceFrame]
                        : 0
                }

                for kernel in interpolationKernels {
                    vDSP_conv(
                        padded, 1,
                        kernel, 1,
                        &phaseOutput, 1,
                        vDSP_Length(count),
                        vDSP_Length(taps)
                    )
                    var phasePeak: Float = 0
                    vDSP_maxmgv(phaseOutput, 1, &phasePeak, vDSP_Length(count))
                    peak = max(peak, phasePeak)
                }
            }
        }

        return linearToDBFS(peak)
    }

    /// Peak of the mono mid signal (L+R)/2 for stereo compatibility checks.
    static func monoPeakDBFS(_ samples: [Float], channels: Int) -> Float {
        let channelCount = max(channels, 1)
        guard channelCount >= 2, !samples.isEmpty else { return peakDBFS(samples) }
        let frameCount = samples.count / channelCount
        var peak: Float = 0
        for frame in 0..<frameCount {
            let base = frame * channelCount
            let mid = 0.5 * (samples[base] + samples[base + 1])
            peak = max(peak, abs(mid))
        }
        return linearToDBFS(peak)
    }

    /// Fold interleaved stereo (or multi-channel) to mono mid for preview.
    static func foldToMono(_ buffer: AudioBuffer) -> AudioBuffer {
        let audio = FloatAudio(buffer)
        guard audio.channels >= 2 else { return buffer }
        return foldToMono(audio).toAudioBuffer()
    }

    static func foldToMono(_ audio: FloatAudio) -> FloatAudio {
        guard audio.channels >= 2 else { return audio }
        let channels = audio.channels
        let frameCount = audio.frameCount
        var mono = [Float](repeating: 0, count: frameCount)
        for frame in 0..<frameCount {
            let base = frame * channels
            // Average first two channels (typical stereo L/R)
            mono[frame] = 0.5 * (audio.samples[base] + audio.samples[base + 1])
        }
        return FloatAudio(samples: mono, sampleRate: audio.sampleRate, channels: 1)
    }

    /// Ungated mean-square loudness with the −0.691 LUFS offset (no K-filter).
    ///
    /// BS.1770 **sums** per-channel powers (`G = 1.0` for L/R) rather than averaging
    /// them. Averaging across interleaved samples reads a correlated stereo mix
    /// 10·log10(2) ≈ 3.01 dB low, which would make the loudness stage over-boost
    /// every stereo track by that amount.
    static func approximateLUFS(_ samples: [Float], channels: Int = 1) -> Float {
        guard !samples.isEmpty else { return -70 }
        let channelCount = max(channels, 1)
        var meanSquare: Float = 0
        vDSP_measqv(samples, 1, &meanSquare, vDSP_Length(samples.count))
        // mean over interleaved × channelCount == sum of per-channel mean squares.
        let channelSum = meanSquare * Float(channelCount)
        guard channelSum > 1e-12 else { return -70 }
        return -0.691 + 10.0 * log10(channelSum)
    }

    static func approximateLUFS(_ audio: FloatAudio) -> Float {
        approximateLUFS(audio.samples, channels: audio.channels)
    }

    static func formatDBFS(_ value: Float) -> String {
        if value <= -119.5 { return "−∞ dB" }
        return String(format: "%.1f dB", value)
    }

    static func formatLUFS(_ value: Float) -> String {
        if value <= -69.5 { return "−∞ LUFS" }
        return String(format: "%.1f LUFS", value)
    }

    private static func linearToDBFS(_ linear: Float) -> Float {
        guard linear > 1e-10 else { return -120 }
        return 20.0 * log10(linear)
    }
}
