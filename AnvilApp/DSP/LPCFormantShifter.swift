import Accelerate

struct LPCFormantShifter {
    private static let lpcOrder = 24
    private static let frameSize = 1024
    private static let hopSize = 256
    private static let silenceThresholdDB: Float = -60.0

    static func correct(original: [Float], shifted: [Float],
                        sampleRate: Int, channels: Int) -> [Float] {
        let origFrameCount = original.count / channels
        let shiftFrameCount = shifted.count / channels

        guard origFrameCount >= frameSize, shiftFrameCount >= frameSize else {
            return shifted
        }

        if channels == 1 {
            return correctChannel(original: original, shifted: shifted,
                                  origFrames: origFrameCount, shiftFrames: shiftFrameCount)
        }

        var origDeinterleaved = [[Float]](repeating: [Float](repeating: 0, count: origFrameCount), count: channels)
        var shiftDeinterleaved = [[Float]](repeating: [Float](repeating: 0, count: shiftFrameCount), count: channels)

        for frame in 0..<origFrameCount {
            for ch in 0..<channels {
                origDeinterleaved[ch][frame] = original[frame * channels + ch]
            }
        }
        for frame in 0..<shiftFrameCount {
            for ch in 0..<channels {
                shiftDeinterleaved[ch][frame] = shifted[frame * channels + ch]
            }
        }

        var correctedChannels = [[Float]]()
        for ch in 0..<channels {
            correctedChannels.append(
                correctChannel(original: origDeinterleaved[ch], shifted: shiftDeinterleaved[ch],
                               origFrames: origFrameCount, shiftFrames: shiftFrameCount)
            )
        }

        let outputFrames = correctedChannels[0].count
        var interleaved = [Float](repeating: 0, count: outputFrames * channels)
        for frame in 0..<outputFrames {
            for ch in 0..<channels {
                interleaved[frame * channels + ch] = correctedChannels[ch][frame]
            }
        }
        return interleaved
    }

    private static func correctChannel(original: [Float], shifted: [Float],
                                        origFrames: Int, shiftFrames: Int) -> [Float] {
        var output = [Float](repeating: 0, count: shiftFrames)
        var envelope = [Float](repeating: 0, count: shiftFrames)

        var window = [Float](repeating: 0, count: frameSize)
        vDSP_hann_window(&window, vDSP_Length(frameSize), Int32(vDSP_HANN_NORM))

        let numFrames = min((origFrames - frameSize) / hopSize, (shiftFrames - frameSize) / hopSize) + 1

        for frameIdx in 0..<numFrames {
            let origStart = frameIdx * hopSize
            let shiftStart = frameIdx * hopSize

            let origFrame = Array(original[origStart..<origStart + frameSize])
            let shiftFrame = Array(shifted[shiftStart..<shiftStart + frameSize])

            if isFrameSilent(shiftFrame) {
                for i in 0..<frameSize where shiftStart + i < shiftFrames {
                    output[shiftStart + i] += shiftFrame[i] * window[i]
                    envelope[shiftStart + i] += window[i]
                }
                continue
            }

            var windowedOrig = [Float](repeating: 0, count: frameSize)
            vDSP_vmul(origFrame, 1, window, 1, &windowedOrig, 1, vDSP_Length(frameSize))

            var windowedShift = [Float](repeating: 0, count: frameSize)
            vDSP_vmul(shiftFrame, 1, window, 1, &windowedShift, 1, vDSP_Length(frameSize))

            let origCoeffs = levinsonDurbin(autocorrelation(windowedOrig))
            let shiftCoeffs = levinsonDurbin(autocorrelation(windowedShift))

            let residual = inverseFilter(windowedShift, coefficients: shiftCoeffs)
            let corrected = synthesisFilter(residual, coefficients: origCoeffs)

            for i in 0..<frameSize where shiftStart + i < shiftFrames {
                output[shiftStart + i] += corrected[i] * window[i]
                envelope[shiftStart + i] += window[i]
            }
        }

        for i in 0..<shiftFrames {
            if envelope[i] > 0.001 {
                output[i] /= envelope[i]
            } else {
                output[i] = shifted[i]
            }
        }

        return output
    }

    private static func autocorrelation(_ frame: [Float]) -> [Float] {
        let n = frame.count
        let order = lpcOrder + 1
        var result = [Float](repeating: 0, count: order)

        for lag in 0..<order {
            var sum: Float = 0
            vDSP_dotpr(Array(frame[0..<n - lag]), 1,
                       Array(frame[lag..<n]), 1,
                       &sum, vDSP_Length(n - lag))
            result[lag] = sum
        }
        return result
    }

    private static func levinsonDurbin(_ r: [Float]) -> [Float] {
        let order = r.count - 1
        guard order > 0, r[0] > 0 else {
            return [Float](repeating: 0, count: order)
        }

        var a = [Float](repeating: 0, count: order)
        var aPrev = [Float](repeating: 0, count: order)
        var error = r[0]

        for i in 0..<order {
            var sum: Float = r[i + 1]
            for j in 0..<i {
                sum += aPrev[j] * r[i - j]
            }

            var k = -sum / error
            k = min(max(k, -0.99), 0.99)

            a[i] = k
            for j in 0..<i {
                a[j] = aPrev[j] + k * aPrev[i - 1 - j]
            }

            error *= (1.0 - k * k)
            if error < 1e-10 { break }

            aPrev = a
        }
        return a
    }

    private static func inverseFilter(_ signal: [Float], coefficients: [Float]) -> [Float] {
        let n = signal.count
        let order = coefficients.count
        var residual = [Float](repeating: 0, count: n)

        for i in 0..<n {
            var prediction: Float = 0
            for j in 0..<min(order, i) {
                prediction += coefficients[j] * signal[i - 1 - j]
            }
            residual[i] = signal[i] - prediction
        }
        return residual
    }

    private static func synthesisFilter(_ residual: [Float], coefficients: [Float]) -> [Float] {
        let n = residual.count
        let order = coefficients.count
        var output = [Float](repeating: 0, count: n)

        for i in 0..<n {
            var prediction: Float = 0
            for j in 0..<min(order, i) {
                prediction += coefficients[j] * output[i - 1 - j]
            }
            output[i] = residual[i] + prediction
        }
        return output
    }

    private static func isFrameSilent(_ frame: [Float]) -> Bool {
        var rms: Float = 0
        vDSP_rmsqv(frame, 1, &rms, vDSP_Length(frame.count))
        let db = 20.0 * log10(max(rms, 1e-10))
        return db < silenceThresholdDB
    }
}
