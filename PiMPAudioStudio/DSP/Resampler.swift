import Foundation

enum Resampler {
    static func resample(_ samples: [Float], ratio: Float, channels: Int) -> [Float] {
        guard ratio != 1.0 else { return samples }

        let clampedRatio = min(max(ratio, 0.25), 4.0)
        let frameCount = samples.count / channels
        let outputFrameCount = Int(Double(frameCount) / Double(clampedRatio))
        guard outputFrameCount > 0 else { return samples }

        if channels == 1 {
            return resampleChannel(
                samples,
                inputFrames: frameCount,
                outputFrames: outputFrameCount,
                ratio: clampedRatio
            )
        }

        var deinterleaved = [[Float]](repeating: [Float](repeating: 0, count: frameCount), count: channels)
        for frame in 0 ..< frameCount {
            for ch in 0 ..< channels {
                deinterleaved[ch][frame] = samples[frame * channels + ch]
            }
        }

        var resampledChannels = [[Float]]()
        for ch in 0 ..< channels {
            resampledChannels.append(
                resampleChannel(
                    deinterleaved[ch],
                    inputFrames: frameCount,
                    outputFrames: outputFrameCount,
                    ratio: clampedRatio
                )
            )
        }

        let actualOutputFrames = resampledChannels[0].count
        var interleaved = [Float](repeating: 0, count: actualOutputFrames * channels)
        for frame in 0 ..< actualOutputFrames {
            for ch in 0 ..< channels {
                interleaved[frame * channels + ch] = resampledChannels[ch][frame]
            }
        }
        return interleaved
    }

    private static func resampleChannel(
        _ input: [Float],
        inputFrames: Int,
        outputFrames: Int,
        ratio: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: outputFrames)
        let doubleRatio = Double(ratio)

        for i in 0 ..< outputFrames {
            let inputPos = Double(i) * doubleRatio
            let floor = Int(inputPos)
            let frac = Float(inputPos - Double(floor))

            let idx0 = min(floor, inputFrames - 1)
            let idx1 = min(floor + 1, inputFrames - 1)

            output[i] = input[idx0] * (1.0 - frac) + input[idx1] * frac
        }
        return output
    }
}
