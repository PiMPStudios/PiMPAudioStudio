import Accelerate

enum WSOLAProcessor {
    private static let windowSize = 1024
    private static let overlapRatio: Float = 0.75
    private static let analysisHop = 256
    private static let searchRange = 64
    private static let silenceThresholdDB: Float = -60.0

    static func stretch(_ samples: [Float], rate: Float, sampleRate: Int, channels: Int) -> [Float] {
        guard rate != 1.0 else { return samples }

        let frameCount = samples.count / channels
        guard frameCount >= windowSize * 2 else { return samples }

        if channels == 1 {
            return stretchChannel(samples, frameCount: frameCount, rate: rate)
        }

        var deinterleaved = [[Float]](repeating: [Float](repeating: 0, count: frameCount), count: channels)
        for frame in 0 ..< frameCount {
            for ch in 0 ..< channels {
                deinterleaved[ch][frame] = samples[frame * channels + ch]
            }
        }

        var stretchedChannels = [[Float]]()
        for ch in 0 ..< channels {
            stretchedChannels.append(stretchChannel(deinterleaved[ch], frameCount: frameCount, rate: rate))
        }

        let outputFrames = stretchedChannels[0].count
        var interleaved = [Float](repeating: 0, count: outputFrames * channels)
        for frame in 0 ..< outputFrames {
            for ch in 0 ..< channels {
                interleaved[frame * channels + ch] = stretchedChannels[ch][frame]
            }
        }
        return interleaved
    }

    private static func stretchChannel(_ input: [Float], frameCount: Int, rate: Float) -> [Float] {
        let synthesisHop = Int(Float(analysisHop) / rate)
        let outputLength = Int(Float(frameCount) / rate)
        var output = [Float](repeating: 0, count: outputLength)
        var window = [Float](repeating: 0, count: windowSize)
        vDSP_hann_window(&window, vDSP_Length(windowSize), Int32(vDSP_HANN_NORM))

        var analysisPos = 0
        var synthesisPos = 0
        var writtenEnd = 0

        while synthesisPos + windowSize <= outputLength, analysisPos + windowSize <= frameCount {
            let bestPos: Int = if isFrameSilent(input, at: analysisPos) {
                analysisPos
            } else if synthesisPos == 0 {
                analysisPos
            } else {
                findBestMatch(
                    input: input,
                    frameCount: frameCount,
                    idealPos: analysisPos,
                    prevOutput: output,
                    prevEnd: synthesisPos
                )
            }

            for i in 0 ..< windowSize {
                let srcIdx = bestPos + i
                guard srcIdx < frameCount else { break }
                output[synthesisPos + i] += input[srcIdx] * window[i]
            }
            writtenEnd = max(writtenEnd, synthesisPos + windowSize)

            analysisPos += analysisHop
            synthesisPos += synthesisHop
        }

        normalizeOverlap(&output, synthesisHop: synthesisHop, window: window)
        return Array(output.prefix(min(outputLength, max(synthesisPos, writtenEnd))))
    }

    private static func findBestMatch(
        input: [Float],
        frameCount: Int,
        idealPos: Int,
        prevOutput: [Float],
        prevEnd: Int
    ) -> Int {
        let searchStart = max(0, idealPos - searchRange)
        let searchEnd = min(frameCount - windowSize, idealPos + searchRange)
        guard searchStart < searchEnd else { return idealPos }

        let refStart = max(0, prevEnd - analysisHop)
        let refLength = min(analysisHop, prevEnd - refStart)
        guard refLength > 0 else { return idealPos }

        var bestPos = idealPos
        var bestCorrelation: Float = -.greatestFiniteMagnitude

        prevOutput.withUnsafeBufferPointer { outputPtr in
            input.withUnsafeBufferPointer { inputPtr in
                for candidate in searchStart ... searchEnd {
                    let overlapLength = min(refLength, frameCount - candidate)
                    guard overlapLength > 0 else { continue }

                    var correlation: Float = 0
                    vDSP_dotpr(
                        outputPtr.baseAddress! + refStart,
                        1,
                        inputPtr.baseAddress! + candidate,
                        1,
                        &correlation,
                        vDSP_Length(overlapLength)
                    )

                    if correlation > bestCorrelation {
                        bestCorrelation = correlation
                        bestPos = candidate
                    }
                }
            }
        }
        return bestPos
    }

    private static func isFrameSilent(_ samples: [Float], at position: Int) -> Bool {
        let end = min(position + windowSize, samples.count)
        let length = end - position
        guard length > 0 else { return true }

        var rms: Float = 0
        samples.withUnsafeBufferPointer { ptr in
            vDSP_rmsqv(ptr.baseAddress! + position, 1, &rms, vDSP_Length(length))
        }

        let db = 20.0 * log10(max(rms, 1e-10))
        return db < silenceThresholdDB
    }

    private static func normalizeOverlap(_ output: inout [Float], synthesisHop: Int, window: [Float]) {
        let windowSize = window.count
        var envelope = [Float](repeating: 0, count: output.count)
        var pos = 0
        while pos + windowSize <= output.count {
            for i in 0 ..< windowSize {
                envelope[pos + i] += window[i]
            }
            pos += synthesisHop
        }

        for i in 0 ..< output.count {
            if envelope[i] > 0.001 {
                output[i] /= envelope[i]
            }
        }
    }
}
