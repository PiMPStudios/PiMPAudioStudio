import AVFoundation
import Accelerate
import MLX

struct AudioBufferConverter {
    enum ConversionError: Error {
        case invalidData
        case bufferCreationFailed
    }

    static func toFloats(_ buffer: AudioBuffer) -> [Float] {
        let int16Count = buffer.data.count / MemoryLayout<Int16>.size
        var int16Samples = [Int16](repeating: 0, count: int16Count)
        buffer.data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Int16.self)
            for i in 0..<int16Count {
                int16Samples[i] = src[i]
            }
        }
        var floats = [Float](repeating: 0, count: int16Count)
        vDSP.convertElements(of: int16Samples, to: &floats)
        var scale: Float = 1.0 / 32768.0
        vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(int16Count))
        return floats
    }

    static func fromFloats(_ samples: [Float], sampleRate: Int, channels: Int) -> AudioBuffer {
        var clamped = samples
        var low: Float = -1.0
        var high: Float = 1.0
        vDSP_vclip(clamped, 1, &low, &high, &clamped, 1, vDSP_Length(samples.count))

        var scaled = [Float](repeating: 0, count: samples.count)
        var scale: Float = 32767.0
        vDSP_vsmul(clamped, 1, &scale, &scaled, 1, vDSP_Length(samples.count))

        var int16Samples = [Int16](repeating: 0, count: samples.count)
        vDSP_vfix16(scaled, 1, &int16Samples, 1, vDSP_Length(samples.count))

        let data = int16Samples.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }
        return AudioBuffer(data: data, sampleRate: sampleRate, channels: channels)
    }

    static func toPCMBuffer(_ buffer: AudioBuffer) throws -> AVAudioPCMBuffer {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(buffer.sampleRate),
            channels: AVAudioChannelCount(buffer.channels),
            interleaved: false
        ) else {
            throw ConversionError.bufferCreationFailed
        }

        let frameCount = AVAudioFrameCount(buffer.frameCount)
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw ConversionError.bufferCreationFailed
        }
        pcmBuffer.frameLength = frameCount

        let floats = toFloats(buffer)
        let channels = buffer.channels

        if channels == 1 {
            floats.withUnsafeBufferPointer { src in
                pcmBuffer.floatChannelData![0].update(from: src.baseAddress!, count: Int(frameCount))
            }
        } else {
            for ch in 0..<channels {
                let channelPtr = pcmBuffer.floatChannelData![ch]
                for frame in 0..<Int(frameCount) {
                    channelPtr[frame] = floats[frame * channels + ch]
                }
            }
        }

        return pcmBuffer
    }

    // MARK: - MLX bridge (for VAE encode/decode in the editor)

    /// Convert an `AudioBuffer` to an `MLXArray` with shape `[1, frames, channels]`
    /// (NLC layout expected by OobleckVAE encoder).
    static func toMLXArray(_ buffer: AudioBuffer) -> MLXArray {
        let floats = toFloats(buffer)
        // toFloats returns interleaved [frame0_ch0, frame0_ch1, frame1_ch0, ...]
        // MLXArray with shape [1, L, C] in row-major order maps identically.
        return MLXArray(floats, [1, buffer.frameCount, buffer.channels])
    }

    /// Convert an `MLXArray` with shape `[1, frames, channels]` or `[frames, channels]`
    /// back to an `AudioBuffer`.  The batch dimension (if present) is dropped first.
    static func fromMLXArray(_ array: MLXArray, sampleRate: Int) -> AudioBuffer {
        // Normalise: accept [1, L, C] or [L, C]
        let framesChannels: MLXArray = array.ndim == 3 ? array[0] : array  // [L, C]
        eval(framesChannels)
        let channelCount = framesChannels.ndim >= 2 ? framesChannels.shape[1] : 1
        // Row-major [L, C] bytes are already interleaved — same layout as toFloats output.
        let rawData = framesChannels.asType(.float32).asData(access: .copy).data
        let floats = rawData.withUnsafeBytes { buf in
            Array(buf.bindMemory(to: Float.self))
        }
        return fromFloats(floats, sampleRate: sampleRate, channels: channelCount)
    }

    static func fromPCMBuffer(_ pcmBuffer: AVAudioPCMBuffer) -> AudioBuffer {
        let frameCount = Int(pcmBuffer.frameLength)
        let channels = Int(pcmBuffer.format.channelCount)
        let totalSamples = frameCount * channels

        var interleaved = [Float](repeating: 0, count: totalSamples)

        if channels == 1 {
            let src = pcmBuffer.floatChannelData![0]
            for i in 0..<frameCount {
                interleaved[i] = src[i]
            }
        } else {
            for frame in 0..<frameCount {
                for ch in 0..<channels {
                    interleaved[frame * channels + ch] = pcmBuffer.floatChannelData![ch][frame]
                }
            }
        }

        return fromFloats(interleaved, sampleRate: Int(pcmBuffer.format.sampleRate), channels: channels)
    }
}
