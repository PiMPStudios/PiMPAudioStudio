import AVFoundation
import Accelerate

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
        vDSP.convertElements(of: scaled, to: &int16Samples)

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
