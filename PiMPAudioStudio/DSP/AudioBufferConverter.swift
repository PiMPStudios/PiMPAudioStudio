import AVFoundation
import Accelerate
import MLX

struct AudioBufferConverter {
    enum ConversionError: LocalizedError {
        case invalidData
        case bufferCreationFailed

        var errorDescription: String? {
            switch self {
            case .invalidData:
                return "The audio data is not in a supported PCM layout."
            case .bufferCreationFailed:
                return "Could not create an AVAudioPCMBuffer for this audio."
            }
        }
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
        var scale = 1.0 / fullScale
        vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(int16Count))
        return floats
    }

    static func fromFloats(_ samples: [Float], sampleRate: Int, channels: Int) -> AudioBuffer {
        var clamped = samples
        var low: Float = -1.0
        var high: Float = 1.0
        vDSP_vclip(clamped, 1, &low, &high, &clamped, 1, vDSP_Length(samples.count))

        var scaled = [Float](repeating: 0, count: samples.count)
        var scale = fullScale
        vDSP_vsmul(clamped, 1, &scale, &scaled, 1, vDSP_Length(samples.count))

        var int16Samples = [Int16](repeating: 0, count: samples.count)
        // Round, not truncate: `vDSP_vfix16` truncates toward zero, which loses a
        // full LSB on every conversion and biases the signal toward silence.
        vDSP_vfixr16(scaled, 1, &int16Samples, 1, vDSP_Length(samples.count))

        let data = int16Samples.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }
        return AudioBuffer(data: data, sampleRate: sampleRate, channels: channels)
    }

    /// Shared Int16 ↔ float scale.
    ///
    /// Deliberately symmetric: decoding with 1/32768 and re-encoding with 32767 meant
    /// a pure round-trip attenuated slightly and, combined with truncation, could not
    /// reproduce its own input. With one factor both ways `i → float → i` is exact, so
    /// a DSP stage that changes nothing really does change nothing. The cost is that
    /// the single most-negative code (−32768) clamps to −32767.
    private static let fullScale: Float = 32_767.0

    static func toPCMBuffer(_ buffer: AudioBuffer) throws -> AVAudioPCMBuffer {
        try toPCMBuffer(FloatAudio(buffer))
    }

    static func toPCMBuffer(_ audio: FloatAudio) throws -> AVAudioPCMBuffer {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(audio.sampleRate),
            channels: AVAudioChannelCount(audio.channels),
            interleaved: false
        ) else {
            throw ConversionError.bufferCreationFailed
        }

        let frameCount = AVAudioFrameCount(audio.frameCount)
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw ConversionError.bufferCreationFailed
        }
        pcmBuffer.frameLength = frameCount

        let floats = audio.samples
        let channels = audio.channels

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
        floatAudio(from: pcmBuffer).toAudioBuffer()
    }

    static func floatAudio(from pcmBuffer: AVAudioPCMBuffer) -> FloatAudio {
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

        return FloatAudio(
            samples: interleaved,
            sampleRate: Int(pcmBuffer.format.sampleRate),
            channels: channels
        )
    }
}
