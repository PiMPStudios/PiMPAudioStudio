import Foundation
import MLX

public struct WAVWriter {
    public static func write(
        samples: MLXArray,
        sampleRate: Int = 44100,
        to url: URL
    ) throws {
        // The MLX VAE decoder returns NLC layout [batch, length, channels].
        // Some training/inference paths return [channels, frames] (CL) or
        // a flat 1-D mono buffer. Normalize to a single shape we understand.
        let channels: Int
        let numFrames: Int
        let normalized: MLXArray

        switch samples.ndim {
        case 1:
            // Mono buffer of audio samples.
            channels = 1
            numFrames = samples.shape[0]
            normalized = samples
        case 2:
            // [channels, frames] — keep as-is; interleave below.
            channels = samples.shape[0]
            numFrames = samples.shape[1]
            normalized = samples
        case 3:
            // [batch, length, channels] from the MLX VAE decoder. We always
            // generate batch=1, so take batch 0 to land on [length, channels].
            channels = samples.shape[2]
            numFrames = samples.shape[1]
            normalized = samples[0]
        default:
            channels = 0
            numFrames = 0
            normalized = samples
            assertionFailure("WAVWriter received samples of unexpected rank \(samples.ndim)")
        }

        let clamped = clip(normalized, min: -1.0, max: 1.0)
        let scaled = (clamped * Float(32767)).asType(.int16)
        eval(scaled)

        let interleaved: MLXArray
        switch samples.ndim {
        case 1:
            // Already a flat mono buffer.
            interleaved = scaled.reshaped([-1])
        case 2:
            // [channels, frames] -> transpose to [frames, channels] -> flatten LRLR…
            interleaved = scaled.transposed(0, 1).reshaped([-1])
        default:
            // [length, channels] (after the [0] indexing above) -> row-major flatten
            // already produces LRLR… interleaving.
            interleaved = scaled.reshaped([-1])
        }
        eval(interleaved)

        let pcmData = interleaved.asData(access: .copy).data

        let bitsPerSample: UInt16 = 16
        let blockAlign = UInt16(channels) * (bitsPerSample / 8)
        let byteRate = UInt32(sampleRate) * UInt32(blockAlign)
        let dataSize = UInt32(pcmData.count)
        let fileSize = 36 + dataSize

        var header = Data(capacity: 44)
        header.append(contentsOf: "RIFF".utf8)
        header.append(littleEndian: fileSize)
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        header.append(littleEndian: UInt32(16))
        header.append(littleEndian: UInt16(1))
        header.append(littleEndian: UInt16(channels))
        header.append(littleEndian: UInt32(sampleRate))
        header.append(littleEndian: byteRate)
        header.append(littleEndian: blockAlign)
        header.append(littleEndian: bitsPerSample)
        header.append(contentsOf: "data".utf8)
        header.append(littleEndian: dataSize)

        var fileData = header
        fileData.append(pcmData)
        try fileData.write(to: url)
    }
}

private extension Data {
    mutating func append(littleEndian value: UInt16) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 2))
    }

    mutating func append(littleEndian value: UInt32) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 4))
    }
}
