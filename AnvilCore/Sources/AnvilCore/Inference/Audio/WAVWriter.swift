import Foundation
import MLX

public struct WAVWriter {
    public static func write(
        samples: MLXArray,
        sampleRate: Int = 44100,
        to url: URL
    ) throws {
        let channels: Int
        let numFrames: Int

        if samples.ndim == 1 {
            channels = 1
            numFrames = samples.shape[0]
        } else {
            channels = samples.shape[0]
            numFrames = samples.shape[1]
        }

        let clamped = clip(samples, min: -1.0, max: 1.0)
        let scaled = (clamped * Float(32767)).asType(.int16)
        eval(scaled)

        let interleaved: MLXArray
        if channels == 1 {
            interleaved = scaled.reshaped([-1])
        } else {
            interleaved = scaled.transposed(0, 1).reshaped([-1])
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
