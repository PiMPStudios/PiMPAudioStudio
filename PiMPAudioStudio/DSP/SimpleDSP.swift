import Foundation
import Accelerate

struct SimpleDSP {
    enum DSPError: LocalizedError {
        case invalidSelection
        case bufferEmpty

        var errorDescription: String? {
            switch self {
            case .invalidSelection:
                return "The selected audio range is not valid."
            case .bufferEmpty:
                return "The audio buffer is empty."
            }
        }
    }

    static func trimToSelection(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let bytesPerSample = MemoryLayout<Int16>.size
        let startByte = startSample * bytesPerSample
        let endByte = min(endSample * bytesPerSample, buffer.data.count)

        guard startByte < buffer.data.count else { throw DSPError.invalidSelection }

        let trimmedData = buffer.data[startByte..<endByte]
        return AudioBuffer(data: Data(trimmedData), sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func deleteSelection(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let bytesPerSample = MemoryLayout<Int16>.size
        let startByte = startSample * bytesPerSample
        let endByte = min(endSample * bytesPerSample, buffer.data.count)

        var result = Data()
        result.append(buffer.data[0..<startByte])
        if endByte < buffer.data.count {
            result.append(buffer.data[endByte...])
        }

        guard !result.isEmpty else { throw DSPError.bufferEmpty }
        return AudioBuffer(data: result, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func fadeIn(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = min(sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels), floats.count)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let length = endSample - startSample
        var ramp = [Float](repeating: 0, count: length)
        var start_val: Float = 0.0
        var end_val: Float = 1.0
        vDSP_vgen(&start_val, &end_val, &ramp, 1, vDSP_Length(length))
        floats.withUnsafeMutableBufferPointer { ptr in
            let base = ptr.baseAddress! + startSample
            vDSP_vmul(base, 1, ramp, 1, base, 1, vDSP_Length(length))
        }

        return AudioBufferConverter.fromFloats(floats, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func fadeOut(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = min(sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels), floats.count)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let length = endSample - startSample
        var ramp = [Float](repeating: 0, count: length)
        var start_val: Float = 1.0
        var end_val: Float = 0.0
        vDSP_vgen(&start_val, &end_val, &ramp, 1, vDSP_Length(length))
        floats.withUnsafeMutableBufferPointer { ptr in
            let base = ptr.baseAddress! + startSample
            vDSP_vmul(base, 1, ramp, 1, base, 1, vDSP_Length(length))
        }

        return AudioBufferConverter.fromFloats(floats, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func normalize(_ buffer: AudioBuffer, mode: NormalizeMode = .peak) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        guard !floats.isEmpty else { throw DSPError.bufferEmpty }

        switch mode {
        case .peak:
            var peak: Float = 0
            vDSP_maxmgv(floats, 1, &peak, vDSP_Length(floats.count))
            guard peak > 0.001 else { return buffer }
            guard peak < 0.99 else { return buffer }
            var scale = 1.0 / peak
            vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(floats.count))

        case .rms:
            return buffer
        }

        return AudioBufferConverter.fromFloats(floats, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    // MARK: - Helpers

    static func sampleIndex(time: TimeInterval, sampleRate: Int, channels: Int) -> Int {
        Int(time * Double(sampleRate)) * channels
    }
}
