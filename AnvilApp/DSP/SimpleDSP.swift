import Foundation
import Accelerate

struct SimpleDSP {
    enum DSPError: Error {
        case invalidSelection
        case bufferEmpty
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

    // MARK: - Helpers

    static func sampleIndex(time: TimeInterval, sampleRate: Int, channels: Int) -> Int {
        Int(time * Double(sampleRate)) * channels
    }
}
