import Testing
import AVFoundation
import Foundation
import MLX
@testable import PiMPAudioCore

@Suite("WAVWriter")
struct WAVWriterTests {
    @Test("Writes valid WAV header and PCM data")
    func writesValidWAV() throws {
        let numFrames = 44100
        let samples = MLXArray.zeros([2, numFrames])
        let url = FileManager.default.temporaryDirectory
            .appending(path: "test_output-\(UUID().uuidString).wav")

        try WAVWriter.write(samples: samples, sampleRate: 44100, to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let data = try Data(contentsOf: url)
        let expectedSize = 44 + (2 * numFrames * 2)
        #expect(data.count == expectedSize)

        let riff = String(data: data[0..<4], encoding: .ascii)
        #expect(riff == "RIFF")

        let wave = String(data: data[8..<12], encoding: .ascii)
        #expect(wave == "WAVE")

        let sampleRate = data[24...27].withUnsafeBytes { $0.load(as: UInt32.self) }
        #expect(sampleRate == 44100)

        let channels = data[22...23].withUnsafeBytes { $0.load(as: UInt16.self) }
        #expect(channels == 2)
    }

    @Test("Produces a WAV file AVFoundation can open")
    func producesAVFoundationReadableWAV() throws {
        let sampleRate = 44100
        let numFrames = 4096
        let tone = (0..<numFrames).map { index in
            Double((index % 64) - 32) / 128.0
        }
        let samples = MLXArray(converting: tone, [1, numFrames])
        let url = FileManager.default.temporaryDirectory
            .appending(path: "test_readable-\(UUID().uuidString).wav")

        try WAVWriter.write(samples: samples, sampleRate: sampleRate, to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let audioFile = try AVAudioFile(forReading: url)
        #expect(audioFile.length == AVAudioFramePosition(numFrames))
        #expect(audioFile.fileFormat.sampleRate == Double(sampleRate))
        #expect(audioFile.fileFormat.channelCount == 1)
    }

    @Test("Clamps samples to [-1, 1] range")
    func clampsSamples() throws {
        let samples = MLXArray(converting: [2.0, -2.0, 0.5, -0.5], [1, 4])
        let url = FileManager.default.temporaryDirectory
            .appending(path: "test_clamp-\(UUID().uuidString).wav")

        try WAVWriter.write(samples: samples, sampleRate: 44100, to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let data = try Data(contentsOf: url)
        let firstSample = data[44...45].withUnsafeBytes { $0.load(as: Int16.self) }
        let secondSample = data[46...47].withUnsafeBytes { $0.load(as: Int16.self) }
        #expect(firstSample == 32767)
        #expect(secondSample == -32767)
    }
}
