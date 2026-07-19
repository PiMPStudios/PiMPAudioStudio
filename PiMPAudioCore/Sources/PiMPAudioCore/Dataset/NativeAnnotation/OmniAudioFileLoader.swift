import AVFoundation
import Foundation

public struct LoadedOmniAudio: Sendable {
    public let samples: [Float]
    public let sampleRate: Int

    public init(samples: [Float], sampleRate: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

public enum OmniAudioFileLoaderError: LocalizedError {
    case allocationFailed(String)
    case conversionFailed(String)
    case missingFloatData
    case emptyAudio

    public var errorDescription: String? {
        switch self {
        case .allocationFailed(let detail):
            return "Audio buffer allocation failed: \(detail)"
        case .conversionFailed(let detail):
            return "Audio conversion failed: \(detail)"
        case .missingFloatData:
            return "The decoded audio buffer does not contain Float32 channel data."
        case .emptyAudio:
            return "The audio file contains no samples."
        }
    }
}

public enum OmniAudioFileLoader {
    public static func loadMono16k(from url: URL) throws -> LoadedOmniAudio {
        try loadMono(from: url, sampleRate: OmniAudioConfiguration.qwen25Omni.sampleRate)
    }

    public static func loadMono(from url: URL, sampleRate: Int) throws -> LoadedOmniAudio {
        let didStartAccessing = url.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat
        let sourceFrameCount = AVAudioFrameCount(file.length)
        guard sourceFrameCount > 0 else {
            throw OmniAudioFileLoaderError.emptyAudio
        }
        guard let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: sourceFrameCount
        ) else {
            throw OmniAudioFileLoaderError.allocationFailed(
                "source buffer (\(sourceFrameCount) frames)"
            )
        }
        try file.read(into: sourceBuffer, frameCount: sourceFrameCount)

        guard let destinationFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw OmniAudioFileLoaderError.conversionFailed(
                "could not create \(sampleRate) Hz mono Float32 format"
            )
        }

        let destinationBuffer: AVAudioPCMBuffer
        if sourceFormat.sampleRate == Double(sampleRate),
           sourceFormat.channelCount == 1,
           sourceFormat.commonFormat == .pcmFormatFloat32,
           !sourceFormat.isInterleaved
        {
            destinationBuffer = sourceBuffer
        } else {
            guard let converter = AVAudioConverter(
                from: sourceFormat,
                to: destinationFormat
            ) else {
                throw OmniAudioFileLoaderError.conversionFailed(
                    "AVAudioConverter initialization failed"
                )
            }

            let ratio = Double(sampleRate) / sourceFormat.sampleRate
            let capacity = AVAudioFrameCount(ceil(Double(sourceFrameCount) * ratio) + 32)
            guard let converted = AVAudioPCMBuffer(
                pcmFormat: destinationFormat,
                frameCapacity: capacity
            ) else {
                throw OmniAudioFileLoaderError.allocationFailed(
                    "destination buffer (\(capacity) frames)"
                )
            }

            var suppliedInput = false
            var conversionError: NSError?
            let status = converter.convert(
                to: converted,
                error: &conversionError
            ) { _, outputStatus in
                if suppliedInput {
                    outputStatus.pointee = .endOfStream
                    return nil
                }
                suppliedInput = true
                outputStatus.pointee = .haveData
                return sourceBuffer
            }
            if let conversionError {
                throw conversionError
            }
            guard status != .error else {
                throw OmniAudioFileLoaderError.conversionFailed(
                    "AVAudioConverter returned status \(status.rawValue)"
                )
            }
            destinationBuffer = converted
        }

        guard let channel = destinationBuffer.floatChannelData?[0] else {
            throw OmniAudioFileLoaderError.missingFloatData
        }
        let frameCount = Int(destinationBuffer.frameLength)
        guard frameCount > 0 else {
            throw OmniAudioFileLoaderError.emptyAudio
        }
        return LoadedOmniAudio(
            samples: Array(UnsafeBufferPointer(start: channel, count: frameCount)),
            sampleRate: sampleRate
        )
    }
}
