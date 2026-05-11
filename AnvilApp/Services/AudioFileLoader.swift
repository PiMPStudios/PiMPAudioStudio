// AnvilApp/Services/AudioFileLoader.swift
import AVFoundation
import MLX
import AnvilCore

/// Loads audio from disk, resamples to 48 kHz stereo, and returns the result
/// as `[1, frames, 2]` in NLC layout (batch=1, length=frames, channels=2).
///
/// Used by both `CoverSongCoordinator` (cover-song preprocessing) and
/// `TrainingService` (LoRA dataset preparation) so that AVFoundation-based
/// resampling isn't duplicated across the two callers.
enum AudioFileLoader {

    enum Error: Swift.Error, LocalizedError {
        case allocationFailed(String)
        case converterFailed(String)
        case noFloatData

        var errorDescription: String? {
            switch self {
            case .allocationFailed(let msg): return "Audio buffer allocation failed: \(msg)"
            case .converterFailed(let msg):  return "Audio conversion failed: \(msg)"
            case .noFloatData:               return "PCM buffer contains no float channel data"
            }
        }
    }

    /// Load the audio file at `url`, resample to 48 kHz stereo, and return
    /// `[1, frames, 2]`. Mono inputs are duplicated to stereo.
    static func load(from url: URL) throws -> MLXArray {
        let file = try AVAudioFile(forReading: url)
        let srcFormat = file.processingFormat
        let srcFrameCount = AVAudioFrameCount(file.length)

        guard let srcBuffer = AVAudioPCMBuffer(
            pcmFormat: srcFormat,
            frameCapacity: srcFrameCount
        ) else {
            throw Error.allocationFailed("source buffer (\(srcFrameCount) frames)")
        }
        try file.read(into: srcBuffer, frameCount: srcFrameCount)

        let targetRate = Double(MusicPipeline.defaultSampleRate)   // 48 000 Hz
        let targetCh: AVAudioChannelCount = 2

        guard let dstFormat = AVAudioFormat(
            standardFormatWithSampleRate: targetRate,
            channels: targetCh
        ) else {
            throw Error.converterFailed("could not create 48 kHz stereo AVAudioFormat")
        }

        // Fast path: already in target format.
        if srcFormat.sampleRate == targetRate && srcFormat.channelCount == targetCh {
            return try bufferToMLX(srcBuffer)
        }

        guard let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else {
            throw Error.converterFailed("AVAudioConverter init failed")
        }

        let ratio = targetRate / srcFormat.sampleRate
        let dstCapacity = AVAudioFrameCount(Double(srcFrameCount) * ratio + 1.0)
        guard let dstBuffer = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: dstCapacity) else {
            throw Error.allocationFailed("destination buffer (\(dstCapacity) frames)")
        }

        // Single-shot conversion: provide the full source on the first call,
        // then signal end-of-stream on subsequent calls.
        var consumed = false
        var convError: NSError?
        let status = converter.convert(to: dstBuffer, error: &convError) { _, outStatus in
            if consumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return srcBuffer
        }

        if let e = convError { throw e }
        guard status != .error else {
            throw Error.converterFailed("AVAudioConverter returned error status \(status.rawValue)")
        }

        return try bufferToMLX(dstBuffer)
    }

    // MARK: - Private

    /// Convert a float-format `AVAudioPCMBuffer` to `[1, frames, 2]` MLXArray.
    /// Mono buffers are duplicated to produce stereo output.
    private static func bufferToMLX(_ buffer: AVAudioPCMBuffer) throws -> MLXArray {
        guard let floatData = buffer.floatChannelData else { throw Error.noFloatData }
        let frames = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)

        let ch0 = Array(UnsafeBufferPointer(start: floatData[0], count: frames))
        let ch1 = channels >= 2
            ? Array(UnsafeBufferPointer(start: floatData[1], count: frames))
            : ch0   // duplicate mono → stereo

        var interleaved = [Float](repeating: 0, count: frames * 2)
        for i in 0..<frames {
            interleaved[i * 2]     = ch0[i]
            interleaved[i * 2 + 1] = ch1[i]
        }
        // [1, frames, 2] — NLC layout required by VAEActor.encodeMean
        return MLXArray(interleaved, [frames, 2]).expandedDimensions(axis: 0)
    }
}
