import Foundation
import MLX

public enum OmniAudioFeatureExtractorError: LocalizedError {
    case emptyWaveform
    case waveformTooShort(minimum: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyWaveform:
            return "The waveform is empty."
        case .waveformTooShort(let minimum, let actual):
            return "The waveform has \(actual) samples; at least \(minimum) are required."
        }
    }
}

/// Native Qwen2.5-Omni log-mel preprocessing.
///
/// The reference implementation centers the STFT with reflect padding, uses a
/// periodic Hann window, drops the final time frame, and constructs the Slaney
/// mel bank in Float64 before casting it to Float32. The FFT and remaining
/// tensor math run through MLX in Float32.
public struct OmniAudioFeatureExtractor: Sendable {
    public let configuration: OmniAudioConfiguration
    private let hannWindow: [Float]
    private let melFilters: [Float]

    public init(configuration: OmniAudioConfiguration = .qwen25Omni) throws {
        try configuration.validate()
        self.configuration = configuration
        self.hannWindow = Self.makePeriodicHannWindow(size: configuration.fftSize)
        self.melFilters = Self.makeSlaneyMelFilters(
            sampleRate: configuration.sampleRate,
            fftSize: configuration.fftSize,
            melCount: configuration.featureSize
        )
    }

    public func logMelSpectrogram(waveform source: [Float]) throws -> MLXArray {
        guard !source.isEmpty else {
            throw OmniAudioFeatureExtractorError.emptyWaveform
        }
        guard source.count >= configuration.fftSize / 2 + 1 else {
            throw OmniAudioFeatureExtractorError.waveformTooShort(
                minimum: configuration.fftSize / 2 + 1,
                actual: source.count
            )
        }

        var waveform = source
        let remainder = waveform.count % configuration.hopLength
        if remainder != 0 {
            waveform.append(
                contentsOf: repeatElement(
                    0,
                    count: configuration.hopLength - remainder
                )
            )
        }

        // A centered STFT would produce one extra frame. The Python reference
        // removes that final time frame before applying the mel matrix.
        let frameCount = waveform.count / configuration.hopLength
        let fftSize = configuration.fftSize
        let centerPadding = fftSize / 2
        var framed = [Float](repeating: 0, count: frameCount * fftSize)

        for frame in 0..<frameCount {
            let frameStart = frame * configuration.hopLength - centerPadding
            let outputBase = frame * fftSize
            for bin in 0..<fftSize {
                let sourceIndex = Self.reflectIndex(
                    frameStart + bin,
                    count: waveform.count
                )
                framed[outputBase + bin] = waveform[sourceIndex] * hannWindow[bin]
            }
        }

        let frameArray = MLXArray(framed, [frameCount, fftSize]).asType(.float32)
        let spectrum = MLXFFT.rfft(frameArray, n: fftSize, axis: -1)
        let power = MLX.square(MLX.abs(spectrum)).transposed(1, 0)
        let melMatrix = MLXArray(
            melFilters,
            [configuration.featureSize, fftSize / 2 + 1]
        ).asType(.float32)
        let mel = MLX.matmul(melMatrix, power)
        let clipped = MLX.maximum(mel, Float(1e-10))
        var logSpec = MLX.log10(clipped)
        logSpec = MLX.maximum(logSpec, logSpec.max() - Float(8))
        logSpec = (logSpec + Float(4)) / Float(4)
        eval(logSpec)
        return logSpec
    }

    public func loadAndExtract(from url: URL) throws -> MLXArray {
        let audio = try OmniAudioFileLoader.loadMono(
            from: url,
            sampleRate: configuration.sampleRate
        )
        return try logMelSpectrogram(waveform: audio.samples)
    }

    public static func makePeriodicHannWindow(size: Int) -> [Float] {
        (0..<size).map { index in
            Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(index) / Double(size)))
        }
    }

    public static func makeSlaneyMelFilters(
        sampleRate: Int,
        fftSize: Int,
        melCount: Int
    ) -> [Float] {
        let fftFrequencies = (0...(fftSize / 2)).map {
            Double($0) * Double(sampleRate) / Double(fftSize)
        }
        let minMel = 0.0
        let maxMel = 45.245640471924965
        let melStep = (maxMel - minMel) / Double(melCount + 1)
        let mels = (0..<(melCount + 2)).map {
            minMel + Double($0) * melStep
        }

        let minimumLogHz = 1_000.0
        let frequencySpacing = 200.0 / 3.0
        let minimumLogMel = minimumLogHz / frequencySpacing
        let logStep = log(6.4) / 27.0
        let frequencies = mels.map { mel -> Double in
            if mel >= minimumLogMel {
                return minimumLogHz * exp(logStep * (mel - minimumLogMel))
            }
            return frequencySpacing * mel
        }

        var result = [Float](
            repeating: 0,
            count: melCount * (fftSize / 2 + 1)
        )
        for melIndex in 0..<melCount {
            let lowerSpacing = frequencies[melIndex + 1] - frequencies[melIndex]
            let upperSpacing = frequencies[melIndex + 2] - frequencies[melIndex + 1]
            let energyScale = 2.0 / (frequencies[melIndex + 2] - frequencies[melIndex])
            let lowerCenter = frequencies[melIndex]
            let upperCenter = frequencies[melIndex + 2]

            for (fftIndex, fftFrequency) in fftFrequencies.enumerated() {
                let lower = (fftFrequency - lowerCenter) / lowerSpacing
                let upper = (upperCenter - fftFrequency) / upperSpacing
                result[melIndex * fftFrequencies.count + fftIndex] =
                    Float(max(0, min(lower, upper)) * energyScale)
            }
        }
        return result
    }

    private static func reflectIndex(_ rawIndex: Int, count: Int) -> Int {
        if rawIndex < 0 {
            return -rawIndex
        }
        if rawIndex >= count {
            return 2 * count - rawIndex - 2
        }
        return rawIndex
    }
}
