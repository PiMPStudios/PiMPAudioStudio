import AVFoundation
import Foundation
import MLX
import PiMPAudioCore

@main
struct QualityEval {
    static func main() async throws {
        do {
            let options = try Options.parse(CommandLine.arguments)
            if options.showHelp {
                print(Options.help)
                return
            }

            let latent = try loadSafetensorArray(
                at: options.latentURL,
                key: options.latentKey,
                description: "latent"
            )
            let referenceAudio = try loadReferenceAudio(options)
            let decodedAudio = try await decodeVAE(
                weightsURL: options.weightsURL,
                latent: latent,
                precision: options.precision
            )

            let metrics = try AudioParityMetrics.compare(
                swiftAudio: decodedAudio,
                referenceAudio: referenceAudio,
                sampleRate: options.sampleRate
            )

            let report = makeMarkdownReport(
                options: options,
                decodedAudio: try AudioParityMetrics.normalizeToNLC(decodedAudio),
                referenceAudio: try AudioParityMetrics.normalizeToNLC(referenceAudio),
                metrics: metrics
            )

            if let reportURL = options.reportURL {
                try FileManager.default.createDirectory(
                    at: reportURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try report.write(to: reportURL, atomically: true, encoding: .utf8)
                print("Wrote report: \(reportURL.path(percentEncoded: false))")
            } else {
                print(report)
            }
        } catch let error as QualityEvalError {
            FileHandle.standardError.write(Data("QualityEval error: \(error.localizedDescription)\n".utf8))
            FileHandle.standardError.write(Data("\n\(Options.help)\n".utf8))
            Foundation.exit(2)
        } catch {
            FileHandle.standardError.write(Data("QualityEval error: \(error.localizedDescription)\n".utf8))
            Foundation.exit(1)
        }
    }
}

private struct Options {
    var showHelp = false
    var caseID = "vae-parity"
    var weightsURL = defaultVAEWeightsURL()
    var latentURL: URL?
    var latentKey = "latents"
    var referenceURL: URL?
    var referenceKey = "audio"
    var reportURL: URL?
    var sampleRate = 48_000
    var precision: Precision = .float32

    static let help = """
    QualityEval VAE parity harness

    Required:
      --latent PATH              Safetensors file containing the upstream latent tensor
      --reference PATH           Upstream decoded WAV, or safetensors with an audio tensor

    Common:
      --weights PATH             ACE-Step VAE safetensors (defaults to the app model cache)
      --case ID                  Short label for the report
      --report PATH              Write Markdown report to this path instead of stdout
      --latent-key KEY           Tensor key in latent safetensors (default: latents)
      --reference-key KEY        Tensor key for safetensors reference audio (default: audio)
      --precision float32|float16
      --sample-rate HZ           Reference sample rate, expected 48000 for ACE-Step

    Example:
      swift run --package-path PiMPAudioCore QualityEval \\
        --case slow-dark-guitar-riff-direct \\
        --latent /tmp/anvil-quality/direct-latents.safetensors \\
        --reference /tmp/anvil-quality/direct-upstream.wav \\
        --report docs/quality-eval/reports/slow-dark-guitar-riff-direct.md
    """

    static func parse(_ arguments: [String]) throws -> Options {
        var options = Options()
        var index = 1

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "-h", "--help":
                options.showHelp = true
                index += 1
            case "--case":
                options.caseID = try value(after: argument, in: arguments, at: &index)
            case "--weights":
                options.weightsURL = URL(fileURLWithPath: expandPath(try value(after: argument, in: arguments, at: &index)))
            case "--latent":
                options.latentURL = URL(fileURLWithPath: expandPath(try value(after: argument, in: arguments, at: &index)))
            case "--latent-key":
                options.latentKey = try value(after: argument, in: arguments, at: &index)
            case "--reference":
                options.referenceURL = URL(fileURLWithPath: expandPath(try value(after: argument, in: arguments, at: &index)))
            case "--reference-key":
                options.referenceKey = try value(after: argument, in: arguments, at: &index)
            case "--report":
                options.reportURL = URL(fileURLWithPath: expandPath(try value(after: argument, in: arguments, at: &index)))
            case "--sample-rate":
                let raw = try value(after: argument, in: arguments, at: &index)
                guard let sampleRate = Int(raw), sampleRate > 0 else {
                    throw QualityEvalError.invalidArgument("--sample-rate must be a positive integer")
                }
                options.sampleRate = sampleRate
            case "--precision":
                let raw = try value(after: argument, in: arguments, at: &index)
                switch raw {
                case "float32":
                    options.precision = .float32
                case "float16":
                    options.precision = .float16
                default:
                    throw QualityEvalError.invalidArgument("--precision must be float32 or float16")
                }
            default:
                throw QualityEvalError.invalidArgument("Unknown argument: \(argument)")
            }
        }

        if !options.showHelp {
            guard options.latentURL != nil else {
                throw QualityEvalError.invalidArgument("Missing required --latent")
            }
            guard options.referenceURL != nil else {
                throw QualityEvalError.invalidArgument("Missing required --reference")
            }
        }

        return options
    }

    private static func value(after flag: String, in arguments: [String], at index: inout Int) throws -> String {
        let valueIndex = index + 1
        guard valueIndex < arguments.count else {
            throw QualityEvalError.invalidArgument("Missing value after \(flag)")
        }
        index += 2
        return arguments[valueIndex]
    }
}

private enum QualityEvalError: Error, LocalizedError {
    case invalidArgument(String)
    case missingFile(URL)
    case missingTensor(path: URL, key: String)
    case unsupportedReference(URL)
    case unsupportedWAVFormat(URL)
    case referenceSampleRate(path: URL, expected: Int, actual: Double)

    var errorDescription: String? {
        switch self {
        case .invalidArgument(let message):
            return message
        case .missingFile(let url):
            return "File does not exist: \(url.path(percentEncoded: false))"
        case .missingTensor(let path, let key):
            return "Tensor key '\(key)' not found in \(path.path(percentEncoded: false))"
        case .unsupportedReference(let url):
            return "Unsupported reference format: \(url.lastPathComponent). Use .wav or .safetensors."
        case .unsupportedWAVFormat(let url):
            return "Could not read WAV samples as Float32: \(url.path(percentEncoded: false))"
        case .referenceSampleRate(let path, let expected, let actual):
            return "Reference sample rate mismatch for \(path.lastPathComponent): expected \(expected), got \(actual)"
        }
    }
}

private func decodeVAE(weightsURL: URL, latent: MLXArray, precision: Precision) async throws -> MLXArray {
    try requireFile(weightsURL)
    let converter = WeightConverter()
    let rawWeights = try await converter.loadSafetensors(from: weightsURL)
    let convertedWeights = try await converter.convertVAEWeights(stateDict: rawWeights)

    let actor = VAEActor()
    await actor.loadWeights(convertedWeights, modelID: "quality-eval")
    let decoded = try await actor.decode(latents: latent, precision: precision)
    eval(decoded)
    return decoded
}

private func loadReferenceAudio(_ options: Options) throws -> MLXArray {
    guard let referenceURL = options.referenceURL else {
        throw QualityEvalError.invalidArgument("Missing required --reference")
    }
    try requireFile(referenceURL)

    switch referenceURL.pathExtension.lowercased() {
    case "wav":
        return try loadWAV(referenceURL, expectedSampleRate: options.sampleRate)
    case "safetensors":
        return try loadSafetensorArray(
            at: referenceURL,
            key: options.referenceKey,
            description: "reference audio"
        )
    default:
        throw QualityEvalError.unsupportedReference(referenceURL)
    }
}

private func loadSafetensorArray(at optionalURL: URL?, key: String, description: String) throws -> MLXArray {
    guard let url = optionalURL else {
        throw QualityEvalError.invalidArgument("Missing \(description) path")
    }
    try requireFile(url)
    let arrays = try MLX.loadArrays(url: url)
    guard let array = arrays[key] else {
        throw QualityEvalError.missingTensor(path: url, key: key)
    }
    eval(array)
    return array
}

private func loadWAV(_ url: URL, expectedSampleRate: Int) throws -> MLXArray {
    let file = try AVAudioFile(forReading: url)
    let format = file.processingFormat
    guard abs(format.sampleRate - Double(expectedSampleRate)) < 0.5 else {
        throw QualityEvalError.referenceSampleRate(
            path: url,
            expected: expectedSampleRate,
            actual: format.sampleRate
        )
    }

    guard let buffer = AVAudioPCMBuffer(
        pcmFormat: format,
        frameCapacity: AVAudioFrameCount(file.length)
    ) else {
        throw QualityEvalError.unsupportedWAVFormat(url)
    }
    try file.read(into: buffer)

    guard let floatChannelData = buffer.floatChannelData else {
        throw QualityEvalError.unsupportedWAVFormat(url)
    }

    let frames = Int(buffer.frameLength)
    let channels = Int(format.channelCount)
    var samples = [Float]()
    samples.reserveCapacity(frames * channels)

    for frame in 0..<frames {
        for channel in 0..<channels {
            samples.append(floatChannelData[channel][frame])
        }
    }

    return MLXArray(samples, [1, frames, channels])
}

private func makeMarkdownReport(
    options: Options,
    decodedAudio: MLXArray,
    referenceAudio: MLXArray,
    metrics: AudioComparisonMetrics
) -> String {
    let weightsPath = options.weightsURL.path(percentEncoded: false)
    let latentPath = options.latentURL?.path(percentEncoded: false) ?? "missing"
    let referencePath = options.referenceURL?.path(percentEncoded: false) ?? "missing"

    return """
    # VAE Parity Report: \(options.caseID)

    **Generated:** \(ISO8601DateFormatter().string(from: Date()))
    **Precision:** \(options.precision)
    **Sample rate:** \(options.sampleRate) Hz
    **Compared duration:** \(format(Double(metrics.durationSeconds))) s

    ## Artifacts

    - Weights: `\(weightsPath)`
    - Latent: `\(latentPath)` (`\(options.latentKey)`)
    - Reference: `\(referencePath)`

    ## Shapes

    | Tensor | Shape |
    | --- | --- |
    | Swift decoded audio | `\(decodedAudio.shape)` |
    | Upstream reference audio | `\(referenceAudio.shape)` |
    | Compared window | `[\(metrics.comparedFrames), \(metrics.comparedChannels)]` |

    ## Metrics

    | Metric | Value |
    | --- | ---: |
    | Mean absolute error | \(format(metrics.meanAbsoluteError)) |
    | Mean squared error | \(format(metrics.meanSquaredError)) |
    | RMSE | \(format(metrics.rootMeanSquaredError)) |
    | Max absolute difference | \(format(metrics.maxAbsoluteDifference)) |
    | Swift RMS | \(format(metrics.swiftRMS)) |
    | Reference RMS | \(format(metrics.referenceRMS)) |
    | Swift high-frequency proxy | \(format(metrics.swiftHighFrequencyProxy)) |
    | Reference high-frequency proxy | \(format(metrics.referenceHighFrequencyProxy)) |
    | Swift stereo correlation | \(formatOptional(metrics.swiftStereoCorrelation)) |
    | Reference stereo correlation | \(formatOptional(metrics.referenceStereoCorrelation)) |
    | RMS envelope correlation | \(formatOptional(metrics.rmsEnvelopeCorrelation)) |

    ## Listening Notes

    - Swift:
    - Reference:
    - Difference:

    ## Initial Read

    \(initialRead(metrics))
    """
}

private func initialRead(_ metrics: AudioComparisonMetrics) -> String {
    if metrics.maxAbsoluteDifference < 1e-4 && metrics.rootMeanSquaredError < 1e-5 {
        return "VAE decode appears numerically aligned for this artifact. Move the next investigation to the next parity case or DiT parity."
    }
    if metrics.rmsEnvelopeCorrelation ?? 0 > 0.98 && metrics.rootMeanSquaredError < 0.01 {
        return "Decode is close in broad dynamics but not bit-level aligned. Inspect VAE dtype, padding, and final scaling before blaming DiT quality."
    }
    return "Decode differs materially. Treat VAE parity as the next bottleneck until this report is explained."
}

private func requireFile(_ url: URL) throws {
    guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
        throw QualityEvalError.missingFile(url)
    }
}

private func defaultVAEWeightsURL() -> URL {
    let path = "~/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/models/ace-step-vae/vae.safetensors"
    return URL(fileURLWithPath: expandPath(path))
}

private func expandPath(_ path: String) -> String {
    (path as NSString).expandingTildeInPath
}

private func format(_ value: Float) -> String {
    String(format: "%.8g", value)
}

private func format(_ value: Double) -> String {
    String(format: "%.3f", value)
}

private func formatOptional(_ value: Float?) -> String {
    guard let value else { return "n/a" }
    return format(value)
}
