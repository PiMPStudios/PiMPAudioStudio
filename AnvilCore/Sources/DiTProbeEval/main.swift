import Foundation
import MLX
import AnvilCore

@main
struct DiTProbeEval {
    static func main() async throws {
        do {
            let options = try Options.parse(CommandLine.arguments)
            if options.showHelp {
                print(Options.help)
                return
            }

            let inputs = try loadInputTensors(options.inputsURL)
            let references = try loadReferenceTensors(options.referenceURL)
            let config = try DiTConfig.load(from: options.configURL)
            let weights = try await loadConvertedWeights(options.weightsURL)
            let actor = DiTActor(config: config)
            await actor.loadWeights(weights, modelID: "dit-probe")

            let traces = try await actor.traceStep(
                noisyLatents: inputs.hiddenStates,
                timestep: inputs.timestep,
                timestepR: inputs.timestepR,
                encoderStates: inputs.encoderHiddenStates,
                contextLatents: inputs.contextLatents,
                encoderAttentionMask: inputs.encoderAttentionMask,
                layerIndices: options.layerIndices
            )

            let report = try makeMarkdownReport(
                options: options,
                config: config,
                swiftTensors: traces,
                referenceTensors: references
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
        } catch let error as DiTProbeEvalError {
            FileHandle.standardError.write(Data("DiTProbeEval error: \(error.localizedDescription)\n".utf8))
            FileHandle.standardError.write(Data("\n\(Options.help)\n".utf8))
            Foundation.exit(2)
        } catch {
            FileHandle.standardError.write(Data("DiTProbeEval error: \(error.localizedDescription)\n".utf8))
            Foundation.exit(1)
        }
    }
}

private struct ProbeInputs {
    let hiddenStates: MLXArray
    let timestep: Float
    let timestepR: Float
    let encoderHiddenStates: MLXArray
    let contextLatents: MLXArray
    let encoderAttentionMask: MLXArray?
}

private struct Options {
    var showHelp = false
    var caseID = "dit-probe"
    var inputsURL: URL?
    var referenceURL: URL?
    var weightsURL = defaultDiTWeightsURL()
    var configURL = defaultDiTConfigURL()
    var reportURL: URL?
    var layerIndices: Set<Int> = [0, 1, 23]

    static let help = """
    DiTProbeEval tensor parity harness

    Required:
      --inputs PATH              Safetensors file with fixed DiT forward inputs
      --reference PATH           Safetensors file with upstream reference tensors

    Common:
      --weights PATH             ACE-Step DiT safetensors or model.safetensors.index.json
                                 (defaults to the app's SFT model cache)
      --config PATH              ACE-Step DiT config.json (defaults to the app's SFT model cache)
      --case ID                  Short label for the report
      --layers LIST              Comma-separated layer outputs to compare (default: 0,1,23)
      --report PATH              Write Markdown report to this path instead of stdout

    Input safetensors keys:
      hidden_states              Required. Aliases: noisy_latents, latents, xt
      encoder_hidden_states      Required. Aliases: encoder_states
      context_latents            Required. Aliases: context
      timestep                   Required scalar or [1]
      timestep_r                 Optional scalar or [1], defaults to timestep
      encoder_attention_mask     Optional. Aliases: encoder_mask, encoderMask

    Reference safetensors keys:
      proj_in, condition_embed, layers.N, post_norm_out, output

    Example:
      swift run --package-path AnvilCore DiTProbeEval \\
        --case sparse-dark-guitar-dit-step \\
        --inputs /tmp/anvil-quality/dit-probe-inputs.safetensors \\
        --reference /tmp/anvil-quality/dit-probe-reference.safetensors \\
        --layers 0,1,23 \\
        --report docs/quality-eval/reports/sparse-dark-guitar-dit.md
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
            case "--inputs":
                options.inputsURL = URL(fileURLWithPath: expandPath(try value(after: argument, in: arguments, at: &index)))
            case "--reference":
                options.referenceURL = URL(fileURLWithPath: expandPath(try value(after: argument, in: arguments, at: &index)))
            case "--weights":
                options.weightsURL = URL(fileURLWithPath: expandPath(try value(after: argument, in: arguments, at: &index)))
            case "--config":
                options.configURL = URL(fileURLWithPath: expandPath(try value(after: argument, in: arguments, at: &index)))
            case "--report":
                options.reportURL = URL(fileURLWithPath: expandPath(try value(after: argument, in: arguments, at: &index)))
            case "--layers":
                options.layerIndices = try parseLayerList(try value(after: argument, in: arguments, at: &index))
            default:
                throw DiTProbeEvalError.invalidArgument("Unknown argument: \(argument)")
            }
        }

        if !options.showHelp {
            guard options.inputsURL != nil else {
                throw DiTProbeEvalError.invalidArgument("Missing required --inputs")
            }
            guard options.referenceURL != nil else {
                throw DiTProbeEvalError.invalidArgument("Missing required --reference")
            }
        }

        return options
    }

    private static func value(after flag: String, in arguments: [String], at index: inout Int) throws -> String {
        let valueIndex = index + 1
        guard valueIndex < arguments.count else {
            throw DiTProbeEvalError.invalidArgument("Missing value after \(flag)")
        }
        index += 2
        return arguments[valueIndex]
    }
}

private enum DiTProbeEvalError: Error, LocalizedError {
    case invalidArgument(String)
    case missingFile(URL)
    case missingTensor(path: URL, aliases: [String])
    case emptyTimestep(path: URL, key: String)

    var errorDescription: String? {
        switch self {
        case .invalidArgument(let message):
            return message
        case .missingFile(let url):
            return "File does not exist: \(url.path(percentEncoded: false))"
        case .missingTensor(let path, let aliases):
            return "None of \(aliases) found in \(path.path(percentEncoded: false))"
        case .emptyTimestep(let path, let key):
            return "Timestep tensor '\(key)' is empty in \(path.path(percentEncoded: false))"
        }
    }
}

private func loadInputTensors(_ optionalURL: URL?) throws -> ProbeInputs {
    guard let url = optionalURL else {
        throw DiTProbeEvalError.invalidArgument("Missing required --inputs")
    }
    try requireFile(url)
    let arrays = try MLX.loadArrays(url: url)

    let hiddenStates = try requireTensor(
        in: arrays,
        aliases: ["hidden_states", "noisy_latents", "latents", "xt"],
        path: url
    )
    let encoderHiddenStates = try requireTensor(
        in: arrays,
        aliases: ["encoder_hidden_states", "encoder_states"],
        path: url
    )
    let contextLatents = try requireTensor(
        in: arrays,
        aliases: ["context_latents", "contextLatents", "context"],
        path: url
    )
    let (timestepKey, timestepArray) = try requireNamedTensor(
        in: arrays,
        aliases: ["timestep", "timesteps", "t"],
        path: url
    )
    let timestep = try scalar(from: timestepArray, path: url, key: timestepKey)
    let timestepR = try optionalNamedTensor(
        in: arrays,
        aliases: ["timestep_r", "timestepR", "tr"],
        path: url
    ).map { key, tensor in
        try scalar(from: tensor, path: url, key: key)
    } ?? timestep
    let encoderAttentionMask = try optionalTensor(
        in: arrays,
        aliases: ["encoder_attention_mask", "encoder_mask", "encoderMask"],
        path: url
    )

    eval(hiddenStates, encoderHiddenStates, contextLatents)
    if let encoderAttentionMask {
        eval(encoderAttentionMask)
    }

    return ProbeInputs(
        hiddenStates: hiddenStates,
        timestep: timestep,
        timestepR: timestepR,
        encoderHiddenStates: encoderHiddenStates,
        contextLatents: contextLatents,
        encoderAttentionMask: encoderAttentionMask
    )
}

private func loadReferenceTensors(_ optionalURL: URL?) throws -> [String: MLXArray] {
    guard let url = optionalURL else {
        throw DiTProbeEvalError.invalidArgument("Missing required --reference")
    }
    try requireFile(url)
    let arrays = try MLX.loadArrays(url: url)
    for tensor in arrays.values {
        eval(tensor)
    }
    return arrays
}

private func loadConvertedWeights(_ weightsURL: URL) async throws -> [(String, MLXArray)] {
    try requireFile(weightsURL)
    let converter = WeightConverter()
    let rawWeights: [(String, MLXArray)]
    if weightsURL.lastPathComponent == "model.safetensors.index.json" {
        rawWeights = try await converter.loadShardedSafetensors(indexURL: weightsURL)
    } else {
        rawWeights = try await converter.loadSafetensors(from: weightsURL)
    }
    return try await converter.convertDiTWeights(stateDict: rawWeights)
}

private func makeMarkdownReport(
    options: Options,
    config: DiTConfig,
    swiftTensors: [String: MLXArray],
    referenceTensors: [String: MLXArray]
) throws -> String {
    var lines: [String] = []
    lines.append("# \(options.caseID) DiT Probe Report")
    lines.append("")
    lines.append("Generated: \(Date())")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append("- Inputs: `\(options.inputsURL?.path(percentEncoded: false) ?? "")`")
    lines.append("- Reference: `\(options.referenceURL?.path(percentEncoded: false) ?? "")`")
    lines.append("- Weights: `\(options.weightsURL.path(percentEncoded: false))`")
    lines.append("- Config: `\(options.configURL.path(percentEncoded: false))`")
    lines.append("- Layers: `\(options.layerIndices.sorted().map(String.init).joined(separator: ","))`")
    lines.append("- DiT variant: `\(config.variantFamily)` (`hiddenSize=\(config.hiddenSize)`, `layers=\(config.numHiddenLayers)`, `heads=\(config.numAttentionHeads)`, `kvHeads=\(config.numKeyValueHeads)`)")
    lines.append("")
    lines.append("## Tensor Comparisons")
    lines.append("")
    lines.append("| Tensor | Shape Match | Swift Shape | Reference Shape | Elements | MAE | RMSE | Max Abs Diff | Swift | Ref | Cosine |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for key in orderedKeys(layerIndices: options.layerIndices, swiftTensors: swiftTensors, referenceTensors: referenceTensors) {
        guard let swiftTensor = swiftTensors[key] else {
            lines.append("| `\(key)` | missing-swift | - | \(shapeString(referenceTensors[key]?.shape)) | - | - | - | - | - | - | - |")
            continue
        }
        guard let referenceTensor = referenceTensors[key] else {
            lines.append("| `\(key)` | missing-reference | \(shapeString(swiftTensor.shape)) | - | - | - | - | - | - | - | - |")
            continue
        }

        let metrics = try TensorParityMetrics.compare(
            name: key,
            lhs: swiftTensor,
            rhs: referenceTensor
        )
        lines.append(
            "| `\(key)` | \(metrics.shapesMatch ? "yes" : "no") | \(shapeString(metrics.lhsShape)) | \(shapeString(metrics.rhsShape)) | \(metrics.comparedElements) | \(format(metrics.meanAbsoluteError)) | \(format(metrics.rootMeanSquaredError)) | \(format(metrics.maxAbsoluteDifference)) | \(format(metrics.lhsMeanAbsoluteValue)) | \(format(metrics.rhsMeanAbsoluteValue)) | \(formatOptional(metrics.cosineSimilarity)) |"
        )
    }

    lines.append("")
    lines.append("## How To Read This")
    lines.append("")
    lines.append("- A bad `proj_in` row points at input packing, patching, or Conv1d weight layout.")
    lines.append("- A good `proj_in` row but bad early `layers.N` row points at attention, q/k norms, RoPE, GQA repeat, or masks.")
    lines.append("- Good early rows but diverging late rows points at accumulated attention/mask drift or timestep/modulation details.")
    lines.append("- Good `post_norm_out` but bad `output` points at `projOut` / de-patchify layout.")
    lines.append("")

    return lines.joined(separator: "\n")
}

private func orderedKeys(
    layerIndices: Set<Int>,
    swiftTensors: [String: MLXArray],
    referenceTensors: [String: MLXArray]
) -> [String] {
    var preferred = ["proj_in", "condition_embed"]
    preferred.append(contentsOf: layerIndices.sorted().map { "layers.\($0)" })
    preferred.append(contentsOf: ["post_norm_out", "output"])

    let allKeys = Set(swiftTensors.keys).union(referenceTensors.keys)
    var ordered: [String] = []
    for key in preferred where allKeys.contains(key) {
        ordered.append(key)
    }
    let remaining = allKeys.subtracting(ordered).sorted()
    ordered.append(contentsOf: remaining)
    return ordered
}

private func requireNamedTensor(
    in arrays: [String: MLXArray],
    aliases: [String],
    path: URL
) throws -> (String, MLXArray) {
    for alias in aliases {
        if let tensor = arrays[alias] {
            return (alias, tensor)
        }
    }
    throw DiTProbeEvalError.missingTensor(path: path, aliases: aliases)
}

private func requireTensor(
    in arrays: [String: MLXArray],
    aliases: [String],
    path: URL
) throws -> MLXArray {
    try requireNamedTensor(in: arrays, aliases: aliases, path: path).1
}

private func optionalNamedTensor(
    in arrays: [String: MLXArray],
    aliases: [String],
    path _: URL
) throws -> (String, MLXArray)? {
    for alias in aliases {
        if let tensor = arrays[alias] {
            return (alias, tensor)
        }
    }
    return nil
}

private func optionalTensor(
    in arrays: [String: MLXArray],
    aliases: [String],
    path: URL
) throws -> MLXArray? {
    try optionalNamedTensor(in: arrays, aliases: aliases, path: path)?.1
}

private func scalar(from tensor: MLXArray, path: URL, key: String) throws -> Float {
    let flat = tensor.asType(.float32).reshaped([-1])
    guard TensorParityMetrics.elementCount(tensor.shape) > 0 else {
        throw DiTProbeEvalError.emptyTimestep(path: path, key: key)
    }
    eval(flat)
    return flat[0].item(Float.self)
}

private func parseLayerList(_ raw: String) throws -> Set<Int> {
    let pieces = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    var layers = Set<Int>()
    for piece in pieces where !piece.isEmpty {
        guard let layer = Int(piece), layer >= 0 else {
            throw DiTProbeEvalError.invalidArgument("--layers must be a comma-separated list of non-negative integers")
        }
        layers.insert(layer)
    }
    return layers
}

private func requireFile(_ url: URL) throws {
    guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
        throw DiTProbeEvalError.missingFile(url)
    }
}

private func expandPath(_ path: String) -> String {
    if path == "~" {
        return FileManager.default.homeDirectoryForCurrentUser.path(percentEncoded: false)
    }
    if path.hasPrefix("~/") {
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(String(path.dropFirst(2)))
            .path(percentEncoded: false)
    }
    return path
}

private func defaultModelCacheRoot() -> URL {
    FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Containers/net.pimp.anvil.AnvilApp/Data/Library/Application Support/Anvil/models")
}

private func defaultDiTWeightsURL() -> URL {
    defaultModelCacheRoot()
        .appendingPathComponent("ace-step-dit-sft")
        .appendingPathComponent("model.safetensors")
}

private func defaultDiTConfigURL() -> URL {
    defaultModelCacheRoot()
        .appendingPathComponent("ace-step-dit-sft")
        .appendingPathComponent("config.json")
}

private func shapeString(_ shape: [Int]?) -> String {
    guard let shape else { return "-" }
    return "[" + shape.map(String.init).joined(separator: ", ") + "]"
}

private func format(_ value: Float) -> String {
    String(format: "%.6g", value)
}

private func formatOptional(_ value: Float?) -> String {
    guard let value else { return "-" }
    return format(value)
}
