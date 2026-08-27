import Darwin
import Dispatch
import Foundation
import MLX
import MLXRandom

enum ProbeMode: String {
    case none
    case implicit
    case explicitKey = "explicit-key"
    case precomputedMask = "precomputed-mask"
}

struct ProbeConfig {
    var mode: ProbeMode = .implicit
    var sites: Int = 128
    var batch: Int = 1
    var tokens: Int = 64
    var inputDim: Int = 64
    var outputDim: Int = 64
    var dropout: Float = 0.08
    var seed: UInt64 = 0
    var timeoutSeconds: Int = 30
}

enum ProbeError: Error, CustomStringConvertible {
    case missingValue(String)
    case invalidValue(String, String)
    case unknownArgument(String)

    var description: String {
        switch self {
        case let .missingValue(key):
            "Missing value for \(key)"
        case let .invalidValue(key, value):
            "Invalid value for \(key): \(value)"
        case let .unknownArgument(key):
            "Unknown argument: \(key)"
        }
    }
}

func parseArguments(_ arguments: [String]) throws -> ProbeConfig {
    var config = ProbeConfig()
    var index = 1

    func value(after key: String) throws -> String {
        let valueIndex = index + 1
        guard valueIndex < arguments.count else {
            throw ProbeError.missingValue(key)
        }
        index = valueIndex
        return arguments[valueIndex]
    }

    while index < arguments.count {
        let key = arguments[index]
        switch key {
        case "--help", "-h":
            printUsageAndExit()
        case "--mode":
            let raw = try value(after: key)
            guard let mode = ProbeMode(rawValue: raw) else {
                throw ProbeError.invalidValue(key, raw)
            }
            config.mode = mode
        case "--sites":
            config.sites = try parsePositiveInt(key, value(after: key))
        case "--batch":
            config.batch = try parsePositiveInt(key, value(after: key))
        case "--tokens":
            config.tokens = try parsePositiveInt(key, value(after: key))
        case "--input-dim":
            config.inputDim = try parsePositiveInt(key, value(after: key))
        case "--output-dim":
            config.outputDim = try parsePositiveInt(key, value(after: key))
        case "--dropout":
            config.dropout = try parseDropout(key, value(after: key))
        case "--seed":
            let raw = try value(after: key)
            guard let seed = UInt64(raw) else {
                throw ProbeError.invalidValue(key, raw)
            }
            config.seed = seed
        case "--timeout":
            config.timeoutSeconds = try parsePositiveInt(key, value(after: key))
        default:
            throw ProbeError.unknownArgument(key)
        }
        index += 1
    }

    return config
}

func parsePositiveInt(_ key: String, _ rawValue: String) throws -> Int {
    guard let value = Int(rawValue), value > 0 else {
        throw ProbeError.invalidValue(key, rawValue)
    }
    return value
}

func parseDropout(_ key: String, _ rawValue: String) throws -> Float {
    guard let value = Float(rawValue), value >= 0, value < 1 else {
        throw ProbeError.invalidValue(key, rawValue)
    }
    return value
}

func printUsageAndExit() -> Never {
    print(
        """
        Usage:
          MLXDropoutVJPProbe --mode implicit|explicit-key|precomputed-mask|none [options]

        Options:
          --sites N        Repeated adapter/dropout sites. Default: 128
          --batch N        Batch dimension. Default: 1
          --tokens N       Sequence length. Default: 64
          --input-dim N    Input hidden size. Default: 64
          --output-dim N   Output hidden size. Default: 64
          --dropout P      Dropout probability in [0, 1). Default: 0.08
          --seed N         MLX random seed. Default: 0
          --timeout N      Force-exit seconds around first valueAndGrad. Default: 30
        """
    )
    exit(0)
}

func ensureMLXMetallibAvailable() {
    let fileManager = FileManager.default
    let executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    let executableDirectory = executableURL.deletingLastPathComponent()
    let colocatedURL = executableDirectory.appendingPathComponent("mlx.metallib", isDirectory: false)

    guard !fileManager.fileExists(atPath: colocatedURL.path),
          let sourceURL = findBundledMLXMetallib() else {
        return
    }

    do {
        try fileManager.copyItem(at: sourceURL, to: colocatedURL)
    } catch {
        fputs("warning: could not copy MLX metallib from \(sourceURL.path): \(error)\n", stderr)
    }
}

func findBundledMLXMetallib() -> URL? {
    let fileManager = FileManager.default
    let homeURL = fileManager.homeDirectoryForCurrentUser
    let derivedDataURL = homeURL
        .appendingPathComponent("Library", isDirectory: true)
        .appendingPathComponent("Developer", isDirectory: true)
        .appendingPathComponent("Xcode", isDirectory: true)
        .appendingPathComponent("DerivedData", isDirectory: true)

    guard let projectDirectories = try? fileManager.contentsOfDirectory(
        at: derivedDataURL,
        includingPropertiesForKeys: nil
    ) else {
        return nil
    }

    for projectURL in projectDirectories where projectURL.lastPathComponent.hasPrefix("PiMPAudioStudio-") {
        let buildProductsURL = projectURL
            .appendingPathComponent("Build", isDirectory: true)
            .appendingPathComponent("Products", isDirectory: true)
            .appendingPathComponent("Debug", isDirectory: true)
        let candidates = [
            buildProductsURL
                .appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
                .appendingPathComponent("Contents", isDirectory: true)
                .appendingPathComponent("Resources", isDirectory: true)
                .appendingPathComponent("default.metallib", isDirectory: false),
            buildProductsURL
                .appendingPathComponent("PiMPAudioStudio.app", isDirectory: true)
                .appendingPathComponent("Contents", isDirectory: true)
                .appendingPathComponent("Resources", isDirectory: true)
                .appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
                .appendingPathComponent("Contents", isDirectory: true)
                .appendingPathComponent("Resources", isDirectory: true)
                .appendingPathComponent("default.metallib", isDirectory: false),
        ]

        if let existingURL = candidates.first(where: { fileManager.fileExists(atPath: $0.path) }) {
            return existingURL
        }
    }

    return nil
}

func makeWatchdog(config: ProbeConfig, startedAt: Date) -> DispatchSourceTimer {
    let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .userInitiated))
    timer.schedule(deadline: .now() + .seconds(config.timeoutSeconds))
    timer.setEventHandler {
        let elapsed = Date().timeIntervalSince(startedAt)
        fputs(
            String(
                format: "timeout mode=%@ sites=%d dropout=%.4f elapsed=%.3fs limit=%ds\n",
                config.mode.rawValue,
                config.sites,
                Double(config.dropout),
                elapsed,
                config.timeoutSeconds
            ),
            stderr
        )
        fflush(stderr)
        exit(124)
    }
    timer.resume()
    return timer
}

func dropoutInput(
    _ input: MLXArray,
    config: ProbeConfig,
    arguments: [MLXArray],
    masksOffset: Int,
    keysOffset: Int,
    site: Int
) -> MLXArray {
    guard config.dropout > 0 else {
        return input
    }

    let keepScale = MLXArray(1.0 / (1.0 - config.dropout))
    let mask: MLXArray
    switch config.mode {
    case .none:
        return input
    case .implicit:
        mask = MLXRandom.uniform(0 ..< 1, input.shape) .> config.dropout
    case .explicitKey:
        mask = MLXRandom.uniform(0 ..< 1, input.shape, key: arguments[keysOffset + site]) .> config.dropout
    case .precomputedMask:
        mask = arguments[masksOffset + site]
    }

    return input * mask.asType(input.dtype) * keepScale
}

func makeSeededKeys(seed: UInt64, count: Int) -> [MLXArray] {
    guard count > 0 else {
        return []
    }
    return MLXRandom.split(key: MLXRandom.key(seed), into: count)
}

func runProbe(_ config: ProbeConfig) {
    MLXRandom.seed(config.seed)

    let inputShape = [config.batch, config.tokens, config.inputDim]
    let input = MLXRandom.uniform(-1 ..< 1, inputShape)
    let weights = (0 ..< config.sites).map { _ in
        MLXRandom.uniform(-0.02 ..< 0.02, [config.outputDim, config.inputDim])
    }

    var masks: [MLXArray] = []
    var keys: [MLXArray] = []
    switch config.mode {
    case .none, .implicit:
        break
    case .explicitKey:
        keys = makeSeededKeys(seed: config.seed + 1000, count: config.sites)
    case .precomputedMask:
        let maskKeys = makeSeededKeys(seed: config.seed + 2000, count: config.sites)
        masks = maskKeys.map { key in
            (MLXRandom.uniform(0 ..< 1, inputShape, key: key) .> config.dropout).asType(input.dtype)
        }
    }

    let masksOffset = config.sites + 1
    let keysOffset = masksOffset + masks.count
    let arguments = weights + [input] + masks + keys
    eval(arguments)

    let gradientArgumentNumbers = Array(0 ..< config.sites)
    let lossGrad = valueAndGrad({ (allArgs: [MLXArray]) -> [MLXArray] in
        let x = allArgs[config.sites]
        var loss = MLXArray(0.0)

        for site in 0 ..< config.sites {
            let adapterInput = dropoutInput(
                x,
                config: config,
                arguments: allArgs,
                masksOffset: masksOffset,
                keysOffset: keysOffset,
                site: site
            )
            let projected = matmul(adapterInput, allArgs[site].T)
            loss = loss + (projected * projected).mean()
        }

        return [loss / MLXArray(Float(config.sites))]
    }, argumentNumbers: gradientArgumentNumbers)

    let startedAt = Date()
    let watchdog = makeWatchdog(config: config, startedAt: startedAt)
    let (lossValues, gradients) = lossGrad(arguments)
    let lossGradElapsed = Date().timeIntervalSince(startedAt)
    watchdog.cancel()

    let itemStartedAt = Date()
    let loss = lossValues[0].item(Float.self)
    let itemElapsed = Date().timeIntervalSince(itemStartedAt)
    let gradientPreview = gradients.prefix(3).map(\.shape)

    print(
        String(
            format: "ok mode=%@ sites=%d input=%@ outputDim=%d dropout=%.4f lossGradElapsed=%.3fs itemElapsed=%.3fs loss=%.6f grads=%d gradPreview=%@",
            config.mode.rawValue,
            config.sites,
            "\(inputShape)",
            config.outputDim,
            Double(config.dropout),
            lossGradElapsed,
            itemElapsed,
            loss,
            gradients.count,
            "\(gradientPreview)"
        )
    )
}

do {
    ensureMLXMetallibAvailable()
    let config = try parseArguments(CommandLine.arguments)
    runProbe(config)
} catch {
    fputs("\(error)\n\n", stderr)
    printUsageAndExit()
}
