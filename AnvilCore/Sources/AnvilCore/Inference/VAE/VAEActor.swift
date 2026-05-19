import Foundation
import MLX
import MLXFast
import MLXNN
import OSLog

private let vaeLogger = Logger(subsystem: "com.pimp.AnvilApp", category: "VAEActor")

public enum VAEActorError: Error, LocalizedError, Sendable {
    case weightsNotLoaded

    public var errorDescription: String? {
        switch self {
        case .weightsNotLoaded: return "VAE weights have not been loaded."
        }
    }
}

public actor VAEActor {
    private var vae: AutoEncoderOobleck?
    public private(set) var loadedModelID: String?

    public init(vae: AutoEncoderOobleck = AutoEncoderOobleck()) {
        self.vae = vae
    }

    public func decode(latents: MLXArray, precision: Precision) throws -> MLXArray {
        guard let vae else { throw VAEActorError.weightsNotLoaded }
        AudioDiagnostics.tensor("vae/in_latents", latents)
        let input = precision == .float16 ? latents.asType(.float16) : latents
        let audio = VAETiledDecode.decode(input, using: vae)
        let resolved = precision == .float16 ? audio.asType(.float32) : audio
        AudioDiagnostics.tensor("vae/out_audio", resolved)
        Self.logAudioEnvelope(resolved)

        let absA = MLX.abs(resolved)
        let absMean = absA.mean().item(Float.self)
        let absMax = absA.max().item(Float.self)
        vaeLogger.info("[VAEActor.decode] in shape=\(latents.shape) → out shape=\(resolved.shape) abs_mean=\(absMean) abs_max=\(absMax)")

        return resolved
    }

    public func encodeSample(audio: MLXArray, precision: Precision) throws -> MLXArray {
        guard let vae else { throw VAEActorError.weightsNotLoaded }
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let latents = vae.encodeSample(input)
        return precision == .float16 ? latents.asType(.float32) : latents
    }

    public func encodeMean(audio: MLXArray, precision: Precision) throws -> MLXArray {
        guard let vae else { throw VAEActorError.weightsNotLoaded }
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let latents = vae.encodeMean(input)
        return precision == .float16 ? latents.asType(.float32) : latents
    }

    public func loadWeights(_ weights: [(String, MLXArray)], modelID: String) {
        if vae == nil {
            vae = AutoEncoderOobleck()
        }

        // Diagnostic: same matched/unmatched key check as DiTActor.
        let modelKeys = Set(vae!.parameters().flattened().map { $0.0 })
        let suppliedKeys = Set(weights.map { $0.0 })
        let matched = suppliedKeys.intersection(modelKeys).count
        let supExtra = suppliedKeys.subtracting(modelKeys)
        let modExtra = modelKeys.subtracting(suppliedKeys)
        vaeLogger.info("[VAEActor.loadWeights] supplied=\(suppliedKeys.count) model=\(modelKeys.count) matched=\(matched)")
        vaeLogger.info("[VAEActor.loadWeights] supplied-but-no-model-param (first 12): \(Array(supExtra.sorted().prefix(12)))")
        vaeLogger.info("[VAEActor.loadWeights] model-param-but-no-supplied (first 12): \(Array(modExtra.sorted().prefix(12)))")

        vae!.update(parameters: ModuleParameters.unflattened(weights))
        eval(vae!.parameters())
        loadedModelID = modelID
    }

    /// Legacy overload for callers that don't track model ID yet.
    public func loadWeights(_ weights: [(String, MLXArray)]) {
        loadWeights(weights, modelID: "")
    }

    public func unloadWeights() {
        vae = nil
        loadedModelID = nil
    }

    private static func logAudioEnvelope(_ audio: MLXArray) {
        guard AudioDiagnostics.isEnabled, audio.shape.count >= 3 else { return }

        let framesPerSecond = 48_000
        let frameCount = audio.shape[1]
        guard frameCount > 0 else { return }

        let seconds = min(60, Int(ceil(Double(frameCount) / Double(framesPerSecond))))
        var absMeans: [Float] = []
        var absMaxes: [Float] = []
        absMeans.reserveCapacity(seconds)
        absMaxes.reserveCapacity(seconds)

        for second in 0..<seconds {
            let start = second * framesPerSecond
            let end = min(start + framesPerSecond, frameCount)
            guard start < end else { break }

            let chunk = audio[0..., start..<end, 0...].asType(.float32)
            let absChunk = MLX.abs(chunk)
            absMeans.append(absChunk.mean().item(Float.self))
            absMaxes.append(absChunk.max().item(Float.self))
        }

        AudioDiagnostics.floats("vae/out_audio_abs_mean_per_sec", absMeans)
        AudioDiagnostics.floats("vae/out_audio_abs_max_per_sec", absMaxes)
    }
}
