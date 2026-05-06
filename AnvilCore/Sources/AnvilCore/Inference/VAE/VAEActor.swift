import Foundation
import MLX
import MLXFast
import MLXNN

public actor VAEActor {
    private var vae: AutoEncoderOobleck

    public init(vae: AutoEncoderOobleck = AutoEncoderOobleck()) {
        self.vae = vae
    }

    public func decode(latents: MLXArray, precision: Precision) -> MLXArray {
        let input = precision == .float16 ? latents.asType(.float16) : latents
        let audio = VAETiledDecode.decode(input, using: vae)
        return precision == .float16 ? audio.asType(.float32) : audio
    }

    public func encodeSample(audio: MLXArray, precision: Precision) -> MLXArray {
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let latents = vae.encodeSample(input)
        return precision == .float16 ? latents.asType(.float32) : latents
    }

    public func encodeMean(audio: MLXArray, precision: Precision) -> MLXArray {
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let latents = vae.encodeMean(input)
        return precision == .float16 ? latents.asType(.float32) : latents
    }

    public func loadWeights(_ weights: [(String, MLXArray)]) {
        vae.update(parameters: ModuleParameters.unflattened(weights))
        eval(vae.parameters())
    }
}
