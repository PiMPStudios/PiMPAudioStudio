// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEModel.swift
import Foundation
import MLX
import MLXNN

public class SFXVAEModel: Module {
    let config: SFXVAEConfig
    let projIn: Linear
    let upsampleBlocks: [SFXUpsampleBlock]
    let projOut: Linear

    public init(config: SFXVAEConfig) {
        self.config = config
        var currentDim = config.latentDim
        self.projIn = Linear(config.latentDim, currentDim)

        var blocks: [SFXUpsampleBlock] = []
        for ratio in config.upsampleRatios {
            let outDim = max(currentDim / 2, config.outputChannels)
            blocks.append(SFXUpsampleBlock(inputDim: currentDim, outputDim: outDim, ratio: ratio))
            currentDim = outDim
        }
        self.upsampleBlocks = blocks
        self.projOut = Linear(currentDim, config.outputChannels)
    }

    public func decode(latents: MLXArray) -> MLXArray {
        var h = projIn(latents)
        for block in upsampleBlocks {
            h = block(h)
        }
        return projOut(h)
    }
}

class SFXUpsampleBlock: Module {
    let linear: Linear
    let ratio: Int

    init(inputDim: Int, outputDim: Int, ratio: Int) {
        self.linear = Linear(inputDim, outputDim * ratio)
        self.ratio = ratio
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let projected = linear(x)
        let outDim = projected.shape[2] / ratio
        let reshaped = projected.reshaped([B, T * ratio, outDim])
        return sfxVAESilu(reshaped)
    }
}

private func sfxVAESilu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}
