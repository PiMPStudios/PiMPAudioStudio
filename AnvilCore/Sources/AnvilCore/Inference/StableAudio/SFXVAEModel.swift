// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXVAEModel.swift
import Foundation
import MLX
import MLXNN

public class SFXVAEModel: Module {
    let config: SFXVAEConfig
    let projIn: Linear
    let upsampleBlocks: [SFXUpsampleBlock]
    let projOut: Linear

    let encoderProjIn: Linear
    let downsampleBlocks: [SFXDownsampleBlock]
    let encoderProjOut: Linear

    public init(config: SFXVAEConfig) {
        self.config = config

        // Decoder setup
        var currentDim = config.latentDim
        self.projIn = Linear(config.latentDim, currentDim)

        var upBlocks: [SFXUpsampleBlock] = []
        for ratio in config.upsampleRatios {
            let outDim = max(currentDim / 2, config.outputChannels)
            upBlocks.append(SFXUpsampleBlock(inputDim: currentDim, outputDim: outDim, ratio: ratio))
            currentDim = outDim
        }
        self.upsampleBlocks = upBlocks
        self.projOut = Linear(currentDim, config.outputChannels)

        // Encoder setup (symmetric — reversed upsample ratios, doubling dims up to latentDim)
        var encDim = config.outputChannels
        self.encoderProjIn = Linear(config.outputChannels, encDim)

        var downBlocks: [SFXDownsampleBlock] = []
        for ratio in config.upsampleRatios.reversed() {
            let outDim = min(encDim * 2, config.latentDim)
            downBlocks.append(SFXDownsampleBlock(inputDim: encDim, outputDim: outDim, ratio: ratio))
            encDim = outDim
        }
        self.downsampleBlocks = downBlocks
        self.encoderProjOut = Linear(encDim, config.latentDim * 2)
    }

    public func decode(latents: MLXArray) -> MLXArray {
        var h = projIn(latents)
        for block in upsampleBlocks {
            h = block(h)
        }
        return projOut(h)
    }

    public func encode(audio: MLXArray) -> MLXArray {
        var h = encoderProjIn(audio)
        for block in downsampleBlocks {
            h = block(h)
        }
        let out = encoderProjOut(h)
        // Split last axis into 2 parts; return first (mean)
        let halfDim = out.shape[out.shape.count - 1] / 2
        return out[.ellipsis, 0..<halfDim]
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

class SFXDownsampleBlock: Module {
    let linear: Linear
    let ratio: Int

    init(inputDim: Int, outputDim: Int, ratio: Int) {
        self.linear = Linear(inputDim * ratio, outputDim)
        self.ratio = ratio
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let C = x.shape[2]
        // Reshape [B, T, C] → [B, T/ratio, C*ratio] then project and apply SiLU
        let reshaped = x.reshaped([B, T / ratio, C * ratio])
        let projected = linear(reshaped)
        return sfxVAESilu(projected)
    }
}

private func sfxVAESilu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}
