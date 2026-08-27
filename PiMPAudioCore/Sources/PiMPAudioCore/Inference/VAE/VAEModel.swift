import Foundation
import MLX
import MLXNN
import MLXRandom

public class OobleckEncoder: Module, UnaryLayer {
    let conv1: Conv1d
    let blocks: [OobleckEncoderBlock]
    let snake1: Snake1d
    let conv2: Conv1d

    public init(
        encoderHiddenSize: Int = 128,
        audioChannels: Int = 2,
        downsamplingRatios: [Int] = [2, 4, 4, 6, 10], // v1.5 VAE config
        channelMultiples: [Int] = [1, 2, 4, 8, 16]
    ) {
        let cm = [1] + channelMultiples
        conv1 = Conv1d(
            inputChannels: audioChannels, outputChannels: encoderHiddenSize,
            kernelSize: 7, padding: 3
        )
        var blocks: [OobleckEncoderBlock] = []
        for (i, stride) in downsamplingRatios.enumerated() {
            blocks.append(OobleckEncoderBlock(
                inputDim: encoderHiddenSize * cm[i],
                outputDim: encoderHiddenSize * cm[i + 1],
                stride: stride
            ))
        }
        self.blocks = blocks
        let dModel = encoderHiddenSize * cm.last!
        snake1 = Snake1d(hiddenDim: dModel)
        conv2 = Conv1d(
            inputChannels: dModel, outputChannels: encoderHiddenSize,
            kernelSize: 3, padding: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for block in blocks {
            h = block(h)
        }
        h = snake1(h)
        h = conv2(h)
        return h
    }
}

public class OobleckDecoder: Module, UnaryLayer {
    let conv1: Conv1d
    let blocks: [OobleckDecoderBlock]
    let snake1: Snake1d
    let conv2: Conv1d

    public init(
        channels: Int = 128,
        inputChannels: Int = 64,
        audioChannels: Int = 2,
        upsamplingRatios: [Int] = [10, 6, 4, 4, 2], // reverse of v1.5 downsamplingRatios
        channelMultiples: [Int] = [1, 2, 4, 8, 16]
    ) {
        let cm = [1] + channelMultiples
        conv1 = Conv1d(
            inputChannels: inputChannels, outputChannels: channels * cm.last!,
            kernelSize: 7, padding: 3
        )
        var blocks: [OobleckDecoderBlock] = []
        for (i, stride) in upsamplingRatios.enumerated() {
            let idx = upsamplingRatios.count - i
            blocks.append(OobleckDecoderBlock(
                inputDim: channels * cm[idx],
                outputDim: channels * cm[idx - 1],
                stride: stride
            ))
        }
        self.blocks = blocks
        snake1 = Snake1d(hiddenDim: channels)
        conv2 = Conv1d(
            inputChannels: channels, outputChannels: audioChannels,
            kernelSize: 7, padding: 3, bias: false
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for block in blocks {
            h = block(h)
        }
        h = snake1(h)
        h = conv2(h)
        return h
    }
}

public class AutoEncoderOobleck: Module {
    public let encoder: OobleckEncoder
    public let decoder: OobleckDecoder
    public let encoderHiddenSize: Int
    public let decoderInputChannels: Int

    public init(
        encoderHiddenSize: Int = 128,
        // ACE-Step v1.5 VAE config: [2, 4, 4, 6, 10] (total stride = 1920).
        // The earlier [2, 4, 4, 8, 8] (stride 2048) is from upstream's docstring
        // for a different/older variant — instantiating with those strides against
        // v1.5 weights produces a Conv1d/ConvTranspose graph whose tensor shapes
        // don't match what's loaded, yielding near-silent decoded audio.
        downsamplingRatios: [Int] = [2, 4, 4, 6, 10],
        channelMultiples: [Int] = [1, 2, 4, 8, 16],
        decoderChannels: Int = 128,
        decoderInputChannels: Int = 64,
        audioChannels: Int = 2
    ) {
        self.encoderHiddenSize = encoderHiddenSize
        self.decoderInputChannels = decoderInputChannels
        encoder = OobleckEncoder(
            encoderHiddenSize: encoderHiddenSize,
            audioChannels: audioChannels,
            downsamplingRatios: downsamplingRatios,
            channelMultiples: channelMultiples
        )
        decoder = OobleckDecoder(
            channels: decoderChannels,
            inputChannels: decoderInputChannels,
            audioChannels: audioChannels,
            upsamplingRatios: downsamplingRatios.reversed(),
            channelMultiples: channelMultiples
        )
    }

    /// Sample from the encoder posterior. Pass a `key` to make the draw
    /// deterministic — required for seed-reproducible cover generation.
    public func encodeSample(_ audioNLC: MLXArray, key: MLXArray? = nil) -> MLXArray {
        let h = encoder(audioNLC)
        let splits = MLX.split(h, parts: 2, axis: -1)
        let mean = splits[0]
        let scale = splits[1]
        let std = log(1.0 + exp(scale)) + 1e-4
        let noise: MLXArray = if let key {
            MLXRandom.normal(mean.shape, key: key)
        } else {
            MLXRandom.normal(mean.shape)
        }
        return mean + std * noise
    }

    public func encodeMean(_ audioNLC: MLXArray) -> MLXArray {
        let h = encoder(audioNLC)
        let splits = MLX.split(h, parts: 2, axis: -1)
        return splits[0]
    }

    public func decode(_ latentsNLC: MLXArray) -> MLXArray {
        decoder(latentsNLC)
    }
}
