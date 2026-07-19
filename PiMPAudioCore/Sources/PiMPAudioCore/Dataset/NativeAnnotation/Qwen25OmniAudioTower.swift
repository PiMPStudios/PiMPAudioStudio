import Foundation
import MLX
import MLXNN

public typealias OmniCancellationCheck = @Sendable () throws -> Void

public struct OmniAudioTowerOutput {
    public let postConv1: MLXArray
    public let postConv2: MLXArray
    public let finalEncoderOutput: MLXArray
    public let projectedAudioEmbeddings: MLXArray

    public init(
        postConv1: MLXArray,
        postConv2: MLXArray,
        finalEncoderOutput: MLXArray,
        projectedAudioEmbeddings: MLXArray
    ) {
        self.postConv1 = postConv1
        self.postConv2 = postConv2
        self.finalEncoderOutput = finalEncoderOutput
        self.projectedAudioEmbeddings = projectedAudioEmbeddings
    }
}

public final class OmniMultiHeadAttention: Module {
    public let headCount: Int
    public let headDimension: Int

    @ModuleInfo(key: "k_proj") public var keyProjection: Linear
    @ModuleInfo(key: "v_proj") public var valueProjection: Linear
    @ModuleInfo(key: "q_proj") public var queryProjection: Linear
    @ModuleInfo(key: "out_proj") public var outputProjection: Linear

    public init(modelDimension: Int, headCount: Int) {
        self.headCount = headCount
        self.headDimension = modelDimension / headCount
        self._keyProjection.wrappedValue = Linear(
            modelDimension,
            modelDimension,
            bias: false
        )
        self._valueProjection.wrappedValue = Linear(
            modelDimension,
            modelDimension,
            bias: true
        )
        self._queryProjection.wrappedValue = Linear(
            modelDimension,
            modelDimension,
            bias: true
        )
        self._outputProjection.wrappedValue = Linear(
            modelDimension,
            modelDimension,
            bias: true
        )
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.shape[0]
        let sequence = input.shape[1]
        let state = input.shape[2]
        let scale = pow(Float(headDimension), -0.25)

        let queries = (
            queryProjection(input)
                .reshaped([batch, sequence, headCount, headDimension])
                .transposed(0, 2, 1, 3)
            * scale
        )
        let keys = (
            keyProjection(input)
                .reshaped([batch, sequence, headCount, headDimension])
                .transposed(0, 2, 3, 1)
            * scale
        )
        let values = valueProjection(input)
            .reshaped([batch, sequence, headCount, headDimension])
            .transposed(0, 2, 1, 3)

        let weights = softmax(MLX.matmul(queries, keys), axis: -1)
        let attended = MLX.matmul(weights, values)
            .transposed(0, 2, 1, 3)
            .reshaped([batch, sequence, state])
        return outputProjection(attended)
    }
}

public final class OmniResidualAttentionBlock: Module {
    @ModuleInfo(key: "self_attn") public var selfAttention: OmniMultiHeadAttention
    @ModuleInfo(key: "self_attn_layer_norm") public var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") public var fc1: Linear
    @ModuleInfo(key: "fc2") public var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") public var finalLayerNorm: LayerNorm

    public init(modelDimension: Int, headCount: Int) {
        self._selfAttention.wrappedValue = OmniMultiHeadAttention(
            modelDimension: modelDimension,
            headCount: headCount
        )
        self._selfAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: modelDimension
        )
        self._fc1.wrappedValue = Linear(
            modelDimension,
            modelDimension * 4,
            bias: true
        )
        self._fc2.wrappedValue = Linear(
            modelDimension * 4,
            modelDimension,
            bias: true
        )
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: modelDimension)
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = input + selfAttention(selfAttentionLayerNorm(input))
        let residual = hidden
        hidden = fc2(gelu(fc1(finalLayerNorm(hidden))))
        return hidden + residual
    }
}

public final class Qwen25OmniAudioTower: Module {
    public let configuration: OmniAudioConfiguration
    public let conv1: Conv1d
    public let conv2: Conv1d
    public let layers: [OmniResidualAttentionBlock]
    public let averagePooler: AvgPool1d
    @ModuleInfo(key: "ln_post") public var postLayerNorm: LayerNorm
    @ModuleInfo(key: "proj") public var proj: Linear

    private let positionalEmbeddingValues: [Float]

    public init(configuration: OmniAudioConfiguration = .qwen25Omni) throws {
        try configuration.validate()
        self.configuration = configuration
        self.conv1 = Conv1d(
            inputChannels: configuration.featureSize,
            outputChannels: configuration.modelDimension,
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
        self.conv2 = Conv1d(
            inputChannels: configuration.modelDimension,
            outputChannels: configuration.modelDimension,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        self.layers = (0..<configuration.encoderLayers).map { _ in
            OmniResidualAttentionBlock(
                modelDimension: configuration.modelDimension,
                headCount: configuration.attentionHeads
            )
        }
        self.averagePooler = AvgPool1d(kernelSize: 2, stride: 2)
        self._postLayerNorm.wrappedValue = LayerNorm(
            dimensions: configuration.modelDimension
        )
        self._proj.wrappedValue = Linear(
            configuration.modelDimension,
            configuration.outputDimension,
            bias: true
        )
        self.positionalEmbeddingValues = Self.makeSinusoids(
            length: configuration.maxSourcePositions,
            channels: configuration.modelDimension
        )
    }

    public func callAsFunction(
        _ rawAudioMel: MLXArray,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> OmniAudioTowerOutput {
        precondition(
            rawAudioMel.ndim == 2
                && rawAudioMel.shape[0] == configuration.featureSize,
            "Expected [\(configuration.featureSize), frames] audio mel input."
        )
        try cancellationCheck()

        var audioMel = rawAudioMel.asType(.bfloat16)
        let sourceFrames = audioMel.shape[1]
        let outputFrameCount = sourceFrames / 2
        let windowInputFrames = configuration.inputFramesPerAttentionWindow
        let remainder = sourceFrames % windowInputFrames
        let lastChunkSize: Int

        if remainder != 0 {
            lastChunkSize = remainder
            audioMel = padded(
                audioMel,
                widths: [IntOrPair(0), IntOrPair((0, windowInputFrames - remainder))]
            )
        } else {
            lastChunkSize = configuration.attentionWindow
        }

        let chunkCount = audioMel.shape[1] / windowInputFrames
        var chunks = audioMel.reshaped([
            audioMel.shape[0],
            chunkCount,
            windowInputFrames,
        ])
        chunks = chunks.transposed(1, 2, 0)

        var postConv1 = gelu(conv1(chunks))
        if remainder != 0, lastChunkSize < windowInputFrames {
            var mask = [Float](
                repeating: 1,
                count: chunkCount * windowInputFrames
            )
            let invalidStart = (chunkCount - 1) * windowInputFrames + lastChunkSize
            for index in invalidStart..<(chunkCount * windowInputFrames) {
                mask[index] = 0
            }
            let maskArray = MLXArray(
                mask,
                [chunkCount, windowInputFrames, 1]
            ).asType(.bfloat16)
            postConv1 = postConv1 * maskArray
        }

        let postConv2 = gelu(conv2(postConv1))
        let positional = MLXArray(
            positionalEmbeddingValues,
            [
                configuration.maxSourcePositions,
                configuration.modelDimension,
            ]
        )
        .asType(.bfloat16)[
            0..<postConv2.shape[1],
            0...
        ]
        .expandedDimensions(axis: 0)
        let positioned = postConv2 + positional

        let finalWindowFrames =
            outputFrameCount - (chunkCount - 1) * configuration.attentionWindow
        var encodedParts: [MLXArray] = []

        if chunkCount > 1 {
            var main = positioned[0..<(chunkCount - 1), 0..., 0...]
            for layer in layers {
                try cancellationCheck()
                main = layer(main)
            }
            encodedParts.append(
                main.reshaped([
                    1,
                    (chunkCount - 1) * configuration.attentionWindow,
                    configuration.modelDimension,
                ])
            )
        }

        if finalWindowFrames > 0 {
            var final = positioned[
                (chunkCount - 1)..<chunkCount,
                0..<finalWindowFrames,
                0...
            ]
            for layer in layers {
                try cancellationCheck()
                final = layer(final)
            }
            encodedParts.append(final)
        }

        let encoderOutput = encodedParts.count == 1
            ? encodedParts[0]
            : concatenated(encodedParts, axis: 1)
        precondition(
            encoderOutput.shape[1] == outputFrameCount,
            "Audio encoder produced \(encoderOutput.shape[1]) frames; expected \(outputFrameCount)."
        )

        let projected = proj(postLayerNorm(averagePooler(encoderOutput)))
        let outputs = OmniAudioTowerOutput(
            postConv1: postConv1.asType(.float32),
            postConv2: postConv2.asType(.float32),
            finalEncoderOutput: encoderOutput.asType(.float32),
            projectedAudioEmbeddings: projected.asType(.float32)
        )
        eval(
            outputs.postConv1,
            outputs.postConv2,
            outputs.finalEncoderOutput,
            outputs.projectedAudioEmbeddings
        )
        return outputs
    }

    public static func makeSinusoids(
        length: Int,
        channels: Int,
        maxTimescale: Double = 10_000
    ) -> [Float] {
        precondition(channels.isMultiple(of: 2))
        let half = channels / 2
        let increment = log(maxTimescale) / Double(half - 1)
        let inverseTimescales = (0..<half).map {
            Float(exp(-increment * Double($0)))
        }
        var result = [Float](repeating: 0, count: length * channels)
        for position in 0..<length {
            for channel in 0..<half {
                let scaledTime = Float(position) * inverseTimescales[channel]
                result[position * channels + channel] = sin(scaledTime)
                result[position * channels + half + channel] = cos(scaledTime)
            }
        }
        return result
    }
}
