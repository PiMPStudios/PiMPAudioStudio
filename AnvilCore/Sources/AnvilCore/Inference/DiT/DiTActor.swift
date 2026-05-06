import Foundation
import MLX
import MLXNN

public actor DiTActor {
    private var decoder: DiTDecoder?
    private let decoderFactory: () -> DiTDecoder
    public private(set) var loadedModelID: String?

    public init(decoder: DiTDecoder) {
        self.decoder = decoder
        self.decoderFactory = { DiTDecoder() }
    }

    public func step(
        noisyLatents: MLXArray,
        timestep: Float,
        timestepR: Float,
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        encoderAttentionMask: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> (output: MLXArray, cache: CrossAttentionCache?) {
        guard let decoder else {
            fatalError("DiTActor.step called without loaded weights")
        }
        let batchSize = noisyLatents.shape[0]
        let tArray = MLXArray.full([batchSize], values: MLXArray(timestep))
        let trArray = MLXArray.full([batchSize], values: MLXArray(timestepR))

        let (output, returnedCache) = decoder(
            hiddenStates: noisyLatents,
            timestep: tArray,
            timestepR: trArray,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            encoderAttentionMask: encoderAttentionMask,
            cache: cache,
            useCache: useCache
        )
        eval(output)
        return (output, returnedCache)
    }

    public func loadWeights(_ weights: [(String, MLXArray)], modelID: String) {
        if decoder == nil {
            decoder = decoderFactory()
        }
        decoder!.update(parameters: ModuleParameters.unflattened(weights))
        eval(decoder!.parameters())
        loadedModelID = modelID
    }

    /// Legacy overload for callers that don't track model ID yet.
    public func loadWeights(_ weights: [(String, MLXArray)]) {
        loadWeights(weights, modelID: "")
    }

    public func unloadWeights() {
        decoder = nil
        loadedModelID = nil
    }
}
