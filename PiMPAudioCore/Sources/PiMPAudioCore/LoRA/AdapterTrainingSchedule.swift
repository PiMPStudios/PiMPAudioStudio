import Foundation
import MLX
import MLXNN
import MLXRandom

struct AdapterTrainingConditioning {
    let encoderStates: MLXArray
    let contextLatents: MLXArray
    let encoderMask: MLXArray?
    let didDropConditioning: Bool
}

enum AdapterTrainingSchedule {
    static let cfgDropoutRatio: Float = 0.15
    static let timestepLogitMean: Float = -0.4
    static let timestepLogitStd: Float = 1.0

    static var diagnosticsSummary: String {
        "timestep=logitNormalMax(mu=\(timestepLogitMean) sigma=\(timestepLogitStd) meanFlow=false) cfgDropout=\(cfgDropoutRatio)"
    }

    static func sampleTimestep() -> MLXArray {
        let first = sampleLogitNormal()
        let second = sampleLogitNormal()
        return MLX.where(first .> second, first, second)
    }

    static func conditionedSample(
        _ sample: TrainingSample,
        model: DiTDecoder
    ) -> AdapterTrainingConditioning {
        guard shouldDropConditioning() else {
            return AdapterTrainingConditioning(
                encoderStates: sample.encoderStates,
                contextLatents: sample.contextLatents,
                encoderMask: sample.encoderMask,
                didDropConditioning: false
            )
        }

        let nullStates = MLX.broadcast(
            model.nullConditionEmb,
            to: sample.encoderStates.shape
        )
        .asType(sample.encoderStates.dtype)

        return AdapterTrainingConditioning(
            encoderStates: stopGradient(nullStates),
            contextLatents: sample.contextLatents,
            encoderMask: sample.encoderMask,
            didDropConditioning: true
        )
    }

    private static func sampleLogitNormal() -> MLXArray {
        sigmoid(MLXRandom.normal([1]) * timestepLogitStd + timestepLogitMean)
    }

    private static func shouldDropConditioning() -> Bool {
        guard cfgDropoutRatio > 0 else { return false }
        return MLXRandom.uniform(0..<1, [1]).item(Float.self) < cfgDropoutRatio
    }
}
