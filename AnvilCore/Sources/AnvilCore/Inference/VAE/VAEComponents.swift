import Foundation
import MLX
import MLXNN

public class Snake1d: Module, UnaryLayer {
    let alpha: MLXArray
    let beta: MLXArray
    let logscale: Bool

    public init(hiddenDim: Int, logscale: Bool = true) {
        self.alpha = MLXArray.zeros([hiddenDim])
        self.beta = MLXArray.zeros([hiddenDim])
        self.logscale = logscale
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = logscale ? exp(alpha) : alpha
        let b = logscale ? exp(beta) : beta
        let sinTerm = sin(a * x)
        let sinSquared = sinTerm * sinTerm
        return x + (1.0 / (b + 1e-9)) * sinSquared
    }
}

public class OobleckResidualUnit: Module, UnaryLayer {
    let snake1: Snake1d
    let conv1: Conv1d
    let snake2: Snake1d
    let conv2: Conv1d

    public init(dimension: Int = 16, dilation: Int = 1) {
        let pad = ((7 - 1) * dilation) / 2
        self.snake1 = Snake1d(hiddenDim: dimension)
        self.conv1 = Conv1d(
            inputChannels: dimension, outputChannels: dimension,
            kernelSize: 7, stride: 1, padding: pad, dilation: dilation
        )
        self.snake2 = Snake1d(hiddenDim: dimension)
        self.conv2 = Conv1d(
            inputChannels: dimension, outputChannels: dimension,
            kernelSize: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = snake1(x)
        output = conv1(output)
        output = snake2(output)
        output = conv2(output)

        let padding = (x.shape[1] - output.shape[1]) / 2
        var residual = x
        if padding > 0 {
            residual = x[0..., padding ..< (x.shape[1] - padding), 0...]
        }
        return residual + output
    }
}

public class OobleckEncoderBlock: Module, UnaryLayer {
    let resUnit1: OobleckResidualUnit
    let resUnit2: OobleckResidualUnit
    let resUnit3: OobleckResidualUnit
    let snake1: Snake1d
    let conv1: Conv1d

    public init(inputDim: Int, outputDim: Int, stride: Int = 1) {
        self.resUnit1 = OobleckResidualUnit(dimension: inputDim, dilation: 1)
        self.resUnit2 = OobleckResidualUnit(dimension: inputDim, dilation: 3)
        self.resUnit3 = OobleckResidualUnit(dimension: inputDim, dilation: 9)
        self.snake1 = Snake1d(hiddenDim: inputDim)
        let kernelSize = 2 * stride
        let pad = Int(ceil(Double(stride) / 2.0))
        self.conv1 = Conv1d(
            inputChannels: inputDim, outputChannels: outputDim,
            kernelSize: kernelSize, stride: stride, padding: pad
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resUnit1(x)
        h = resUnit2(h)
        h = resUnit3(h)
        h = snake1(h)
        h = conv1(h)
        return h
    }
}

public class OobleckDecoderBlock: Module, UnaryLayer {
    let snake1: Snake1d
    let convT1: ConvTransposed1d
    let resUnit1: OobleckResidualUnit
    let resUnit2: OobleckResidualUnit
    let resUnit3: OobleckResidualUnit

    public init(inputDim: Int, outputDim: Int, stride: Int = 1) {
        self.snake1 = Snake1d(hiddenDim: inputDim)
        let kernelSize = 2 * stride
        let pad = Int(ceil(Double(stride) / 2.0))
        self.convT1 = ConvTransposed1d(
            inputChannels: inputDim, outputChannels: outputDim,
            kernelSize: kernelSize, stride: stride, padding: pad
        )
        self.resUnit1 = OobleckResidualUnit(dimension: outputDim, dilation: 1)
        self.resUnit2 = OobleckResidualUnit(dimension: outputDim, dilation: 3)
        self.resUnit3 = OobleckResidualUnit(dimension: outputDim, dilation: 9)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake1(x)
        h = convT1(h)
        h = resUnit1(h)
        h = resUnit2(h)
        h = resUnit3(h)
        return h
    }
}
