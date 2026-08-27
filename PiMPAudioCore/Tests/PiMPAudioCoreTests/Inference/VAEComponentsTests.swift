import MLX
import MLXNN
@testable import PiMPAudioCore
import Testing

@Suite("VAE Components")
struct VAEComponentsTests {
    @Test("Snake1d output shape matches input")
    func snake1dShape() {
        let snake = Snake1d(hiddenDim: 16)
        let input = MLXArray.zeros([1, 100, 16])
        let output = snake(input)
        #expect(output.shape == [1, 100, 16])
    }

    @Test("Snake1d is not identity")
    func snake1dNotIdentity() {
        let snake = Snake1d(hiddenDim: 4)
        let input = MLXArray(converting: [1.0, 2.0, 3.0, 4.0], [1, 1, 4])
        let output = snake(input)
        eval(output)
        // With alpha=0, beta=0: exp(0)=1, so output = x + sin(x)^2
        // sin(1)^2 ≈ 0.708, so output[0] ≈ 1.708
        let flat = output.reshaped([-1])
        let val = flat[0].item(Float.self)
        #expect(val > 1.5)
    }

    @Test("OobleckResidualUnit preserves shape")
    func residualUnitShape() {
        let unit = OobleckResidualUnit(dimension: 16, dilation: 1)
        let input = MLXArray.zeros([1, 64, 16])
        let output = unit(input)
        eval(output)
        #expect(output.shape == [1, 64, 16])
    }

    @Test("OobleckResidualUnit with dilation 3")
    func residualUnitDilation3() {
        let unit = OobleckResidualUnit(dimension: 16, dilation: 3)
        let input = MLXArray.zeros([1, 64, 16])
        let output = unit(input)
        eval(output)
        #expect(output.shape == [1, 64, 16])
    }

    @Test("OobleckEncoderBlock downsamples by stride")
    func encoderBlockDownsample() {
        let block = OobleckEncoderBlock(inputDim: 16, outputDim: 32, stride: 4)
        let input = MLXArray.zeros([1, 256, 16])
        let output = block(input)
        eval(output)
        #expect(output.shape[0] == 1)
        #expect(output.shape[2] == 32)
        #expect(output.shape[1] >= 63 && output.shape[1] <= 65)
    }

    @Test("OobleckDecoderBlock upsamples by stride")
    func decoderBlockUpsample() {
        let block = OobleckDecoderBlock(inputDim: 32, outputDim: 16, stride: 4)
        let input = MLXArray.zeros([1, 64, 32])
        let output = block(input)
        eval(output)
        #expect(output.shape[0] == 1)
        #expect(output.shape[2] == 16)
        #expect(output.shape[1] >= 255 && output.shape[1] <= 257)
    }
}
