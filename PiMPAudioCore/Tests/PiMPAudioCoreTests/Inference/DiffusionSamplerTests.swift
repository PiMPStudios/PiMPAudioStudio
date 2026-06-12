import Testing
import MLX
import MLXRandom
@testable import PiMPAudioCore

@Suite("Diffusion Sampler")
struct DiffusionSamplerTests {
    @Test("ODE step reduces noise")
    func odeStepBasic() {
        let xt = MLXArray(converting: [1.0, 2.0, 3.0]).reshaped([1, 1, 3])
        let vt = MLXArray(converting: [0.5, 0.5, 0.5]).reshaped([1, 1, 3])
        let result = DiffusionSampler.odeStep(xt: xt, vt: vt, currentT: 1.0, nextT: 0.5)
        eval(result)
        let vals = result.reshaped([-1]).asArray(Float.self)
        #expect(abs(vals[0] - 0.75) < 1e-5)
        #expect(abs(vals[1] - 1.75) < 1e-5)
        #expect(abs(vals[2] - 2.75) < 1e-5)
    }

    @Test("Final step computes x0 correctly")
    func finalStep() {
        let xt = MLXArray(converting: [2.0, 4.0]).reshaped([1, 1, 2])
        let vt = MLXArray(converting: [1.0, 2.0]).reshaped([1, 1, 2])
        let result = DiffusionSampler.finalStep(xt: xt, vt: vt, currentT: 0.5)
        eval(result)
        let vals = result.reshaped([-1]).asArray(Float.self)
        #expect(abs(vals[0] - 1.5) < 1e-5)
        #expect(abs(vals[1] - 3.0) < 1e-5)
    }

    @Test("APG with scale 1.0 returns pred_cond unchanged")
    func apgNoGuidance() {
        let predCond = MLXArray(converting: [1.0, 2.0, 3.0]).reshaped([1, 1, 3])
        let predUncond = MLXArray(converting: [0.5, 1.0, 1.5]).reshaped([1, 1, 3])
        let result = DiffusionSampler.apgForward(
            predCond: predCond, predUncond: predUncond,
            guidanceScale: 1.0, momentum: nil
        )
        eval(result)
        let vals = result.reshaped([-1]).asArray(Float.self)
        #expect(abs(vals[0] - 1.0) < 1e-4)
        #expect(abs(vals[1] - 2.0) < 1e-4)
        #expect(abs(vals[2] - 3.0) < 1e-4)
    }

    @Test("APG with scale > 1.0 differs from pred_cond")
    func apgWithGuidance() {
        let predCond = MLXArray(converting: [1.0, 0.0, 0.0]).reshaped([1, 1, 3])
        let predUncond = MLXArray(converting: [0.0, 1.0, 0.0]).reshaped([1, 1, 3])
        let result = DiffusionSampler.apgForward(
            predCond: predCond, predUncond: predUncond,
            guidanceScale: 3.0, momentum: nil
        )
        eval(result)
        let vals = result.reshaped([-1]).asArray(Float.self)
        let diffFromCond = abs(vals[0] - 1.0) + abs(vals[1]) + abs(vals[2])
        #expect(diffFromCond > 0.01)
    }

    @Test("APG momentum matches upstream MomentumBuffer recurrence")
    func apgMomentumRecurrence() {
        // Upstream (acestep apg_guidance.MomentumBuffer, momentum = -0.75):
        //   running = diff + momentum * running
        let momentum = APGMomentum()
        let uncond = MLXArray.zeros([1, 1, 3])

        let cond1 = MLXArray(converting: [1.0, 0.0, 0.0]).reshaped([1, 1, 3])
        _ = DiffusionSampler.apgForward(
            predCond: cond1, predUncond: uncond,
            guidanceScale: 2.0, momentum: momentum, normThreshold: 0
        )
        // running == d1 == [1, 0, 0]

        let cond2 = MLXArray(converting: [0.0, 2.0, 0.0]).reshaped([1, 1, 3])
        _ = DiffusionSampler.apgForward(
            predCond: cond2, predUncond: uncond,
            guidanceScale: 2.0, momentum: momentum, normThreshold: 0
        )
        // running == d2 - 0.75 * d1 == [-0.75, 2, 0]
        eval(momentum.running)
        let vals = momentum.running.reshaped([-1]).asArray(Float.self)
        #expect(abs(vals[0] + 0.75) < 1e-5)
        #expect(abs(vals[1] - 2.0) < 1e-5)
        #expect(abs(vals[2]) < 1e-5)
    }

    @Test("Noise generation with seed is deterministic")
    func deterministicNoise() {
        let noise1 = DiffusionSampler.generateNoise(shape: [1, 10, 64], seed: 42)
        let noise2 = DiffusionSampler.generateNoise(shape: [1, 10, 64], seed: 42)
        eval(noise1, noise2)
        let diff = abs(noise1 - noise2).sum()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-6)
    }

    @Test("Noise generation without seed is random")
    func randomNoise() {
        let noise1 = DiffusionSampler.generateNoise(shape: [1, 10, 64], seed: nil)
        let noise2 = DiffusionSampler.generateNoise(shape: [1, 10, 64], seed: nil)
        eval(noise1, noise2)
        let diff = abs(noise1 - noise2).sum()
        eval(diff)
        #expect(diff.item(Float.self) > 0.1)
    }
}
