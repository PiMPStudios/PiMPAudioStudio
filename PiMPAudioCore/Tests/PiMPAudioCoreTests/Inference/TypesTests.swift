import Testing
@testable import PiMPAudioCore

@Suite("Inference Types")
struct TypesTests {
    @Test("Precision raw values")
    func precisionRawValues() {
        #expect(Precision.float32.rawValue == "float32")
        #expect(Precision.float16.rawValue == "float16")
    }

    @Test("InferMethod raw values")
    func inferMethodRawValues() {
        #expect(InferMethod.ode.rawValue == "ode")
        #expect(InferMethod.sde.rawValue == "sde")
    }

    @Test("ModelComponent cases exist")
    func modelComponentCases() {
        let vae = ModelComponent.vae
        let dit = ModelComponent.dit
        #expect(vae != dit)
    }

    @Test("DiffusionConfig defaults")
    func diffusionConfigDefaults() {
        let config = DiffusionConfig()
        #expect(config.seed == nil)
        #expect(config.method == .ode)
        #expect(config.shift == 3.0)
        #expect(config.steps == nil)
        #expect(config.guidanceScale == 1.0)
        #expect(config.cfgIntervalStart == 0.0)
        #expect(config.cfgIntervalEnd == 1.0)
        #expect(config.audioCoverStrength == 1.0)
        #expect(config.precision == .float32)
    }

    @Test("TimeCosts stores values")
    func timeCosts() {
        let costs = TimeCosts(diffusionTime: 1.5, perStepTime: 0.2, totalTime: 2.0)
        #expect(costs.diffusionTime == 1.5)
        #expect(costs.perStepTime == 0.2)
        #expect(costs.totalTime == 2.0)
    }

    @Test("DiffusionProgress stores values")
    func diffusionProgress() {
        let progress = DiffusionProgress(currentStep: 3, totalSteps: 8, elapsedTime: 0.6)
        #expect(progress.currentStep == 3)
        #expect(progress.totalSteps == 8)
    }

    @Test("PipelineState cases")
    func pipelineStateCases() {
        let state = PipelineState.generating(step: 2, totalSteps: 8)
        if case .generating(let step, let total) = state {
            #expect(step == 2)
            #expect(total == 8)
        } else {
            Issue.record("Expected .generating")
        }
    }

    @Test("ConversionProgress stores values")
    func conversionProgress() {
        let p = ConversionProgress(current: 5, total: 100, currentTensorName: "layers.0.weight")
        #expect(p.current == 5)
        #expect(p.total == 100)
        #expect(p.currentTensorName == "layers.0.weight")
    }
}
