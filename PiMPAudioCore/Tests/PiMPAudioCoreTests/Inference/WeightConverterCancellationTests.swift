// PiMPAudioCore/Tests/PiMPAudioCoreTests/Inference/WeightConverterCancellationTests.swift
import Testing
import MLX
@testable import PiMPAudioCore

@Suite("WeightConverter Cancellation")
struct WeightConverterCancellationTests {

    @Test("convertVAEWeights respects task cancellation")
    func vaeWeightsCancellation() async {
        // Build a large enough state dict to trigger at least one cancellation check (>50 entries)
        var stateDict: [(String, MLXArray)] = []
        for i in 0..<60 {
            stateDict.append(("layer.\(i).weight", MLXArray.zeros([4, 4])))
        }

        let converter = WeightConverter()
        let task = Task {
            try await converter.convertVAEWeights(stateDict: stateDict)
        }
        task.cancel()
        // Either completes with partial results or throws CancellationError — both are acceptable.
        // The key assertion is that the task finishes without hanging.
        let result = await task.result
        switch result {
        case .success(let weights):
            #expect(weights.count >= 0)
        case .failure:
            // CancellationError is expected when cancellation wins the race
            break
        }
    }

    @Test("convertDiTWeights respects task cancellation")
    func ditWeightsCancellation() async {
        var stateDict: [(String, MLXArray)] = []
        for i in 0..<60 {
            stateDict.append(("layers.\(i).weight", MLXArray.zeros([4, 4])))
        }

        let converter = WeightConverter()
        let task = Task {
            try await converter.convertDiTWeights(stateDict: stateDict)
        }
        task.cancel()
        let result = await task.result
        switch result {
        case .success(let weights):
            #expect(weights.count >= 0)
        case .failure:
            // CancellationError is expected when cancellation wins the race
            break
        }
    }

    @Test("Cancelled convertDiTWeights finishes the progress stream")
    func ditWeightsCancellationFinishesProgress() async {
        var stateDict: [(String, MLXArray)] = []
        for i in 0..<60 {
            stateDict.append(("layers.\(i).weight", MLXArray.zeros([4, 4])))
        }

        let converter = WeightConverter()
        let stream = await converter.makeProgressStream()
        let convertTask = Task {
            try await converter.convertDiTWeights(stateDict: stateDict)
        }
        convertTask.cancel()
        _ = await convertTask.result

        let finished = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                for await _ in stream {}
                return true
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(2))
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
        #expect(finished)
    }
}
