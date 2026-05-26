import Foundation
import Testing
import MLX
@testable import AnvilCore

@Suite("ACE-Step LM Audio Codes")
struct AceStepLMTests {
    @Test("AudioCodeVocabulary loads contiguous audio-code tokens")
    func audioCodeVocabularyLoad() throws {
        let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let tokenizerURL = tempDir.appending(path: "tokenizer.json")
        let json = """
        {
          "added_tokens": [
            { "content": "<|audio_code_0|>", "id": 151669 },
            { "content": "<|audio_code_1|>", "id": 151670 },
            { "content": "<|audio_code_2|>", "id": 151671 }
          ]
        }
        """
        try Data(json.utf8).write(to: tokenizerURL)

        let vocabulary = try AudioCodeVocabulary.load(from: tokenizerURL, requiredCodeCount: 3)
        #expect(vocabulary.tokenStartID == 151_669)
        #expect(vocabulary.codeCount == 3)
        #expect(vocabulary.tokenEndID == 151_672)
    }

    @Test("AudioCodeVocabulary rejects non-contiguous audio-code tokens")
    func audioCodeVocabularyRejectsNonContiguousTokens() throws {
        let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let tokenizerURL = tempDir.appending(path: "tokenizer.json")
        let json = """
        {
          "added_tokens": [
            { "content": "<|audio_code_0|>", "id": 10 },
            { "content": "<|audio_code_1|>", "id": 12 },
            { "content": "<|audio_code_2|>", "id": 13 }
          ]
        }
        """
        try Data(json.utf8).write(to: tokenizerURL)

        do {
            _ = try AudioCodeVocabulary.load(from: tokenizerURL, requiredCodeCount: 3)
            Issue.record("Expected non-contiguous audio-code token IDs to throw")
        } catch let error as AudioCodeVocabularyError {
            #expect(error == .nonContiguousAudioCodes(expected: 11, actual: 12))
        }
    }

    @Test("AceStepLM predicts packed audio-code shape")
    func aceStepLMPredictAudioCodesShape() {
        let config = AceStepLMConfig(
            hiddenSize: 8,
            numHiddenLayers: 1,
            intermediateSize: 16,
            numAttentionHeads: 2,
            numKeyValueHeads: 1,
            headDim: 4,
            rmsNormEps: 1e-6,
            vocabSize: 16,
            maxPositionEmbeddings: 16,
            ropeTheta: 1_000_000.0
        )
        let lm = AceStepLM(config: config)
        let sourceCodes = MLXArray([0, 1, 2]).asType(.int32).reshaped([1, 3, 1])
        let vocabulary = AudioCodeVocabulary(tokenStartID: 10, codeCount: 3)

        let predicted = lm.predictAudioCodes(sourceCodes: sourceCodes, vocabulary: vocabulary)
        #expect(predicted.shape == [1, 3, 1])
    }

    @Test("ResidualFSQ reconstructs from trailing singleton packed-code axis")
    func residualFSQAcceptsTokenizerIndexShape() {
        let fsq = ResidualFSQ(hiddenSize: 4, levels: [2, 2])
        let packed2D = MLXArray([0, 1, 2, 3]).asType(.int32).reshaped([1, 4])
        let packed3D = packed2D.expandedDimensions(axis: -1)

        let reconstructed2D = fsq.getOutputFromIndices(packed2D)
        let reconstructed3D = fsq.getOutputFromIndices(packed3D)

        #expect(reconstructed2D.shape == [1, 4, 4])
        #expect(reconstructed3D.shape == [1, 4, 4])
    }
}
