// AnvilCore/Tests/AnvilCoreTests/Inference/StableAudio/CLAPTokenizerTests.swift
import Testing
import Foundation
@testable import AnvilCore

@Suite("CLAPTokenizer")
struct CLAPTokenizerTests {
    @Test("Encodes text to token IDs with correct length")
    func encodesText() throws {
        let tokenizer = CLAPTokenizer.forTesting()
        let tokens = tokenizer.encode(text: "thunder rolling in the distance")
        #expect(tokens.count == tokenizer.maxLength)
    }

    @Test("Empty text produces padded sequence")
    func emptyText() throws {
        let tokenizer = CLAPTokenizer.forTesting()
        let tokens = tokenizer.encode(text: "")
        #expect(tokens.count == tokenizer.maxLength)
        #expect(tokens[0] == tokenizer.bosToken)
        #expect(tokens[1] == tokenizer.eosToken)
        // Rest should be padding
        for i in 2..<tokens.count {
            #expect(tokens[i] == tokenizer.padToken)
        }
    }

    @Test("Long text is truncated to maxLength")
    func truncation() throws {
        let tokenizer = CLAPTokenizer.forTesting()
        let longText = String(repeating: "word ", count: 200)
        let tokens = tokenizer.encode(text: longText)
        #expect(tokens.count == tokenizer.maxLength)
    }

    @Test("Max length property is accessible")
    func maxLengthAccessible() {
        let tokenizer = CLAPTokenizer.forTesting()
        #expect(tokenizer.maxLength == 77)
    }
}
