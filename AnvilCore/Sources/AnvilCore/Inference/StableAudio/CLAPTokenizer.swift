// AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPTokenizer.swift
import Foundation

public struct CLAPTokenizer: Sendable {
    public let maxLength: Int
    public let bosToken: Int
    public let eosToken: Int
    public let padToken: Int
    private let vocabulary: [String: Int]

    public init(
        vocabulary: [String: Int],
        maxLength: Int = 77,
        bosToken: Int = 49406,
        eosToken: Int = 49407,
        padToken: Int = 0
    ) {
        self.vocabulary = vocabulary
        self.maxLength = maxLength
        self.bosToken = bosToken
        self.eosToken = eosToken
        self.padToken = padToken
    }

    public init(vocabularyURL: URL, configURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabularyURL)
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(TokenizerConfig.self, from: configData)

        self.vocabulary = vocab
        self.maxLength = config.maxLength ?? 77
        self.bosToken = config.bosTokenId ?? 49406
        self.eosToken = config.eosTokenId ?? 49407
        self.padToken = config.padTokenId ?? 0
    }

    public func encode(text: String) -> [Int] {
        var tokens: [Int] = [bosToken]

        let words = text.lowercased().split(separator: " ")
        for word in words {
            if tokens.count >= maxLength - 1 { break }
            let wordStr = String(word)
            if let id = vocabulary[wordStr] {
                tokens.append(id)
            } else {
                // Character-level fallback for unknown words
                for char in wordStr {
                    if tokens.count >= maxLength - 1 { break }
                    let charStr = String(char)
                    tokens.append(vocabulary[charStr] ?? padToken)
                }
            }
        }

        tokens.append(eosToken)

        // Pad or truncate to maxLength
        if tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength - 1)) + [eosToken]
        }
        while tokens.count < maxLength {
            tokens.append(padToken)
        }

        return tokens
    }

    public static func forTesting() -> CLAPTokenizer {
        var vocab: [String: Int] = [:]
        // Build a basic test vocabulary
        let chars = "abcdefghijklmnopqrstuvwxyz 0123456789"
        for (i, char) in chars.enumerated() {
            vocab[String(char)] = i + 1
        }
        let commonWords = [
            "thunder", "rolling", "in", "the", "distance",
            "rain", "wind", "explosion", "footsteps", "word"
        ]
        for (i, word) in commonWords.enumerated() {
            vocab[word] = 1000 + i
        }
        return CLAPTokenizer(vocabulary: vocab)
    }
}

struct TokenizerConfig: Codable {
    let maxLength: Int?
    let bosTokenId: Int?
    let eosTokenId: Int?
    let padTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case maxLength = "max_length"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }
}
