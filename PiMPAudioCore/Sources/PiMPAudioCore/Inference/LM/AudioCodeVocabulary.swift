// PiMPAudioCore/Sources/PiMPAudioCore/Inference/LM/AudioCodeVocabulary.swift
import Foundation

/// Maps ACE-Step packed FSQ audio-code IDs (`0...63999`) to the tokenizer
/// vocabulary IDs used by the 5 Hz LM (`<|audio_code_N|>` special tokens).
public struct AudioCodeVocabulary: Sendable, Equatable {
    public let tokenStartID: Int
    public let codeCount: Int

    public var tokenEndID: Int {
        tokenStartID + codeCount
    }

    public init(tokenStartID: Int, codeCount: Int = 64000) {
        self.tokenStartID = tokenStartID
        self.codeCount = codeCount
    }

    public static func load(
        from tokenizerURL: URL,
        requiredCodeCount: Int = 64000
    ) throws -> AudioCodeVocabulary {
        let data = try Data(contentsOf: tokenizerURL)
        let json = try JSONSerialization.jsonObject(with: data)
        guard let root = json as? [String: Any],
              let addedTokens = root["added_tokens"] as? [[String: Any]] else {
            throw AudioCodeVocabularyError.missingAddedTokens
        }

        var tokenIDsByCode: [Int: Int] = [:]
        for token in addedTokens {
            guard let content = token["content"] as? String,
                  let id = token["id"] as? Int,
                  let code = Self.audioCodeID(from: content) else {
                continue
            }
            tokenIDsByCode[code] = id
        }

        guard let startID = tokenIDsByCode[0] else {
            throw AudioCodeVocabularyError.missingAudioCodeZero
        }

        for code in 0 ..< requiredCodeCount {
            guard let tokenID = tokenIDsByCode[code] else {
                throw AudioCodeVocabularyError.missingAudioCode(code)
            }
            let expected = startID + code
            guard tokenID == expected else {
                throw AudioCodeVocabularyError.nonContiguousAudioCodes(
                    expected: expected,
                    actual: tokenID
                )
            }
        }

        return AudioCodeVocabulary(tokenStartID: startID, codeCount: requiredCodeCount)
    }

    private static func audioCodeID(from content: String) -> Int? {
        let prefix = "<|audio_code_"
        let suffix = "|>"
        guard content.hasPrefix(prefix), content.hasSuffix(suffix) else {
            return nil
        }
        let start = content.index(content.startIndex, offsetBy: prefix.count)
        let end = content.index(content.endIndex, offsetBy: -suffix.count)
        return Int(content[start ..< end])
    }
}

public enum AudioCodeVocabularyError: Error, LocalizedError, Equatable {
    case missingAddedTokens
    case missingAudioCodeZero
    case missingAudioCode(Int)
    case nonContiguousAudioCodes(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .missingAddedTokens:
            "ACE-Step LM tokenizer is missing added audio-code tokens"
        case .missingAudioCodeZero:
            "ACE-Step LM tokenizer is missing <|audio_code_0|>"
        case let .missingAudioCode(code):
            "ACE-Step LM tokenizer is missing <|audio_code_\(code)|>"
        case let .nonContiguousAudioCodes(expected, actual):
            "ACE-Step LM audio-code token IDs are not contiguous (expected \(expected), got \(actual))"
        }
    }
}
