import Foundation
import MLX
import Tokenizers

public enum OmniAnnotationTask: String, Codable, Sendable {
    case caption
    case transcribe

    public var prompt: String {
        switch self {
        case .caption:
            "*Task* Describe this audio in detail"
        case .transcribe:
            """
            *Task* Transcribe this audio in detail. Use section tags like \
            [Verse], [Chorus], [Intro] etc. Output the structured lyrics.
            """
        }
    }
}

public struct OmniPreparedPrompt: Sendable {
    public let tokenIDs: [Int]
    public let audioPlaceholderIndex: Int
    public let audioEmbeddingCount: Int

    public init(
        tokenIDs: [Int],
        audioPlaceholderIndex: Int,
        audioEmbeddingCount: Int
    ) {
        self.tokenIDs = tokenIDs
        self.audioPlaceholderIndex = audioPlaceholderIndex
        self.audioEmbeddingCount = audioEmbeddingCount
    }

    public var expandedTokenIDs: [Int] {
        var result = tokenIDs
        result.replaceSubrange(
            audioPlaceholderIndex ... audioPlaceholderIndex,
            with: repeatElement(
                OmniPromptBuilder.extendedAudioTokenSeed,
                count: audioEmbeddingCount
            )
        )
        return result
    }

    public func mixedEmbeddings(
        using model: OmniQwen2DecoderModel,
        audioEmbeddings: MLXArray
    ) throws -> MLXArray {
        guard audioEmbeddings.ndim == 3,
              audioEmbeddings.dim(0) == 1,
              audioEmbeddings.dim(1) == audioEmbeddingCount,
              audioEmbeddings.dim(2) == model.configuration.hiddenSize
        else {
            throw OmniPromptBuilderError.audioEmbeddingShape(
                expected: [1, audioEmbeddingCount, model.configuration.hiddenSize],
                actual: audioEmbeddings.shape
            )
        }

        let prefix = Array(tokenIDs[..<audioPlaceholderIndex])
        let suffix = Array(tokenIDs[(audioPlaceholderIndex + 1)...])
        var parts: [MLXArray] = []
        if !prefix.isEmpty {
            parts.append(
                model.embeddings(
                    for: MLXArray(prefix.map(Int32.init), [1, prefix.count])
                )
            )
        }
        parts.append(audioEmbeddings)
        if !suffix.isEmpty {
            parts.append(
                model.embeddings(
                    for: MLXArray(suffix.map(Int32.init), [1, suffix.count])
                )
            )
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
    }

    public static func fromExpandedReference(
        _ expandedTokenIDs: [Int],
        threshold: Int = OmniPromptBuilder.extendedAudioTokenSeed
    ) throws -> Self {
        guard let start = expandedTokenIDs.firstIndex(where: { $0 >= threshold }),
              let end = expandedTokenIDs.lastIndex(where: { $0 >= threshold })
        else {
            throw OmniPromptBuilderError.missingAudioPlaceholder
        }
        guard expandedTokenIDs[start ... end].allSatisfy({ $0 >= threshold }) else {
            throw OmniPromptBuilderError.nonContiguousAudioSpan
        }
        var collapsed = expandedTokenIDs
        collapsed.replaceSubrange(start ... end, with: [151_646])
        return Self(
            tokenIDs: collapsed,
            audioPlaceholderIndex: start,
            audioEmbeddingCount: end - start + 1
        )
    }
}

public final class OmniPromptBuilder: @unchecked Sendable {
    public static let audioPlaceholder = "<|AUDIO|>"
    public static let audioStart = "<|audio_bos|>"
    public static let audioEnd = "<|audio_eos|>"
    public static let extendedAudioTokenSeed = 5_000_000
    public static let systemPrompt =
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            + "capable of perceiving auditory and visual inputs, as well as generating text and speech."

    public let tokenizer: any Tokenizer
    public let specialTokens: OmniSpecialTokenConfiguration

    public init(
        tokenizer: any Tokenizer,
        specialTokens: OmniSpecialTokenConfiguration
    ) {
        self.tokenizer = tokenizer
        self.specialTokens = specialTokens
    }

    public static func load(
        from modelDirectory: URL,
        specialTokens: OmniSpecialTokenConfiguration
    ) async throws -> Self {
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
        return Self(tokenizer: tokenizer, specialTokens: specialTokens)
    }

    public func prepare(
        task: OmniAnnotationTask,
        audioEmbeddingCount: Int
    ) throws -> OmniPreparedPrompt {
        guard audioEmbeddingCount > 0 else {
            throw OmniPromptBuilderError.emptyAudioEmbeddings
        }
        let audioMarkup =
            Self.audioStart + Self.audioPlaceholder + Self.audioEnd
        let messages: [Message] = [
            [
                "role": "system",
                "content": Self.systemPrompt,
            ],
            [
                "role": "user",
                "content": task.prompt + audioMarkup,
            ],
        ]
        let tokenIDs = try tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: nil
        )
        let positions = tokenIDs.indices.filter {
            tokenIDs[$0] == specialTokens.audioToken
        }
        guard positions.count == 1, let position = positions.first else {
            throw OmniPromptBuilderError.audioPlaceholderCount(positions.count)
        }
        return OmniPreparedPrompt(
            tokenIDs: tokenIDs,
            audioPlaceholderIndex: position,
            audioEmbeddingCount: audioEmbeddingCount
        )
    }

    public func decode(_ tokenIDs: [Int]) -> String {
        var text = tokenizer.decode(
            tokens: tokenIDs,
            skipSpecialTokens: false
        )
        for stop in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"] {
            if let range = text.range(of: stop) {
                text = String(text[..<range.lowerBound])
            }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public enum OmniPromptBuilderError: LocalizedError {
    case emptyAudioEmbeddings
    case missingAudioPlaceholder
    case audioPlaceholderCount(Int)
    case nonContiguousAudioSpan
    case audioEmbeddingShape(expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .emptyAudioEmbeddings:
            "The audio tower produced no embeddings."
        case .missingAudioPlaceholder:
            "The token sequence does not contain an extended audio span."
        case let .audioPlaceholderCount(count):
            "The prompt contains \(count) audio placeholders; expected exactly one."
        case .nonContiguousAudioSpan:
            "The extended audio tokens are not one contiguous prompt span."
        case let .audioEmbeddingShape(expected, actual):
            "Audio embedding shape \(actual) does not match \(expected)."
        }
    }
}
