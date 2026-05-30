import Foundation
import Hub
import MLX
import Tokenizers

/// Thin wrapper around a swift-transformers BPE tokenizer.
///
/// Loads a tokenizer from a local `tokenizer.json` file and provides
/// `encodeText` and `encodeLyrics` convenience methods that return
/// `MLXArray` tensors shaped `[1, tokenCount]`, capped by `maxLength`.
public struct PiMPAudioTokenizer: Sendable {
    // Box the protocol existential so the struct stays Sendable without
    // requiring the upstream type to declare conformance itself.
    private final class Box: @unchecked Sendable {
        let encode: (String) -> [Int]

        init(_ tokenizer: any Tokenizer) {
            self.encode = { text in tokenizer.encode(text: text) }
        }

        init(encode: @escaping (String) -> [Int]) {
            self.encode = encode
        }
    }

    private let box: Box

    // MARK: - Init

    /// Load a tokenizer from a `tokenizer.json` file on disk.
    ///
    /// Supports BPE (GPT-2, Qwen, Llama…) and Unigram/SentencePiece (T5…)
    /// tokenizers. A companion `tokenizer_config.json` in the same directory
    /// is used when present; otherwise the tokenizer class is inferred from
    /// the `model.type` field in `tokenizer.json` to avoid routing T5/Unigram
    /// tokenizers through `BPETokenizer` (which crashes when merges is empty).
    ///
    /// swift-transformers class mapping:
    ///   model.type "Unigram" → tokenizer_class "T5Tokenizer"  → UnigramTokenizer
    ///   anything else        → tokenizer_class "PreTrainedTokenizer" → BPETokenizer
    public init(configURL: URL) throws {
        // Parse the raw JSON once so we can inspect model.type before building
        // the Config — Config's dynamic-member API doesn't expose NSString dicts.
        let rawJSON = try Data(contentsOf: configURL)
        let rawDict = (try? JSONSerialization.jsonObject(with: rawJSON)) as? [NSString: Any] ?? [:]
        let tokenizerData = Config(rawDict)

        let tokenizerConfigURL = configURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer_config.json")

        let tokenizerConfig: Config
        if FileManager.default.fileExists(atPath: tokenizerConfigURL.path(percentEncoded: false)) {
            tokenizerConfig = try Self.loadConfig(from: tokenizerConfigURL)
        } else {
            // Infer the tokenizer class from model.type so that Unigram/T5
            // tokenizers (which have empty merges) don't route to BPETokenizer
            // and crash with "BPETokenizer requires merges".
            let modelType = (rawDict["model"] as? [NSString: Any])
                .flatMap { $0["type"] as? String } ?? "BPE"
            let tokenizerClass = modelType == "Unigram" ? "T5Tokenizer" : "PreTrainedTokenizer"
            tokenizerConfig = Config([NSString("tokenizer_class"): tokenizerClass] as [NSString: Any])
        }

        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData
        )
        self.box = Box(tokenizer)
    }

    private init(testEncoder: @escaping (String) -> [Int]) {
        self.box = Box(encode: testEncoder)
    }

    // MARK: - Encode

    /// Encode text for conditioning inputs (e.g. prompt, genre tags).
    ///
    /// - Parameters:
    ///   - text: The text to encode.
    ///   - maxLength: Sequence length of the returned arrays. Default `256`.
    /// - Returns: A tuple `(ids, mask)` of `Int32 MLXArray` shaped `[1, tokenCount]`.
    public func encodeText(
        _ text: String,
        maxLength: Int = 256
    ) -> (ids: MLXArray, mask: MLXArray) {
        encode(text, maxLength: maxLength)
    }

    /// Encode lyrics for the lyric encoder input.
    ///
    /// - Parameters:
    ///   - lyrics: The lyrics string to encode.
    ///   - maxLength: Sequence length of the returned arrays. Default `2048`.
    /// - Returns: A tuple `(ids, mask)` of `Int32 MLXArray` shaped `[1, tokenCount]`.
    public func encodeLyrics(
        _ lyrics: String,
        maxLength: Int = 2048
    ) -> (ids: MLXArray, mask: MLXArray) {
        encode(lyrics, maxLength: maxLength)
    }

    // MARK: - Testing Factory

    /// Create a minimal functional tokenizer suitable for unit tests.
    ///
    /// Uses an in-memory scalar encoder so structural tests do not depend on
    /// swift-transformers BPE edge cases or external checkpoint files.
    public static func forTesting() throws -> PiMPAudioTokenizer {
        PiMPAudioTokenizer { text in
            text.unicodeScalars.compactMap { scalar in
                guard !CharacterSet.whitespacesAndNewlines.contains(scalar) else { return nil }
                return Int(scalar.value % 10_000) + 1
            }
        }
    }

    // MARK: - Private Helpers

    private func encode(_ text: String, maxLength: Int) -> (ids: MLXArray, mask: MLXArray) {
        var tokenIds = box.encode(text)

        // Truncate if the text encodes to more tokens than maxLength.
        // (maxLength is now a truncation CAP, not a pad-to length — matches
        // upstream's `padding="longest", truncation=True, max_length=N` in
        // acestep/core/generation/handler/conditioning_text.py.)
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }

        // For an empty/whitespace input the encoder may produce zero tokens.
        // Returning a [1, 0] tensor breaks downstream concatenation; emit a
        // single PAD-with-mask=0 entry as a degenerate-but-shaped placeholder.
        let realCount = max(tokenIds.count, 1)
        var idsFlat = tokenIds.map { Int32($0) }
        var maskFlat = [Int32](repeating: 1, count: tokenIds.count)
        if idsFlat.isEmpty {
            idsFlat = [0]
            maskFlat = [0]
        }

        let ids = MLXArray(idsFlat).reshaped([1, realCount])
        let mask = MLXArray(maskFlat).reshaped([1, realCount])
        return (ids, mask)
    }

    /// Load a JSON file as a `Config` value, replicating the internal
    /// `HubApi.configuration(fileURL:)` logic (which is not `public`).
    private static func loadConfig(from url: URL) throws -> Config {
        let data = try Data(contentsOf: url)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dict = parsed as? [NSString: Any] else {
            throw TokenizerError.malformedVocab
        }
        return Config(dict)
    }

}
