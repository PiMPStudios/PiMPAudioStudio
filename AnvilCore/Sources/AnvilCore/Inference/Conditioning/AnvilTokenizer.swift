import Foundation
import Hub
import MLX
import Tokenizers

/// Thin wrapper around a swift-transformers BPE tokenizer.
///
/// Loads a tokenizer from a local `tokenizer.json` file and provides
/// `encodeText` and `encodeLyrics` convenience methods that return
/// padded `MLXArray` tensors shaped `[1, maxLength]`.
public struct AnvilTokenizer: Sendable {
    // Box the protocol existential so the struct stays Sendable without
    // requiring the upstream type to declare conformance itself.
    private final class Box: @unchecked Sendable {
        let tokenizer: any Tokenizer
        init(_ tokenizer: any Tokenizer) { self.tokenizer = tokenizer }
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

    // MARK: - Encode

    /// Encode text for conditioning inputs (e.g. prompt, genre tags).
    ///
    /// - Parameters:
    ///   - text: The text to encode.
    ///   - maxLength: Sequence length of the returned arrays. Default `256`.
    /// - Returns: A tuple `(ids, mask)` of `Int32 MLXArray` shaped `[1, maxLength]`.
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
    /// - Returns: A tuple `(ids, mask)` of `Int32 MLXArray` shaped `[1, maxLength]`.
    public func encodeLyrics(
        _ lyrics: String,
        maxLength: Int = 2048
    ) -> (ids: MLXArray, mask: MLXArray) {
        encode(lyrics, maxLength: maxLength)
    }

    // MARK: - Testing Factory

    /// Create a minimal functional tokenizer suitable for unit tests.
    ///
    /// Writes a compact BPE `tokenizer.json` to a temporary directory and
    /// loads it. The vocabulary contains every printable ASCII character so
    /// the test strings used by `AnvilTokenizerTests` all produce real tokens.
    /// No external checkpoint files are required.
    public static func forTesting() throws -> AnvilTokenizer {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnvilTokenizerForTesting", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let tokenizerURL = dir.appendingPathComponent("tokenizer.json")
        let json = Self.minimalTokenizerJSON()
        try json.write(to: tokenizerURL, atomically: true, encoding: .utf8)

        return try AnvilTokenizer(configURL: tokenizerURL)
    }

    // MARK: - Private Helpers

    private func encode(_ text: String, maxLength: Int) -> (ids: MLXArray, mask: MLXArray) {
        var tokenIds = box.tokenizer.encode(text: text)

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

    /// Returns the JSON string for a minimal valid BPE tokenizer.
    ///
    /// The vocabulary maps every printable ASCII character (and space) to a
    /// unique integer ID starting at 1. Unknown characters fall back through
    /// `hexaEncode` in BPETokenizer, which is fine for testing purposes.
    private static func minimalTokenizerJSON() -> String {
        // Build a vocab covering printable ASCII (0x20–0x7E) + common chars.
        // IDs start at 1; 0 is reserved as the padding token.
        var vocab: [String: Int] = [:]
        var nextID = 1
        for codePoint in 32...126 {
            let char = String(UnicodeScalar(codePoint)!)
            vocab[char] = nextID
            nextID += 1
        }

        let vocabEntries = vocab
            .sorted { $0.value < $1.value }
            .map { pair -> String in
                let escaped = pair.key
                    .replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "\"", with: "\\\"")
                return "      \"\(escaped)\": \(pair.value)"
            }
            .joined(separator: ",\n")

        return """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [],
          "normalizer": null,
          "pre_tokenizer": null,
          "post_processor": null,
          "decoder": null,
          "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": {
        \(vocabEntries)
            },
            "merges": []
          }
        }
        """
    }
}
