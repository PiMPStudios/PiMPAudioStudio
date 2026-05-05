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

    /// Load a BPE tokenizer from a `tokenizer.json` file on disk.
    ///
    /// The file is expected to follow the Hugging Face `tokenizer.json`
    /// schema, with `model.type`, `model.vocab`, and `model.merges` keys.
    /// A companion `tokenizer_config.json` is optional but must be present
    /// in the same directory if the tokenizer class cannot be inferred.
    public init(configURL: URL) throws {
        let tokenizerData = try Self.loadConfig(from: configURL)

        // Build a minimal tokenizerConfig that declares PreTrainedTokenizer.
        // The `from(tokenizerConfig:tokenizerData:)` factory uses the
        // `tokenizer_class` key to select an implementation class.
        let tokenizerConfigURL = configURL
            .deletingLastPathComponent()
            .appendingPathComponent("tokenizer_config.json")

        let tokenizerConfig: Config
        if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
            tokenizerConfig = try Self.loadConfig(from: tokenizerConfigURL)
        } else {
            // Fall back to a minimal config pointing at the generic BPE class.
            tokenizerConfig = Config(["tokenizer_class": "PreTrainedTokenizer"] as [NSString: Any])
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
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }

        let realCount = tokenIds.count
        let padCount = maxLength - realCount

        // Build Int32 arrays padded with 0.
        var idsFlat = tokenIds.map { Int32($0) }
        idsFlat += [Int32](repeating: 0, count: padCount)

        var maskFlat = [Int32](repeating: 1, count: realCount)
        maskFlat += [Int32](repeating: 0, count: padCount)

        let ids = MLXArray(idsFlat).reshaped([1, maxLength])
        let mask = MLXArray(maskFlat).reshaped([1, maxLength])
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
