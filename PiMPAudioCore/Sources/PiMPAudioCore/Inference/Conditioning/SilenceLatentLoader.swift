import Foundation
import MLX

/// Reads the `silence_latent.pt` tensor that ships with each ACE-Step DiT
/// checkpoint and returns it as an MLXArray.
///
/// Upstream stores the tensor as a 3D PyTorch save with shape `[1, 64, 15000]`
/// (channels-first, NCL). We transpose to `[1, 15000, 64]` to match the
/// channels-last NLC convention used everywhere else in the PiMPAudioCore
/// inference pipeline.
///
/// Used by `ConditioningActor` to build the text2music `context_latents` as
/// `cat([silence_latent[:, :tFrames, :], chunk_mask_ones], axis=-1)` —
/// matching upstream's
/// `acestep/models/turbo/modeling_acestep_v15_turbo.py:1680, 1931`.
public enum SilenceLatentLoader {
    /// Shape of the raw tensor stored in `silence_latent.pt`.
    /// Verified empirically against
    /// `~/Developer/GitRepos/ACE-Step-1.5/checkpoints/acestep-v15-{turbo,sft}/silence_latent.pt`.
    /// 1 batch × 64 latent channels × 15,000 frames (= 600 s at 25 fps).
    public static let storedShape: [Int] = [1, 64, 15000]

    public enum LoaderError: Error, LocalizedError {
        case unzipFailed(stderr: String)
        case unexpectedSize(bytes: Int, expected: Int)
        case fileNotFound(path: String)

        public var errorDescription: String? {
            switch self {
            case .unzipFailed(let stderr):
                return "Failed to extract silence_latent.pt: \(stderr)"
            case .unexpectedSize(let bytes, let expected):
                return "silence_latent.pt extracted \(bytes) bytes, expected \(expected)."
            case .fileNotFound(let path):
                return "silence_latent.pt not found at \(path)"
            }
        }
    }

    /// Load `silence_latent.pt` and return shape `[1, 15000, 64]` MLXArray.
    ///
    /// The .pt file is a PyTorch save = a zip archive with this layout:
    ///   silence_latent/data.pkl              (pickle metadata)
    ///   silence_latent/data/0                (raw float32 binary, 3.84 MB)
    ///   silence_latent/byteorder
    ///   silence_latent/version
    ///
    /// We use `/usr/bin/unzip -p` to stream the `*/data/0` entry to stdout —
    /// no temp file, no third-party zip library, no pickle parsing. The shape
    /// is hardcoded since this is a fixed-format file shipped by upstream.
    public static func load(from url: URL) throws -> MLXArray {
        let path = url.path(percentEncoded: false)
        guard FileManager.default.fileExists(atPath: path) else {
            throw LoaderError.fileNotFound(path: path)
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        proc.arguments = ["-p", path, "*/data/0"]
        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardOutput = stdout
        proc.standardError = stderr
        try proc.run()

        let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()

        guard proc.terminationStatus == 0 else {
            let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
            let stderrStr = String(data: stderrData, encoding: .utf8) ?? "<binary>"
            throw LoaderError.unzipFailed(stderr: stderrStr)
        }

        let totalElements = storedShape.reduce(1, *)
        let expectedBytes = totalElements * MemoryLayout<Float>.size
        guard stdoutData.count == expectedBytes else {
            throw LoaderError.unexpectedSize(bytes: stdoutData.count, expected: expectedBytes)
        }

        let floats = stdoutData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> [Float] in
            let buffer = raw.bindMemory(to: Float.self)
            return Array(buffer)
        }

        // [1, 64, 15000] (channels-first) → [1, 15000, 64] (channels-last)
        let raw = MLXArray(floats, [storedShape[0], storedShape[1], storedShape[2]])
        return raw.transposed(0, 2, 1)
    }
}
