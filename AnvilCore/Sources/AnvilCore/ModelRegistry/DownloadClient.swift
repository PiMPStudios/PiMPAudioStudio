import Foundation
import CryptoKit

public actor DownloadClient {
    public init() {}

    public func huggingFaceURL(repo: String, file: String) -> URL {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repo)/resolve/main/\(file)"
        guard let url = components.url else {
            preconditionFailure("Invalid HuggingFace URL for repo: \(repo), file: \(file)")
        }
        return url
    }

    /// Downloads a file and verifies its SHA256 checksum if provided.
    /// On verification failure, deletes the file and retries once.
    /// If `expectedHash` is nil or empty, skips verification.
    public func downloadAndVerify(
        repo: String,
        file: String,
        to destination: URL,
        expectedHash: String?,
        bearerToken: String? = nil,
        onProgress: @Sendable @escaping (Double) -> Void,
        onRetry: @Sendable @escaping () -> Void
    ) async throws {
        try await downloadFile(repo: repo, file: file, to: destination, bearerToken: bearerToken, onProgress: onProgress)

        guard let hash = expectedHash, !hash.isEmpty else { return }

        let verified = await verifySHA256(file: destination, expected: hash)
        if verified { return }

        // First verification failed — delete and retry once
        try? FileManager.default.removeItem(at: destination)
        onRetry()

        try await downloadFile(repo: repo, file: file, to: destination, bearerToken: bearerToken, onProgress: onProgress)

        let retryVerified = await verifySHA256(file: destination, expected: hash)
        if retryVerified { return }

        // Second verification failed — hard fail
        try? FileManager.default.removeItem(at: destination)
        throw DownloadError.checksumMismatch(file: file)
    }

    /// Raw download without verification. Used internally by downloadAndVerify.
    public func download(
        repo: String,
        file: String,
        to destination: URL,
        bearerToken: String? = nil,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        try await downloadFile(
            repo: repo,
            file: file,
            to: destination,
            bearerToken: bearerToken,
            onProgress: onProgress
        )
    }

    private func downloadFile(
        repo: String,
        file: String,
        to destination: URL,
        bearerToken: String?,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        var request = URLRequest(url: huggingFaceURL(repo: repo, file: file))
        if let token = bearerToken, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let delegate = DownloadProgressDelegate(onProgress: onProgress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

        try await withTaskCancellationHandler {
            defer { session.invalidateAndCancel() }
            let (tempURL, response) = try await session.download(for: request)

            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                throw DownloadError.httpError(
                    statusCode: (response as? HTTPURLResponse)?.statusCode ?? -1
                )
            }

            let destDir = destination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: destination.path()) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: tempURL, to: destination)
        } onCancel: {
            session.invalidateAndCancel()
        }
    }

    public nonisolated func verifySHA256(file: URL, expected: String) async -> Bool {
        await Task.detached(priority: .userInitiated) {
            guard let handle = try? FileHandle(forReadingFrom: file) else { return false }
            defer { try? handle.close() }

            var hasher = SHA256()
            let bufferSize = 1_024 * 1_024  // 1 MB chunks

            while true {
                let chunk: Data? = autoreleasepool {
                    let data = handle.readData(ofLength: bufferSize)
                    return data.isEmpty ? nil : data
                }
                guard let chunk else { break }
                hasher.update(data: chunk)
            }

            let digest = hasher.finalize()
            let hex = digest.map { String(format: "%02x", $0) }.joined()
            return hex == expected.lowercased()
        }.value
    }
}

public enum DownloadError: Error, Sendable, LocalizedError {
    case httpError(statusCode: Int)
    case checksumMismatch(file: String)
    case authRequired(repo: String)
    case authForbidden(repo: String)

    public var errorDescription: String? {
        switch self {
        case .httpError(let statusCode):
            let meaning: String
            switch statusCode {
            case 401: meaning = "Unauthorized — authentication or model access may be required"
            case 403: meaning = "Forbidden — you may need to accept the model's license on HuggingFace"
            case 404: meaning = "Not Found — the model repository or file does not exist"
            case 429: meaning = "Too Many Requests — rate limited, please try again later"
            case 500...599: meaning = "Server error — HuggingFace may be experiencing issues"
            case -1: meaning = "Unknown — no HTTP response received"
            default: meaning = HTTPURLResponse.localizedString(forStatusCode: statusCode)
            }
            return "Download failed (HTTP \(statusCode)): \(meaning)"
        case .checksumMismatch(let file):
            return "Checksum verification failed for '\(file)' — the file may be corrupted"
        case .authRequired(let repo):
            return "\(repo) is a gated model. Add a HuggingFace access token in Settings → Models, and accept the model's license on huggingface.co."
        case .authForbidden(let repo):
            return "HuggingFace rejected the access token for \(repo). Verify the token is valid and that the model license has been accepted."
        }
    }
}

private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Double) -> Void

    init(onProgress: @Sendable @escaping (Double) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {}

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }
}
