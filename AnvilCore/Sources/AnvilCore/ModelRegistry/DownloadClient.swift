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
        onProgress: @Sendable @escaping (Double) -> Void,
        onRetry: @Sendable @escaping () -> Void
    ) async throws {
        try await downloadFile(repo: repo, file: file, to: destination, onProgress: onProgress)

        guard let hash = expectedHash, !hash.isEmpty else { return }

        let verified = await verifySHA256(file: destination, expected: hash)
        if verified { return }

        // First verification failed — delete and retry once
        try? FileManager.default.removeItem(at: destination)
        onRetry()

        try await downloadFile(repo: repo, file: file, to: destination, onProgress: onProgress)

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
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        try await downloadFile(repo: repo, file: file, to: destination, onProgress: onProgress)
    }

    private func downloadFile(
        repo: String,
        file: String,
        to destination: URL,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        let url = huggingFaceURL(repo: repo, file: file)
        let delegate = DownloadProgressDelegate(onProgress: onProgress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

        try await withTaskCancellationHandler {
            defer { session.invalidateAndCancel() }
            let (tempURL, response) = try await session.download(from: url)

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

public enum DownloadError: Error, Sendable {
    case httpError(statusCode: Int)
    case checksumMismatch(file: String)
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
