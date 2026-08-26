import Foundation
import CryptoKit

public actor DownloadClient: NativeModelPackDownloading {
    public init() {}

    public func huggingFaceURL(
        repo: String,
        revision: String = "main",
        file: String
    ) throws -> URL {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repo)/resolve/\(revision)/\(file)"
        guard let url = components.url else {
            throw DownloadError.invalidURL("\(repo)/\(revision)/\(file)")
        }
        return url
    }

    /// Non-empty SHA-256 hex to verify after download, or `nil` to skip.
    /// Missing/blank pins must not block a download.
    public nonisolated static func checksumToVerify(_ expectedHash: String?) -> String? {
        guard let expectedHash else { return nil }
        let trimmed = expectedHash.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Downloads a file and verifies its SHA256 checksum if provided.
    /// On verification failure, deletes the file and retries once.
    /// If `expectedHash` is nil or empty, skips verification.
    public func downloadAndVerify(
        repo: String,
        revision: String = "main",
        file: String,
        to destination: URL,
        expectedHash: String?,
        bearerToken: String? = nil,
        onProgress: @Sendable @escaping (Double) -> Void,
        onRetry: @Sendable @escaping () -> Void
    ) async throws {
        let source = try huggingFaceURL(
            repo: repo,
            revision: revision,
            file: file
        )
        try await downloadFile(from: source, to: destination, bearerToken: bearerToken, onProgress: onProgress)

        guard let hash = Self.checksumToVerify(expectedHash) else { return }

        let verified = await verifySHA256(file: destination, expected: hash)
        if verified { return }

        // First verification failed — delete and retry once
        try? FileManager.default.removeItem(at: destination)
        onRetry()

        try await downloadFile(from: source, to: destination, bearerToken: bearerToken, onProgress: onProgress)

        let retryVerified = await verifySHA256(file: destination, expected: hash)
        if retryVerified { return }

        // Second verification failed — hard fail
        try? FileManager.default.removeItem(at: destination)
        throw DownloadError.checksumMismatch(file: file)
    }

    /// Raw download without verification. Used internally by downloadAndVerify.
    public func download(
        repo: String,
        revision: String = "main",
        file: String,
        to destination: URL,
        bearerToken: String? = nil,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        try await downloadFile(
            from: try huggingFaceURL(
                repo: repo,
                revision: revision,
                file: file
            ),
            to: destination,
            bearerToken: bearerToken,
            onProgress: onProgress
        )
    }

    /// Stream a remote URL to `destination` via a download task (disk-backed).
    ///
    /// Progress is real byte progress when the server provides Content-Length.
    /// Does not buffer the entire payload in RAM.
    public func download(
        from url: URL,
        to destination: URL,
        bearerToken: String? = nil,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        try await downloadFile(
            from: url,
            to: destination,
            bearerToken: bearerToken,
            onProgress: onProgress
        )
    }

    private func downloadFile(
        from url: URL,
        to destination: URL,
        bearerToken: String?,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        var request = URLRequest(url: url)
        if let token = bearerToken, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // Apple docs: when a downloadTask is created with a completionHandler,
        // the session delegate's per-task callbacks are NOT invoked. So we use
        // the delegate-only API (downloadTask(with:) — no closure) and bridge
        // didFinishDownloadingTo / didCompleteWithError to a continuation.
        let orchestrator = DownloadOrchestrator(onProgress: onProgress)
        let session = URLSession(configuration: .default, delegate: orchestrator, delegateQueue: nil)

        // Emit an immediate 0% so the UI can switch out of "Waiting…" before
        // the first delegate callback fires.
        onProgress(0.0)

        let downloadTask = session.downloadTask(with: request)

        let (tempURL, response): (URL, URLResponse)
        do {
            (tempURL, response) = try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<(URL, URLResponse), Error>) in
                    orchestrator.start(downloadTask: downloadTask, continuation: continuation)
                }
            } onCancel: {
                downloadTask.cancel()
                session.invalidateAndCancel()
            }
        } catch {
            session.invalidateAndCancel()
            throw error
        }

        session.invalidateAndCancel()

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            try? FileManager.default.removeItem(at: tempURL)
            throw DownloadError.httpError(
                statusCode: (response as? HTTPURLResponse)?.statusCode ?? -1
            )
        }

        let destDir = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
        // Unconditionally remove any stale destination. Use URL-based removeItem
        // (not the String path() variant) to avoid percent-encoding ambiguity
        // around paths containing spaces like "Application Support". try? swallows
        // the "no such file" error when the destination doesn't exist yet.
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: tempURL, to: destination)

        // Final 100% in case the delegate stopped firing slightly before the end.
        onProgress(1.0)
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
    case invalidURL(String)

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
        case .invalidURL(let spec):
            return "Invalid Hugging Face URL for \(spec)"
        }
    }
}

/// Bridges URLSessionDownloadDelegate callbacks to a checked continuation,
/// throttles per-byte progress events to ~10 Hz, and hands the final temp
/// file URL to the awaiting caller. One orchestrator per session/task —
/// session must be invalidated after use.
private final class DownloadOrchestrator: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Double) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<(URL, URLResponse), Error>?
    private var resolved = false
    private var lastFiredAt: Date = .distantPast
    private static let throttleInterval: TimeInterval = 0.1   // 10 Hz

    init(onProgress: @Sendable @escaping (Double) -> Void) {
        self.onProgress = onProgress
    }

    func start(
        downloadTask: URLSessionDownloadTask,
        continuation: CheckedContinuation<(URL, URLResponse), Error>
    ) {
        lock.lock()
        self.continuation = continuation
        lock.unlock()
        downloadTask.resume()
    }

    private func resolve(_ result: Result<(URL, URLResponse), Error>) {
        lock.lock()
        guard !resolved, let continuation else {
            lock.unlock()
            return
        }
        resolved = true
        self.continuation = nil
        lock.unlock()

        switch result {
        case .success(let pair): continuation.resume(returning: pair)
        case .failure(let err):  continuation.resume(throwing: err)
        }
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // URLSession reclaims `location` as soon as this method returns, so
        // we move it to our own temp location SYNCHRONOUSLY here, then resume.
        let safeURL = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString)
        do {
            try? FileManager.default.removeItem(at: safeURL)
            try FileManager.default.moveItem(at: location, to: safeURL)
            let response = downloadTask.response ?? URLResponse()
            resolve(.success((safeURL, response)))
        } catch {
            resolve(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        // Only meaningful on error; success arrives via didFinishDownloadingTo.
        if let error {
            resolve(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)

        // Throttle to ~10 Hz so we don't fan out hundreds of @MainActor
        // publishes per second to SwiftUI. Always pass 100% through.
        lock.lock()
        let now = Date()
        let elapsed = now.timeIntervalSince(lastFiredAt)
        let shouldFire = progress >= 1.0 || elapsed >= Self.throttleInterval
        if shouldFire { lastFiredAt = now }
        lock.unlock()

        guard shouldFire else { return }
        onProgress(progress)
    }
}
