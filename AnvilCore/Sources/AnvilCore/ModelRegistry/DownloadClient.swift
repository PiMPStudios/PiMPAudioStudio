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

        // Emit an immediate 0% so the UI can switch out of "Waiting…" before
        // the first delegate callback fires.
        onProgress(0.0)

        // Use the completion-handler downloadTask API wrapped in a continuation.
        // The async `download(for:)` overload does not reliably fire the
        // session delegate's didWriteData callback, so we use the older API
        // that guarantees per-chunk progress events.
        let task = ManagedTaskHolder()
        let result: Result<(URL, URLResponse), Error> = await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Result<(URL, URLResponse), Error>, Never>) in
                let downloadTask = session.downloadTask(with: request) { tempURL, response, error in
                    if let error {
                        continuation.resume(returning: .failure(error))
                        return
                    }
                    guard let tempURL, let response else {
                        continuation.resume(returning: .failure(
                            DownloadError.httpError(statusCode: -1)
                        ))
                        return
                    }
                    // URLSession reclaims the temp file as soon as this completion
                    // returns — move it into our own temp location synchronously.
                    let safeURL = FileManager.default.temporaryDirectory
                        .appending(path: UUID().uuidString)
                    do {
                        try? FileManager.default.removeItem(at: safeURL)
                        try FileManager.default.moveItem(at: tempURL, to: safeURL)
                        continuation.resume(returning: .success((safeURL, response)))
                    } catch {
                        continuation.resume(returning: .failure(error))
                    }
                }
                task.set(downloadTask)
                downloadTask.resume()
            }
        } onCancel: {
            task.cancel()
            session.invalidateAndCancel()
        }

        session.invalidateAndCancel()

        let (tempURL, response): (URL, URLResponse)
        switch result {
        case .success(let pair):
            (tempURL, response) = pair
        case .failure(let error):
            throw error
        }

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

/// Thread-safe holder for the in-flight URLSessionDownloadTask so the
/// cancellation handler can signal it without a race against task creation.
private final class ManagedTaskHolder: @unchecked Sendable {
    private let lock = NSLock()
    private var task: URLSessionDownloadTask?
    private var cancelled = false

    func set(_ task: URLSessionDownloadTask) {
        lock.lock()
        defer { lock.unlock() }
        if cancelled {
            task.cancel()
        } else {
            self.task = task
        }
    }

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        cancelled = true
        task?.cancel()
    }
}

private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Double) -> Void
    private let lock = NSLock()
    private var lastFiredAt: Date = .distantPast
    private static let throttleInterval: TimeInterval = 0.1   // 10 Hz

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
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)

        // Throttle delegate callbacks to ~10 Hz so we don't fan out hundreds
        // of @MainActor publishes per second to SwiftUI. Always pass 100%
        // through so the final state is never lost.
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
