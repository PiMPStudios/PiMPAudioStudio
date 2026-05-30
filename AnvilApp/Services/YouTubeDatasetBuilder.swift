import Foundation
import AVFoundation
import Darwin

// MARK: - Config & Models

struct YouTubeDatasetConfig {
    var clipLength: TimeInterval = 35
    var clipsPerTrack: Int = 8
    var maxTotalClips: Int = 40
}

struct YouTubeVideoInfo {
    let id: String
    let title: String
    let duration: TimeInterval?
}

struct YouTubeDatasetResult {
    let outputDirectory: URL
    let clips: [(url: URL, caption: String)]
    var clipCount: Int { clips.count }
}

enum YouTubeImportPhase: Equatable {
    case idle
    case fetchingPlaylist
    case downloading(index: Int, total: Int, title: String)
    case chunking(index: Int, total: Int)
    case complete(outputDirectory: URL, clipCount: Int)
    case failed(String)

    var isRunning: Bool {
        switch self {
        case .fetchingPlaylist, .downloading, .chunking: return true
        default: return false
        }
    }
}

// MARK: - Service

@MainActor
final class YouTubeDatasetBuilder: ObservableObject {
    @Published var phase: YouTubeImportPhase = .idle
    @Published var resultClips: [(url: URL, caption: String)] = []

    private var task: Task<Void, Never>?

    func start(channelURL: URL, config: YouTubeDatasetConfig, outputBase: URL, styleTemplate: String = "") {
        cancel()
        resultClips = []
        phase = .fetchingPlaylist

        task = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await YouTubeDatasetBuilder.run(
                    channelURL: channelURL,
                    config: config,
                    outputBase: outputBase,
                    styleTemplate: styleTemplate,
                    onPhase: { [weak self] p in await MainActor.run { self?.phase = p } }
                )
                self.resultClips = result.clips
                self.phase = .complete(outputDirectory: result.outputDirectory, clipCount: result.clipCount)
            } catch is CancellationError {
                self.phase = .idle
            } catch {
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
    }

    // MARK: - yt-dlp Discovery

    /// /usr/bin/env is always reachable from a sandboxed app.
    /// We route every yt-dlp call through it so we never need to touch
    /// /opt/homebrew directly — the sandbox blocks fileExists and direct
    /// execution of arbitrary Homebrew paths even with Full Disk Access.
    private static let envURL = URL(fileURLWithPath: "/usr/bin/env")

    /// PATH passed to every subprocess.  Sandboxed processes inherit a minimal
    /// environment that does not include /opt/homebrew/bin, so we set it explicitly.
    private static let ytDlpEnvironment: [String: String] = [
        "PATH": "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": NSHomeDirectory(),
    ]

    /// Returns the installed yt-dlp version string, or nil if not reachable.
    static func ytDlpVersion() -> String? {
        let result = try? runProcessSync(
            arguments: ["yt-dlp", "--version"],
            environment: ytDlpEnvironment
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        // Sanity check: a real version string is non-empty and starts with a digit
        guard let v = result, !v.isEmpty, v.first?.isNumber == true else { return nil }
        return v
    }

    // MARK: - Core Pipeline

    static func run(
        channelURL: URL,
        config: YouTubeDatasetConfig,
        outputBase: URL,
        styleTemplate: String = "",
        onPhase: @escaping (YouTubeImportPhase) async -> Void
    ) async throws -> YouTubeDatasetResult {
        guard ytDlpVersion() != nil else { throw YouTubeDatasetError.ytDlpNotFound }

        let maxVideos = max(1,
            Int(ceil(Double(config.maxTotalClips) / Double(max(1, config.clipsPerTrack)))) + 1
        )
        let videos = try await fetchPlaylist(url: channelURL, limit: maxVideos)
        guard !videos.isEmpty else { throw YouTubeDatasetError.emptyPlaylist }

        // Create output dirs (app has write access here via security-scoped bookmark)
        let folderTag = sanitizeName(videos.first?.title ?? "youtube-dataset")
        let dateTag = String(ISO8601DateFormatter().string(from: Date()).prefix(10))
        let outputDir = outputBase
            .appendingPathComponent("TrainingDatasets", isDirectory: true)
            .appendingPathComponent("\(folderTag)_\(dateTag)", isDirectory: true)
        let clipsDir = outputDir.appendingPathComponent("clips", isDirectory: true)
        try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)

        var allClips: [(url: URL, caption: String)] = []

        for (idx, video) in videos.prefix(maxVideos).enumerated() {
            try Task.checkCancellation()
            guard allClips.count < config.maxTotalClips else { break }

            let videoCount = min(videos.count, maxVideos)
            await onPhase(.downloading(index: idx + 1, total: videoCount, title: video.title))

            // Download to temp (sandbox always allows writes here; subprocess runs under same sandbox)
            let tempDir = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

            do {
                let audioURL = try await downloadAudio(video: video, to: tempDir)

                try Task.checkCancellation()
                await onPhase(.chunking(index: idx + 1, total: videoCount))

                let remaining = config.maxTotalClips - allClips.count
                let chunks = try await chunkAudio(
                    from: audioURL,
                    title: video.title,
                    videoIndex: idx + 1,
                    clipLength: config.clipLength,
                    maxClips: min(config.clipsPerTrack, remaining),
                    styleTemplate: styleTemplate,
                    outputDirectory: clipsDir
                )
                allClips.append(contentsOf: chunks)
            } catch is CancellationError {
                try? FileManager.default.removeItem(at: tempDir)
                throw CancellationError()
            } catch {
                // Skip failed video — log and continue
                print("[YouTubeDatasetBuilder] Skipping \(video.id): \(error.localizedDescription)")
            }
            try? FileManager.default.removeItem(at: tempDir)
        }

        guard !allClips.isEmpty else { throw YouTubeDatasetError.noClipsProduced }

        try writeCaptionsFile(clips: allClips, to: outputDir)
        return YouTubeDatasetResult(outputDirectory: outputDir, clips: allClips)
    }

    // MARK: - Playlist Fetch

    static func fetchPlaylist(url: URL, limit: Int) async throws -> [YouTubeVideoInfo] {
        let raw = try await runProcess(
            arguments: [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                "--no-warnings",
                "--no-cache-dir",
                "--playlist-end", "\(limit)",
                url.absoluteString,
            ]
        )
        var videos: [YouTubeVideoInfo] = []
        for line in raw.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty,
                  let data = trimmed.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let id = json["id"] as? String,
                  let title = json["title"] as? String
            else { continue }
            videos.append(YouTubeVideoInfo(id: id, title: title, duration: json["duration"] as? Double))
        }
        return videos
    }

    // MARK: - Audio Download

    static func downloadAudio(video: YouTubeVideoInfo, to outputDir: URL) async throws -> URL {
        _ = try await runProcess(
            arguments: [
                "yt-dlp",
                "-x",
                "--audio-quality", "0",
                "--no-cache-dir",
                "--no-warnings",
                "--no-playlist",
                "-o", outputDir.path + "/audio.%(ext)s",
                "https://www.youtube.com/watch?v=\(video.id)",
            ]
        )
        // yt-dlp picks the extension — find whichever it chose
        for ext in ["m4a", "mp3", "webm", "opus", "ogg", "wav", "aac"] {
            let candidate = outputDir.appendingPathComponent("audio.\(ext)")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        throw YouTubeDatasetError.downloadFailed(video.id)
    }

    // MARK: - Audio Chunking (pure AVFoundation)

    static func chunkAudio(
        from sourceURL: URL,
        title: String,
        videoIndex: Int,
        clipLength: TimeInterval,
        maxClips: Int,
        styleTemplate: String = "",
        outputDirectory: URL,
        filenamePrefix: String = "v",
        samplingMode: LocalDatasetClipSamplingMode = .sequential,
        samplingSeed: Int = 1337,
        sourceIdentifier: String? = nil
    ) async throws -> [(url: URL, caption: String)] {
        let asset = AVURLAsset(url: sourceURL)
        let cmDuration = try await asset.load(.duration)
        let totalSeconds = cmDuration.seconds
        guard totalSeconds > 0 else { return [] }

        // Short track: single clip
        if totalSeconds <= clipLength {
            let dest = outputDirectory.appendingPathComponent(
                clipFilename(prefix: filenamePrefix, sourceIndex: videoIndex, clipIndex: 1)
            )
            try await exportRange(asset: asset,
                                  range: CMTimeRange(start: .zero, duration: cmDuration),
                                  to: dest)
            let caption = styleTemplate.isEmpty ? title : styleTemplate
            return [(url: dest, caption: caption)]
        }

        let startTimes = LocalDatasetClipSampler.startTimes(
            totalSeconds: totalSeconds,
            clipLength: clipLength,
            maxClips: maxClips,
            mode: samplingMode,
            seed: samplingSeed,
            sourceIdentifier: sourceIdentifier ?? sourceURL.standardizedFileURL.path
        )
        var clips: [(url: URL, caption: String)] = []

        for (i, startSec) in startTimes.enumerated() {
            try Task.checkCancellation()
            let start = CMTime(seconds: startSec, preferredTimescale: 600)
            let rawEnd = startSec + clipLength
            let actualLen = rawEnd <= totalSeconds ? clipLength : totalSeconds - startSec
            let dur = CMTime(seconds: actualLen, preferredTimescale: 600)

            let dest = outputDirectory.appendingPathComponent(
                clipFilename(prefix: filenamePrefix, sourceIndex: videoIndex, clipIndex: i + 1)
            )
            try await exportRange(asset: asset,
                                  range: CMTimeRange(start: start, duration: dur),
                                  to: dest)
            let caption: String
            if styleTemplate.isEmpty {
                caption = startTimes.count > 1 ? "\(title) (clip \(i + 1))" : title
            } else {
                caption = styleTemplate
            }
            clips.append((url: dest, caption: caption))
        }
        return clips
    }

    private static func clipFilename(prefix: String, sourceIndex: Int, clipIndex: Int) -> String {
        let source = String(format: "%03d", sourceIndex)
        let clip = String(format: "%03d", clipIndex)
        return "\(prefix)\(source)_clip_\(clip).m4a"
    }

    static func exportRange(asset: AVAsset, range: CMTimeRange, to dest: URL) async throws {
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        guard let audioTrack = tracks.first else { throw YouTubeDatasetError.noAudioTrack }

        let composition = AVMutableComposition()
        guard let compTrack = composition.addMutableTrack(
            withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid
        ) else { throw YouTubeDatasetError.compositionFailed }

        try compTrack.insertTimeRange(range, of: audioTrack, at: .zero)

        guard let session = AVAssetExportSession(
            asset: composition, presetName: AVAssetExportPresetAppleM4A
        ) else { throw YouTubeDatasetError.exportSessionFailed }

        session.outputURL = dest
        session.outputFileType = .m4a
        await session.export()
        guard session.status == .completed else {
            throw YouTubeDatasetError.exportFailed(session.error?.localizedDescription ?? "unknown")
        }
    }

    // MARK: - Captions File

    static func writeCaptionsFile(clips: [(url: URL, caption: String)], to outputDir: URL) throws {
        let entries = clips.map { ["file": "clips/\($0.url.lastPathComponent)", "caption": $0.caption] }
        let data = try JSONSerialization.data(withJSONObject: entries, options: .prettyPrinted)
        try data.write(to: outputDir.appendingPathComponent("captions.json"))
    }

    // MARK: - Process Helpers

    /// Run a command via /usr/bin/env with the Homebrew PATH injected.
    /// `arguments` should start with the command name (e.g. "yt-dlp").
    static func runProcess(arguments: [String]) async throws -> String {
        let runner = CapturedProcess(
            executableURL: envURL,
            arguments: arguments,
            environment: ytDlpEnvironment
        )
        return try await runner.run()
    }

    /// Synchronous variant — used only for the version check on the UI thread.
    static func runProcessSync(arguments: [String],
                               environment: [String: String]) throws -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = envURL
        process.arguments = arguments
        process.environment = environment
        process.standardOutput = pipe
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }

    // MARK: - Helpers

    static func sanitizeName(_ input: String) -> String {
        let clean = input
            .replacingOccurrences(of: "[^a-zA-Z0-9 \\-]", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: " +", with: "-", options: .regularExpression)
        return String(clean.prefix(40))
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .lowercased()
    }
}

private final class CapturedProcess: @unchecked Sendable {
    private let process = Process()
    private let stdout = Pipe()
    private let stderr = Pipe()
    private let lock = NSLock()
    private var stdoutData = Data()
    private var stderrData = Data()
    private var continuation: CheckedContinuation<String, Error>?
    private var didResume = false
    private var didCancel = false

    init(executableURL: URL, arguments: [String], environment: [String: String]) {
        process.executableURL = executableURL
        process.arguments = arguments
        process.environment = environment
        process.standardOutput = stdout
        process.standardError = stderr
    }

    func run() async throws -> String {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
                start(continuation: continuation)
            }
        } onCancel: {
            cancel()
        }
    }

    private func start(continuation: CheckedContinuation<String, Error>) {
        lock.lock()
        if didCancel {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
            return
        }
        self.continuation = continuation
        lock.unlock()

        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.appendStdout(handle.availableData)
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.appendStderr(handle.availableData)
        }
        process.terminationHandler = { [weak self] process in
            self?.finish(process: process)
        }

        do {
            try process.run()
        } catch {
            finish(error: error)
        }
    }

    private func appendStdout(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock()
        stdoutData.append(data)
        lock.unlock()
    }

    private func appendStderr(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock()
        stderrData.append(data)
        lock.unlock()
    }

    private func finish(process: Process) {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil
        let remainingStdout = stdout.fileHandleForReading.readDataToEndOfFile()
        let remainingStderr = stderr.fileHandleForReading.readDataToEndOfFile()

        lock.lock()
        stdoutData.append(remainingStdout)
        stderrData.append(remainingStderr)
        guard !didResume, let continuation else {
            lock.unlock()
            return
        }
        didResume = true
        self.continuation = nil
        let outData = stdoutData
        let errData = stderrData
        lock.unlock()

        if process.terminationStatus == 0 {
            continuation.resume(returning: String(data: outData, encoding: .utf8) ?? "")
        } else {
            let errStr = String(data: errData, encoding: .utf8) ?? ""
            continuation.resume(
                throwing: YouTubeDatasetError.processFailed(process.terminationStatus, errStr)
            )
        }
    }

    private func finish(error: Error) {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil

        lock.lock()
        guard !didResume, let continuation else {
            lock.unlock()
            return
        }
        didResume = true
        self.continuation = nil
        lock.unlock()

        continuation.resume(throwing: error)
    }

    private func cancel() {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil

        lock.lock()
        didCancel = true
        let continuation = didResume ? nil : self.continuation
        if continuation != nil {
            didResume = true
            self.continuation = nil
        }
        lock.unlock()

        if process.isRunning {
            let pid = process.processIdentifier
            process.terminate()
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 2.0) { [process] in
                if process.isRunning {
                    kill(pid, SIGKILL)
                }
            }
        }

        continuation?.resume(throwing: CancellationError())
    }
}

// MARK: - Errors

enum YouTubeDatasetError: LocalizedError {
    case ytDlpNotFound
    case emptyPlaylist
    case downloadFailed(String)
    case noAudioTrack
    case compositionFailed
    case exportSessionFailed
    case exportFailed(String)
    case noClipsProduced
    case processFailed(Int32, String)

    var errorDescription: String? {
        switch self {
        case .ytDlpNotFound:
            return "yt-dlp not found. Install it with: brew install yt-dlp\n\nIf already installed, AnvilApp may need Full Disk Access in System Settings → Privacy & Security."
        case .emptyPlaylist:
            return "No videos found at this URL. Make sure it's a public YouTube channel or playlist."
        case .downloadFailed(let id):
            return "Could not download video \(id) — it may be private, age-restricted, or unavailable."
        case .noAudioTrack:
            return "Downloaded file has no audio track."
        case .compositionFailed:
            return "Failed to create audio composition for chunking."
        case .exportSessionFailed:
            return "Failed to initialise AVAssetExportSession."
        case .exportFailed(let msg):
            return "Audio export failed: \(msg)"
        case .noClipsProduced:
            return "No clips were produced. All videos may have failed to download or contain no audio."
        case .processFailed(let code, let output):
            let detail = String(output.prefix(400))
            return "yt-dlp exited with code \(code).\n\n\(detail)"
        }
    }
}
