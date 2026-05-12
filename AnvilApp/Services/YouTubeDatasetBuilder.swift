import Foundation
import AVFoundation

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

    func start(channelURL: URL, config: YouTubeDatasetConfig, outputBase: URL) {
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

    static func locateYtDlp() -> URL? {
        let paths = [
            "/opt/homebrew/bin/yt-dlp",
            "/usr/local/bin/yt-dlp",
            "/opt/local/bin/yt-dlp",
            "/usr/bin/yt-dlp",
        ]
        // isExecutableFile(atPath:) always returns false in a sandboxed app for
        // paths outside the container even with Full Disk Access, because the
        // execute-permission check is a separate sandbox gate. fileExists is
        // sufficient — Homebrew always installs with correct execute bits.
        return paths
            .first { FileManager.default.fileExists(atPath: $0) }
            .map { URL(fileURLWithPath: $0) }
    }

    /// Returns the installed yt-dlp version string, or nil if not found.
    static func ytDlpVersion() -> String? {
        guard let ytDlp = locateYtDlp() else { return nil }
        return try? runProcessSync(executable: ytDlp, arguments: ["--version"])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Core Pipeline

    static func run(
        channelURL: URL,
        config: YouTubeDatasetConfig,
        outputBase: URL,
        onPhase: @escaping (YouTubeImportPhase) async -> Void
    ) async throws -> YouTubeDatasetResult {
        guard let ytDlp = locateYtDlp() else { throw YouTubeDatasetError.ytDlpNotFound }

        let maxVideos = max(1,
            Int(ceil(Double(config.maxTotalClips) / Double(max(1, config.clipsPerTrack)))) + 1
        )
        let videos = try await fetchPlaylist(ytDlp: ytDlp, url: channelURL, limit: maxVideos)
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
                let audioURL = try await downloadAudio(ytDlp: ytDlp, video: video, to: tempDir)

                try Task.checkCancellation()
                await onPhase(.chunking(index: idx + 1, total: videoCount))

                let remaining = config.maxTotalClips - allClips.count
                let chunks = try await chunkAudio(
                    from: audioURL,
                    title: video.title,
                    videoIndex: idx + 1,
                    clipLength: config.clipLength,
                    maxClips: min(config.clipsPerTrack, remaining),
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

    static func fetchPlaylist(ytDlp: URL, url: URL, limit: Int) async throws -> [YouTubeVideoInfo] {
        let raw = try await runProcess(
            executable: ytDlp,
            arguments: [
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

    static func downloadAudio(ytDlp: URL, video: YouTubeVideoInfo, to outputDir: URL) async throws -> URL {
        _ = try await runProcess(
            executable: ytDlp,
            arguments: [
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
        outputDirectory: URL
    ) async throws -> [(url: URL, caption: String)] {
        let asset = AVURLAsset(url: sourceURL)
        let cmDuration = try await asset.load(.duration)
        let totalSeconds = cmDuration.seconds
        guard totalSeconds > 0 else { return [] }

        // Short track: single clip
        if totalSeconds <= clipLength {
            let dest = outputDirectory.appendingPathComponent(
                String(format: "v%03d_clip_001.m4a", videoIndex)
            )
            try await exportRange(asset: asset,
                                  range: CMTimeRange(start: .zero, duration: cmDuration),
                                  to: dest)
            return [(url: dest, caption: title)]
        }

        let clipCount = min(maxClips, Int(totalSeconds / clipLength))
        var clips: [(url: URL, caption: String)] = []

        for i in 0..<clipCount {
            try Task.checkCancellation()
            let startSec = Double(i) * clipLength
            let start = CMTime(seconds: startSec, preferredTimescale: 600)
            let rawEnd = startSec + clipLength
            let actualLen = rawEnd <= totalSeconds ? clipLength : totalSeconds - startSec
            let dur = CMTime(seconds: actualLen, preferredTimescale: 600)

            let dest = outputDirectory.appendingPathComponent(
                String(format: "v%03d_clip_%03d.m4a", videoIndex, i + 1)
            )
            try await exportRange(asset: asset,
                                  range: CMTimeRange(start: start, duration: dur),
                                  to: dest)
            let caption = clipCount > 1 ? "\(title) (clip \(i + 1))" : title
            clips.append((url: dest, caption: caption))
        }
        return clips
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

    static func runProcess(executable: URL, arguments: [String]) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            let stdout = Pipe()
            let stderr = Pipe()
            process.executableURL = executable
            process.arguments = arguments
            process.standardOutput = stdout
            process.standardError = stderr
            process.terminationHandler = { p in
                let outData = stdout.fileHandleForReading.readDataToEndOfFile()
                if p.terminationStatus == 0 {
                    continuation.resume(returning: String(data: outData, encoding: .utf8) ?? "")
                } else {
                    let errData = stderr.fileHandleForReading.readDataToEndOfFile()
                    let errStr = String(data: errData, encoding: .utf8) ?? ""
                    continuation.resume(throwing: YouTubeDatasetError.processFailed(p.terminationStatus, errStr))
                }
            }
            do { try process.run() } catch { continuation.resume(throwing: error) }
        }
    }

    static func runProcessSync(executable: URL, arguments: [String]) throws -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = executable
        process.arguments = arguments
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
