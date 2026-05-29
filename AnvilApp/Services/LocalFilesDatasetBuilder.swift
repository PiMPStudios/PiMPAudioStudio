import AVFoundation
import Foundation

enum LocalDatasetClipFormat: String, CaseIterable, Hashable, Identifiable {
    case wav
    case m4a

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .wav: return "WAV"
        case .m4a: return "M4A"
        }
    }

    var helpText: String {
        switch self {
        case .wav:
            return "Lossless PCM clips. Recommended for training quality."
        case .m4a:
            return "Compact AAC clips. Useful for quick tests or smaller packages."
        }
    }
}

struct LocalFilesDatasetConfig {
    var clipLength: TimeInterval = 35
    var clipsPerTrack: Int = 8
    var maxTotalClips: Int = 40
    var clipFormat: LocalDatasetClipFormat = .wav
}

struct LocalFilesDatasetResult {
    let outputDirectory: URL
    let clips: [TrainingImportedFile]
    var clipCount: Int { clips.count }
}

enum LocalAudioFileDiscovery {
    static let supportedExtensions = Set(["aac", "aif", "aiff", "flac", "m4a", "mp3", "ogg", "opus", "wav", "webm"])

    static func audioFiles(in directories: [URL]) -> [URL] {
        directories.flatMap { audioFiles(in: $0) }.deduplicatedByFilePath()
    }

    static func audioFiles(in directory: URL) -> [URL] {
        let didStartAccessing = directory.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                directory.stopAccessingSecurityScopedResource()
            }
        }

        let keys: [URLResourceKey] = [.isRegularFileKey]
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }

        var files: [URL] = []
        for case let fileURL as URL in enumerator {
            guard supportedExtensions.contains(fileURL.pathExtension.lowercased()) else { continue }
            let values = try? fileURL.resourceValues(forKeys: Set(keys))
            guard values?.isRegularFile != false else { continue }
            files.append(fileURL)
        }

        return files.sorted { lhs, rhs in
            lhs.path.localizedStandardCompare(rhs.path) == .orderedAscending
        }
    }
}

private extension Array where Element == URL {
    func deduplicatedByFilePath() -> [URL] {
        var seen = Set<String>()
        return filter { url in
            seen.insert(url.standardizedFileURL.path).inserted
        }
    }
}

enum LocalFilesImportPhase: Equatable {
    case idle
    case chunking(index: Int, total: Int, name: String)
    case complete(outputDirectory: URL, clipCount: Int)
    case failed(String)

    var isRunning: Bool {
        if case .chunking = self { return true }
        return false
    }
}

enum LocalFilesDatasetError: LocalizedError {
    case noFilesSelected
    case noClipsProduced
    case audioBufferCreationFailed

    var errorDescription: String? {
        switch self {
        case .noFilesSelected:
            return "Select at least one local audio file."
        case .noClipsProduced:
            return "No clips were produced from the selected files."
        case .audioBufferCreationFailed:
            return "Could not allocate an audio buffer for the selected file."
        }
    }
}

@MainActor
final class LocalFilesDatasetBuilder: ObservableObject {
    @Published var phase: LocalFilesImportPhase = .idle
    @Published var resultClips: [TrainingImportedFile] = []

    private var task: Task<Void, Never>?

    func start(
        sourceFiles: [URL],
        sourceDirectories: [URL] = [],
        config: LocalFilesDatasetConfig,
        outputBase: URL,
        styleTemplate: String = ""
    ) {
        cancel()
        resultClips = []
        phase = .chunking(index: 0, total: sourceFiles.count, name: "Preparing")

        task = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await Self.run(
                    sourceFiles: sourceFiles,
                    sourceDirectories: sourceDirectories,
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

    nonisolated static func run(
        sourceFiles: [URL],
        sourceDirectories: [URL] = [],
        config: LocalFilesDatasetConfig,
        outputBase: URL,
        styleTemplate: String = "",
        onPhase: @escaping (LocalFilesImportPhase) async -> Void
    ) async throws -> LocalFilesDatasetResult {
        guard !sourceFiles.isEmpty else { throw LocalFilesDatasetError.noFilesSelected }

        let scopedDirectories = sourceDirectories.filter { $0.startAccessingSecurityScopedResource() }
        defer {
            scopedDirectories.forEach { $0.stopAccessingSecurityScopedResource() }
        }

        let folderTag = datasetFolderTag(for: sourceFiles.first)
        let outputDir = outputBase
            .appendingPathComponent("TrainingDatasets", isDirectory: true)
            .appendingPathComponent("\(folderTag)_local_\(timestampTag())", isDirectory: true)
        let clipsDir = outputDir.appendingPathComponent("clips", isDirectory: true)
        try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)

        var allClips: [TrainingImportedFile] = []
        let trimmedStyle = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)

        for (idx, sourceURL) in sourceFiles.enumerated() {
            try Task.checkCancellation()
            guard allClips.count < config.maxTotalClips else { break }

            let title = sourceURL.deletingPathExtension().lastPathComponent
            await onPhase(.chunking(index: idx + 1, total: sourceFiles.count, name: sourceURL.lastPathComponent))

            let didStartAccessing = sourceURL.startAccessingSecurityScopedResource()
            defer {
                if didStartAccessing {
                    sourceURL.stopAccessingSecurityScopedResource()
                }
            }

            do {
                let remaining = config.maxTotalClips - allClips.count
                let chunks = try await chunkAudio(
                    from: sourceURL,
                    title: title,
                    videoIndex: idx + 1,
                    clipLength: config.clipLength,
                    maxClips: min(config.clipsPerTrack, remaining),
                    styleTemplate: trimmedStyle,
                    outputDirectory: clipsDir,
                    clipFormat: config.clipFormat
                )
                allClips.append(contentsOf: chunks.map {
                    TrainingImportedFile(url: $0.url, description: $0.caption)
                })
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                print("[LocalFilesDatasetBuilder] Skipping \(sourceURL.lastPathComponent): \(error.localizedDescription)")
            }
        }

        guard !allClips.isEmpty else { throw LocalFilesDatasetError.noClipsProduced }

        try TrainingDatasetPackageWriter.writeLocalPackage(
            outputDirectory: outputDir,
            clips: allClips,
            sourceFiles: sourceFiles,
            config: config,
            styleTemplate: trimmedStyle
        )
        return LocalFilesDatasetResult(outputDirectory: outputDir, clips: allClips)
    }

    nonisolated private static func chunkAudio(
        from sourceURL: URL,
        title: String,
        videoIndex: Int,
        clipLength: TimeInterval,
        maxClips: Int,
        styleTemplate: String,
        outputDirectory: URL,
        clipFormat: LocalDatasetClipFormat
    ) async throws -> [(url: URL, caption: String)] {
        switch clipFormat {
        case .wav:
            return try chunkAudioToWAV(
                from: sourceURL,
                title: title,
                videoIndex: videoIndex,
                clipLength: clipLength,
                maxClips: maxClips,
                styleTemplate: styleTemplate,
                outputDirectory: outputDirectory
            )
        case .m4a:
            return try await YouTubeDatasetBuilder.chunkAudio(
                from: sourceURL,
                title: title,
                videoIndex: videoIndex,
                clipLength: clipLength,
                maxClips: maxClips,
                styleTemplate: styleTemplate,
                outputDirectory: outputDirectory,
                filenamePrefix: "f"
            )
        }
    }

    nonisolated private static func chunkAudioToWAV(
        from sourceURL: URL,
        title: String,
        videoIndex: Int,
        clipLength: TimeInterval,
        maxClips: Int,
        styleTemplate: String,
        outputDirectory: URL
    ) throws -> [(url: URL, caption: String)] {
        let sourceFile = try AVAudioFile(forReading: sourceURL)
        let format = sourceFile.processingFormat
        let sampleRate = format.sampleRate
        let totalFrames = sourceFile.length
        guard sampleRate > 0, totalFrames > 0 else { return [] }

        let totalSeconds = Double(totalFrames) / sampleRate
        let framesPerClip = max(1, AVAudioFramePosition((clipLength * sampleRate).rounded(.down)))
        let clipCount = totalSeconds <= clipLength
            ? 1
            : min(maxClips, Int(totalFrames / framesPerClip))
        guard clipCount > 0 else { return [] }

        var clips: [(url: URL, caption: String)] = []
        for i in 0..<clipCount {
            try Task.checkCancellation()
            let startFrame = AVAudioFramePosition(i) * framesPerClip
            let frameCount = min(framesPerClip, totalFrames - startFrame)
            guard frameCount > 0 else { continue }

            let dest = outputDirectory.appendingPathComponent(
                clipFilename(prefix: "f", sourceIndex: videoIndex, clipIndex: i + 1, extension: "wav")
            )
            try writeWAVSegment(
                from: sourceFile,
                format: format,
                startFrame: startFrame,
                frameCount: frameCount,
                to: dest
            )

            let caption = styleTemplate.isEmpty
                ? (clipCount > 1 ? "\(title) (clip \(i + 1))" : title)
                : styleTemplate
            clips.append((url: dest, caption: caption))
        }
        return clips
    }

    nonisolated private static func writeWAVSegment(
        from sourceFile: AVAudioFile,
        format: AVAudioFormat,
        startFrame: AVAudioFramePosition,
        frameCount: AVAudioFramePosition,
        to dest: URL
    ) throws {
        if FileManager.default.fileExists(atPath: dest.path(percentEncoded: false)) {
            try FileManager.default.removeItem(at: dest)
        }

        let outputFile = try AVAudioFile(forWriting: dest, settings: format.settings)
        sourceFile.framePosition = startFrame

        var framesRemaining = frameCount
        let maxBufferFrames: AVAudioFrameCount = 262_144
        while framesRemaining > 0 {
            try Task.checkCancellation()
            let framesToRead = AVAudioFrameCount(min(Int64(maxBufferFrames), framesRemaining))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRead) else {
                throw LocalFilesDatasetError.audioBufferCreationFailed
            }

            try sourceFile.read(into: buffer, frameCount: framesToRead)
            guard buffer.frameLength > 0 else { break }
            try outputFile.write(from: buffer)
            framesRemaining -= AVAudioFramePosition(buffer.frameLength)
        }
    }

    nonisolated private static func clipFilename(
        prefix: String,
        sourceIndex: Int,
        clipIndex: Int,
        extension fileExtension: String
    ) -> String {
        let source = String(format: "%03d", sourceIndex)
        let clip = String(format: "%03d", clipIndex)
        return "\(prefix)\(source)_clip_\(clip).\(fileExtension)"
    }

    nonisolated private static func datasetFolderTag(for url: URL?) -> String {
        let name = url?.deletingPathExtension().lastPathComponent ?? "local-dataset"
        let sanitized = sanitizeDatasetName(name)
        return sanitized.isEmpty ? "local-dataset" : sanitized
    }

    nonisolated private static func sanitizeDatasetName(_ input: String) -> String {
        let clean = input
            .replacingOccurrences(of: "[^a-zA-Z0-9 \\-]", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: " +", with: "-", options: .regularExpression)
        return String(clean.prefix(40))
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .lowercased()
    }

    nonisolated private static func timestampTag() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        return formatter.string(from: Date())
    }
}
