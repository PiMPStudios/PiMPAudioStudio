import AVFoundation
import Foundation
import PiMPAudioCore

enum LocalDatasetBuildMode: String, CaseIterable, Hashable, Identifiable {
    case wholeTracks = "whole_tracks"
    case clipExtraction = "clip_extraction"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .wholeTracks: return "Whole Tracks"
        case .clipExtraction: return "Extract Clips"
        }
    }

    var helpText: String {
        switch self {
        case .wholeTracks:
            return "Preserves every complete source and trains from full mixes by default. Recommended for ACE-Step 1.5."
        case .clipExtraction:
            return "Creates fixed-length derived clips for experiments or tightly controlled sample counts."
        }
    }
}

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

enum LocalDatasetClipSamplingMode: String, CaseIterable, Hashable, Identifiable {
    case sequential
    case evenlySpread
    case randomSeeded

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .sequential: return "Sequential"
        case .evenlySpread: return "Evenly Spread"
        case .randomSeeded: return "Random Seeded"
        }
    }

    var helpText: String {
        switch self {
        case .sequential:
            return "Takes clips from the start of each file in order."
        case .evenlySpread:
            return "Spreads clips across each full source file while avoiding exact starts and endings."
        case .randomSeeded:
            return "Uses reproducible random clip positions from across each source file."
        }
    }
}

struct LocalFilesDatasetConfig {
    static let maximumTotalClips = 1_000

    var buildMode: LocalDatasetBuildMode = .wholeTracks
    var clipLength: TimeInterval = 35
    var clipsPerTrack: Int = 8
    var maxTotalClips: Int = 40
    var clipFormat: LocalDatasetClipFormat = .wav
    var samplingMode: LocalDatasetClipSamplingMode = .evenlySpread
    var samplingSeed: Int = 1337

    init() {}

    init(
        buildMode: LocalDatasetBuildMode,
        clipLength: TimeInterval = 35,
        clipsPerTrack: Int = 8,
        maxTotalClips: Int = 40,
        clipFormat: LocalDatasetClipFormat = .wav,
        samplingMode: LocalDatasetClipSamplingMode = .evenlySpread,
        samplingSeed: Int = 1337
    ) {
        self.buildMode = buildMode
        self.clipLength = clipLength
        self.clipsPerTrack = clipsPerTrack
        self.maxTotalClips = maxTotalClips
        self.clipFormat = clipFormat
        self.samplingMode = samplingMode
        self.samplingSeed = samplingSeed
    }

    /// Compatibility initializer for callers that explicitly configure clip
    /// extraction values. New UI state uses `init()` and defaults to whole tracks.
    init(
        clipLength: TimeInterval,
        clipsPerTrack: Int,
        maxTotalClips: Int,
        clipFormat: LocalDatasetClipFormat = .wav,
        samplingMode: LocalDatasetClipSamplingMode = .evenlySpread,
        samplingSeed: Int = 1337
    ) {
        self.init(
            buildMode: .clipExtraction,
            clipLength: clipLength,
            clipsPerTrack: clipsPerTrack,
            maxTotalClips: maxTotalClips,
            clipFormat: clipFormat,
            samplingMode: samplingMode,
            samplingSeed: samplingSeed
        )
    }
}

enum LocalDatasetClipSampler {
    static func startTimes(
        totalSeconds: TimeInterval,
        clipLength: TimeInterval,
        maxClips: Int,
        mode: LocalDatasetClipSamplingMode,
        seed: Int,
        sourceIdentifier: String
    ) -> [TimeInterval] {
        guard totalSeconds > 0, clipLength > 0, maxClips > 0 else { return [] }
        if totalSeconds <= clipLength { return [0] }

        let clipCount = min(maxClips, Int(totalSeconds / clipLength))
        guard clipCount > 0 else { return [] }

        switch mode {
        case .sequential:
            return (0..<clipCount).map { Double($0) * clipLength }
        case .evenlySpread:
            let maxStart = max(0, totalSeconds - clipLength)
            return (0..<clipCount).map { index in
                maxStart * (Double(index) + 0.5) / Double(clipCount)
            }
        case .randomSeeded:
            let availableSlots = max(1, Int(totalSeconds / clipLength))
            return selectedRandomSlots(
                availableSlots: availableSlots,
                clipCount: clipCount,
                seed: seed,
                sourceIdentifier: sourceIdentifier
            )
            .map { Double($0) * clipLength }
        }
    }

    static func frameStarts(
        totalFrames: AVAudioFramePosition,
        framesPerClip: AVAudioFramePosition,
        maxClips: Int,
        mode: LocalDatasetClipSamplingMode,
        seed: Int,
        sourceIdentifier: String
    ) -> [AVAudioFramePosition] {
        guard totalFrames > 0, framesPerClip > 0, maxClips > 0 else { return [] }
        if totalFrames <= framesPerClip { return [0] }

        let clipCount = min(maxClips, Int(totalFrames / framesPerClip))
        guard clipCount > 0 else { return [] }

        switch mode {
        case .sequential:
            return (0..<clipCount).map { AVAudioFramePosition($0) * framesPerClip }
        case .evenlySpread:
            let maxStart = totalFrames - framesPerClip
            return (0..<clipCount).map { index in
                AVAudioFramePosition((Double(maxStart) * (Double(index) + 0.5) / Double(clipCount)).rounded(.down))
            }
        case .randomSeeded:
            let availableSlots = max(1, Int(totalFrames / framesPerClip))
            return selectedRandomSlots(
                availableSlots: availableSlots,
                clipCount: clipCount,
                seed: seed,
                sourceIdentifier: sourceIdentifier
            )
            .map { AVAudioFramePosition($0) * framesPerClip }
        }
    }

    private static func selectedRandomSlots(
        availableSlots: Int,
        clipCount: Int,
        seed: Int,
        sourceIdentifier: String
    ) -> [Int] {
        guard availableSlots > 0, clipCount > 0 else { return [] }
        var generator = SeededRandomNumberGenerator(seed: mixedSeed(seed, sourceIdentifier: sourceIdentifier))
        var slots = Array(0..<availableSlots)
        slots.shuffle(using: &generator)
        return slots.prefix(min(clipCount, availableSlots)).sorted()
    }

    private static func mixedSeed(_ seed: Int, sourceIdentifier: String) -> UInt64 {
        let input = "\(seed)|\(sourceIdentifier)"
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in input.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01b3
        }
        return hash == 0 ? 0x9e37_79b9_7f4a_7c15 : hash
    }

    private struct SeededRandomNumberGenerator: RandomNumberGenerator {
        var state: UInt64

        init(seed: UInt64) {
            state = seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9e37_79b9_7f4a_7c15
            var value = state
            value = (value ^ (value >> 30)) &* 0xbf58_476d_1ce4_e5b9
            value = (value ^ (value >> 27)) &* 0x94d0_49bb_1331_11eb
            return value ^ (value >> 31)
        }
    }
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
    private var taskRunID: UUID?
    private var activeRunID: UUID?

    func start(
        sourceFiles: [URL],
        sourceDirectories: [URL] = [],
        config: LocalFilesDatasetConfig,
        outputBase: URL,
        datasetName: String = "",
        styleTemplate: String = ""
    ) {
        let previousTask = task
        previousTask?.cancel()
        let runID = UUID()
        taskRunID = runID
        activeRunID = runID
        resultClips = []
        phase = .chunking(index: 0, total: sourceFiles.count, name: "Preparing")

        task = Task { [weak self, previousTask] in
            await previousTask?.value
            guard let self else { return }
            defer { self.finishRun(runID) }
            guard self.isCurrentRun(runID), !Task.isCancelled else { return }
            do {
                let result = try await Self.run(
                    sourceFiles: sourceFiles,
                    sourceDirectories: sourceDirectories,
                    config: config,
                    outputBase: outputBase,
                    datasetName: datasetName,
                    styleTemplate: styleTemplate,
                    onPhase: { [weak self] phase in
                        await MainActor.run {
                            guard self?.isCurrentRun(runID) == true else { return }
                            self?.phase = phase
                        }
                    }
                )
                guard self.isCurrentRun(runID) else { return }
                self.resultClips = result.clips
                self.phase = .complete(outputDirectory: result.outputDirectory, clipCount: result.clipCount)
            } catch is CancellationError {
                guard self.isCurrentRun(runID) else { return }
                AudioDiagnostics.note("dataset/local/taskCancelled", "sources=\(sourceFiles.count)")
                self.phase = .idle
            } catch {
                guard self.isCurrentRun(runID) else { return }
                AudioDiagnostics.note("dataset/local/taskFailed", "error=\(error.localizedDescription)")
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        task?.cancel()
        activeRunID = nil
        phase = .idle
    }

    private func isCurrentRun(_ runID: UUID) -> Bool {
        activeRunID == runID
    }

    private func finishRun(_ runID: UUID) {
        if activeRunID == runID {
            activeRunID = nil
        }
        if taskRunID == runID {
            taskRunID = nil
            task = nil
        }
    }

    nonisolated static func run(
        sourceFiles: [URL],
        sourceDirectories: [URL] = [],
        config: LocalFilesDatasetConfig,
        outputBase: URL,
        datasetName: String = "",
        styleTemplate: String = "",
        onPhase: @escaping (LocalFilesImportPhase) async -> Void
    ) async throws -> LocalFilesDatasetResult {
        guard !sourceFiles.isEmpty else { throw LocalFilesDatasetError.noFilesSelected }

        let buildStartedAt = Date()
        let trimmedStyle = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)
        let requestedItems = config.buildMode == .wholeTracks
            ? sourceFiles.count
            : sourceFiles.count * config.clipsPerTrack
        let plannedItems = config.buildMode == .wholeTracks
            ? sourceFiles.count
            : min(requestedItems, config.maxTotalClips)
        AudioDiagnostics.note(
            "dataset/local/buildBegin",
            "sources=\(sourceFiles.count) directories=\(sourceDirectories.count) "
                + "mode=\(config.buildMode.rawValue) "
                + "clipLength=\(secondsText(config.clipLength)) clipsPerFile=\(config.clipsPerTrack) "
                + "maxTotal=\(config.maxTotalClips) format=\(config.clipFormat.rawValue) "
                + "sampling=\(config.samplingMode.rawValue) seed=\(config.samplingSeed) "
                + "datasetNameProvided=\(!datasetName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) "
                + "styleProvided=\(!trimmedStyle.isEmpty)"
        )
        AudioDiagnostics.memory("dataset/local/buildBegin")

        let scopedDirectories = sourceDirectories.filter { $0.startAccessingSecurityScopedResource() }
        defer {
            scopedDirectories.forEach { $0.stopAccessingSecurityScopedResource() }
        }

        let folderTag = datasetFolderTag(for: datasetName, fallbackURL: sourceFiles.first)
        let outputDir = outputBase
            .appendingPathComponent("TrainingDatasets", isDirectory: true)
            .appendingPathComponent("\(folderTag)_local_\(timestampTag())", isDirectory: true)
        let mediaDirectoryName = config.buildMode == .wholeTracks ? "sources" : "clips"
        let mediaDirectory = outputDir.appendingPathComponent(mediaDirectoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: mediaDirectory, withIntermediateDirectories: true)
        AudioDiagnostics.note(
            "dataset/local/outputReady",
            "output=\(outputDir.path(percentEncoded: false)) media=\(mediaDirectory.path(percentEncoded: false))"
        )
        AudioDiagnostics.note(
            "dataset/local/coverage",
            "requested=\(requestedItems) planned=\(plannedItems) "
                + "capped=\(config.buildMode == .clipExtraction && requestedItems > config.maxTotalClips) "
                + "sources=\(sourceFiles.count)"
        )

        var trainingItems: [TrainingImportedFile] = []

        for (idx, sourceURL) in sourceFiles.enumerated() {
            try Task.checkCancellation()
            guard config.buildMode == .wholeTracks || trainingItems.count < config.maxTotalClips else {
                AudioDiagnostics.note(
                    "dataset/local/capReached",
                    "atSource=\(idx + 1) totalSources=\(sourceFiles.count) totalItems=\(trainingItems.count) cap=\(config.maxTotalClips)"
                )
                break
            }

            let title = sourceURL.deletingPathExtension().lastPathComponent
            await onPhase(.chunking(index: idx + 1, total: sourceFiles.count, name: sourceURL.lastPathComponent))

            let didStartAccessing = sourceURL.startAccessingSecurityScopedResource()
            defer {
                if didStartAccessing {
                    sourceURL.stopAccessingSecurityScopedResource()
                }
            }

            do {
                let packagedSourceURL = try TrainingDatasetPackageWriter.stageWholeTrackSource(
                    sourceURL,
                    index: idx,
                    outputDirectory: outputDir
                )
                if config.buildMode == .wholeTracks {
                    let caption = trimmedStyle.isEmpty ? title : trimmedStyle
                    trainingItems.append(
                        TrainingImportedFile(
                            url: packagedSourceURL,
                            description: caption,
                            sourceClip: TrainingSourceClipMetadata(
                                fileName: sourceURL.lastPathComponent,
                                startSeconds: 0
                            )
                        )
                    )
                    AudioDiagnostics.note(
                        "dataset/local/sourcePreserved",
                        "index=\(idx + 1)/\(sourceFiles.count) file=\(sourceURL.lastPathComponent)"
                    )
                    continue
                }

                let remaining = config.maxTotalClips - trainingItems.count
                let maxClipsForSource = min(config.clipsPerTrack, remaining)
                let sourceStartedAt = Date()
                AudioDiagnostics.note(
                    "dataset/local/sourceBegin",
                    "index=\(idx + 1)/\(sourceFiles.count) file=\(sourceURL.lastPathComponent) "
                        + "remaining=\(remaining) maxClips=\(maxClipsForSource)"
                )
                let chunks = try await chunkAudio(
                    from: sourceURL,
                    title: title,
                    videoIndex: idx + 1,
                    clipLength: config.clipLength,
                    maxClips: maxClipsForSource,
                    styleTemplate: trimmedStyle,
                    outputDirectory: mediaDirectory,
                    clipFormat: config.clipFormat,
                    samplingMode: config.samplingMode,
                    samplingSeed: config.samplingSeed
                )
                trainingItems.append(contentsOf: chunks)
                AudioDiagnostics.note(
                    "dataset/local/sourceEnd",
                    "index=\(idx + 1)/\(sourceFiles.count) file=\(sourceURL.lastPathComponent) "
                        + "produced=\(chunks.count) totalItems=\(trainingItems.count) "
                        + "elapsed=\(secondsText(Date().timeIntervalSince(sourceStartedAt)))"
                )
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                AudioDiagnostics.note(
                    "dataset/local/sourceSkipped",
                    "index=\(idx + 1)/\(sourceFiles.count) file=\(sourceURL.lastPathComponent) "
                        + "error=\(error.localizedDescription)"
                )
                print("[LocalFilesDatasetBuilder] Skipping \(sourceURL.lastPathComponent): \(error.localizedDescription)")
            }
        }

        guard !trainingItems.isEmpty else {
            AudioDiagnostics.note("dataset/local/buildEmpty", "sources=\(sourceFiles.count)")
            throw LocalFilesDatasetError.noClipsProduced
        }

        let packageStartedAt = Date()
        AudioDiagnostics.note(
            "dataset/local/packageWriteBegin",
            "items=\(trainingItems.count) output=\(outputDir.path(percentEncoded: false))"
        )
        AudioDiagnostics.memory("dataset/local/packageWriteBegin")
        let packagedItems = try TrainingDatasetPackageWriter.writeLocalPackage(
            outputDirectory: outputDir,
            clips: trainingItems,
            sourceFiles: sourceFiles,
            config: config,
            styleTemplate: trimmedStyle
        )
        AudioDiagnostics.note(
            "dataset/local/packageWriteEnd",
            "items=\(packagedItems.count) elapsed=\(secondsText(Date().timeIntervalSince(packageStartedAt)))"
        )
        AudioDiagnostics.note(
            "dataset/local/buildEnd",
            "items=\(packagedItems.count) output=\(outputDir.path(percentEncoded: false)) "
                + "elapsed=\(secondsText(Date().timeIntervalSince(buildStartedAt)))"
        )
        AudioDiagnostics.memory("dataset/local/buildEnd")
        return LocalFilesDatasetResult(outputDirectory: outputDir, clips: packagedItems)
    }

    nonisolated private static func chunkAudio(
        from sourceURL: URL,
        title: String,
        videoIndex: Int,
        clipLength: TimeInterval,
        maxClips: Int,
        styleTemplate: String,
        outputDirectory: URL,
        clipFormat: LocalDatasetClipFormat,
        samplingMode: LocalDatasetClipSamplingMode,
        samplingSeed: Int
    ) async throws -> [TrainingImportedFile] {
        switch clipFormat {
        case .wav:
            return try chunkAudioToWAV(
                from: sourceURL,
                title: title,
                videoIndex: videoIndex,
                clipLength: clipLength,
                maxClips: maxClips,
                styleTemplate: styleTemplate,
                outputDirectory: outputDirectory,
                samplingMode: samplingMode,
                samplingSeed: samplingSeed
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
                filenamePrefix: "f",
                samplingMode: samplingMode,
                samplingSeed: samplingSeed,
                sourceIdentifier: sourceURL.standardizedFileURL.path
            ).map {
                TrainingImportedFile(url: $0.url, description: $0.caption)
            }
        }
    }

    nonisolated private static func chunkAudioToWAV(
        from sourceURL: URL,
        title: String,
        videoIndex: Int,
        clipLength: TimeInterval,
        maxClips: Int,
        styleTemplate: String,
        outputDirectory: URL,
        samplingMode: LocalDatasetClipSamplingMode,
        samplingSeed: Int
    ) throws -> [TrainingImportedFile] {
        let sourceFile = try AVAudioFile(forReading: sourceURL)
        let format = sourceFile.processingFormat
        let sampleRate = format.sampleRate
        let totalFrames = sourceFile.length
        guard sampleRate > 0, totalFrames > 0 else { return [] }

        let framesPerClip = max(1, AVAudioFramePosition((clipLength * sampleRate).rounded(.down)))
        let startFrames = LocalDatasetClipSampler.frameStarts(
            totalFrames: totalFrames,
            framesPerClip: framesPerClip,
            maxClips: maxClips,
            mode: samplingMode,
            seed: samplingSeed,
            sourceIdentifier: sourceURL.standardizedFileURL.path
        )
        guard !startFrames.isEmpty else { return [] }
        AudioDiagnostics.note(
            "dataset/local/wavPlan",
            "source=\(videoIndex) file=\(sourceURL.lastPathComponent) "
                + "totalSeconds=\(secondsText(Double(totalFrames) / sampleRate)) "
                + "sampleRate=\(String(format: "%.0f", sampleRate)) clips=\(startFrames.count) "
                + "clipLength=\(secondsText(clipLength))"
        )

        var clips: [TrainingImportedFile] = []
        for (i, startFrame) in startFrames.enumerated() {
            try Task.checkCancellation()
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
                ? (startFrames.count > 1 ? "\(title) (clip \(i + 1))" : title)
                : styleTemplate
            clips.append(TrainingImportedFile(
                url: dest,
                description: caption,
                sourceClip: TrainingSourceClipMetadata(
                    fileName: sourceURL.lastPathComponent,
                    startSeconds: Double(startFrame) / sampleRate,
                    startFrame: Int64(startFrame),
                    frameCount: Int64(frameCount),
                    sampleRate: sampleRate
                )
            ))
            let clipIndex = i + 1
            if shouldLogClipProgress(index: clipIndex, total: startFrames.count) {
                AudioDiagnostics.note(
                    "dataset/local/wavClip",
                    "source=\(videoIndex) clip=\(clipIndex)/\(startFrames.count) "
                        + "start=\(secondsText(Double(startFrame) / sampleRate)) "
                        + "duration=\(secondsText(Double(frameCount) / sampleRate)) "
                        + "file=\(dest.lastPathComponent)"
                )
            }
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

    nonisolated private static func datasetFolderTag(for datasetName: String, fallbackURL: URL?) -> String {
        let trimmedName = datasetName.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallbackName = fallbackURL?.deletingPathExtension().lastPathComponent ?? "local-dataset"
        let sanitized = sanitizeDatasetName(trimmedName.isEmpty ? fallbackName : trimmedName)
        let fallback = sanitizeDatasetName(fallbackName)
        if !sanitized.isEmpty {
            return sanitized
        }
        return fallback.isEmpty ? "local-dataset" : fallback
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

    nonisolated private static func secondsText(_ seconds: TimeInterval) -> String {
        String(format: "%.2f", seconds)
    }

    nonisolated private static func shouldLogClipProgress(index: Int, total: Int) -> Bool {
        index == 1 || index == total || index % 10 == 0
    }
}
