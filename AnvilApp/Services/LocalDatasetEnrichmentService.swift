import Darwin
import Foundation

enum LocalDatasetEnrichmentStep: String, Equatable {
    case separatingStems
    case exportingVocalsBundle
    case exportingInstrumentalBundle
    case refreshingMetadata

    var title: String {
        switch self {
        case .separatingStems:
            return "Separating stems"
        case .exportingVocalsBundle:
            return "Exporting vocal bundle"
        case .exportingInstrumentalBundle:
            return "Exporting instrumental bundle"
        case .refreshingMetadata:
            return "Refreshing package metadata"
        }
    }
}

enum LocalDatasetEnrichmentPhase: Equatable {
    case idle
    case running(step: LocalDatasetEnrichmentStep)
    case complete(LocalDatasetEnrichmentResult)
    case failed(String)

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }
}

struct LocalDatasetEnrichmentResult: Equatable {
    let datasetDirectory: URL
    let separatedClipCount: Int
    let vocalBundleExists: Bool
    let instrumentalBundleExists: Bool
    let bpmEstimateCount: Int
    let averageBPM: Double?
    let transcribedClipCount: Int

    var summary: String {
        var parts: [String] = []
        parts.append("Separated \(separatedClipCount) clip\(separatedClipCount == 1 ? "" : "s")")
        if vocalBundleExists && instrumentalBundleExists {
            parts.append("wrote vocal and instrumental bundles")
        } else if vocalBundleExists {
            parts.append("wrote vocal bundle")
        } else if instrumentalBundleExists {
            parts.append("wrote instrumental bundle")
        }
        if let averageBPM, bpmEstimateCount > 0 {
            parts.append("\(bpmEstimateCount) BPM estimate\(bpmEstimateCount == 1 ? "" : "s"), avg \(averageBPM.formatted(.number.precision(.fractionLength(1))))")
        }
        if transcribedClipCount > 0 {
            parts.append("\(transcribedClipCount) transcript\(transcribedClipCount == 1 ? "" : "s") found")
        } else {
            parts.append("transcription pending")
        }
        parts.append("key analysis pending")
        return parts.joined(separator: "; ") + "."
    }
}

struct LocalDatasetEnrichmentCommand: Equatable {
    let step: LocalDatasetEnrichmentStep
    let executableURL: URL
    let arguments: [String]

    var displayCommand: String {
        ([executableURL.path] + arguments).map(Self.shellQuoted).joined(separator: " ")
    }

    private static func shellQuoted(_ value: String) -> String {
        guard value.rangeOfCharacter(from: .whitespacesAndNewlines) != nil else {
            return value
        }
        return "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }
}

enum LocalDatasetEnrichmentError: LocalizedError {
    case missingAnvilCLI
    case commandFailed(step: LocalDatasetEnrichmentStep, status: Int32, command: String, output: String)

    var errorDescription: String? {
        switch self {
        case .missingAnvilCLI:
            return """
            Anvil audio CLI was not found. Install anvil-audio-v2, add `anvil` to PATH, or set ANVIL_AUDIO_CLI to the anvil executable.
            """
        case .commandFailed(let step, let status, let command, let output):
            let detail = output.trimmingCharacters(in: .whitespacesAndNewlines)
            let clipped = String(detail.prefix(1_200))
            return """
            \(step.title) failed with exit code \(status).

            \(command)

            \(clipped)
            """
        }
    }
}

@MainActor
final class LocalDatasetEnrichmentService: ObservableObject {
    @Published var phase: LocalDatasetEnrichmentPhase = .idle

    private var task: Task<Void, Never>?

    func enrich(datasetDirectory: URL, force: Bool = false) {
        cancel()
        phase = .running(step: .separatingStems)

        task = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await Self.run(
                    datasetDirectory: datasetDirectory,
                    force: force,
                    onStep: { [weak self] step in
                        await MainActor.run {
                            self?.phase = .running(step: step)
                        }
                    }
                )
                phase = .complete(result)
            } catch is CancellationError {
                phase = .idle
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
    }

    func reset() {
        cancel()
        phase = .idle
    }

    nonisolated static func run(
        datasetDirectory: URL,
        force: Bool = false,
        onStep: @escaping (LocalDatasetEnrichmentStep) async -> Void
    ) async throws -> LocalDatasetEnrichmentResult {
        let didStartAccessing = datasetDirectory.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                datasetDirectory.stopAccessingSecurityScopedResource()
            }
        }

        let executableURL = try resolveAnvilExecutableURL()
        let environment = commandEnvironment(executableURL: executableURL)
        let commands = commandPlan(
            datasetDirectory: datasetDirectory,
            executableURL: executableURL,
            force: force
        )

        for command in commands {
            try Task.checkCancellation()
            await onStep(command.step)
            let runner = LocalDatasetCapturedProcess(command: command, environment: environment)
            _ = try await runner.run()
        }

        try Task.checkCancellation()
        await onStep(.refreshingMetadata)
        var result = inspectResult(datasetDirectory: datasetDirectory)
        try refreshDatasetMetadata(datasetDirectory: datasetDirectory, result: result)
        try TrainingDatasetPackageWriter.refreshValidationReport(at: datasetDirectory)
        result = inspectResult(datasetDirectory: datasetDirectory)
        return result
    }

    nonisolated static func commandPlan(
        datasetDirectory: URL,
        executableURL: URL,
        force: Bool = false
    ) -> [LocalDatasetEnrichmentCommand] {
        let datasetPath = datasetDirectory.path
        let vocalBundle = datasetDirectory
            .appendingPathComponent(TrainingDatasetPackageWriter.vocalBundleFileName)
            .path
        let instrumentalBundle = datasetDirectory
            .appendingPathComponent(TrainingDatasetPackageWriter.instrumentalBundleFileName)
            .path

        var separateArguments = [
            "dataset",
            "separate",
            datasetPath,
            "--mode",
            "instrumental",
            "--backend",
            "audio-separator",
            "--output-format",
            "wav",
        ]
        if force {
            separateArguments.append("--force")
        }

        return [
            LocalDatasetEnrichmentCommand(
                step: .separatingStems,
                executableURL: executableURL,
                arguments: separateArguments
            ),
            LocalDatasetEnrichmentCommand(
                step: .exportingVocalsBundle,
                executableURL: executableURL,
                arguments: [
                    "dataset",
                    "export-training-bundle",
                    datasetPath,
                    "--profile",
                    "acestep-lora",
                    "--include",
                    "vocals",
                    "--output",
                    vocalBundle,
                    "--strict",
                ]
            ),
            LocalDatasetEnrichmentCommand(
                step: .exportingInstrumentalBundle,
                executableURL: executableURL,
                arguments: [
                    "dataset",
                    "export-training-bundle",
                    datasetPath,
                    "--profile",
                    "acestep-lora",
                    "--include",
                    "instrumental",
                    "--output",
                    instrumentalBundle,
                    "--strict",
                ]
            ),
        ]
    }

    nonisolated static func resolveAnvilExecutableURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> URL {
        for url in candidateAnvilExecutableURLs(environment: environment)
            where FileManager.default.isExecutableFile(atPath: url.path) {
            return url
        }
        throw LocalDatasetEnrichmentError.missingAnvilCLI
    }

    nonisolated static func candidateAnvilExecutableURLs(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [URL] {
        var candidates: [URL] = []
        if let configured = environment["ANVIL_AUDIO_CLI"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: (configured as NSString).expandingTildeInPath))
        }

        let pathDirs = (environment["PATH"] ?? "")
            .split(separator: ":")
            .map(String.init)
            .filter { !$0.isEmpty }
        candidates.append(contentsOf: pathDirs.map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)
                .appendingPathComponent("anvil")
        })

        let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        candidates.append(
            home
                .appendingPathComponent("Developer/GitRepos/anvil-audio-v2/.venv/bin/anvil")
        )
        candidates.append(
            home
                .appendingPathComponent("Developer/anvil-audio-v2/.venv/bin/anvil")
        )

        let sourcePath = URL(fileURLWithPath: #filePath)
        let gitRepos = sourcePath
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        candidates.append(
            gitRepos
                .appendingPathComponent("anvil-audio-v2/.venv/bin/anvil")
        )

        return candidates.deduplicatedByPath()
    }

    nonisolated private static func commandEnvironment(
        executableURL: URL,
        base: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var environment = base
        let home = NSHomeDirectory()
        let executableDir = executableURL.deletingLastPathComponent().path
        let separatorToolDir = "\(home)/.cache/anvil-audio/tools/audio-separator/bin"
        let pathParts = [
            executableDir,
            separatorToolDir,
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/opt/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
            base["PATH"] ?? "",
        ]
        environment["PATH"] = pathParts
            .flatMap { $0.split(separator: ":").map(String.init) }
            .filter { !$0.isEmpty }
            .deduplicated()
            .joined(separator: ":")
        environment["PYTHONUNBUFFERED"] = "1"
        environment["NO_COLOR"] = "1"
        environment["CLICOLOR"] = "0"
        environment["TERM"] = "dumb"

        let separatorExecutable = "\(separatorToolDir)/audio-separator"
        if environment["ANVIL_AUDIO_SEPARATOR_BIN"]?.isEmpty != false,
           FileManager.default.isExecutableFile(atPath: separatorExecutable) {
            environment["ANVIL_AUDIO_SEPARATOR_BIN"] = separatorExecutable
        }
        return environment
    }

    nonisolated private static func inspectResult(datasetDirectory: URL) -> LocalDatasetEnrichmentResult {
        let captions = readJSONArray(from: datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName))
        let separatedClipCount = captions.filter { record in
            guard let separation = record["separation"] as? [String: Any],
                  let stems = separation["stems"] as? [String: Any] else {
                return false
            }
            return !stems.isEmpty
        }.count
        let transcribedClipCount = captions.filter { !transcriptionText(from: $0).isEmpty }.count
        let bpmEstimates = stemBPMEstimates(datasetDirectory: datasetDirectory, captions: captions)
        let averageBPM = bpmEstimates.isEmpty
            ? nil
            : rounded(bpmEstimates.reduce(0, +) / Double(bpmEstimates.count), places: 1)

        return LocalDatasetEnrichmentResult(
            datasetDirectory: datasetDirectory,
            separatedClipCount: separatedClipCount,
            vocalBundleExists: FileManager.default.fileExists(
                atPath: datasetDirectory
                    .appendingPathComponent(TrainingDatasetPackageWriter.vocalBundleFileName)
                    .path
            ),
            instrumentalBundleExists: FileManager.default.fileExists(
                atPath: datasetDirectory
                    .appendingPathComponent(TrainingDatasetPackageWriter.instrumentalBundleFileName)
                    .path
            ),
            bpmEstimateCount: bpmEstimates.count,
            averageBPM: averageBPM,
            transcribedClipCount: transcribedClipCount
        )
    }

    nonisolated private static func refreshDatasetMetadata(
        datasetDirectory: URL,
        result: LocalDatasetEnrichmentResult
    ) throws {
        let manifestURL = datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.datasetManifestFileName)
        var manifest = readJSONObject(from: manifestURL) ?? [:]
        let now = ISO8601DateFormatter().string(from: Date())

        var separation = manifest["separation"] as? [String: Any] ?? [:]
        var availableAssets = ["full_mix"]
        if result.vocalBundleExists {
            availableAssets.append("vocals")
        }
        if result.instrumentalBundleExists {
            availableAssets.append("instrumental")
        }
        separation["enabled"] = result.separatedClipCount > 0
        separation["status"] = result.separatedClipCount > 0 ? "complete" : "not_run"
        separation["available_assets"] = availableAssets
        separation["updated_at"] = separation["updated_at"] ?? now
        separation["vocal_bundle"] = TrainingDatasetPackageWriter.vocalBundleFileName
        separation["instrumental_bundle"] = TrainingDatasetPackageWriter.instrumentalBundleFileName
        manifest["separation"] = separation

        var transcription = manifest["transcription"] as? [String: Any] ?? [:]
        transcription["enabled"] = result.transcribedClipCount > 0
        transcription["status"] = transcriptionStatus(for: result)
        transcription["clip_count"] = result.transcribedClipCount
        manifest["transcription"] = transcription

        var analysis = manifest["analysis"] as? [String: Any] ?? [:]
        analysis["enabled"] = result.bpmEstimateCount > 0
        analysis["status"] = result.bpmEstimateCount > 0 ? "stem_analysis_available" : (analysis["status"] ?? "not_run")
        analysis["bpm_status"] = result.bpmEstimateCount > 0 ? "estimated_from_stems" : "not_run"
        analysis["key_status"] = "not_run"
        if let averageBPM = result.averageBPM {
            analysis["stem_tempo_bpm"] = [
                "average": averageBPM,
                "estimate_count": result.bpmEstimateCount,
            ]
        }
        manifest["analysis"] = analysis

        var training = manifest["training"] as? [String: Any] ?? [:]
        training["default_bundle"] = TrainingDatasetPackageWriter.trainingBundleFileName
        training["full_mix_bundle"] = TrainingDatasetPackageWriter.fullMixBundleFileName
        training["vocal_bundle"] = TrainingDatasetPackageWriter.vocalBundleFileName
        training["instrumental_bundle"] = TrainingDatasetPackageWriter.instrumentalBundleFileName
        training["dataset_config"] = TrainingDatasetPackageWriter.datasetConfigFileName
        manifest["training"] = training

        manifest["enrichment"] = [
            "updated_at": now,
            "status": result.separatedClipCount > 0 ? "partial" : "not_run",
            "separated_clip_count": result.separatedClipCount,
            "vocal_bundle": result.vocalBundleExists ? TrainingDatasetPackageWriter.vocalBundleFileName : "",
            "instrumental_bundle": result.instrumentalBundleExists ? TrainingDatasetPackageWriter.instrumentalBundleFileName : "",
            "transcription_status": transcriptionStatus(for: result),
            "bpm_status": result.bpmEstimateCount > 0 ? "estimated_from_stems" : "not_run",
            "key_status": "not_run",
        ]

        try writeJSONObject(manifest, to: manifestURL)
    }

    nonisolated private static func transcriptionStatus(for result: LocalDatasetEnrichmentResult) -> String {
        guard result.transcribedClipCount > 0 else { return "not_run" }
        if result.separatedClipCount > 0, result.transcribedClipCount < result.separatedClipCount {
            return "partial"
        }
        return "complete"
    }

    nonisolated private static func stemBPMEstimates(
        datasetDirectory: URL,
        captions: [[String: Any]]
    ) -> [Double] {
        captions.flatMap { record -> [Double] in
            guard let separation = record["separation"] as? [String: Any],
                  let separationFile = separation["separation_file"] as? String,
                  !separationFile.isEmpty else {
                return []
            }
            let separationURL = resolve(separationFile, relativeTo: datasetDirectory)
            guard let payload = readJSONObject(from: separationURL),
                  let stems = payload["stems"] as? [String: Any] else {
                return []
            }
            return stems.values.compactMap { value in
                guard let stem = value as? [String: Any],
                      let analysis = stem["analysis"] as? [String: Any] else {
                    return nil
                }
                return doubleValue(analysis["tempo_bpm_estimate"])
            }
        }
    }

    nonisolated private static func transcriptionText(from record: [String: Any]) -> String {
        if let transcription = record["transcription"] as? [String: Any],
           let text = transcription["text"] as? String,
           !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let transcript = record["transcript"] as? String,
           !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return ""
    }

    nonisolated private static func readJSONObject(from url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    nonisolated private static func readJSONArray(from url: URL) -> [[String: Any]] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] ?? []
    }

    nonisolated private static func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    nonisolated private static func resolve(_ path: String, relativeTo baseDirectory: URL) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL
        }
        return baseDirectory.appendingPathComponent(expanded).standardizedFileURL
    }

    nonisolated private static func doubleValue(_ value: Any?) -> Double? {
        if let number = value as? NSNumber {
            return number.doubleValue
        }
        if let string = value as? String {
            return Double(string)
        }
        return nil
    }

    nonisolated private static func rounded(_ value: Double, places: Int) -> Double {
        let scale = pow(10, Double(places))
        return (value * scale).rounded() / scale
    }
}

private final class LocalDatasetCapturedProcess: @unchecked Sendable {
    private let process = Process()
    private let stdout = Pipe()
    private let stderr = Pipe()
    private let lock = NSLock()
    private let command: LocalDatasetEnrichmentCommand
    private var stdoutData = Data()
    private var stderrData = Data()
    private var continuation: CheckedContinuation<String, Error>?
    private var didResume = false
    private var didCancel = false

    init(command: LocalDatasetEnrichmentCommand, environment: [String: String]) {
        self.command = command
        process.executableURL = command.executableURL
        process.arguments = command.arguments
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

        let output = [
            String(data: outData, encoding: .utf8) ?? "",
            String(data: errData, encoding: .utf8) ?? "",
        ]
        .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        .joined(separator: "\n")

        if process.terminationStatus == 0 {
            continuation.resume(returning: output)
        } else {
            continuation.resume(
                throwing: LocalDatasetEnrichmentError.commandFailed(
                    step: command.step,
                    status: process.terminationStatus,
                    command: command.displayCommand,
                    output: output
                )
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

private extension Array where Element == URL {
    func deduplicatedByPath() -> [URL] {
        var seen = Set<String>()
        return filter { url in
            seen.insert(url.standardizedFileURL.path).inserted
        }
    }
}

private extension Array where Element == String {
    func deduplicated() -> [String] {
        var seen = Set<String>()
        return filter { seen.insert($0).inserted }
    }
}
