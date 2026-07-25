import Foundation

protocol LyricWriter: Sendable {
    var id: String { get }
    var displayName: String { get }
    var requiresAPIKey: Bool { get }
    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncThrowingStream<String, Error>
}

enum LyricWriterError: LocalizedError {
    case requestEncodingFailed(String)
    case requestFailed(String, String)
    case providerExited(String, Int32, String?)

    var errorDescription: String? {
        switch self {
        case .requestEncodingFailed(let provider):
            return "\(provider) lyric request could not be encoded."
        case .requestFailed(let provider, let detail):
            return "\(provider) lyric request failed: \(detail)"
        case .providerExited(let provider, let status, let detail):
            if let detail, !detail.isEmpty {
                return "\(provider) lyric writer exited with status \(status): \(detail)"
            }
            return "\(provider) lyric writer exited with status \(status)."
        }
    }
}

enum LyricWriterRequest {
    static func jsonBody(_ body: [String: Any], provider: String) throws -> Data {
        do {
            return try JSONSerialization.data(withJSONObject: body)
        } catch {
            throw LyricWriterError.requestEncodingFailed(provider)
        }
    }

    static func validate(response: URLResponse, provider: String) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw LyricWriterError.requestFailed(provider, "HTTP \(http.statusCode)")
        }
    }
}

struct ProviderCLILyricWriter: LyricWriter {
    let id: String
    let displayName: String
    let requiresAPIKey = false

    private let command: String
    private let mode: ProviderCLILyricMode
    private let homeDirectoryMode: ProviderCLIHomeDirectoryMode

    init?(
        providerID: String,
        displayName: String,
        command: String,
        homeDirectoryMode: ProviderCLIHomeDirectoryMode
    ) {
        guard let mode = ProviderCLILyricMode(providerID: providerID) else {
            return nil
        }

        self.id = providerID
        self.displayName = displayName
        self.command = command
        self.mode = mode
        self.homeDirectoryMode = homeDirectoryMode
    }

    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                let plan = LyricWritingPlan.make(duration: duration)
                let userPrompt = plan.userPrompt(prompt: prompt, style: style)
                let fullPrompt = "\(plan.systemPrompt)\n\n\(userPrompt)"

                do {
                    let invocation = try mode.invocation(
                        systemPrompt: plan.systemPrompt,
                        userPrompt: userPrompt,
                        fullPrompt: fullPrompt
                    )
                    defer { invocation.cleanup() }

                    // Provider CLIs often need to write session/cache files under HOME.
                    // Sandbox entitlements make real `~/.grok` / `~/.codex` read-only (Gemini is
                    // read-write, Claude uses process home + Keychain). Lyric writes are one-shot
                    // and do not need durable sessions, so seed a temporary writable HOME.
                    let isolation = try ProviderCLILyricIsolation.prepare(
                        mode: mode,
                        displayName: displayName
                    )
                    defer { isolation.cleanup() }

                    let result = try await ProviderCommandRunner.run(
                        command: command,
                        arguments: invocation.arguments,
                        standardInput: invocation.standardInput,
                        homeDirectoryMode: isolation.homeDirectoryMode ?? homeDirectoryMode,
                        currentDirectoryURL: isolation.workingDirectoryURL,
                        timeoutSeconds: 120
                    )

                    guard result.exitStatus == 0 else {
                        let detail = LyricCLIFailureDetail.message(from: result)
                        LyricCLIFailureDetail.persist(
                            provider: displayName,
                            command: command,
                            arguments: invocation.arguments,
                            result: result
                        )
                        throw LyricWriterError.providerExited(
                            displayName,
                            result.exitStatus,
                            detail
                        )
                    }

                    let text = LyricCLIResponseParser.responseText(
                        from: result,
                        providerID: id
                    )
                    if !text.isEmpty {
                        continuation.yield(text)
                    }
                } catch {
                    continuation.finish(throwing: error)
                    return
                }

                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

/// Isolation package for one-shot provider CLI lyric runs.
///
/// Real home provider dirs are often sandbox **read-only** (`~/.grok`, `~/.codex`). Lyric
/// generation does not need durable sessions, so we seed a temporary writable HOME with only
/// the credentials needed for auth, then discard it.
struct ProviderCLILyricIsolation {
    let homeDirectoryMode: ProviderCLIHomeDirectoryMode?
    let workingDirectoryURL: URL
    let cleanupURLs: [URL]

    func cleanup(fileManager: FileManager = .default) {
        for url in cleanupURLs {
            try? fileManager.removeItem(at: url)
        }
    }

    static func prepare(
        mode: ProviderCLILyricMode,
        displayName: String,
        fileManager: FileManager = .default,
        realUserHome: String = ProviderCLIPaths.realUserHomeDirectory()
    ) throws -> ProviderCLILyricIsolation {
        let workRoot = fileManager.temporaryDirectory
            .appendingPathComponent(
                "pimpaudio-lyric-cli-\(mode.rawValue)-\(UUID().uuidString)",
                isDirectory: true
            )
        try fileManager.createDirectory(at: workRoot, withIntermediateDirectories: true)

        switch mode {
        case .claude:
            // Claude auth is Keychain-backed; process/container HOME is already writable.
            // Still pin cwd to a sandbox-writable directory.
            return ProviderCLILyricIsolation(
                homeDirectoryMode: nil,
                workingDirectoryURL: workRoot,
                cleanupURLs: [workRoot]
            )

        case .codex:
            let home = try seedHome(
                providerFolder: ".codex",
                displayName: displayName,
                requiredFiles: ["auth.json"],
                optionalFiles: ["config.toml"],
                extraFiles: ["hooks.json": Data("{}\n".utf8)],
                workRoot: workRoot,
                realUserHome: realUserHome,
                fileManager: fileManager,
                notSignedInHint: "Run `codex login` in Terminal, then retry."
            )
            return ProviderCLILyricIsolation(
                homeDirectoryMode: .custom(home.path),
                workingDirectoryURL: workRoot,
                cleanupURLs: [workRoot]
            )

        case .gemini:
            let home = try seedHome(
                providerFolder: ".gemini",
                displayName: displayName,
                requiredFiles: ["settings.json"],
                optionalFiles: [
                    "oauth_creds.json",
                    "google_accounts.json",
                    "installation_id",
                    "state.json",
                    "trustedFolders.json",
                    "projects.json",
                ],
                extraFiles: [:],
                workRoot: workRoot,
                realUserHome: realUserHome,
                fileManager: fileManager,
                notSignedInHint: "Run `gemini` and complete Google OAuth, then retry."
            )
            return ProviderCLILyricIsolation(
                homeDirectoryMode: .custom(home.path),
                workingDirectoryURL: workRoot,
                cleanupURLs: [workRoot]
            )

        case .grok:
            let home = try seedHome(
                providerFolder: ".grok",
                displayName: displayName,
                requiredFiles: ["auth.json"],
                optionalFiles: ["config.toml", "agent_id"],
                extraFiles: [:],
                workRoot: workRoot,
                realUserHome: realUserHome,
                fileManager: fileManager,
                notSignedInHint: "Run `grok login --oauth` or `grok login --device-auth` in Terminal, then retry."
            )
            return ProviderCLILyricIsolation(
                homeDirectoryMode: .custom(home.path),
                workingDirectoryURL: workRoot,
                cleanupURLs: [workRoot]
            )
        }
    }

    private static func seedHome(
        providerFolder: String,
        displayName: String,
        requiredFiles: [String],
        optionalFiles: [String],
        extraFiles: [String: Data],
        workRoot: URL,
        realUserHome: String,
        fileManager: FileManager,
        notSignedInHint: String
    ) throws -> URL {
        let home = workRoot.appendingPathComponent("home", isDirectory: true)
        let providerDir = home.appendingPathComponent(providerFolder, isDirectory: true)
        try fileManager.createDirectory(at: providerDir, withIntermediateDirectories: true)

        let realProvider = URL(fileURLWithPath: realUserHome, isDirectory: true)
            .appendingPathComponent(providerFolder, isDirectory: true)

        for name in requiredFiles {
            let source = realProvider.appendingPathComponent(name)
            guard fileManager.fileExists(atPath: source.path) else {
                throw LyricWriterError.requestFailed(
                    displayName,
                    "Not signed in. \(notSignedInHint)"
                )
            }
            try fileManager.copyItem(at: source, to: providerDir.appendingPathComponent(name))
        }

        for name in optionalFiles {
            let source = realProvider.appendingPathComponent(name)
            guard fileManager.fileExists(atPath: source.path) else { continue }
            try? fileManager.copyItem(at: source, to: providerDir.appendingPathComponent(name))
        }

        for (name, data) in extraFiles {
            try data.write(to: providerDir.appendingPathComponent(name), options: .atomic)
        }

        return home
    }
}

/// Back-compat alias used by existing unit tests.
enum GrokLyricIsolatedHome {
    static func prepare(
        fileManager: FileManager = .default,
        realUserHome: String = ProviderCLIPaths.realUserHomeDirectory()
    ) throws -> URL {
        let isolation = try ProviderCLILyricIsolation.prepare(
            mode: .grok,
            displayName: "Grok (xAI)",
            fileManager: fileManager,
            realUserHome: realUserHome
        )
        // Caller owns cleanup of the full isolation package; expose HOME root only.
        guard case .custom(let path) = isolation.homeDirectoryMode else {
            throw LyricWriterError.requestFailed("Grok (xAI)", "Failed to prepare isolated home.")
        }
        // Stash cleanup list on a marker file so cleanup(_:) can remove parents.
        let home = URL(fileURLWithPath: path, isDirectory: true)
        let marker = home.appendingPathComponent(".pimpaudio-cleanup-roots")
        let roots = isolation.cleanupURLs.map(\.path).joined(separator: "\n")
        try? Data(roots.utf8).write(to: marker)
        return home
    }

    static func cleanup(_ url: URL, fileManager: FileManager = .default) {
        let marker = url.appendingPathComponent(".pimpaudio-cleanup-roots")
        if let data = try? Data(contentsOf: marker),
           let text = String(data: data, encoding: .utf8) {
            for path in text.split(separator: "\n").map(String.init) where !path.isEmpty {
                try? fileManager.removeItem(atPath: path)
            }
            return
        }
        try? fileManager.removeItem(at: url)
    }
}

enum LyricCLIFailureDetail {
    static func message(from result: ProviderCommandResult) -> String? {
        let candidates = [result.stderr, result.stdout, result.combinedOutput]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        // Pass 1: prefer structured JSON "message" fields (Grok/Codex style) from any
        // stream. Do not return a plain stderr first-line until every candidate has been
        // checked for JSON — otherwise "Error: …" on stderr masks a cleaner stdout JSON body.
        for candidate in candidates {
            if let message = firstJSONMessage(in: candidate) {
                return message
            }
        }

        // Pass 2: first non-empty line (stderr first, then stdout).
        for candidate in candidates {
            let firstLine = candidate
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .first { !$0.isEmpty }
            if let firstLine {
                return firstLine
            }
        }
        return nil
    }

    static func persist(
        provider: String,
        command: String,
        arguments: [String],
        result: ProviderCommandResult
    ) {
        let directory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("PiMPAudioStudio/Logs", isDirectory: true)
        guard let directory else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("lyric-cli-last-failure.log")
        let body = """
        provider=\(provider)
        command=\(command)
        arguments=\(arguments.joined(separator: " "))
        exit=\(result.exitStatus)
        launch=\(result.launchDiagnostic ?? "")
        --- stdout ---
        \(result.stdout)
        --- stderr ---
        \(result.stderr)
        """
        try? Data(body.utf8).write(to: url, options: .atomic)
    }

    private static func firstJSONMessage(in text: String) -> String? {
        if let data = text.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let message = object["message"] as? String {
            let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }

        // Scan for embedded JSON objects with a message field (stderr wrappers, JSONL, etc.).
        var depth = 0
        var start: String.Index?
        var index = text.startIndex
        while index < text.endIndex {
            let ch = text[index]
            if ch == "{" {
                if depth == 0 { start = index }
                depth += 1
            } else if ch == "}" {
                if depth > 0 {
                    depth -= 1
                    if depth == 0, let start {
                        let end = text.index(after: index)
                        let slice = String(text[start..<end])
                        if let data = slice.data(using: .utf8),
                           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                           let message = object["message"] as? String {
                            let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmed.isEmpty { return trimmed }
                        }
                    }
                }
            }
            index = text.index(after: index)
        }
        return nil
    }
}

struct ProviderCLILyricInvocation {
    let arguments: [String]
    let standardInput: String?
    let temporaryPromptFileURL: URL?

    func cleanup() {
        guard let temporaryPromptFileURL else { return }
        try? FileManager.default.removeItem(at: temporaryPromptFileURL)
    }
}

enum ProviderCLILyricMode: String, Sendable {
    case claude
    case codex
    case gemini
    case grok

    init?(providerID: String) {
        switch providerID {
        case "anthropic":
            self = .claude
        case "openai":
            self = .codex
        case "google":
            self = .gemini
        case "xai":
            self = .grok
        default:
            return nil
        }
    }

    func invocation(
        systemPrompt: String,
        userPrompt: String,
        fullPrompt: String
    ) throws -> ProviderCLILyricInvocation {
        switch self {
        case .claude:
            return ProviderCLILyricInvocation(
                arguments: [
                    "--print",
                    "--output-format",
                    "json",
                    "--permission-mode",
                    "plan",
                    "--no-session-persistence",
                ],
                standardInput: fullPrompt,
                temporaryPromptFileURL: nil
            )
        case .codex:
            return ProviderCLILyricInvocation(
                arguments: [
                    "exec",
                    "--skip-git-repo-check",
                    "--ignore-rules",
                    "--ignore-user-config",
                    "--sandbox",
                    "read-only",
                    "--color",
                    "never",
                    "--json",
                    "-",
                ],
                standardInput: fullPrompt,
                temporaryPromptFileURL: nil
            )
        case .gemini:
            return ProviderCLILyricInvocation(
                arguments: [
                    "--skip-trust",
                    "--approval-mode",
                    "plan",
                    "--output-format",
                    "json",
                    "--prompt",
                    "",
                ],
                standardInput: fullPrompt,
                temporaryPromptFileURL: nil
            )
        case .grok:
            let promptFileURL = try ProviderCLIPromptFile.write(fullPrompt)
            return ProviderCLILyricInvocation(
                arguments: [
                    "--no-memory",
                    "--no-subagents",
                    "--disable-web-search",
                    "--permission-mode",
                    "plan",
                    "--output-format",
                    "json",
                    "--prompt-file",
                    promptFileURL.path,
                ],
                standardInput: nil,
                temporaryPromptFileURL: promptFileURL
            )
        }
    }
}

enum ProviderCLIPromptFile {
    static func write(_ prompt: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pimpaudio-lyric-prompt-\(UUID().uuidString).txt")
        let didCreate = FileManager.default.createFile(
            atPath: url.path,
            contents: Data(prompt.utf8),
            attributes: [.posixPermissions: 0o600]
        )
        guard didCreate else {
            throw CocoaError(.fileWriteUnknown)
        }
        return url
    }
}

enum LyricCLIResponseParser {
    static func responseText(from result: ProviderCommandResult, providerID: String) -> String {
        let outputs = [result.stdout, result.combinedOutput]
        let objects = outputs.flatMap(jsonObjects)

        let parsed: String?
        switch providerID {
        case "anthropic":
            parsed = lastStringValue(in: objects, key: "result")
        case "openai":
            parsed = lastCodexAgentMessage(in: objects)
        case "google":
            parsed = lastStringValue(in: objects, key: "response")
        case "xai":
            parsed = lastStringValue(in: objects, key: "text")
        default:
            parsed = nil
        }

        return clean(parsed ?? result.stdout)
    }

    private static func lastStringValue(in objects: [[String: Any]], key: String) -> String? {
        objects.reversed().compactMap { $0[key] as? String }.first
    }

    private static func lastCodexAgentMessage(in objects: [[String: Any]]) -> String? {
        for object in objects.reversed() {
            guard object["type"] as? String == "item.completed",
                  let item = object["item"] as? [String: Any],
                  item["type"] as? String == "agent_message",
                  let text = item["text"] as? String else {
                continue
            }
            return text
        }
        return nil
    }

    private static func jsonObjects(in text: String) -> [[String: Any]] {
        var objects: [[String: Any]] = []
        var startIndex: String.Index?
        var depth = 0
        var isInsideString = false
        var isEscaped = false

        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]

            if isInsideString {
                if isEscaped {
                    isEscaped = false
                } else if character == "\\" {
                    isEscaped = true
                } else if character == "\"" {
                    isInsideString = false
                }
                index = text.index(after: index)
                continue
            }

            switch character {
            case "\"":
                isInsideString = true
            case "{":
                if depth == 0 {
                    startIndex = index
                }
                depth += 1
            case "}":
                guard depth > 0 else { break }
                depth -= 1
                if depth == 0, let objectStartIndex = startIndex {
                    let endIndex = text.index(after: index)
                    let candidate = String(text[objectStartIndex..<endIndex])
                    if let object = jsonObject(from: candidate) {
                        objects.append(object)
                    }
                    startIndex = nil
                }
            default:
                break
            }

            index = text.index(after: index)
        }

        return objects
    }

    private static func jsonObject(from text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return object
    }

    private static func clean(_ text: String) -> String {
        text.replacingOccurrences(
            of: "\u{001B}\\[[0-9;]*[A-Za-z]",
            with: "",
            options: .regularExpression
        )
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

struct LyricWritingPlan {
    let durationSeconds: Int
    let lineBudget: String
    let sectionPlan: String
    let maxTokens: Int

    static func make(duration: TimeInterval) -> LyricWritingPlan {
        let seconds = max(Int(duration.rounded()), 15)

        switch seconds {
        case ...35:
            return LyricWritingPlan(
                durationSeconds: seconds,
                lineBudget: "4 to 6 total lyric lines",
                sectionPlan: "Use [Verse] with 2 to 4 lines and optional [Hook] with 1 to 2 lines. Do not use [Verse 2], [Bridge], or [Outro].",
                maxTokens: 96
            )
        case ...60:
            return LyricWritingPlan(
                durationSeconds: seconds,
                lineBudget: "6 to 10 total lyric lines",
                sectionPlan: "Use exactly these section markers in this order: [Verse 1], [Chorus]. Keep each section short. Do not use [Bridge].",
                maxTokens: 140
            )
        case ...90:
            return LyricWritingPlan(
                durationSeconds: seconds,
                lineBudget: "10 to 14 total lyric lines",
                sectionPlan: "Use exactly these section markers in this order: [Verse 1], [Chorus], optional [Verse 2]. Do not use [Bridge] unless it replaces [Verse 2].",
                maxTokens: 200
            )
        case ...150:
            return LyricWritingPlan(
                durationSeconds: seconds,
                lineBudget: "14 to 20 total lyric lines",
                sectionPlan: "Use exactly these section markers in this order: [Verse 1], [Chorus], [Verse 2], [Chorus], optional [Bridge]. Keep the bridge to 2 lines.",
                maxTokens: 300
            )
        default:
            return LyricWritingPlan(
                durationSeconds: seconds,
                lineBudget: "20 to 32 total lyric lines",
                sectionPlan: "Use exactly these section markers in this order: [Verse 1], [Chorus], [Verse 2], [Chorus], [Bridge], optional [Outro]. Keep sections compact and avoid repeated filler.",
                maxTokens: 420
            )
        }
    }

    var systemPrompt: String {
        """
        You write tight, singable lyrics for an AI music generator (not a poem essay).
        Fit lyric density to the exact clip duration.
        Target duration: \(durationSeconds) seconds.
        Write \(lineBudget), total, across the whole song.
        \(sectionPlan)

        World and voice:
        - Read the song description as the sonic world. Match its mood, genre, and energy.
        - Pull concrete images from the description into the lyrics when they appear \
        (place, weather, time of night, objects, instruments, vocal character). Prefer \
        those over generic heartbreak language.
        - If the description names a vocal character (e.g. gritty raspy male), write lines \
        that sound like that singer, not a neutral pop narrator.
        - Let instruments color the scene when natural (slide answering from the next room, \
        harmonica cutting the gutter drip, porch screen, juke joint dark). Do not inventory \
        the band or paste production notes as lyrics.

        Line craft (recipe):
        - Each verse line = one concrete image or physical action (what is seen, touched, \
        heard, or done). Not a stack of mood adjectives.
        - Emotion words need an object or action ("throat catches on the long note", not \
        "mournful longing and despair").
        - Chorus = 1–2 short memorable hooks that return. Do not restate "I'm sad in the rain" \
        with new adjectives.
        - Prefer scene over synonym lists of mournful / lonely / empty / heartfelt.
        - End on a complete line. Never trail off mid-sentence.

        Hard bans (never write these, or close paraphrases):
        - "searching for a love", "searching for a light", "love that's gone"
        - "memories linger", "memories remain", "face the rain alone"
        - "broken heart", "cracks my heart", "tears fall like rain"
        - "lonely melody", "endless night", "ray of hope", "why did you leave"
        - "mournful sigh", "lonely thrill", "mournful star", "empty sky" as empty fillers
        - Production terms sung as lyrics: "minor key", "BPM", "tempo", "in the mix", \
        "slide guitar solo", "guitar weeps" as a band inventory line
        - Invented song titles, or a chorus that only names weather + sadness

        Form and output (strict):
        - If the prompt sounds instrumental, riff-based, ambient, or like a backing track, \
        write only a sparse chant or hook instead of a full vocal song.
        - The first non-empty character of your answer must be "[".
        - Every requested section marker must appear on its own line. Do not collapse the whole song into one [Verse].
        - Do not put more than 6 lyric lines under a single section marker.
        - Output only section markers and singable lyric lines.
        - Do not include commentary, titles, key, tempo, duration, line counts, explanations, \
        markdown bullets, chord symbols like (Am), timestamps like [0:00 - 0:10], or performance \
        directions like [Ambient fade-out] or [Guitar Solo].
        """
    }

    /// On-device Llama needs a short worked example; API/CLI backends skip this to save tokens.
    var systemPromptForLocalLLM: String {
        """
        \(systemPrompt)

        Example of good shape (style only — write fresh lines for the real song, do not copy):
        [Verse 1]
        Gray sheet on the porch screen
        Thumb on the third frets, same two bars again
        Slide answers from the kitchen dark
        My throat catches on the long notes

        [Chorus]
        Harmonica cuts once through the gutter drip
        I don't call your name, I just hold the bend
        """
    }

    func userPrompt(prompt: String, style: String) -> String {
        let trimmedStyle = style.trimmingCharacters(in: .whitespacesAndNewlines)
        let instruction = """
        Use the song description as the world and sound. Weave its concrete details into \
        the lyric imagery (place, weather, objects, vocal grain, instrument as action).

        Recipe reminder: one image or action per verse line; chorus is a short hook; no \
        pure emotion stacks; no production metadata as lyrics.

        Never write (or paraphrase): searching for a love/light, love that's gone, \
        memories linger/remain, face the rain alone, broken/cracks my heart, tears fall \
        like rain, lonely melody, mournful sigh, lonely thrill, empty sky, minor key, \
        BPM, tempo, slide guitar solo as a lyric line.

        Do not paste instrument lists, mix notes, timestamps, fade directions, or duration \
        as lyric lines.
        """
        guard !trimmedStyle.isEmpty else {
            return "Song description:\n\(prompt)\n\n\(instruction)"
        }
        return "Song description:\n\(prompt)\n\nStyle:\n\(trimmedStyle)\n\n\(instruction)"
    }
}

enum LyricOutputSanitizer {
    static func sanitize(_ rawText: String, targetDuration: TimeInterval? = nil) -> String {
        let normalized = rawText
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")

        var sanitizedLines: [String] = []
        var hasSeenSection = false

        for rawLine in normalized.components(separatedBy: "\n") {
            guard var line = cleanLine(rawLine) else { continue }
            guard !line.isEmpty else { continue }

            if isSectionMarker(line) {
                line = normalizedSectionMarker(line)
                appendLine(line, to: &sanitizedLines)
                hasSeenSection = true
                continue
            }

            guard !isBracketedDirective(line) else { continue }
            guard !isCommentary(line) else { continue }

            line = stripChordPrefixes(from: line)
            line = stripTrailingMetaSentence(from: line)
            guard let cleaned = cleanLine(line), !isCommentary(cleaned) else { continue }
            guard !cleaned.isEmpty else { continue }

            if !hasSeenSection {
                appendLine("[Verse]", to: &sanitizedLines)
                hasSeenSection = true
            }
            appendLine(cleaned, to: &sanitizedLines)
        }

        let collapsedLines = collapseBlankLines(in: sanitizedLines)
        let repairedLines = repairOversizedSingleVerse(
            collapsedLines,
            targetDuration: targetDuration
        )
        return repairedLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func cleanLine(_ line: String) -> String? {
        var cleaned = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return "" }
        guard cleaned != "```" else { return nil }

        cleaned = cleaned.replacingOccurrences(
            of: #"^[-*]\s+"#,
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "\"'` "))
        return cleaned.isEmpty ? nil : cleaned
    }

    private static func isSectionMarker(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        let pattern = #"^\[(intro|verse|pre[- ]?chorus|chorus|hook|refrain|bridge|outro|instrumental|break|drop|chant)(?:[^\]]*)\]"#
        return trimmed.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
    }

    private static func isBracketedDirective(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("["), trimmed.hasSuffix("]") else { return false }
        if isSectionMarker(trimmed) { return false }

        let timeRangePattern = #"^\[\d{1,2}:\d{2}(?::\d{2})?\s*[-–]\s*\d{1,2}:\d{2}(?::\d{2})?\]$"#
        if trimmed.range(of: timeRangePattern, options: .regularExpression) != nil {
            return true
        }

        let timestampPattern = #"^\[\d{1,2}:\d{2}(?::\d{2})?\]$"#
        if trimmed.range(of: timestampPattern, options: .regularExpression) != nil {
            return true
        }

        let lowercased = trimmed.lowercased()
        let directiveWords = [
            "ambient",
            "fade",
            "fade-out",
            "fade out",
            "guitar solo",
            "instrumental",
            "silence",
            "spoken",
            "whisper",
            "repeat",
            "hold"
        ]
        return directiveWords.contains(where: { lowercased.contains($0) })
    }

    private static func normalizedSectionMarker(_ line: String) -> String {
        guard let endIndex = line.firstIndex(of: "]") else { return line }
        return String(line[...endIndex])
    }

    private static func isCommentary(_ line: String) -> Bool {
        let lowercased = line.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let commentaryPrefixes = [
            "here's ",
            "here are ",
            "this song",
            "the song",
            "these lyrics",
            "lyrics:",
            "title:",
            "key:",
            "tempo:",
            "duration:",
            "total duration",
            "line count:",
            "note:",
            "notes:",
            "explanation:",
            "i hope "
        ]

        if commentaryPrefixes.contains(where: { lowercased.hasPrefix($0) }) {
            return true
        }

        let metaFragments = [
            " total duration",
            " total lyric",
            "approximately ",
            " approx.",
            " bpm",
            " seconds"
        ]
        return metaFragments.contains(where: { lowercased.contains($0) })
    }

    private static func stripChordPrefixes(from line: String) -> String {
        line.replacingOccurrences(
            of: #"^(\s*\([A-G](?:#|b)?(?:m|min|maj|dim|aug|sus|add)?[0-9]*(?:/[A-G](?:#|b)?)?\)\s*)+"#,
            with: "",
            options: .regularExpression
        )
    }

    private static func stripTrailingMetaSentence(from line: String) -> String {
        let pattern = #"\s+This song has.*$|\s+The song has.*$|\s+These lyrics.*$"#
        return line.replacingOccurrences(of: pattern, with: "", options: [.regularExpression, .caseInsensitive])
    }

    private static func appendLine(_ line: String, to lines: inout [String]) {
        if isSectionMarker(line), !lines.isEmpty, lines.last != "" {
            lines.append("")
        }
        lines.append(line)
    }

    private static func repairOversizedSingleVerse(
        _ lines: [String],
        targetDuration: TimeInterval?
    ) -> [String] {
        let nonBlankLines = lines.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let sectionMarkers = nonBlankLines.filter { isSectionMarker($0) }
        guard sectionMarkers.count == 1,
              isGenericVerseMarker(sectionMarkers[0])
        else {
            return lines
        }

        let lyricLines = nonBlankLines.filter { !isSectionMarker($0) }
        guard lyricLines.count >= oversizedVerseThreshold(targetDuration: targetDuration) else {
            return lines
        }

        let markers = repairSectionMarkers(
            targetDuration: targetDuration,
            lyricLineCount: lyricLines.count
        )
        guard markers.count > 1 else { return lines }

        var repaired: [String] = []
        var lyricIndex = 0
        let baseCount = lyricLines.count / markers.count
        let extraCount = lyricLines.count % markers.count

        for markerIndex in markers.indices {
            appendLine(markers[markerIndex], to: &repaired)
            let sectionLineCount = baseCount + (markerIndex < extraCount ? 1 : 0)
            for _ in 0..<sectionLineCount where lyricIndex < lyricLines.count {
                appendLine(lyricLines[lyricIndex], to: &repaired)
                lyricIndex += 1
            }
        }

        return collapseBlankLines(in: repaired)
    }

    private static func isGenericVerseMarker(_ line: String) -> Bool {
        let lowercased = line.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return lowercased == "[verse]" || lowercased == "[verse 1]"
    }

    private static func oversizedVerseThreshold(targetDuration: TimeInterval?) -> Int {
        guard let targetDuration else { return 16 }
        switch targetDuration {
        case ..<60: return 10
        case ..<90: return 12
        case ..<150: return 14
        default: return 12
        }
    }

    private static func repairSectionMarkers(
        targetDuration: TimeInterval?,
        lyricLineCount: Int
    ) -> [String] {
        if (targetDuration ?? 0) >= 150 || lyricLineCount >= 16 {
            if lyricLineCount >= 28 {
                return ["[Verse 1]", "[Chorus]", "[Verse 2]", "[Chorus]", "[Bridge]", "[Chorus]", "[Outro]"]
            }
            if lyricLineCount >= 22 {
                return ["[Verse 1]", "[Chorus]", "[Verse 2]", "[Chorus]", "[Bridge]", "[Chorus]"]
            }
            return ["[Verse 1]", "[Chorus]", "[Verse 2]", "[Chorus]", "[Bridge]"]
        }

        if (targetDuration ?? 0) >= 90 || lyricLineCount >= 10 {
            return ["[Verse 1]", "[Chorus]", "[Verse 2]", "[Chorus]"]
        }

        return ["[Verse 1]", "[Chorus]"]
    }

    private static func collapseBlankLines(in lines: [String]) -> [String] {
        var collapsed: [String] = []
        for line in lines {
            if line.isEmpty, collapsed.last == "" { continue }
            collapsed.append(line)
        }
        return collapsed
    }
}
