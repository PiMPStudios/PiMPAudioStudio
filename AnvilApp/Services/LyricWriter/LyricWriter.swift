import Foundation

protocol LyricWriter: Sendable {
    var id: String { get }
    var displayName: String { get }
    var requiresAPIKey: Bool { get }
    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncStream<String>
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
                sectionPlan: "Use [Verse] and [Chorus]. Keep each section short. Do not use [Bridge].",
                maxTokens: 140
            )
        case ...90:
            return LyricWritingPlan(
                durationSeconds: seconds,
                lineBudget: "10 to 14 total lyric lines",
                sectionPlan: "Use [Verse], [Chorus], and optional [Verse 2]. Do not use [Bridge] unless it replaces [Verse 2].",
                maxTokens: 200
            )
        case ...150:
            return LyricWritingPlan(
                durationSeconds: seconds,
                lineBudget: "14 to 20 total lyric lines",
                sectionPlan: "Use [Verse 1], [Chorus], [Verse 2], [Chorus], and optional [Bridge]. Keep the bridge to 2 lines.",
                maxTokens: 300
            )
        default:
            return LyricWritingPlan(
                durationSeconds: seconds,
                lineBudget: "20 to 32 total lyric lines",
                sectionPlan: "Use a full song structure, but keep sections compact and avoid repeated filler.",
                maxTokens: 420
            )
        }
    }

    var systemPrompt: String {
        """
        You are a concise songwriter writing lyrics for an AI music generator.
        Match the user's mood, genre, and energy, but fit the lyric density to the exact clip duration.
        Target duration: \(durationSeconds) seconds.
        Write \(lineBudget), total, across the whole song.
        \(sectionPlan)
        If the prompt sounds instrumental, riff-based, ambient, or like a backing track, write only a sparse chant or hook instead of a full vocal song.
        Prefer short concrete phrases over long narrative lines.
        End on a complete line. Never trail off mid-sentence.
        The first non-empty character of your answer must be "[".
        Output only section markers and singable lyric lines.
        Do not include commentary, titles, key, tempo, duration, line counts, explanations, markdown bullets, chord symbols like (Am), timestamps like [0:00 - 0:10], or performance directions like [Ambient fade-out].
        """
    }

    func userPrompt(prompt: String, style: String) -> String {
        let trimmedStyle = style.trimmingCharacters(in: .whitespacesAndNewlines)
        let instruction = """
        Use this as creative direction only. Do not repeat production notes, BPM, key, instrument lists, timestamps, fade directions, or duration metadata in the lyrics.
        """
        guard !trimmedStyle.isEmpty else {
            return "Song description:\n\(prompt)\n\n\(instruction)"
        }
        return "Song description:\n\(prompt)\n\nStyle:\n\(trimmedStyle)\n\n\(instruction)"
    }
}

enum LyricOutputSanitizer {
    static func sanitize(_ rawText: String) -> String {
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

        return collapseBlankLines(in: sanitizedLines).joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
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

    private static func collapseBlankLines(in lines: [String]) -> [String] {
        var collapsed: [String] = []
        for line in lines {
            if line.isEmpty, collapsed.last == "" { continue }
            collapsed.append(line)
        }
        return collapsed
    }
}
