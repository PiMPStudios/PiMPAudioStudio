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
        Output only lyrics with section markers. No commentary.
        """
    }

    func userPrompt(prompt: String, style: String) -> String {
        let trimmedStyle = style.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedStyle.isEmpty else {
            return "Song description:\n\(prompt)"
        }
        return "Song description:\n\(prompt)\n\nStyle:\n\(trimmedStyle)"
    }
}
