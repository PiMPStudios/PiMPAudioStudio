import Foundation
import PiMPAudioCore
import SwiftUI

@MainActor
final class PromptEnhancer: ObservableObject {
    @Published var isEnhancing: Bool = false

    private var llm: LLMActor?
    private var enhanceTask: Task<String, Never>?

    /// Instructions tuned for ACE-Step / DiT music prompts: dense audio caption prose,
    /// not a production brief, title card, or mix-engineer essay.
    private static let systemPromptTemplate = """
        You rewrite short audio ideas into a single dense prompt for an AI music \
        generator (mode: %@).

        Output rules (strict):
        - Output ONLY the enhanced prompt as one continuous paragraph of plain prose.
        - No title, no song name, no labels, no markdown, no bullets, no numbered lists, \
        no headings, no quotes around the whole prompt.
        - Do not write "Genre:", "Instruments:", "Mood:", "Tempo:", or similar section labels.
        - Do not discuss frequency ranges, mix EQ, midrange, stereo field, or DAW technique.
        - Do not invent a track title or meta framing like "A Music Generation Mode Prompt".
        - Keep under 120 words. Prefer concrete nouns over abstract analysis.
        - Preserve every concrete detail from the user (instruments, voice, place, weather, \
        era, key/tempo if given). Expand missing useful cues (genre, tempo band, mood, \
        arrangement density, production vibe) without replacing what they already said.
        - Write like an audio caption a model can hear, e.g. "slow delta blues, gritty \
        raspy male vocals, national steel guitar, mournful harmonica..." not a studio memo.
        """

    init(llm: LLMActor?) {
        self.llm = llm
    }

    func updateLLM(_ llm: LLMActor?) {
        self.llm = llm
    }

    var isAvailable: Bool {
        llm != nil
    }

    func enhance(prompt: String, mode: GenerationMode) async -> String {
        guard let llm else { return prompt }

        enhanceTask?.cancel()

        let task = Task {
            isEnhancing = true
            defer { isEnhancing = false }

            let systemPrompt = String(format: Self.systemPromptTemplate, mode.displayName)

            var accumulated = ""
            let stream = await llm.generate(
                systemPrompt: systemPrompt,
                userPrompt: prompt,
                maxTokens: 180,
                temperature: 0.55
            )
            // Exiting this loop (cancel or finish) terminates the stream and
            // cancels the LLMActor producer via onTermination.
            for await token in stream {
                if Task.isCancelled { break }
                accumulated += token
            }
            if Task.isCancelled { return prompt }

            let enhanced = Self.sanitizeEnhancedPrompt(accumulated)
            return enhanced.isEmpty ? prompt : enhanced
        }
        enhanceTask = task

        return await task.value
    }

    func cancel() {
        enhanceTask?.cancel()
        enhanceTask = nil
        isEnhancing = false
    }

    /// Light cleanup when the model ignores format rules (titles, bullets, labels).
    static func sanitizeEnhancedPrompt(_ raw: String) -> String {
        var text = raw
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !text.isEmpty else { return "" }

        // Drop fenced code blocks if the model wraps output.
        text = text.replacingOccurrences(
            of: #"^```(?:\w+)?\s*"#,
            with: "",
            options: .regularExpression
        )
        text = text.replacingOccurrences(
            of: #"\s*```$"#,
            with: "",
            options: .regularExpression
        )

        var lines = text
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        // Strip a leading title line like `"Delta Storm": ...` or plain Title Case heading.
        if let first = lines.first {
            let looksLikeTitle =
                first.hasPrefix("\"") ||
                first.hasPrefix("#") ||
                first.localizedCaseInsensitiveContains("music generation mode prompt") ||
                (first.count < 80 && first.hasSuffix(":") && !first.contains(","))
            if looksLikeTitle {
                lines.removeFirst()
            }
        }

        // Turn bullets / numbered list markers into plain clauses.
        lines = lines.map { line in
            var s = line
            s = s.replacingOccurrences(of: #"^#+\s*"#, with: "", options: .regularExpression)
            s = s.replacingOccurrences(of: #"^[-*•]\s+"#, with: "", options: .regularExpression)
            s = s.replacingOccurrences(of: #"^\d+[\.\)]\s+"#, with: "", options: .regularExpression)
            // "Genre: Foo" / "Mood: Bar" → "Foo" / "Bar"
            s = s.replacingOccurrences(
                of: #"^(Genre|Instruments?|Mood|Tempo|Texture|Production(?: Style)?|Key|BPM|Vocals?)\s*:\s*"#,
                with: "",
                options: [.regularExpression, .caseInsensitive]
            )
            return s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        .filter { !$0.isEmpty }

        text = lines.joined(separator: " ")
        // Collapse whitespace and stray list debris.
        text = text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: #"\s+\*\s+"#, with: ", ", options: .regularExpression)
        text = text.trimmingCharacters(in: CharacterSet(charactersIn: "\"'` "))

        // If the model still rambling-cut off mid-word, prefer complete trailing punctuation
        // by trimming a trailing incomplete fragment after the last comma/period when cut short.
        if let last = text.last, !".!?".contains(last), text.count > 40 {
            if let cut = text.lastIndex(where: { ",.;:".contains($0) }) {
                let after = text[text.index(after: cut)...]
                if after.split(separator: " ").count <= 4 {
                    text = String(text[...cut]).trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        }

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
