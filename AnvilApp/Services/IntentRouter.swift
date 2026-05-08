import Foundation
import AnvilCore
import SwiftUI

@MainActor
final class IntentRouter: ObservableObject {
    @Published var detectedMode: GenerationMode = .music
    @Published var isClassifying: Bool = false

    private var llm: LLMActor?
    private var debounceTask: Task<Void, Never>?

    private static let systemPrompt = """
        You classify audio generation prompts. Reply with exactly one word: "music", "soundfx", or "ambient". No explanation.

        Rules (apply in order):
        1. If the prompt mentions a musical instrument (guitar, bass, drums, piano, synth, vocals, etc.), a genre (rock, jazz, pop, metal, hiphop, classical, EDM, doom, ambient music, etc.), a tempo/BPM, key, or song-structure terms (riff, verse, chorus, hook, melody, harmony, beat, groove) → "music".
        2. If the prompt describes a SHORT, single, real-world sound effect like "door slam", "siren", "thunderclap", "knock", "explosion", "gunshot", "footsteps", "glass breaking" → "soundfx".
        3. If the prompt describes a NON-MUSICAL environmental loop or background like "rainforest at night", "wind blowing", "ocean waves", "office room tone", "cafe chatter", "city traffic" with no musical instruments → "ambient".
        4. Mood/atmosphere words alone ("dark", "atmospheric", "soundscape", "foreboding", "ominous", "ethereal", "haunting", "dreamy", "moody") do NOT mean ambient. Only choose ambient if the prompt clearly describes environmental sound rather than music.
        5. If genuinely ambiguous, default to "music".

        Examples:
        - "slow dark guitar riff with foreboding atmosphere" → music (rule 1: guitar, riff)
        - "atmospheric soundscape with heavy distorted guitar at 90 BPM" → music (rule 1: guitar, BPM)
        - "doom metal song with crushing oppressive mix" → music (rule 1: doom metal)
        - "rain on a tin roof at night" → ambient (rule 3, no instruments)
        - "single explosion sound" → soundfx (rule 2)
        - "ethereal ambient pad with shimmering reverb" → music (rule 1: pad is an instrument; ambient music ≠ ambient SFX)
        - "ominous dark soundscape" → music (rule 5: ambiguous, default music)
        """

    private static let debounceInterval: Duration = .milliseconds(300)

    init(llm: LLMActor?) {
        self.llm = llm
    }

    func updateLLM(_ llm: LLMActor?) {
        self.llm = llm
    }

    func classify(prompt: String) {
        debounceTask?.cancel()

        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            detectedMode = .music
            return
        }

        guard let llm else { return }

        debounceTask = Task { [weak self] in
            try? await Task.sleep(for: Self.debounceInterval)
            guard !Task.isCancelled else { return }
            guard let self else { return }

            self.isClassifying = true
            let result = await llm.classify(
                systemPrompt: Self.systemPrompt,
                userPrompt: trimmed
            )
            guard !Task.isCancelled else {
                self.isClassifying = false
                return
            }

            if let mode = GenerationMode(fromClassification: result) {
                self.detectedMode = mode
            }
            self.isClassifying = false
        }
    }

    func cancel() {
        debounceTask?.cancel()
        isClassifying = false
    }
}
