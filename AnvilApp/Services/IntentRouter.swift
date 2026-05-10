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
        Classify this audio prompt. Reply with exactly one word: "music", "soundfx", or "ambient". No explanation.

        Choose "music" if the prompt contains instrument names (guitar, piano, bass, drums, synth, pad, vocals), genre names (rock, jazz, pop, metal, hiphop, EDM, classical, doom), musical terms (BPM, tempo, riff, melody, chorus, beat, groove, hook, key), or clearly describes a song or track.

        Choose "soundfx" if the prompt describes a physical action, creature, or real-world sound event — with or without a surface, material, or location qualifier. Key signals: footsteps, steps, walking, running, stomping, door, slam, creak, click, tap, knock, drip, pour, shatter, break, crack, snap, bark, growl, meow, roar, engine, motor, alarm, siren, explosion, gunshot, thunder, rain hitting something, typing, crunching, rolling, bouncing.

        Choose "ambient" if the prompt describes a continuous non-musical environmental background with no specific foreground event: "rainforest at night", "ocean waves", "city traffic", "wind blowing", "cafe chatter", "room tone", "crowd murmur", "nature ambience".

        When ambiguous between soundfx and ambient, prefer "soundfx".
        When ambiguous between all three, prefer "music" only if musical content is clearly implied.

        Examples:
        - "footsteps on a wooden floor" → soundfx
        - "footsteps on gravel" → soundfx
        - "dog barking" → soundfx
        - "rain hitting a window" → soundfx
        - "door creaking open" → soundfx
        - "glass shattering" → soundfx
        - "thunder crack" → soundfx
        - "engine revving" → soundfx
        - "ocean waves on a beach" → ambient
        - "rainforest at night" → ambient
        - "city traffic ambience" → ambient
        - "slow dark guitar riff" → music
        - "doom metal with heavy bass" → music
        - "ethereal ambient pad with reverb" → music
        - "ominous dark soundscape" → music
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
