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

        // Apply keyword classifier immediately — works with no download required.
        detectedMode = Self.keywordClassify(trimmed)

        // When an LLM is available, enqueue a debounced refinement pass.
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

    // MARK: - Keyword classifier

    /// Classifies a prompt without requiring any LLM download.
    ///
    /// Scores the prompt against word-level signal sets for music, soundfx, and ambient,
    /// with phrase-level boosts for common multi-word patterns. Music is the default
    /// when no signal is found — it covers the widest range of ambiguous prompts.
    private static func keywordClassify(_ prompt: String) -> GenerationMode {
        let lower = prompt.lowercased()
        let words = Set(
            lower.components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty }
        )

        let musicWords: Set<String> = [
            "guitar", "piano", "bass", "drums", "drum", "synth", "synthesizer", "pad",
            "violin", "cello", "trumpet", "saxophone", "sax", "flute", "keyboard", "organ",
            "harp", "banjo", "ukulele", "vocals", "voice", "strings", "percussion", "brass",
            "rock", "jazz", "pop", "metal", "edm", "classical", "doom", "folk", "country",
            "blues", "soul", "funk", "reggae", "lofi", "synthwave", "trap", "indie",
            "punk", "grunge", "disco", "house", "techno",
            "bpm", "tempo", "riff", "melody", "chorus", "verse", "hook", "groove", "rhythm",
            "chord", "orchestral", "instrumental", "song", "music", "beat", "solo", "ballad"
        ]

        let sfxWords: Set<String> = [
            "footsteps", "footstep", "steps", "walking", "running", "stomping", "stomp",
            "knock", "knocking", "click", "clicking", "tap", "tapping",
            "creak", "creaking", "slam", "slamming", "crash", "crashing", "bang", "clang",
            "drip", "dripping", "pour", "pouring", "splash", "splashing",
            "explosion", "gunshot", "gunfire", "shattering", "shatter", "breaking",
            "cracking", "snapping", "bouncing", "rolling", "impact", "typing",
            "bark", "barking", "growl", "growling", "meow", "roar", "howl", "howling",
            "chirp", "chirping", "squeak", "squeaking", "hiss", "grunt",
            "engine", "motor", "revving", "rev", "alarm", "siren", "horn", "buzzing", "drill",
            "thunder", "lightning", "thunderclap"
        ]

        let ambientWords: Set<String> = [
            "rainforest", "jungle", "forest", "ocean", "waves", "beach",
            "wind", "breeze", "rain", "rainfall", "rainstorm", "thunderstorm", "snowfall",
            "city", "traffic", "cafe", "crowd", "murmur", "chatter",
            "nature", "birdsong", "crickets", "insects", "underwater",
            "stream", "river", "waterfall", "fireplace", "campfire",
            "ambience", "ambiance"
        ]

        var musicScore   = words.intersection(musicWords).count * 3
        var sfxScore     = words.intersection(sfxWords).count * 3
        var ambientScore = words.intersection(ambientWords).count * 2

        // Phrase boosts for multi-word signals
        if lower.contains("hip hop") || lower.contains("hip-hop") { musicScore += 3 }
        if lower.contains("lo-fi") || lower.contains("lo fi")     { musicScore += 3 }
        if lower.contains("ambient music")                         { musicScore += 4 }
        if lower.contains("rain hit") || lower.contains("rain on ") || lower.contains("raining on") {
            sfxScore += 4
        }
        if lower.contains("thunder crack") || lower.contains("thunderclap") { sfxScore += 3 }
        if lower.contains("room tone")    { ambientScore += 4 }
        if lower.contains("ocean waves")  { ambientScore += 3 }
        if lower.contains("city traffic") { ambientScore += 3 }
        if lower.contains("city ambien")  { ambientScore += 3 }

        // "pad" is a synth instrument — suppress any ambient score when present
        if words.contains("pad") { ambientScore = max(0, ambientScore - 3) }

        // Default to music when no signal is found
        guard musicScore > 0 || sfxScore > 0 || ambientScore > 0 else { return .music }

        // SFX beats ambient when tied (specific foreground events take priority)
        if sfxScore >= ambientScore && sfxScore > musicScore { return .soundFX }
        if ambientScore > sfxScore && ambientScore > musicScore { return .ambient }
        return .music
    }
}
