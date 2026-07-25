import Foundation

/// Pending History → Generate restore (settings + adapter stack).
struct HistoryGenerateRestore {
    let preset: GenerationPreset
    /// Adapter stack payloads for `applyAdapterSelection` (`id` + `weight`).
    let adapterStack: [[String: Any]]
}

/// Manages CRUD operations for saved generation presets.
///
/// Presets are stored as individual JSON files in
/// `~/Library/Application Support/PiMP Audio Studio/presets/`.
@MainActor
final class PresetStore: ObservableObject {
    @Published var presets: [GenerationPreset] = []

    /// Set this to trigger a preset load in GenerateView.
    /// GenerateView clears it after applying.
    @Published var presetToLoad: GenerationPreset?

    /// Set this to restore a History take into Generate (prompt, knobs, adapters).
    /// GenerateView clears it after applying.
    @Published var historyRestore: HistoryGenerateRestore?

    static var presetsDirectory: URL {
        AppStorageLocations.presetsDirectory
    }

    init() {
        loadAll()
    }

    // MARK: - CRUD

    func save(_ preset: GenerationPreset) {
        let dir = Self.presetsDirectory
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(preset) else { return }

        let fileURL = dir.appending(path: "\(preset.id.uuidString).json")
        try? data.write(to: fileURL)

        if let idx = presets.firstIndex(where: { $0.id == preset.id }) {
            presets[idx] = preset
        } else {
            presets.insert(preset, at: 0)
        }
    }

    func delete(_ preset: GenerationPreset) {
        let fileURL = Self.presetsDirectory.appending(path: "\(preset.id.uuidString).json")
        try? FileManager.default.removeItem(at: fileURL)
        presets.removeAll { $0.id == preset.id }
    }

    func rename(_ preset: GenerationPreset, to newName: String) {
        guard var updated = presets.first(where: { $0.id == preset.id }) else { return }
        updated.name = newName
        save(updated)
    }

    func duplicate(_ preset: GenerationPreset) {
        let copy = GenerationPreset(
            name: "\(preset.name) Copy",
            prompt: preset.prompt,
            lyrics: preset.lyrics,
            mode: preset.mode,
            duration: preset.duration,
            seedText: preset.seedText,
            guidanceScale: preset.guidanceScale,
            guidancePreset: preset.guidancePreset,
            pipelineOverride: preset.pipelineOverride,
            method: preset.method,
            precision: preset.precision,
            stepsOverride: preset.stepsOverride,
            shiftOverride: preset.shiftOverride,
            adapterID: preset.adapterID,
            coverStrength: preset.coverStrength
        )
        save(copy)
    }

    // MARK: - Private

    private func loadAll() {
        let dir = Self.presetsDirectory
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles
        ) else {
            return
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        presets = files
            .filter { $0.pathExtension == "json" }
            .compactMap { try? Data(contentsOf: $0) }
            .compactMap { try? decoder.decode(GenerationPreset.self, from: $0) }
            .sorted { $0.createdAt > $1.createdAt }
    }
}
