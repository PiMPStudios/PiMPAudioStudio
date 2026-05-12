import Foundation

/// Records the training data sources for a LoRA adapter so the session can be
/// reloaded later for retraining with the same (or expanded) dataset.
///
/// Saved as `training_manifest.json` alongside `adapter.json` in the adapter
/// directory. Absence of the file is gracefully handled — adapters created
/// before this feature was added simply have no manifest.
struct TrainingManifest: Codable {
    /// IDs of `HistoryEntry` items used as training data.
    let historyEntryIDs: [UUID]

    /// Imported audio files (not from history) with their training captions.
    let importedFiles: [ImportedFileRef]

    struct ImportedFileRef: Codable {
        /// Absolute file URL at the time of training.
        let url: URL
        /// Caption / style description used for this clip.
        let description: String
    }
}
