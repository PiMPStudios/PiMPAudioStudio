import SwiftUI
import PiMPAudioCore

struct PresetsView: View {
    @EnvironmentObject private var presetStore: PresetStore

    @State private var renamingPreset: GenerationPreset?
    @State private var renameText: String = ""

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Presets")
                    .font(.headline)
                Spacer()
                Text("\(presetStore.presets.count) saved")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding()

            Divider()

            if presetStore.presets.isEmpty {
                emptyState
            } else {
                presetList
            }
        }
        .alert("Rename Preset", isPresented: Binding(
            get: { renamingPreset != nil },
            set: { if !$0 { renamingPreset = nil } }
        )) {
            TextField("New name", text: $renameText)
            Button("Cancel", role: .cancel) { renamingPreset = nil }
            Button("Rename") {
                if let preset = renamingPreset {
                    presetStore.rename(preset, to: renameText)
                }
                renamingPreset = nil
            }
        } message: {
            Text("Enter a new name for \"\(renamingPreset?.name ?? "")\"")
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 8) {
            Spacer()
            Image(systemName: "slider.horizontal.3")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("No Presets Yet")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Save your current generation settings as a preset\nfrom the Generate page.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding()
    }

    // MARK: - Preset List

    private var presetList: some View {
        List {
            ForEach(presetStore.presets) { preset in
                presetRow(preset)
                    .contentShape(Rectangle())
                    .onTapGesture { loadPreset(preset) }
                    .contextMenu {
                        Button("Load") { loadPreset(preset) }
                        Divider()
                        Button("Duplicate") { presetStore.duplicate(preset) }
                        Button("Rename\u{2026}") {
                            renameText = preset.name
                            renamingPreset = preset
                        }
                        Divider()
                        Button("Delete", role: .destructive) { presetStore.delete(preset) }
                    }
            }
            .onDelete { indexSet in
                for idx in indexSet {
                    presetStore.delete(presetStore.presets[idx])
                }
            }
        }
    }

    private func presetRow(_ preset: GenerationPreset) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(preset.name)
                    .font(.body)
                    .fontWeight(.medium)
                    .lineLimit(1)

                Text(preset.prompt)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)

                HStack(spacing: 6) {
                    // Mode badge
                    Text(preset.mode?.displayName ?? "Auto")
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.blue.opacity(0.1))
                        .clipShape(Capsule())

                    // Pipeline badge
                    if preset.pipelineOverride != .auto {
                        Text(preset.pipelineOverride.displayName)
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.secondary.opacity(0.15))
                            .clipShape(Capsule())
                    }

                    // Duration
                    Text("\(Int(preset.duration))s")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)

                    // Adapter
                    if let adapterID = preset.adapterID, !adapterID.isEmpty {
                        HStack(spacing: 2) {
                            Image(systemName: "puzzlepiece.extension")
                                .font(.caption2)
                            Text(adapterID)
                                .font(.caption2)
                        }
                        .foregroundStyle(.orange)
                        .lineLimit(1)
                    }
                }
            }

            Spacer()

            Text(preset.createdAt, style: .date)
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
    }

    // MARK: - Actions

    private func loadPreset(_ preset: GenerationPreset) {
        presetStore.presetToLoad = preset
    }
}
