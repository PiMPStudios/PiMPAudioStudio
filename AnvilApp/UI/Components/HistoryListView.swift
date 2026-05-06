import SwiftUI

struct HistoryListView: View {
    @EnvironmentObject private var historyStore: HistoryStore
    @EnvironmentObject private var playerService: AudioPlayerService
    @EnvironmentObject private var trainingService: TrainingService

    @State private var isSelecting = false
    @State private var selectedIDs: Set<UUID> = []
    @State private var showTrainingSheet = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            if historyStore.entries.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .sheet(isPresented: $showTrainingSheet) {
            let selected = historyStore.entries.filter { selectedIDs.contains($0.id) }
            TrainingSheet(selectedEntries: selected)
        }
    }

    private var header: some View {
        HStack {
            Text("HISTORY")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)

            if isSelecting && !selectedIDs.isEmpty {
                Text("\(selectedIDs.count) selected")
                    .font(.caption2)
                    .foregroundStyle(.accent)
            }

            Spacer()

            if !historyStore.entries.isEmpty {
                if isSelecting {
                    if !selectedIDs.isEmpty {
                        Button("Train LoRA") {
                            showTrainingSheet = true
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.mini)
                    }

                    Button("Done") {
                        isSelecting = false
                        selectedIDs.removeAll()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                } else {
                    Button("Select") {
                        isSelecting = true
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)

                    Button("Delete All", role: .destructive) {
                        historyStore.deleteAll()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform")
                .font(.title)
                .foregroundStyle(.quaternary)
            Text("No generations yet")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var list: some View {
        List {
            ForEach(historyStore.entries) { entry in
                HStack(spacing: 8) {
                    if isSelecting {
                        Image(systemName: selectedIDs.contains(entry.id) ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(selectedIDs.contains(entry.id) ? Color.accentColor : .secondary)
                    }
                    HistoryRow(entry: entry)
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    if isSelecting {
                        toggleSelection(entry.id)
                    } else {
                        let url = historyStore.fileURL(for: entry)
                        playerService.load(url: url)
                    }
                }
            }
            .onDelete { indices in
                let toDelete = indices.map { historyStore.entries[$0] }
                for entry in toDelete {
                    historyStore.delete(entry)
                }
            }
        }
        .listStyle(.plain)
    }

    private func toggleSelection(_ id: UUID) {
        if selectedIDs.contains(id) {
            selectedIDs.remove(id)
        } else {
            selectedIDs.insert(id)
        }
    }
}

struct HistoryRow: View {
    let entry: HistoryEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.prompt)
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.tail)

            HStack(spacing: 8) {
                Text("\(Int(entry.duration))s")
                Text("\u{2022}")
                Text(entry.createdAt, style: .date)
                Text("\u{2022}")
                Text("seed: \(entry.seed)")
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}
