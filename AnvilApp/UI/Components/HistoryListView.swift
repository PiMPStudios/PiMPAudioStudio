import SwiftUI

struct HistoryListView: View {
    @EnvironmentObject private var historyStore: HistoryStore
    @EnvironmentObject private var playerService: AudioPlayerService

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
    }

    private var header: some View {
        HStack {
            Text("HISTORY")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
            Spacer()
            if !historyStore.entries.isEmpty {
                Button("Delete All", role: .destructive) {
                    historyStore.deleteAll()
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
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
                HistoryRow(entry: entry)
                    .onTapGesture {
                        let url = historyStore.fileURL(for: entry)
                        playerService.load(url: url)
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
