import SwiftUI
import AnvilCore

struct AdaptersSettingsView: View {
    enum Tab: String, CaseIterable {
        case myAdapters = "My Adapters"
        case community = "Community"
    }

    @State private var selectedTab: Tab = .myAdapters
    @StateObject private var communityService = CommunityService()

    var body: some View {
        VStack(spacing: 0) {
            // Segmented control
            Picker("", selection: $selectedTab) {
                ForEach(Tab.allCases, id: \.self) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.top, 12)
            .padding(.bottom, 8)

            Divider()

            // Tab content
            switch selectedTab {
            case .myAdapters:
                MyAdaptersListView()
            case .community:
                CommunityBrowserView(service: communityService)
            }
        }
    }
}

// MARK: - My Adapters (extracted from original)

private struct MyAdaptersListView: View {
    @State private var adapters: [LoRAAdapter] = []
    @State private var showingMergeSheet = false
    @State private var renamingAdapter: LoRAAdapter?
    @State private var renameText: String = ""
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Toolbar
            HStack {
                Text("LoRA Adapters")
                    .font(.headline)
                Spacer()
                Button("Merge Two") {
                    showingMergeSheet = true
                }
                .disabled(adapters.count < 2)
            }
            .padding()

            Divider()

            if adapters.isEmpty {
                emptyState
            } else {
                adapterList
            }

            // Error banner
            if let error = errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle")
                    Text(error)
                    Spacer()
                    Button("Dismiss") { errorMessage = nil }
                        .buttonStyle(.borderless)
                }
                .font(.caption)
                .foregroundStyle(.red)
                .padding(8)
                .background(.red.opacity(0.05))
            }
        }
        .onAppear { refreshAdapters() }
        .sheet(isPresented: $showingMergeSheet) {
            MergeSheet(onComplete: { refreshAdapters() })
                .frame(minWidth: 400, minHeight: 300)
        }
        .alert("Rename Adapter", isPresented: Binding(
            get: { renamingAdapter != nil },
            set: { if !$0 { renamingAdapter = nil } }
        )) {
            TextField("New name", text: $renameText)
            Button("Cancel", role: .cancel) { renamingAdapter = nil }
            Button("Rename") { performRename() }
        } message: {
            Text("Enter a new name for \"\(renamingAdapter?.name ?? "")\"")
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Spacer()
            Image(systemName: "puzzlepiece.extension")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("No Adapters Yet")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Train one from History or download from the Community tab.")
                .font(.caption)
                .foregroundStyle(.tertiary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding()
    }

    private var adapterList: some View {
        List {
            ForEach(adapters) { adapter in
                adapterRow(adapter)
                    .contextMenu {
                        Button("Rename...") {
                            renameText = adapter.name
                            renamingAdapter = adapter
                        }
                        Button("Export...") { beginExport(adapter) }
                        Divider()
                        Button("Delete", role: .destructive) { deleteAdapter(adapter) }
                    }
            }
            .onDelete { indexSet in
                for idx in indexSet {
                    deleteAdapter(adapters[idx])
                }
            }
        }
    }

    private func adapterRow(_ adapter: LoRAAdapter) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(adapter.name)
                        .font(.body)
                        .fontWeight(.medium)
                    // Source badge
                    if adapter.source != .local {
                        Text(adapter.source.rawValue.capitalized)
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(.blue.opacity(0.1))
                            .clipShape(Capsule())
                    }
                }
                HStack(spacing: 8) {
                    Text(adapter.baseModelCategory.rawValue)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.blue.opacity(0.1))
                        .clipShape(Capsule())
                    Text("Rank \(adapter.config.rank)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if let loss = adapter.finalLoss {
                        Text("Loss: \(String(format: "%.4f", loss))")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            Text(adapter.createdAt, style: .date)
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
    }

    // MARK: - Actions

    private func refreshAdapters() {
        adapters = AdapterRegistry.listAdapters()
    }

    private func performRename() {
        guard let adapter = renamingAdapter else { return }
        do {
            try AdapterRegistry.rename(adapterID: adapter.id, to: renameText)
            refreshAdapters()
        } catch {
            errorMessage = error.localizedDescription
        }
        renamingAdapter = nil
    }

    private func deleteAdapter(_ adapter: LoRAAdapter) {
        do {
            try AdapterRegistry.delete(adapterID: adapter.id)
            refreshAdapters()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func beginExport(_ adapter: LoRAAdapter) {
        let panel = NSSavePanel()
        panel.title = "Export Adapter"
        panel.nameFieldStringValue = adapter.name
        panel.canCreateDirectories = true

        if panel.runModal() == .OK, let url = panel.url {
            do {
                try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
                try AdapterRegistry.export(adapterID: adapter.id, to: url)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
