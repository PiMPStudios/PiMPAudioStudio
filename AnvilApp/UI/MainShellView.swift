import SwiftUI

struct MainShellView: View {
    @EnvironmentObject private var outputManager: OutputDirectoryManager
    @State private var selectedSidebarItem: SidebarItem = .generate

    var body: some View {
        if !outputManager.hasValidDirectory() {
            OutputDirectoryPicker()
        } else {
            NavigationSplitView {
                sidebar
            } content: {
                canvas
            } detail: {
                inspector
            }
            .navigationSplitViewStyle(.balanced)
        }
    }

    private var sidebar: some View {
        List(selection: $selectedSidebarItem) {
            Section("Create") {
                Label("Generate", systemImage: "waveform")
                    .tag(SidebarItem.generate)
            }
            Section("Library") {
                Label("History", systemImage: "clock")
                    .tag(SidebarItem.history)
                Label("Presets", systemImage: "slider.horizontal.3")
                    .tag(SidebarItem.presets)
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("Anvil")
    }

    @ViewBuilder
    private var canvas: some View {
        switch selectedSidebarItem {
        case .generate:
            GenerateView()
        case .history:
            HistoryListView()
        case .presets:
            Text("Presets coming in Phase 4b")
                .foregroundStyle(.secondary)
        }
    }

    private var inspector: some View {
        Form {
            Section("Parameters") {
                LabeledContent("Duration") { TextField("30s", text: .constant("")) }
                LabeledContent("Seed") { TextField("Random", text: .constant("")) }
                LabeledContent("Guidance") { TextField("1.0", text: .constant("")) }
            }
            Section("Model") {
                Text("ACE-Step DiT v1.5")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 250)
    }
}

enum SidebarItem: Hashable {
    case generate
    case history
    case presets
}
