import SwiftUI

struct MainShellView: View {
    @EnvironmentObject private var appState: AppState
    @State private var selectedSidebarItem: SidebarItem = .generate

    var body: some View {
        NavigationSplitView {
            sidebar
        } content: {
            canvas
        } detail: {
            inspector
        }
        .navigationSplitViewStyle(.balanced)
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

    private var canvas: some View {
        VStack(spacing: 16) {
            Picker("Mode", selection: .constant("auto")) {
                Text("Auto").tag("auto")
                Text("Music").tag("music")
                Text("Sound FX").tag("fx")
                Text("Ambient").tag("ambient")
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            TextEditor(text: .constant(""))
                .font(.body)
                .frame(minHeight: 100)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.quaternary)
                )
                .padding(.horizontal)

            Spacer()

            HStack {
                Button("Write Lyrics") {}
                    .buttonStyle(.bordered)
                Button("Generate") {}
                    .buttonStyle(.borderedProminent)
            }
            .padding()
        }
        .padding(.top)
    }

    private var inspector: some View {
        Form {
            Section("Parameters") {
                LabeledContent("BPM") { TextField("120", text: .constant("")) }
                LabeledContent("Key") { TextField("C Major", text: .constant("")) }
                LabeledContent("Duration") { TextField("30s", text: .constant("")) }
                LabeledContent("Seed") { TextField("Random", text: .constant("")) }
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
