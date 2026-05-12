import SwiftUI
import AnvilCore

struct EditorToolbar: View {
    @ObservedObject var state: EditorState
    let onOpen: () -> Void
    let onExport: (ExportFormat) -> Void
    let onUndo: () -> Void
    let onRedo: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Button(action: onOpen) {
                Label("Open", systemImage: "folder")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            if state.isLoaded {
                Menu {
                    ForEach(ExportFormat.allCases, id: \.self) { fmt in
                        Button(fmt.menuLabel) { onExport(fmt) }
                    }
                } label: {
                    Label("Export", systemImage: "square.and.arrow.down")
                }
                .menuStyle(.borderlessButton)
                .buttonStyle(.bordered)
                .controlSize(.small)
                .fixedSize()
            }

            if state.isLoaded {
                Divider().frame(height: 20)
                Text(state.fileName).font(.headline).lineLimit(1)
                Text("— \(formattedDuration) · \(state.sampleRate / 1000)kHz")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Spacer()

            if state.isLoaded {
                Picker("Mode", selection: $state.editMode) {
                    Text("Inpaint").tag(EditMode.inpaint)
                    Text("Repaint").tag(EditMode.repaint)
                }
                .pickerStyle(.segmented)
                .frame(width: 160)

                Divider().frame(height: 20)

                Picker("Pipeline", selection: $state.pipeline) {
                    ForEach(PipelineChoice.allCases, id: \.self) { choice in
                        Text(choice.displayName).tag(choice)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 140)

                Divider().frame(height: 20)

                HStack(spacing: 4) {
                    Button(action: onUndo) {
                        Image(systemName: "arrow.uturn.backward")
                    }
                    .disabled(!state.canUndo)
                    .keyboardShortcut("z", modifiers: .command)

                    Button(action: onRedo) {
                        Image(systemName: "arrow.uturn.forward")
                    }
                    .disabled(!state.canRedo)
                    .keyboardShortcut("z", modifiers: [.command, .shift])
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }

    private var formattedDuration: String {
        let minutes = Int(state.duration) / 60
        let seconds = Int(state.duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}
