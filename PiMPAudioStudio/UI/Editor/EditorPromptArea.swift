import SwiftUI
import PiMPAudioCore

struct EditorPromptArea: View {
    @ObservedObject var state: EditorState

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Original Prompt (context)")
                    .font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $state.originalPrompt)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .background(Color.gray.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.3), lineWidth: 1))
                    .frame(minHeight: 60)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(state.regionPromptLabel)
                    .font(.caption).foregroundStyle(.orange)
                TextEditor(text: $state.regionPrompt)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .background(Color.orange.opacity(0.05))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.orange.opacity(0.4), lineWidth: 1))
                    .frame(minHeight: 60)
            }
        }
        .padding(.horizontal, 16)
    }
}
