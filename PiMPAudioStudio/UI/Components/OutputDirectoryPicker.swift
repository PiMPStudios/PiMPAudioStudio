import SwiftUI

struct OutputDirectoryPicker: View {
    @EnvironmentObject private var outputManager: OutputDirectoryManager

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)

            Text("Choose Output Folder")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Select where PiMP Audio Studio saves your generated audio files.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)

            Button("Select Folder") {
                Task {
                    await outputManager.pickDirectory()
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(40)
    }
}
