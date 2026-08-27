import PiMPAudioCore
import SwiftUI

struct DownloadProgressRow: View {
    let spec: ModelSpec
    let progress: Double?

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(spec.id)
                    .font(.body.monospaced())
                Text(spec.totalSizeFormatted)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if let progress {
                if progress >= 1.0 {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else {
                    ProgressView(value: progress)
                        .frame(width: 120)
                }
            } else {
                Text("Waiting…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}
