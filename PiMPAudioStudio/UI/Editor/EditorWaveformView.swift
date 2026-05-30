import SwiftUI

struct EditorWaveformView: View {
    let peaks: [Float]
    let duration: TimeInterval
    let playbackProgress: Double
    @Binding var selectionStart: TimeInterval
    @Binding var selectionEnd: TimeInterval
    let isSelectionEnabled: Bool
    let onSeek: (Double) -> Void

    @State private var isDragging = false
    @State private var dragOrigin: Double = 0

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Waveform bars
                Canvas { context, size in
                    let barWidth = size.width / CGFloat(max(peaks.count, 1))
                    let midY = size.height / 2
                    for (index, peak) in peaks.enumerated() {
                        let x = CGFloat(index) * barWidth
                        let barHeight = CGFloat(peak) * size.height * 0.8
                        let rect = CGRect(x: x, y: midY - barHeight / 2, width: max(barWidth - 1, 1), height: max(barHeight, 1))
                        context.fill(Path(roundedRect: rect, cornerRadius: 1), with: .color(.blue.opacity(0.7)))
                    }
                }

                // Selection overlay
                if isSelectionEnabled && selectionEnd > selectionStart {
                    let startFrac = selectionStart / max(duration, 0.001)
                    let endFrac = selectionEnd / max(duration, 0.001)
                    let x = geo.size.width * startFrac
                    let width = geo.size.width * (endFrac - startFrac)

                    Rectangle()
                        .fill(Color.orange.opacity(0.15))
                        .frame(width: max(width, 0), height: geo.size.height)
                        .position(x: x + width / 2, y: geo.size.height / 2)
                        .overlay(alignment: .leading) {
                            Rectangle().fill(Color.orange).frame(width: 2, height: geo.size.height)
                                .position(x: x, y: geo.size.height / 2)
                        }
                        .overlay(alignment: .trailing) {
                            Rectangle().fill(Color.orange).frame(width: 2, height: geo.size.height)
                                .position(x: x + width, y: geo.size.height / 2)
                        }
                }

                // Playhead
                let playX = geo.size.width * playbackProgress
                Rectangle()
                    .fill(Color.white)
                    .frame(width: 2, height: geo.size.height)
                    .position(x: playX, y: geo.size.height / 2)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        guard isSelectionEnabled else {
                            seek(to: value.location.x, width: geo.size.width)
                            return
                        }

                        let dragDistance = abs(value.location.x - value.startLocation.x)
                        guard dragDistance >= 4 else { return }

                        let startFrac = max(0, min(1, value.startLocation.x / geo.size.width))
                        let currentFrac = max(0, min(1, value.location.x / geo.size.width))
                        if !isDragging {
                            isDragging = true
                            dragOrigin = startFrac * duration
                        }
                        let currentTime = currentFrac * duration
                        selectionStart = min(dragOrigin, currentTime)
                        selectionEnd = max(dragOrigin, currentTime)
                    }
                    .onEnded { value in
                        if !isDragging {
                            seek(to: value.location.x, width: geo.size.width)
                        }
                        isDragging = false
                    }
            )
        }
        .frame(height: 120)
        .background(Color.black.opacity(0.3))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.3), lineWidth: 1))
    }

    private func seek(to xPosition: CGFloat, width: CGFloat) {
        guard width > 0 else { return }
        let fraction = max(0, min(1, xPosition / width))
        onSeek(Double(fraction))
    }
}
