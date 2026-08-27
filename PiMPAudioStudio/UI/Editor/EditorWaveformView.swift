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
        VStack(spacing: 0) {
            timeRuler
            waveformCanvas
        }
        .background(Color.black.opacity(0.3))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.3), lineWidth: 1))
    }

    // MARK: - Time ruler

    private var timeRuler: some View {
        Canvas { context, size in
            guard duration > 0 else { return }
            let interval = Self.niceTickInterval(forDuration: duration)
            var t: TimeInterval = 0
            while t <= duration {
                let x = size.width * (t / duration)
                context.stroke(
                    Path { path in
                        path.move(to: CGPoint(x: x, y: size.height - 4))
                        path.addLine(to: CGPoint(x: x, y: size.height))
                    },
                    with: .color(.white.opacity(0.35)),
                    lineWidth: 1
                )
                if t > 0 {
                    context.draw(
                        Text(Self.formatRulerTime(t))
                            .font(.system(size: 8.5))
                            .foregroundStyle(.white.opacity(0.5)),
                        at: CGPoint(x: x, y: size.height / 2 - 2)
                    )
                }
                t += interval
            }
        }
        .frame(height: 14)
        .padding(.top, 2)
    }

    private static func niceTickInterval(forDuration duration: TimeInterval) -> TimeInterval {
        let candidates: [TimeInterval] = [1, 2, 5, 10, 15, 30, 60, 120]
        for candidate in candidates where duration / candidate <= 10 {
            return candidate
        }
        return 300
    }

    private static func formatRulerTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return minutes > 0 ? String(format: "%d:%02d", minutes, seconds) : "\(seconds)s"
    }

    // MARK: - Waveform, selection, playhead

    private var waveformCanvas: some View {
        GeometryReader { geo in
            ZStack {
                Canvas { context, size in
                    let barWidth = size.width / CGFloat(max(peaks.count, 1))
                    let midY = size.height / 2
                    for (index, peak) in peaks.enumerated() {
                        let x = CGFloat(index) * barWidth
                        let barHeight = CGFloat(peak) * size.height * 0.8
                        let rect = CGRect(
                            x: x,
                            y: midY - barHeight / 2,
                            width: max(barWidth - 1, 1),
                            height: max(barHeight, 1)
                        )
                        context.fill(
                            Path(roundedRect: rect, cornerRadius: 1),
                            with: .color(Self.barColor(for: peak))
                        )
                    }
                }

                if isSelectionEnabled, selectionEnd > selectionStart {
                    selectionOverlay(in: geo.size)
                }

                playhead(in: geo.size)
            }
            .contentShape(Rectangle())
            .gesture(dragGesture(in: geo.size))
        }
        .frame(height: 104)
    }

    private func selectionOverlay(in size: CGSize) -> some View {
        let startFrac = selectionStart / max(duration, 0.001)
        let endFrac = selectionEnd / max(duration, 0.001)
        let x = size.width * startFrac
        let width = size.width * (endFrac - startFrac)

        return ZStack {
            Rectangle()
                .fill(Color.orange.opacity(0.15))
                .frame(width: max(width, 0), height: size.height)
                .position(x: x + width / 2, y: size.height / 2)

            Rectangle().fill(Color.orange).frame(width: 2, height: size.height)
                .position(x: x, y: size.height / 2)
            Rectangle().fill(Color.orange).frame(width: 2, height: size.height)
                .position(x: x + width, y: size.height / 2)

            // Duration badge sits right on the selection so the number is
            // visible while dragging, not just in the transport bar below.
            if width > 36 {
                Text(Self.formatSelectionDuration(selectionEnd - selectionStart))
                    .font(.system(size: 9, weight: .semibold))
                    .monospacedDigit()
                    .foregroundStyle(.white)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .background(Color.orange, in: Capsule())
                    .position(x: x + width / 2, y: 10)
            }
        }
    }

    private func playhead(in size: CGSize) -> some View {
        let playX = size.width * playbackProgress
        return ZStack {
            Rectangle()
                .fill(Color.white)
                .frame(width: 2, height: size.height)
                .position(x: playX, y: size.height / 2)
            Circle()
                .fill(Color.white)
                .frame(width: 6, height: 6)
                .position(x: playX, y: 3)
        }
    }

    private func dragGesture(in size: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                guard isSelectionEnabled else {
                    seek(to: value.location.x, width: size.width)
                    return
                }

                let dragDistance = abs(value.location.x - value.startLocation.x)
                guard dragDistance >= 4 else { return }

                let startFrac = max(0, min(1, value.startLocation.x / size.width))
                let currentFrac = max(0, min(1, value.location.x / size.width))
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
                    seek(to: value.location.x, width: size.width)
                }
                isDragging = false
            }
    }

    private func seek(to xPosition: CGFloat, width: CGFloat) {
        guard width > 0 else { return }
        let fraction = max(0, min(1, xPosition / width))
        onSeek(Double(fraction))
    }

    /// Maps amplitude to a dim-teal → bright-cyan gradient so louder
    /// passages read as visually "hotter" instead of one flat color.
    private static func barColor(for peak: Float) -> Color {
        let t = Double(min(max(peak, 0), 1))
        let quiet = (r: 0.20, g: 0.55, b: 0.62)
        let loud = (r: 0.40, g: 0.88, b: 0.95)
        return Color(
            red: quiet.r + (loud.r - quiet.r) * t,
            green: quiet.g + (loud.g - quiet.g) * t,
            blue: quiet.b + (loud.b - quiet.b) * t
        ).opacity(0.6 + 0.4 * t)
    }

    private static func formatSelectionDuration(_ time: TimeInterval) -> String {
        String(format: "%.1fs", time)
    }
}
