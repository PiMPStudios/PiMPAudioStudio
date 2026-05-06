import SwiftUI

struct WaveformView: View {
    let peaks: [Float]
    let progress: Double
    let onSeek: (Double) -> Void

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                let barWidth = size.width / CGFloat(peaks.count)
                let midY = size.height / 2

                for (index, peak) in peaks.enumerated() {
                    let x = CGFloat(index) * barWidth
                    let barHeight = CGFloat(peak) * size.height * 0.8
                    let progressFraction = Double(index) / Double(peaks.count)
                    let color: Color = progressFraction <= progress ? .accentColor : .secondary.opacity(0.4)

                    let rect = CGRect(
                        x: x,
                        y: midY - barHeight / 2,
                        width: max(barWidth - 1, 1),
                        height: max(barHeight, 1)
                    )
                    context.fill(Path(roundedRect: rect, cornerRadius: 1), with: .color(color))
                }

                let playheadX = size.width * progress
                let playheadRect = CGRect(x: playheadX - 1, y: 0, width: 2, height: size.height)
                context.fill(Path(playheadRect), with: .color(.white))
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let fraction = max(0, min(1, value.location.x / geo.size.width))
                        onSeek(fraction)
                    }
            )
        }
    }
}
