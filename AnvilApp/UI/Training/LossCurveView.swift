import SwiftUI
import Charts

/// A mini line chart displaying training loss history.
struct LossCurveView: View {
    let lossHistory: [Float]

    var body: some View {
        if lossHistory.isEmpty {
            Text("No data yet")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(height: 120)
        } else {
            Chart {
                ForEach(Array(lossHistory.enumerated()), id: \.offset) { step, loss in
                    LineMark(
                        x: .value("Step", step),
                        y: .value("Loss", Double(loss))
                    )
                    .foregroundStyle(.blue)
                }
            }
            .chartYScale(domain: .automatic(includesZero: false))
            .chartXAxisLabel("Step")
            .chartYAxisLabel("Loss")
            .frame(height: 120)
        }
    }
}
