import SwiftUI
import Charts
import PiMPAudioCore

/// A mini line chart displaying training loss history.
struct LossCurveView: View {
    let lossHistory: [Float]
    let validationHistory: [TrainingValidationPoint]

    init(lossHistory: [Float], validationHistory: [TrainingValidationPoint] = []) {
        self.lossHistory = lossHistory
        self.validationHistory = validationHistory
    }

    var body: some View {
        if lossHistory.isEmpty && validationHistory.isEmpty {
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
                ForEach(Array(validationHistory.enumerated()), id: \.offset) { _, point in
                    LineMark(
                        x: .value("Step", point.step),
                        y: .value("Validation", Double(point.loss))
                    )
                    .foregroundStyle(.orange)
                    .lineStyle(.init(lineWidth: 1.5, dash: [4, 3]))

                    PointMark(
                        x: .value("Step", point.step),
                        y: .value("Validation", Double(point.loss))
                    )
                    .foregroundStyle(.orange)
                }
            }
            .chartYScale(domain: .automatic(includesZero: false))
            .chartXAxisLabel("Step")
            .chartYAxisLabel("Loss")
            .frame(height: 120)
        }
    }
}
