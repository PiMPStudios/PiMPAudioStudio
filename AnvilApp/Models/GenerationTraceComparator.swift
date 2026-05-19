import Foundation

struct TraceComparisonGroup: Identifiable, Codable, Sendable, Equatable {
    let title: String
    let systemImage: String
    let rows: [TraceComparisonRow]

    var id: String { title }
}

struct TraceComparisonRow: Identifiable, Codable, Sendable, Equatable {
    let label: String
    let left: String
    let right: String

    var id: String { label }

    var isDifferent: Bool {
        left != right
    }
}

enum GenerationTraceComparator {
    static func groups(left: GenerationTrace, right: GenerationTrace) -> [TraceComparisonGroup] {
        [
            TraceComparisonGroup(
                title: "Model",
                systemImage: "cube.box",
                rows: [
                    row("Pipeline", left.pipeline, right.pipeline),
                    row("Model", modelSummary(left.primaryModel), modelSummary(right.primaryModel)),
                    row("Supporting", modelListSummary(left.supportingModels), modelListSummary(right.supportingModels))
                ]
            ),
            TraceComparisonGroup(
                title: "Adapters",
                systemImage: "slider.horizontal.2.square",
                rows: [
                    row("LoRA", adapterSummary(left.adapters), adapterSummary(right.adapters))
                ]
            ),
            TraceComparisonGroup(
                title: "Sampling",
                systemImage: "dial.low",
                rows: [
                    row("Duration", "\(Int(left.request.duration))s", "\(Int(right.request.duration))s"),
                    row("Seed", left.request.seed.map(String.init) ?? "Random", right.request.seed.map(String.init) ?? "Random"),
                    row("Method", left.request.method, right.request.method),
                    row("Guidance", format(left.request.guidanceScale), format(right.request.guidanceScale)),
                    row("Preset", left.request.guidancePresetName, right.request.guidancePresetName),
                    row(
                        "CFG Window",
                        "\(left.request.cfgIntervalStart) -> \(left.request.cfgIntervalEnd)",
                        "\(right.request.cfgIntervalStart) -> \(right.request.cfgIntervalEnd)"
                    ),
                    row("Precision", left.request.precision, right.request.precision),
                    row("Steps", left.request.stepsOverride.map(String.init) ?? "Auto", right.request.stepsOverride.map(String.init) ?? "Auto"),
                    row("Shift", optionalFormat(left.request.shiftOverride), optionalFormat(right.request.shiftOverride)),
                    row("Mode", left.request.resolvedMode, right.request.resolvedMode)
                ]
            ),
            TraceComparisonGroup(
                title: "Prompt & Lyrics",
                systemImage: "text.quote",
                rows: [
                    row("Prompt", compact(left.prompt), compact(right.prompt)),
                    row("Lyrics", lyricsSummary(left.lyrics), lyricsSummary(right.lyrics))
                ]
            ),
            TraceComparisonGroup(
                title: "Audio",
                systemImage: "waveform",
                rows: [
                    row("Abs Mean", format(left.audio.absMean, digits: 4), format(right.audio.absMean, digits: 4)),
                    row("Abs Max", format(left.audio.absMax, digits: 3), format(right.audio.absMax, digits: 3)),
                    row("Duration", String(format: "%.1fs", left.audio.duration), String(format: "%.1fs", right.audio.duration))
                ]
            )
        ]
    }

    private static func row(_ label: String, _ left: String, _ right: String) -> TraceComparisonRow {
        TraceComparisonRow(label: label, left: left, right: right)
    }

    private static func modelSummary(_ model: HistoryModelSnapshot?) -> String {
        guard let model else { return "None" }
        return model.displayName ?? model.id
    }

    private static func modelListSummary(_ models: [HistoryModelSnapshot]) -> String {
        guard !models.isEmpty else { return "None" }
        return models.map { $0.displayName ?? $0.id }.joined(separator: ", ")
    }

    private static func adapterSummary(_ adapters: [HistoryAdapterSnapshot]) -> String {
        guard !adapters.isEmpty else { return "None" }
        return adapters.map { adapter in
            let label = adapter.name ?? adapter.id
            return "\(label) x\(String(format: "%.2f", adapter.weight))"
        }.joined(separator: ", ")
    }

    private static func lyricsSummary(_ lyrics: String) -> String {
        let trimmed = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "None" }
        let lines = trimmed.components(separatedBy: .newlines).filter { !$0.isEmpty }
        return "\(lines.count) lines"
    }

    private static func compact(_ text: String) -> String {
        let oneLine = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if oneLine.count > 180 {
            return String(oneLine.prefix(177)) + "..."
        }
        return oneLine
    }

    private static func optionalFormat(_ value: Float?) -> String {
        guard let value else { return "Auto" }
        return format(value)
    }

    private static func format(_ value: Float, digits: Int = 1) -> String {
        String(format: "%.\(digits)f", value)
    }
}
