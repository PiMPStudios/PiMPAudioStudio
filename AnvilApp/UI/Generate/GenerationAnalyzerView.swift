import SwiftUI

struct GenerationAnalyzerView: View {
    @EnvironmentObject private var traceMonitor: GenerationTraceMonitor
    @EnvironmentObject private var historyStore: HistoryStore
    @EnvironmentObject private var playerService: AudioPlayerService

    var body: some View {
        Group {
            if let comparison = traceMonitor.comparison {
                comparisonContent(comparison)
            } else if let session = traceMonitor.inspectedSession ?? traceMonitor.currentSession {
                analyzerContent(for: session)
            } else {
                emptyState
            }
        }
        .frame(minWidth: 760, minHeight: 520)
    }

    private func comparisonContent(_ comparison: GenerationTraceComparison) -> some View {
        let rows = comparisonRows(comparison)
        let changedCount = rows.filter(\.isDifferent).count

        return ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Trace Compare")
                            .font(.title2.weight(.semibold))
                        Text("\(changedCount) changed fields")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    if traceMonitor.currentSession != nil {
                        Button("Show Live") {
                            traceMonitor.clearComparison()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }

                compareSummary(comparison)
                changedFieldsPanel(rows)
                promptDiffPanel(comparison)
                audioComparePanel(comparison)
            }
            .padding(18)
        }
    }

    private func analyzerContent(for session: LiveGenerationTraceSession) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header(for: session)
                liveProgress(for: session)

                HStack(alignment: .top, spacing: 16) {
                    requestPanel(for: session)
                    modelPanel(for: session)
                }

                if let audio = session.audio {
                    audioPanel(audio)
                }

                eventTimeline(session.events)
            }
            .padding(18)
        }
        .navigationTitle("Generation Analyzer")
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 44))
                .foregroundStyle(.secondary)
            Text("No generation is being traced")
                .font(.title3.weight(.semibold))
            Text("Open this window before clicking Generate to watch the live backend flow.")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private func compareSummary(_ comparison: GenerationTraceComparison) -> some View {
        HStack(alignment: .top, spacing: 16) {
            traceSummaryColumn(title: "A", trace: comparison.left)
            traceSummaryColumn(title: "B", trace: comparison.right)
        }
    }

    private func traceSummaryColumn(title: String, trace: GenerationTrace) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                panelTitle("Run \(title)", systemImage: "waveform.path.ecg")
                Spacer()
                traceActionButtons(title: title, trace: trace)
            }
            metricRow("Created", trace.createdAt.formatted(date: .abbreviated, time: .standard))
            metricRow("Pipeline", trace.pipeline)
            metricRow("Model", trace.primaryModel?.displayName ?? trace.primaryModel?.id ?? "Unknown")
            metricRow("LoRA", adapterSummary(trace.adapters))
            metricRow("Seed", trace.request.seed.map(String.init) ?? "Random")
            metricRow("Trace", trace.traceFileName)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private func changedFieldsPanel(_ rows: [ComparisonRow]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            panelTitle("Changed Settings", systemImage: "list.bullet.rectangle")
            VStack(spacing: 0) {
                ForEach(rows) { row in
                    HStack(alignment: .top, spacing: 12) {
                        Text(row.label)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 110, alignment: .leading)
                        Text(row.left)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Text(row.right)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.callout)
                    .lineLimit(3)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(row.isDifferent ? Color.accentColor.opacity(0.12) : Color.clear)

                    if row.id != rows.last?.id {
                        Divider()
                    }
                }
            }
            .background(.black.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        }
        .padding(12)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private func promptDiffPanel(_ comparison: GenerationTraceComparison) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            panelTitle("Prompt & Lyrics", systemImage: "text.quote")
            HStack(alignment: .top, spacing: 16) {
                textCompareColumn(title: "A Prompt", text: comparison.left.prompt)
                textCompareColumn(title: "B Prompt", text: comparison.right.prompt)
            }
            HStack(alignment: .top, spacing: 16) {
                textCompareColumn(title: "A Lyrics", text: emptyLabel(comparison.left.lyrics))
                textCompareColumn(title: "B Lyrics", text: emptyLabel(comparison.right.lyrics))
            }
        }
        .padding(12)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private func audioComparePanel(_ comparison: GenerationTraceComparison) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            panelTitle("Audio", systemImage: "waveform")
            HStack(alignment: .top, spacing: 16) {
                audioSummaryColumn(title: "A", trace: comparison.left)
                audioSummaryColumn(title: "B", trace: comparison.right)
            }
        }
        .padding(12)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private func textCompareColumn(title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(text)
                .font(.caption)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private func audioSummaryColumn(title: String, trace: GenerationTrace) -> some View {
        let audio = trace.audio
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Run \(title)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                traceActionButtons(title: title, trace: trace)
            }
            metricRow("Duration", String(format: "%.1fs", audio.duration))
            metricRow("Abs Mean", String(format: "%.4f", audio.absMean))
            metricRow("Abs Max", String(format: "%.3f", audio.absMax))
            metricRow("Shape", audio.sampleShape.map(String.init).joined(separator: " x "))
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func traceActionButtons(title: String, trace: GenerationTrace) -> some View {
        HStack(spacing: 6) {
            tracePlaybackButton(title: title, trace: trace)
            Button {
                openTraceInEditor(trace)
            } label: {
                Label("Edit \(title)", systemImage: "waveform.badge.minus")
            }
            .buttonStyle(.bordered)
            .controlSize(.mini)
            .help("Edit Run \(title)")
        }
    }

    private func tracePlaybackButton(title: String, trace: GenerationTrace) -> some View {
        let isSelected = isTraceLoaded(trace)
        let isPlaying = isSelected && playerService.isPlaying

        return Button {
            toggleTracePlayback(trace)
        } label: {
            Label(
                isPlaying ? "Pause \(title)" : "Play \(title)",
                systemImage: isPlaying ? "pause.fill" : "play.fill"
            )
        }
        .buttonStyle(.bordered)
        .controlSize(.mini)
        .help(isPlaying ? "Pause Run \(title)" : "Play Run \(title)")
    }

    private func header(for session: LiveGenerationTraceSession) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Circle()
                .fill(statusColor(session.status))
                .frame(width: 10, height: 10)
            VStack(alignment: .leading, spacing: 4) {
                Text(session.status.displayName)
                    .font(.title2.weight(.semibold))
                Text("\(session.source.displayName) · \(session.pipeline)")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if session.source == .saved, traceMonitor.currentSession != nil {
                Button("Show Live") {
                    traceMonitor.clearInspection()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            VStack(alignment: .trailing, spacing: 4) {
                Text(session.startedAt.formatted(date: .omitted, time: .standard))
                    .font(.callout.monospacedDigit())
                if let completedAt = session.completedAt {
                    Text("Finished \(completedAt.formatted(date: .omitted, time: .standard))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Updated \(session.updatedAt.formatted(date: .omitted, time: .standard))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func liveProgress(for session: LiveGenerationTraceSession) -> some View {
        if let progress = session.progress {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(progress.label)
                        .font(.headline)
                    Spacer()
                    if let step = progress.step, let totalSteps = progress.totalSteps {
                        Text("\(step) / \(totalSteps)")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    if let eta = progress.eta, eta > 0 {
                        Text("~\(Int(eta))s")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }

                if let fraction = progress.fraction {
                    ProgressView(value: fraction)
                        .progressViewStyle(.linear)
                } else {
                    ProgressView()
                        .progressViewStyle(.linear)
                }
            }
            .padding(12)
            .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func requestPanel(for session: LiveGenerationTraceSession) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            panelTitle("Request", systemImage: "slider.horizontal.3")
            metricRow("Duration", "\(Int(session.request.duration))s")
            metricRow("Seed", session.request.seed.map(String.init) ?? "Random")
            metricRow("Guidance", String(format: "%.1f", session.request.guidanceScale))
            metricRow("Preset", session.request.guidancePresetName)
            metricRow("CFG Window", "\(session.request.cfgIntervalStart) → \(session.request.cfgIntervalEnd)")
            metricRow("Precision", session.request.precision)
            metricRow("Steps", session.request.stepsOverride.map(String.init) ?? "Auto")

            Divider()

            Text("Prompt")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(session.prompt)
                .font(.callout)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

            if !session.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text("Lyrics")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
                Text(session.lyrics)
                    .font(.caption)
                    .textSelection(.enabled)
                    .lineLimit(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private func modelPanel(for session: LiveGenerationTraceSession) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            panelTitle("Models", systemImage: "cube.box")
            if let primaryModel = session.primaryModel {
                metricRow("Primary", primaryModel.displayName ?? primaryModel.id)
                metricRow("Category", primaryModel.category)
            } else {
                metricRow("Primary", "Resolving")
            }

            if !session.supportingModels.isEmpty {
                Divider()
                Text("Supporting")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ForEach(session.supportingModels, id: \.id) { model in
                    Text(model.displayName ?? model.id)
                        .font(.callout)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }

            Divider()

            Text("LoRA")
                .font(.caption)
                .foregroundStyle(.secondary)
            if session.adapters.isEmpty {
                Text("None")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(session.adapters, id: \.id) { adapter in
                    HStack {
                        Text(adapter.name ?? adapter.id)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Text("x\(String(format: "%.2f", adapter.weight))")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    .font(.callout)
                }
            }

            if let traceFileName = session.traceFileName {
                Divider()
                metricRow("Trace", traceFileName)
            }
        }
        .padding(12)
        .frame(width: 280, alignment: .topLeading)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private func audioPanel(_ audio: GenerationTraceAudioSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            panelTitle("Audio", systemImage: "waveform")
            HStack(spacing: 24) {
                metricColumn("Sample Rate", "\(audio.sampleRate) Hz")
                metricColumn("Duration", String(format: "%.1fs", audio.duration))
                metricColumn("Abs Mean", String(format: "%.4f", audio.absMean))
                metricColumn("Abs Max", String(format: "%.3f", audio.absMax))
                metricColumn("Shape", audio.sampleShape.map(String.init).joined(separator: " x "))
            }
        }
        .padding(12)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private func eventTimeline(_ events: [LiveGenerationTraceEvent]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            panelTitle("Events", systemImage: "list.bullet.rectangle")

            if events.isEmpty {
                Text("Waiting for events")
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(events.reversed()) { event in
                        eventRow(event)
                        if event.id != events.first?.id {
                            Divider()
                        }
                    }
                }
            }
        }
        .padding(12)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private func eventRow(_ event: LiveGenerationTraceEvent) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: eventIcon(event.level))
                .foregroundStyle(eventColor(event.level))
                .frame(width: 18)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(event.stage)
                        .font(.caption.weight(.semibold))
                    Text(event.timestamp.formatted(date: .omitted, time: .standard))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                    if let progress = event.progress {
                        Text("\(Int(progress * 100))%")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                Text(event.message)
                    .font(.callout)
                if let detail = event.detail, !detail.isEmpty {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            Spacer()
        }
        .padding(.vertical, 8)
    }

    private func panelTitle(_ text: String, systemImage: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
                .foregroundStyle(.secondary)
            Text(text)
                .font(.headline)
        }
    }

    private func metricRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 86, alignment: .leading)
            Text(value)
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 0)
        }
    }

    private func metricColumn(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.callout.monospacedDigit())
        }
    }

    private func statusColor(_ status: LiveGenerationTraceStatus) -> Color {
        switch status {
        case .running: return .blue
        case .complete: return .green
        case .failed: return .red
        case .cancelled: return .orange
        }
    }

    private func eventColor(_ level: LiveTraceEventLevel) -> Color {
        switch level {
        case .info: return .secondary
        case .success: return .green
        case .warning: return .orange
        case .error: return .red
        }
    }

    private func eventIcon(_ level: LiveTraceEventLevel) -> String {
        switch level {
        case .info: return "circle"
        case .success: return "checkmark.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .error: return "xmark.octagon.fill"
        }
    }

    private func comparisonRows(_ comparison: GenerationTraceComparison) -> [ComparisonRow] {
        let left = comparison.left
        let right = comparison.right

        return [
            row("Pipeline", left.pipeline, right.pipeline),
            row("Model", modelSummary(left.primaryModel), modelSummary(right.primaryModel)),
            row("Supporting", modelListSummary(left.supportingModels), modelListSummary(right.supportingModels)),
            row("LoRA", adapterSummary(left.adapters), adapterSummary(right.adapters)),
            row("Duration", "\(Int(left.request.duration))s", "\(Int(right.request.duration))s"),
            row("Seed", left.request.seed.map(String.init) ?? "Random", right.request.seed.map(String.init) ?? "Random"),
            row("Method", left.request.method, right.request.method),
            row("Guidance", format(left.request.guidanceScale), format(right.request.guidanceScale)),
            row("Preset", left.request.guidancePresetName, right.request.guidancePresetName),
            row(
                "CFG Window",
                "\(left.request.cfgIntervalStart) → \(left.request.cfgIntervalEnd)",
                "\(right.request.cfgIntervalStart) → \(right.request.cfgIntervalEnd)"
            ),
            row("Precision", left.request.precision, right.request.precision),
            row("Steps", left.request.stepsOverride.map(String.init) ?? "Auto", right.request.stepsOverride.map(String.init) ?? "Auto"),
            row("Shift", optionalFormat(left.request.shiftOverride), optionalFormat(right.request.shiftOverride)),
            row("Mode", left.request.resolvedMode, right.request.resolvedMode),
            row("Prompt", compact(left.prompt), compact(right.prompt)),
            row("Lyrics", lyricsSummary(left.lyrics), lyricsSummary(right.lyrics)),
            row("Abs Mean", format(left.audio.absMean, digits: 4), format(right.audio.absMean, digits: 4)),
            row("Abs Max", format(left.audio.absMax, digits: 3), format(right.audio.absMax, digits: 3))
        ]
    }

    private func row(_ label: String, _ left: String, _ right: String) -> ComparisonRow {
        ComparisonRow(label: label, left: left, right: right)
    }

    private func modelSummary(_ model: HistoryModelSnapshot?) -> String {
        guard let model else { return "None" }
        return model.displayName ?? model.id
    }

    private func modelListSummary(_ models: [HistoryModelSnapshot]) -> String {
        guard !models.isEmpty else { return "None" }
        return models.map { $0.displayName ?? $0.id }.joined(separator: ", ")
    }

    private func adapterSummary(_ adapters: [HistoryAdapterSnapshot]) -> String {
        guard !adapters.isEmpty else { return "None" }
        return adapters.map { adapter in
            let label = adapter.name ?? adapter.id
            return "\(label) x\(String(format: "%.2f", adapter.weight))"
        }.joined(separator: ", ")
    }

    private func lyricsSummary(_ lyrics: String) -> String {
        let trimmed = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "None" }
        let lines = trimmed.components(separatedBy: .newlines).filter { !$0.isEmpty }
        return "\(lines.count) lines"
    }

    private func emptyLabel(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "None" : trimmed
    }

    private func compact(_ text: String) -> String {
        let oneLine = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if oneLine.count > 180 {
            return String(oneLine.prefix(177)) + "..."
        }
        return oneLine
    }

    private func isTraceLoaded(_ trace: GenerationTrace) -> Bool {
        playerService.loadedURL == historyStore.fileURL(forAudioFileName: trace.audioFileName)
    }

    private func toggleTracePlayback(_ trace: GenerationTrace) {
        let url = historyStore.fileURL(forAudioFileName: trace.audioFileName)

        if playerService.loadedURL == url {
            if playerService.isPlaying {
                playerService.pause()
            } else {
                if playerService.duration > 0,
                   playerService.currentTime >= playerService.duration - 0.1 {
                    playerService.seek(to: 0)
                }
                playerService.play()
            }
            return
        }

        playerService.load(url: url, precomputedPeaks: trace.audio.waveformPeaks)
        playerService.play()
    }

    private func openTraceInEditor(_ trace: GenerationTrace) {
        playerService.pause()
        NotificationCenter.default.post(
            name: .navigateToEditor,
            object: nil,
            userInfo: [
                "url": historyStore.fileURL(forAudioFileName: trace.audioFileName),
                "prompt": trace.originalPrompt ?? trace.prompt
            ]
        )
    }

    private func format(_ value: Float, digits: Int = 1) -> String {
        String(format: "%.\(digits)f", value)
    }

    private func optionalFormat(_ value: Float?) -> String {
        value.map { String(format: "%.1f", $0) } ?? "Auto"
    }
}

private struct ComparisonRow: Identifiable, Equatable {
    let id = UUID()
    let label: String
    let left: String
    let right: String

    var isDifferent: Bool {
        left != right
    }
}
