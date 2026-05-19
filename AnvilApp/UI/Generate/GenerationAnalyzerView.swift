import SwiftUI

struct GenerationAnalyzerView: View {
    @EnvironmentObject private var traceMonitor: GenerationTraceMonitor

    var body: some View {
        Group {
            if let session = traceMonitor.inspectedSession ?? traceMonitor.currentSession {
                analyzerContent(for: session)
            } else {
                emptyState
            }
        }
        .frame(minWidth: 760, minHeight: 520)
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
}
