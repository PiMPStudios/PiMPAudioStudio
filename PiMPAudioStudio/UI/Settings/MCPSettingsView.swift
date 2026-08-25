import AppKit
import SwiftUI

struct MCPSettingsView: View {
    @EnvironmentObject private var mcpServer: PiMPAudioMCPServer
    @State private var portText: String = ""
    @State private var portError: String?

    var body: some View {
        Form {
            serverSection
            networkSection
            tokenSection
            toolsSection
        }
        .formStyle(.grouped)
        .frame(minWidth: 500, minHeight: 400)
        .onAppear {
            portText = String(mcpServer.port)
            portError = nil
        }
        .onChange(of: mcpServer.port) { _, newValue in
            portText = String(newValue)
        }
    }

    private var serverSection: some View {
        Section("Local MCP Server") {
            Toggle("Enable MCP Server", isOn: Binding(
                get: { mcpServer.isEnabled },
                set: { mcpServer.setEnabled($0) }
            ))

            LabeledContent("State") {
                HStack(spacing: 6) {
                    Circle()
                        .fill(stateColor)
                        .frame(width: 8, height: 8)
                    Text(mcpServer.state.label)
                }
            }

            if case .running = mcpServer.state {
                LabeledContent("Bound") {
                    Text(mcpServer.bindDescription)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }

                endpointRow(
                    title: "Local endpoint",
                    value: mcpServer.endpoint,
                    help: "Copy local endpoint"
                )

                if let networkEndpoint = mcpServer.networkEndpoint {
                    endpointRow(
                        title: "Network endpoint",
                        value: networkEndpoint,
                        help: "Copy LAN endpoint"
                    )
                }
            }

            if case .failed(let message) = mcpServer.state {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            Text("MCP endpoint for local agents (and optional LAN access). Covers generation, editing, datasets, local training, adapters/models, Remote GPU lifecycle, history, diagnostics, and operations. Expensive or destructive tools default to dry-run. Actual mutations need confirm: true or an MCP form prompt. Clients that cannot prompt must send confirm: true.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var networkSection: some View {
        Section("Bind Address & Port") {
            Picker("Bind address", selection: Binding(
                get: { mcpServer.bindAddress },
                set: { mcpServer.setBindAddress($0) }
            )) {
                ForEach(MCPBindAddress.allCases) { address in
                    Text(address.title).tag(address)
                }
            }

            Text(mcpServer.bindAddress.detail)
                .font(.caption)
                .foregroundStyle(.secondary)

            LabeledContent("Port") {
                HStack(spacing: 8) {
                    TextField("38765", text: $portText)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.body, design: .monospaced))
                        .frame(maxWidth: 100)
                        .onSubmit(applyPort)
                    Button("Apply") {
                        applyPort()
                    }
                    .controlSize(.small)
                    .disabled(!canApplyPort)
                }
            }

            if let portError {
                Text(portError)
                    .font(.caption)
                    .foregroundStyle(.red)
            } else {
                Text("Port range \(PiMPAudioMCPServer.minimumPort)–\(PiMPAudioMCPServer.maximumPort). Default is \(PiMPAudioMCPServer.defaultPort). Changing bind address or port restarts the server when it is enabled.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if mcpServer.isNetworkExposed {
                Label(
                    "All-interfaces binding exposes MCP on your LAN. A bearer token is required and cannot be disabled. Avoid untrusted networks.",
                    systemImage: "exclamationmark.triangle.fill"
                )
                .font(.caption)
                .foregroundStyle(.orange)
            }
        }
    }

    private var tokenSection: some View {
        Section(mcpServer.isNetworkExposed ? "Bearer Token (Recommended)" : "Optional Local Token") {
            Toggle("Require bearer token", isOn: Binding(
                get: { mcpServer.isTokenRequired },
                set: { mcpServer.setTokenRequired($0) }
            ))
            .disabled(mcpServer.isNetworkExposed)

            if mcpServer.isTokenRequired {
                LabeledContent("Token") {
                    HStack {
                        Text(mcpServer.token)
                            .font(.system(.caption, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(mcpServer.token, forType: .string)
                        } label: {
                            Image(systemName: "doc.on.doc")
                        }
                        .buttonStyle(.borderless)
                        .help("Copy token")
                        Button("Regenerate") {
                            mcpServer.regenerateToken()
                        }
                        .controlSize(.small)
                    }
                }
            }

            Text(
                mcpServer.isNetworkExposed
                    ? "All-interfaces binding requires a bearer token. LAN clients must send `Authorization: Bearer <token>`. The toggle cannot be turned off until you bind to localhost. The token is stored in the Keychain."
                    : "For localhost-only use this is optional. Turn it on if you want local clients to send `Authorization: Bearer <token>`. The token is stored in the Keychain."
            )
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var canApplyPort: Bool {
        let trimmed = portText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = UInt16(trimmed), PiMPAudioMCPServer.isValidPort(value) else {
            return false
        }
        return value != mcpServer.port
    }

    private func applyPort() {
        let trimmed = portText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = UInt16(trimmed), PiMPAudioMCPServer.isValidPort(value) else {
            portError = "Enter a port between \(PiMPAudioMCPServer.minimumPort) and \(PiMPAudioMCPServer.maximumPort)."
            portText = String(mcpServer.port)
            return
        }
        portError = nil
        _ = mcpServer.setPort(value)
        portText = String(mcpServer.port)
    }

    private func endpointRow(title: String, value: String, help: String) -> some View {
        LabeledContent(title) {
            HStack {
                Text(value)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(value, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless)
                .help(help)
            }
        }
    }

    private var toolsSection: some View {
        Group {
            Section("MCP Tools · Models & Adapters") {
                toolRow("list_models", "Model registry, readiness, and active model state")
                toolRow("download_model", "Download a registry model (dry-run default; operation when submitted)")
                toolRow("set_active_model", "Activate a ready model for its category")
                toolRow("list_adapters", "Installed LoRA/LoKR adapters and checkpoint status")
                toolRow("apply_adapter", "Select, stack, clear, and set adapter gain for generation")
                toolRow("import_adapter", "Import a local native/PEFT/Side-Step adapter folder")
                toolRow("delete_adapter", "Delete an installed adapter")
                toolRow("merge_adapters", "Merge two LoRA adapters into a new adapter")
                toolRow("run_adapter_audition", "Fixed-seed base/gain audition snapshots (operation)")
            }

            Section("MCP Tools · Generation & Intelligence") {
                toolRow("generate", "Direct generation submission")
                toolRow("generate_song", "Enhance/write/apply/generate orchestration")
                toolRow("generation_status", "Live generation status and latest result")
                toolRow("cancel_generation", "Cancel the active generation run")
                toolRow("enhance_prompt", "Prompt enhancement (returns text only)")
                toolRow("write_lyrics", "Duration-aware lyric writing (returns text only)")
            }

            Section("MCP Tools · History & Traces") {
                toolRow("list_history", "Recent generation history rows")
                toolRow("get_trace", "One generation trace sidecar")
                toolRow("compare_traces", "Analyzer-style grouped trace diff")
                toolRow("mark_preferred_take", "Preferred flag and listening note updates")
                toolRow("delete_history_entries", "Delete selected history takes or all history")
            }

            Section("MCP Tools · Editor") {
                toolRow("editor_status", "Editor readiness, loaded audio, and supported actions")
                toolRow("load_editor_audio", "Load history/audio id or path into the editor (operation)")
                toolRow("submit_editor_inpaint", "Confirmed region inpaint/repaint (operation)")
                toolRow("apply_editor_effect_chain", "Confirmed DSP effect chain (operation)")
                toolRow("render_editor_audio", "Export editor buffer/selection to WAV/AIFF (operation)")
            }

            Section("MCP Tools · Datasets & Local Training") {
                toolRow("list_training_datasets", "Visible local dataset packages")
                toolRow("inspect_training_dataset", "Package health, annotations, and recommendations")
                toolRow("build_training_dataset", "Build package from local files (dry-run default; operation)")
                toolRow("enrich_training_dataset", "Caption/lyrics/stem enrichment (dry-run default; operation)")
                toolRow("update_training_dataset_review", "Edit captions, include flags, and approvals")
                toolRow("training_status", "Active local LoRA/LoKR training state and metrics")
                toolRow("list_training_runs", "Installed adapters with training_log summaries")
                toolRow("inspect_training_run", "Run config, progress/event tails, and health")
                toolRow("compare_training_runs", "Compact comparison table for recent runs")
                toolRow("training_preflight", "Resolve dataset/config and memory guards without training")
                toolRow("start_training", "Start local LoRA/LoKR training (dry-run default)")
                toolRow("stop_training", "Stop active local training, optionally saving a checkpoint")
            }

            Section("MCP Tools · Remote GPU") {
                toolRow("runpod_status", "API key state, GPU/templates, pods, and recommended template")
                toolRow("refresh_runpod_launch_options", "Refresh live GPU stock and official templates")
                toolRow("launch_runpod_pod", "Launch Secure Cloud pod (dry-run default; optional region)")
                toolRow("stop_runpod_pod", "Terminate/delete a RunPod pod (not pause)")
                toolRow("create_remote_training_job", "Export reviewed package + remote_job zip (dry-run default)")
                toolRow("upload_remote_training_job", "Upload job zip and portable worker over SSH")
                toolRow("start_remote_training_job", "Start detached remote setup/training on a pod")
                toolRow("remote_training_job_status", "Monitor remote job phase, message, and log tail")
                toolRow("pull_remote_training_job", "Download remote_artifacts and import final adapter")
            }

            Section("MCP Tools · Library, Diagnostics & Operations") {
                toolRow("save_generation_preset", "Save a named generation preset")
                toolRow("delete_generation_preset", "Delete a saved generation preset")
                toolRow("set_verbose_diagnostics", "Enable or disable AUDIODIAG file logging")
                toolRow("tail_diagnostics_log", "Recent diagnostics lines (unfiltered by default)")
                toolRow("tail_training_log", "Recent diagnostics lines filtered to training by default")
                toolRow("list_operations", "Pollable long-running operation records")
                toolRow("get_operation", "One operation by id (status, progress, result/error)")
                toolRow("cancel_operation", "Cancel a running MCP-observable operation")
                toolRow("release_memory", "Confirmed model unload and MLX cache flush")
            }
        }
    }

    private func toolRow(_ name: String, _ description: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(name)
                .font(.system(.caption, design: .monospaced))
            Text(description)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }

    private var stateColor: Color {
        switch mcpServer.state {
        case .stopped:
            return .secondary
        case .starting:
            return .orange
        case .running:
            return .green
        case .failed:
            return .red
        }
    }
}
