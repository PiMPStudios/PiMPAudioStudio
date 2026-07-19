import AppKit
import SwiftUI

struct MCPSettingsView: View {
    @EnvironmentObject private var mcpServer: PiMPAudioMCPServer

    var body: some View {
        Form {
            serverSection
            tokenSection
            toolsSection
        }
        .formStyle(.grouped)
        .frame(minWidth: 500, minHeight: 400)
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
                LabeledContent("Endpoint") {
                    HStack {
                        Text(mcpServer.endpoint)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(mcpServer.endpoint, forType: .string)
                        } label: {
                            Image(systemName: "doc.on.doc")
                        }
                        .buttonStyle(.borderless)
                        .help("Copy endpoint")
                    }
                }
            }

            if case .failed(let message) = mcpServer.state {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            Text("Localhost-only MCP endpoint for local agents. Covers generation, editing, datasets, local training, adapters/models, Remote GPU lifecycle, history, diagnostics, and operations. Expensive or destructive tools use dry-run defaults and confirm/elicitation gates.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var tokenSection: some View {
        Section("Optional Local Token") {
            Toggle("Require bearer token", isOn: Binding(
                get: { mcpServer.isTokenRequired },
                set: { mcpServer.setTokenRequired($0) }
            ))

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

            Text("For localhost-only use this is optional. Turn it on if you want local clients to send `Authorization: Bearer <token>`.")
                .font(.caption)
                .foregroundStyle(.secondary)
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
