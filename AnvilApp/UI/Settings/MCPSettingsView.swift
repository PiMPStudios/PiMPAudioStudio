import AppKit
import SwiftUI

struct MCPSettingsView: View {
    @EnvironmentObject private var mcpServer: AnvilMCPServer

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

            Text("The v1 server is localhost-only and exposes generation, prompt, lyric, adapter, history, and trace tools.")
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
        Section("MCP Tools") {
            toolRow("list_models", "Model registry, readiness, and active model state")
            toolRow("list_adapters", "Installed LoRA adapters and checkpoint status")
            toolRow("list_history", "Recent generation history rows")
            toolRow("get_trace", "One generation trace sidecar")
            toolRow("compare_traces", "Analyzer-style grouped trace diff")
            toolRow("mark_preferred_take", "Preferred flag and listening note updates")
            toolRow("enhance_prompt", "Prompt enhancement")
            toolRow("write_lyrics", "Duration-aware lyric writing")
            toolRow("apply_adapter", "LoRA selection and strength")
            toolRow("generate_song", "Enhance/write/apply/generate orchestration")
            toolRow("generate", "Direct generation submission")
            toolRow("generation_status", "Live generation status")
            toolRow("release_memory", "Confirmed model unload and cache flush")
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
