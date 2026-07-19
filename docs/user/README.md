# PiMP Audio Studio User Documentation

These docs explain how to use and troubleshoot PiMP Audio Studio from the point
of view of someone running the app, not someone changing the code.

The source of truth is this `docs/user/` folder. For public distribution, these
pages are intended to ship inside the app (for example under a future
Documentation menu or Help window) so users do not need the source repository.

## Start Here

| Topic | Guide |
| --- | --- |
| First launch, layout, output folder | [Getting Started](getting-started.md) |
| Models, tokens, memory | [First Launch And Model Setup](model-setup.md) |
| Music generation and covers | [Generating Music](generating.md) |
| Sound effects and ambient | [Sound Effects](sound-effects.md) |
| Inpaint, repaint, DSP | [Audio Editor](editor.md) |
| History, traces, presets | [History And Presets](history-and-presets.md) |
| Datasets, enrichment, local training | [Datasets And Training](datasets-and-training.md) |
| Import, gain, stacking, merge | [Adapters](adapters.md) |
| RunPod remote training | [Remote GPU Training](remote-gpu-training.md) |
| Prompt enhance, lyrics, providers | [Intelligence And Lyrics](intelligence.md) |
| Claude / Codex / Gemini / Grok / HF CLIs | [Local CLI Access](local-cli-access.md) |
| Localhost automation | [MCP For Users](mcp.md) |
| Diagnostics folder and breadcrumbs | [Diagnostics And Logs](diagnostics-and-logs.md) |
| Editor and common shortcuts | [Keyboard Shortcuts](keyboard-shortcuts.md) |
| Symptom-first fixes | [Troubleshooting](troubleshooting.md) |
| What to collect for a bug report | [Bug Report Checklist](bug-report-checklist.md) |

## App Map

The main window uses a three-column shell:

1. **Sidebar**
   - **Create:** Generate, Edit
   - **Build:** Datasets, Training
   - **Library:** History, Presets
2. **Canvas** — the active workspace for the selected sidebar item.
3. **Inspector** — pipeline override, duration, seed, guidance, and active model
   summary for generation-related work.

Settings live under the app Settings window and cover Models, Adapters,
Intelligence, MCP, and related preferences.

## What Belongs Here

User docs should answer:

- What does this feature do?
- What should I click?
- What does good output look like?
- What command or log can I use to verify it?
- What should I collect when something fails?

Internal design notes, research, roadmaps, and long-form investigations belong
in the top-level `docs/` folder or `docs/developer/`.
