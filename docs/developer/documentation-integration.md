# Documentation Integration

PiMP Audio Studio's public docs should not depend on users having the source
repository. When the app is distributed as a signed `.app` or `.dmg`, user docs
need to be bundled or reachable from inside the app.

Code wiring for a Documentation/Help menu is intentionally **not** part of the
docs-only preparation work. This page describes the intended integration once
product decides on presentation.

## Canonical Sources

| Audience | Path |
| --- | --- |
| End users | `docs/user/` |
| Developers | `docs/developer/` |
| Engineering deep dives | top-level `docs/*.md`, `docs/debug/`, `docs/superpowers/` |

Author in Markdown. Keep user docs free of repo-only jargon unless labeled.

## Recommended App Bundle Layout

At build or release time, copy approved **user** docs into:

```text
PiMP Audio Studio.app/Contents/Resources/Help/
  README.md
  getting-started.md
  model-setup.md
  generating.md
  sound-effects.md
  editor.md
  history-and-presets.md
  datasets-and-training.md
  adapters.md
  remote-gpu-training.md
  intelligence.md
  local-cli-access.md
  mcp.md
  diagnostics-and-logs.md
  keyboard-shortcuts.md
  troubleshooting.md
  bug-report-checklist.md
```

Optional later: ship a subset of developer docs under a separate
`Resources/DeveloperHelp/` only in internal builds.

## UI Options (Undecided)

Any of these can consume the same Markdown source:

1. **Help menu** — standard macOS Help book / custom Help window
2. **Documentation menu** — top-level menu beside Help
3. **In-app browser** — SwiftUI Markdown renderer with TOC + search
4. **Hybrid** — offline bundle + online docs site

Recommended Help view behaviors:

- Render bundled Markdown offline
- Table of contents from `docs/user/README.md`
- Search titles and body text
- "Copy Diagnostics Info" on troubleshooting pages
- Do not require the git checkout for normal help

## Build System Options

Good first implementation:

1. Add `PiMPAudioStudio/Resources/Help/`.
2. Sync from `docs/user/` via script.
3. Register as target resources in `project.yml`.
4. Render in a SwiftUI window.

Better release implementation:

1. Keep `docs/user/` as sole authoring source.
2. CI fails if bundled Help is stale versus `docs/user/`.
3. Release checklist verifies Help opens offline.

## Writing Rules For Bundled Docs

- Prefer app menu paths and user-visible labels.
- Avoid repo-only paths unless labeled as developer reference.
- Include verification commands when helpful.
- Never include secrets, private tokens, or personal machine paths.
- Show what good and bad output look like.
- Keep markdownlint-friendly ATX headings and fenced code languages.

## Do Not Bundle By Default

```text
docs/debug/
docs/superpowers/
docs/*-roadmap.md
docs/*-investigation.md
docs/developer/   (unless internal build explicitly wants it)
```

## Maintenance

When features change:

1. Update the matching `docs/user/` page the same PR when possible.
2. Update `docs/developer/` if architecture/APIs changed.
3. Note user-facing doc work in `CHANGELOG.md`.
4. Keep `docs/user/README.md` index accurate.
