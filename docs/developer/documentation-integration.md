# Documentation Integration

PiMP Audio Studio's public docs should not depend on users having the source
repository. When the app is distributed as a signed `.app` or `.dmg`, user docs
need to be bundled or reachable from inside the app.

**Shipped (MVP):** offline Help window + menu, bundling `docs/user` Markdown
into the app at build time.

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

## Shipped UI

| Piece | Location |
| --- | --- |
| Help menu | **Help → PiMP Audio Studio Help** (`⌘?`) |
| Shortcuts jump | **Help → Keyboard Shortcuts** |
| Window | `WindowGroup` id `documentation` → `DocumentationView` |
| Catalog | `HelpCatalog` loads `Bundle.main/.../Help/*.md` |
| Search | Sidebar searchable filter over titles + body |

Presentation:

- Left: TOC (prefer links from `README.md`, then defaults, then leftovers)
- Right: block Markdown via `HelpMarkdown` + `MarkdownDocumentView` (headings,
  paragraphs with soft-wrap join, lists, fenced code, tables, thematic breaks;
  inline bold/code/links via `AttributedString` inline syntax)
- Leading H1 matching the page title is dropped so the nav title is not doubled
- Relative `*.md` links navigate within Help; other URLs open normally
- Offline only for the MVP (no online site yet)

> Note: A single `Text(AttributedString(markdown: fullDocument))` collapses all
> blocks into one run with no spacing — do not use that for multi-section pages.

## Build System

| Mechanism | Role |
| --- | --- |
| `Scripts/sync-user-help.sh` | Optional local sync into a Resources tree |
| Xcode **Sync User Help Docs** script phase | Copies `docs/user/*.md` → app bundle `Help/` on every build |
| `project.yml` `preBuildScripts` | Same intent when regenerating the project with XcodeGen |

DEBUG builds fall back to the repo `docs/user` path if the bundle folder is
missing, so Help still works when iterating without a clean install.

### Better release follow-ups

1. CI fails if Help is empty after build.
2. Release checklist verifies Help opens offline on a clean machine.
3. Optional online docs site + hybrid “Open online…” later.

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
