# Documentation Integration

PiMP Audio Studio's public docs should not depend on users having access to the
source repository. If the app is distributed as a signed `.app` or `.dmg`, the
user docs need to be bundled or made available from inside the app.

## Recommended Source Layout

Keep Markdown source in:

```text
docs/user/
```

Use this folder as the canonical authoring location. The files are plain
Markdown so they can be rendered in GitHub, bundled in the app, or published to
a website later.

## Recommended App Layout

At build or release time, copy selected user docs into the app bundle:

```text
PiMP Audio Studio.app/
  Contents/
    Resources/
      Help/
        README.md
        model-setup.md
        datasets-and-training.md
        adapters.md
        local-cli-access.md
        remote-gpu-training.md
        diagnostics-and-logs.md
        troubleshooting.md
        bug-report-checklist.md
```

Then add an in-app Help entry that renders those Markdown files:

```text
Help -> PiMP Audio Studio Help
```

The Help view should:

- Render bundled Markdown offline.
- Offer a table of contents.
- Support search across page titles and body text.
- Include a "Copy Diagnostics Info" action on troubleshooting pages.
- Avoid opening the source repository for normal user help.

## Build System Options

Good first implementation:

1. Add a `PiMPAudioStudio/Resources/Help/` folder.
2. Copy the user Markdown files there.
3. Register the folder as target resources in `project.yml`.
4. Render the files in a SwiftUI Help window.

Better release implementation:

1. Keep `docs/user/` as source.
2. Add a script that syncs approved docs into `PiMPAudioStudio/Resources/Help/`.
3. Fail CI or release packaging if bundled docs are stale.

## Writing Rules For Bundled Docs

Bundled docs should:

- Avoid repo-only paths unless they are clearly labeled as developer paths.
- Prefer app menu paths and user-visible labels.
- Include verification commands when helpful.
- Avoid secrets, private tokens, and personal machine paths.
- Explain what good output and bad output look like.

## Docs That Should Stay Developer-Only

Do not bundle raw investigation notes by default:

```text
docs/debug/
docs/superpowers/
docs/*-roadmap.md
docs/*-investigation.md
```

They are useful for engineering, but too noisy and implementation-specific for
normal users.
