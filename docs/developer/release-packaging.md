# Release And Packaging

This page captures the packaging path for a standalone PiMP Audio Studio build:
a signed and notarized macOS app that users can mount, drag into Applications,
and run without the source repository.

## Current Release Identity

```text
Bundle ID: com.pimpstudios.PiMPAudioStudio
Display name: PiMP Audio Studio
Team ID: F3S7J7L552
Marketing version: 0.8.0
Build version: 8
Minimum macOS: 15.0
Hardened runtime: enabled
Sandbox: enabled
```

The repository is licensed under Apache License 2.0. See the root `LICENSE`,
`NOTICE`, and `TRADEMARKS.md`. Third-party components and downloadable models
keep their own terms (`THIRD_PARTY_NOTICES.md`).

## Standalone App Goal

1. User downloads a `.dmg`.
2. User opens the disk image.
3. User drags `PiMP Audio Studio.app` into Applications.
4. User launches the app.
5. App guides model downloads and optional account/token setup.

Do not require users to clone the repository, open Xcode, or know local project
paths.

## What Goes In The App Bundle

Bundle:

- Compiled app
- Swift package runtime products
- App icons and static UI assets
- Bundled user docs from `docs/user/`
- Small templates/config files needed at runtime
- Portable remote-worker payload as inert resources if still required for upload
- `LICENSE`, `NOTICE`, and the third-party license bundle
- A published privacy policy only if that distribution channel needs one

Do not bundle:

- Large ACE-Step model weights
- Gated Hugging Face weights
- User API keys or SSH keys
- Local training datasets
- Remote GPU job artifacts

## Legal Release Gate

Before public binary distribution:

1. Ship `LICENSE`, `NOTICE`, and the third-party notices required by bundled
   dependencies (MIT copyright lines, Apache License 2.0 text, and any
   dependency NOTICE files).
2. Do not present `docs/legal/eula-draft.md` as in-force terms. Source and
   self-built binaries are Apache License 2.0.
3. Publish a privacy policy if the distribution channel needs one, based on
   a current data-flow audit.
4. Audit every model, dataset, and remote-worker dependency independently from
   the application source license.
5. Keep the product name and icon consistent with `TRADEMARKS.md`.

## Sandboxing Notes

See [Storage, Sandbox, Security](storage-sandbox-security.md). Audit temporary
exceptions before a public Developer ID build.

## Docs Bundling

Recommended:

```text
PiMP Audio Studio.app/Contents/Resources/Help/
```

Source:

```text
docs/user/
```

See [Documentation Integration](documentation-integration.md).

## Release Checklist

Before cutting a build:

- Update `MARKETING_VERSION` and `CURRENT_PROJECT_VERSION` in `project.yml` /
  project settings
- Update `CHANGELOG.md`
- Confirm bundled Help docs match `docs/user/`
- Run a clean build
- Run practical unit tests on the release machine
- Smoke test first launch on a clean user profile
- Download required model components through the app
- Generate one base music clip
- Generate one SFX clip if Stable Audio is ready
- Import and audition one adapter
- Build a tiny dataset and run a local training preflight
- Run remote GPU dry-run packaging without launching a paid pod (or launch only
  with explicit intent)
- Confirm verbose diagnostics write to the expected sandbox path
- Confirm Local CLI Access and MCP settings do not expose secrets
- Confirm MCP remains off by default

## Signing And Notarization Shape

1. Archive or build a Release app.
2. Sign with a Developer ID Application certificate.
3. Preserve hardened runtime and sandbox entitlements.
4. Submit the app or DMG for Apple notarization.
5. Staple the notarization ticket.
6. Build a DMG with the app plus an Applications symlink.
7. Mount the DMG on a second machine and verify Gatekeeper launch.

## Packaging Smoke Commands

```bash
codesign --verify --deep --strict --verbose=2 "PiMP Audio Studio.app"
spctl --assess --type execute --verbose=4 "PiMP Audio Studio.app"
xcrun stapler validate "PiMP Audio Studio.app"
```

For a DMG:

```bash
spctl --assess --type open --verbose=4 "PiMP Audio Studio.dmg"
```

## Known Packaging Work Still Needed

- Add a repeatable release script or CI workflow
- Add a Help bundle sync step with staleness checks
- Build the in-app Documentation/Help presentation
- Decide whether public docs also publish on a website
- Audit temporary sandbox exceptions before paid public release
