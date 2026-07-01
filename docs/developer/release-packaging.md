# Release And Packaging

This page captures the packaging path for a standalone PiMP Audio Studio build:
a signed and notarized macOS app that users can mount, drag into Applications,
and run without access to the source repository.

## Current Release Identity

Current project settings:

```text
Bundle ID: com.pimpstudios.PiMPAudioStudio
Display name: PiMP Audio Studio
Marketing version: 0.8.0
Build version: 8
Minimum macOS: 15.0
Hardened runtime: enabled
Sandbox: enabled
```

The repository currently states:

```text
Private repository. All rights reserved.
```

There is no public open-source license at the top level. Do not publish source
or redistributable assets as open source unless the licensing decision changes.

## Standalone App Goal

The public install flow should be:

1. User downloads a `.dmg`.
2. User opens the disk image.
3. User drags `PiMP Audio Studio.app` into `Applications`.
4. User launches the app.
5. App guides model downloads and optional account/token setup.

Do not require users to clone the repository, open Xcode, or know local project
paths.

## What Goes In The App Bundle

Bundle:

- Compiled app.
- Swift package runtime products.
- App icons and static UI assets.
- Bundled user docs from `docs/user/`.
- Small templates/config files needed at runtime.

Do not bundle:

- Large ACE-Step model weights.
- Gated Hugging Face weights.
- User API keys.
- User SSH keys.
- Local training datasets.
- Remote GPU job artifacts.

## Sandboxing Notes

The app is sandboxed and currently has entitlements for:

- User-selected read/write files.
- Network client.
- Network server for localhost MCP.
- Read-only temporary exceptions for common CLI install prefixes.
- Read-only home exceptions for selected provider CLI credential folders.
- Read/write home exception for Gemini CLI state.

Before public release, review whether each temporary exception is still required
and document why it exists. Prefer security-scoped user selection wherever a
workflow can reasonably use it.

## Docs Bundling

User docs should ship inside the app. Recommended bundle location:

```text
PiMP Audio Studio.app/Contents/Resources/Help/
```

The source Markdown remains in:

```text
docs/user/
```

Release packaging should either copy those files during build or run a sync step
before signing. See [Documentation Integration](documentation-integration.md).

## Release Checklist

Before cutting a build:

- Update `MARKETING_VERSION` and `CURRENT_PROJECT_VERSION`.
- Update `CHANGELOG.md`.
- Confirm bundled Help docs match `docs/user/`.
- Run a clean build.
- Run unit tests that are practical on the release machine.
- Smoke test first launch on a clean user profile or clean machine.
- Download required model components through the app.
- Generate one base music clip.
- Generate one SFX clip if Stable Audio is ready.
- Import and audition one adapter.
- Build a tiny dataset and run a local training preflight.
- Run remote GPU dry-run packaging without launching a paid pod.
- Confirm verbose diagnostics write to the expected sandbox path.
- Confirm Local CLI Access rows do not expose secrets.

## Signing And Notarization Shape

The exact CI/release script can evolve, but the high-level shape is:

1. Archive or build a Release app.
2. Sign with a Developer ID Application certificate.
3. Preserve hardened runtime and sandbox entitlements.
4. Submit the app or DMG for Apple notarization.
5. Staple the notarization ticket.
6. Build a DMG with the app plus an Applications symlink.
7. Mount the DMG on a second machine and verify Gatekeeper launch.

## Packaging Smoke Commands

Useful local checks after export:

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

- Add a repeatable release script or CI workflow.
- Add a Help bundle sync step.
- Build an in-app Help window.
- Decide whether public docs should also be published on a website.
- Audit temporary sandbox exceptions before a paid public release.
