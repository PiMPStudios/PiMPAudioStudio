# Contributing

PiMP Audio Studio is open source under [Apache License 2.0](LICENSE).
Contributions are welcome.

## License Of Contributions

By submitting a pull request, issue patch, or other contribution, you
license that contribution under Apache License 2.0, with no additional
terms, unless you and Professional Interactive Media Productions LLC agree
otherwise in writing.

You certify that:

- you have the right to submit the contribution (you wrote it, or you have
  permission from the copyright holder)
- the contribution is your original work, or you clearly identify third-party
  material and its license
- you are not submitting secrets, private datasets, gated model weights, or
  confidential third-party source

See Apache License 2.0 section 5.

## Before You Start

1. Read [Building From Source](README.md#building-from-source) and
   [Build And Test](docs/developer/build-and-test.md).
2. Search existing issues before opening a new one.
3. Keep changes focused. Do not mix formatting, refactors, and features in
   one pull request.

## What Not To Include

Do not commit:

- API keys, tokens, SSH keys, or `.env` files
- model weights, Hugging Face caches, or generated audio
- training datasets or `remote_artifacts`
- `__pycache__`, `.pyc`, or Xcode user state

Vendored third-party code under `Vendor/` keeps its original license. Do not
relicense those files as Apache 2.0.

## Code And Tests

- Format Swift with SwiftFormat 0.62.1 using `.swiftformat` (same as CI).
  `Vendor/` and `Spikes/` are excluded.
- Add or update tests for the behavior you change. App tests live in
  `PiMPAudioStudioTests/`; core tests live in `PiMPAudioCore/Tests/`.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible or packaging
  changes.
- User-facing docs live in `docs/user/`; developer docs live in
  `docs/developer/`.

## Pull Requests

Open a pull request against `main` with:

- a short description of what changed and why
- test evidence (`xcodebuild test`, the Core serial script, or a reason the
  change is docs-only)
- notes if the change touches sandbox entitlements, MCP, or remote GPU

## Trademarks

Using or forking this repository does not grant rights in the PiMP name or
app icon. See [TRADEMARKS.md](TRADEMARKS.md).
