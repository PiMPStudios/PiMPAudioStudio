# Legal Documents

PiMP-owned source in this repository is licensed under Apache License 2.0.
See the root [`LICENSE`](../../LICENSE) and [`NOTICE`](../../NOTICE).

These files are drafting and operational aids. They are not a substitute for
advice from a licensed attorney.

## Current Documents

- [`../../LICENSE`](../../LICENSE) is Apache License 2.0, the operative
  license for PiMP-owned source, documentation, scripts, and configuration
  in this repository.
- [`../../NOTICE`](../../NOTICE) is the Apache attribution notice.
- [`../../TRADEMARKS.md`](../../TRADEMARKS.md) records that the Apache
  license does not grant rights in the PiMP name, logo, or app icon.
- [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) separates
  third-party components and records the current direct dependency inventory.
- [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) explains how to contribute
  and that contributions are licensed under Apache License 2.0.
- [`eula-draft.md`](eula-draft.md) is a leftover customer EULA draft from
  when a paid binary was under consideration. It is **not operative**. Source
  distribution is Apache License 2.0. A compiled app that you build yourself
  is also covered by Apache License 2.0; this draft is not required for that.
- [`privacy-policy-draft.md`](privacy-policy-draft.md) is a non-operative
  customer privacy policy draft. It is not in force until counsel review,
  confirmed contact details, and publication.
- [`privacy-data-flow-inventory.md`](privacy-data-flow-inventory.md) is the
  working inventory of on-device storage and outbound network flows that
  the privacy draft is based on. It is not a privacy policy.
- [`source-access-acknowledgment.md`](source-access-acknowledgment.md) is
  retired. It applied to invited readers of the former private proprietary
  tree.

## Models And Third-Party Code

Apache License 2.0 does not relicense:

- Hugging Face model weights, gated assets, or dataset files the app
  downloads
- MLX Swift, Swift Transformers, Speech Swift, the MCP Swift SDK, or
  vendored Demucs MLX Swift
- Python/CUDA packages the remote worker may install on a user-approved pod

Keep [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) current when
direct dependencies change.

## Before A Public Binary Or Website

If you publish a notarized `.app`, a download page, or a privacy URL:

1. Confirm the Apache license, `NOTICE`, and third-party notices travel with
   the distribution.
2. Have counsel review whether a published privacy policy is required for
   that channel, using the data-flow inventory.
3. Do not present `eula-draft.md` as in-force terms.
4. Keep trademark use consistent with [`TRADEMARKS.md`](../../TRADEMARKS.md).
