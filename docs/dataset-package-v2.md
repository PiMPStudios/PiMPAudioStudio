# Dataset Package v2

Status: contract, compatibility, whole-track/clip writers, and role-aware
Dataset Review integration implemented.

Package v2 makes complete source tracks canonical. Full mixes, stems, and clips
are separate assets connected to a stable source track instead of being
independent clip rows.

The normative JSON contract is
[`schemas/dataset-package-v2.schema.json`](schemas/dataset-package-v2.schema.json).
An annotated whole-track example is available at
[`examples/dataset-package-v2-whole-track.json`](examples/dataset-package-v2-whole-track.json).
The matching Swift types and validator live under
`PiMPAudioCore/Sources/PiMPAudioCore/Dataset/`.

## Contract

The package contains three related collections:

1. `source_tracks` records each complete imported track.
2. `assets` records the preserved full mix and any optional stems or clips.
3. `training_plans` records which assets a specific backend will use and the
   exact range/window policy it will apply.

Every asset has:

- a stable `asset_id`;
- its canonical `source_track_id`;
- an `asset_role` of `full_mix`, `vocals`, `instrumental`, or `clip`;
- a package-relative audio path;
- its `source_range`;
- optional `derived_from` and derivation provenance;
- independent caption and lyrics annotations with provenance.

Annotations keep caption and lyrics provenance separate because the native ACE
captioner and transcriber are different model packs and may have different
revisions, prompt versions, quantization, and raw-output artifacts.

## Training Ranges

The complete asset duration and the effective backend training behavior are
different facts:

- `asset.source_range` is the complete time range represented by the asset.
- `selection.training_range` is the portion made available to that backend.
- `selection.sample_window_seconds` is the smaller window sampled during
  individual training steps, when applicable.
- `selection.range_policy` explains whether the range is the full source, a
  backend limit, a manual choice, or a preserved legacy clip.

For example, a 305-second source may be recorded as:

```json
{
  "source_range": {
    "start_seconds": 0,
    "end_seconds": 305
  },
  "training_range": {
    "start_seconds": 0,
    "end_seconds": 240
  },
  "range_policy": "backend_limit",
  "sample_window_seconds": 60
}
```

This distinction prevents a complete source, a preprocess cap, and a sampled
training window from being collapsed into one misleading duration field.

## Default Selection

New whole-track packages create a full-mix asset for every source track and
include only those full mixes in the default training plan. Explicit clip mode
still preserves those complete sources, then creates selected `clip` assets
with `derived_from` relationships to the canonical full mix.

Stem and alternate assets are opt-in. Dataset Review exposes the plan selection,
source relationship, effective range, and a warning when more than one
representation of a source is selected. The validator enforces the same
relationship and weight-safety rules outside the UI.

## Legacy Package Rules

Version-1 packages remain read-only inputs:

- `TrainingDatasetPackageEditor.loadPackageV2(at:)` detects the manifest
  version.
- Native v2 manifests decode and validate directly.
- V1 manifests and `captions.json` pass through
  `DatasetPackageV1Adapter`.
- The adapter returns an in-memory v2 graph and never rewrites the original
  manifest, captions, clips, stems, or review metadata.

Legacy full-mix rows are classified as `clip` assets. In v1, “full mix” means
the full mix of an already-cut clip; it does not prove that the complete source
track exists in the package. Reclassifying those rows as canonical whole-track
assets would invent data that was not preserved.

Vocal and instrumental files already referenced by a legacy record become
optional derived assets attached to that clip asset. Only the original clip
rows are selected in the compatibility training plan, preserving existing
training weight.

## Existing Training Consumers

`DatasetTrainingBundleV1Projection` converts one validated v2 training plan to
the existing `training_bundle_*.json` shape. It preserves additive v2 fields
such as:

- `source_package_version`;
- `training_plan_id`;
- `backend_id`;
- `source_track_id`;
- `asset_id` and `asset_role`;
- `source_range` and `training_range`;
- `sample_window_seconds`;
- `include_in_training`.

The existing local importer and remote worker can ignore these new fields and
continue consuming `file`, `assets`, `caption`, `lyrics`, and `clips` during
the transition.

Direct package-v2 consumption by the remote worker remains a later milestone.

## Validation and Path Safety

The Swift validator fails packages closed for:

- unsupported package versions;
- empty or duplicate stable IDs;
- missing track, asset, parent-asset, or training-plan references;
- cross-track or cyclic derivation relationships;
- invalid or out-of-bounds time ranges;
- invalid sample windows;
- empty training plans;
- absolute paths, URL paths, path traversal, and malformed package-relative
  paths;
- malformed SHA-256 values.

Unknown range ends are allowed for incomplete legacy metadata but produce a
warning because effective coverage cannot be fully reviewed.

Schema validation does not replace filesystem validation. Archive extraction
and remote execution must continue rejecting symlink escapes after paths are
resolved on disk.

## Current Writer and Review Flow

The local-file builder now:

1. copies every complete imported source into `sources/`;
2. emits stable source-track and full-mix asset IDs;
3. selects full mixes by default, with a visible 240-second backend range when
   required;
4. retains clip extraction as an explicit derived-asset mode;
5. adds separated stems as opt-in derived assets;
6. projects the selected v2 plan to existing local/remote bundle files.

Dataset Review reads all v2 assets, not only the compatibility bundle. Review
saves update annotations and `include_in_training` selections in the canonical
manifest before regenerating projections.

Native ACE annotation now runs against this stable whole-track graph. Caption
and lyrics provenance survive Dataset Review edits, while the next package
milestone is direct package-v2 remote-worker consumption.
