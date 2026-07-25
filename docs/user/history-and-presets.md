# History And Presets

## History

Open:

```text
Library -> History
```

Every completed generation (music, cover, SFX, training audition snapshot) is
recorded as a History entry with audio and, when available, a sibling
`.trace.json` file.

### Playback

Click a row (outside select mode) to load it into the app player.

### Select Mode

Click **Select**, then choose one or more rows:

| Action | Requirement | Purpose |
| --- | --- | --- |
| Open in Generate | Exactly one selected | Reloads prompt, lyrics, seed, duration, guidance, pipeline, and adapters into Generate for a quick tweak |
| Edit | Exactly one selected with audio | Loads the take into the Editor for polish / DSP |
| Compare | Exactly two selected takes with traces | Opens A/B / Trace Compare |
| Train LoRA | One or more selected | Opens the training sheet with those takes |
| Export… | One or more selected | Copies selected audio to a folder you choose |
| Delete | One or more selected | Removes selected audio + trace sidecars only |
| Done | — | Leaves select mode |

**Open in Generate** prefers the sibling `.trace.json` when present so guidance
preset, steps, and related knobs match the original run as closely as possible.
Row context menus also offer **Open in Generate** and **Edit**.

**Delete All** clears the entire History list (destructive).

### Analyzer

When a take has an analyzable trace, use the analyzer button on the row (or open
**Analyzer** from Generate) to inspect stage timings, settings, and adapters.

Preferred-take flags and listening notes can be stored on traces and mirrored
into History metadata.

### Safety Notes

- History paths are confined to the History directory. Malformed names cannot
  escape that folder.
- If `history.json` is corrupt, the app preserves it as `history.json.corrupt`
  before resetting the in-memory list.

## Presets

Open:

```text
Library -> Presets
```

Presets store generation settings saved from the Generate page (prompt-related
configuration, duration-related fields, mode, and related knobs captured at save
time).

### Managing Presets

| Action | How |
| --- | --- |
| Load | Click a preset row (switches to Generate and applies) |
| Duplicate | Context menu → Duplicate |
| Rename | Context menu → Rename… |
| Delete | Context menu → Delete, or swipe/delete in the list |

If the list is empty, save a preset from Generate with the save control first.

### When To Use Presets

- Repeatable product tones ("trailer hit", "lofi loop", "podcast bed")
- Fixed A/B labs (same duration/guidance/mode, different prompts)
- Team handoff of known-good settings without retyping

Presets do not replace adapters. For style capture across many songs, train or
import an adapter; use presets for session-level generation knobs.

## Related

- [Generating Music](generating.md)
- [Adapters](adapters.md)
- [Datasets And Training](datasets-and-training.md)
