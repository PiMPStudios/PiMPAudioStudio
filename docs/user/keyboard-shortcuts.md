# Keyboard Shortcuts

Shortcuts currently focus on the **Editor**. Menu bar standard macOS shortcuts
(copy/paste, quit, etc.) behave as usual.

## Editor

| Shortcut | Action |
| --- | --- |
| `⌘T` | Trim to selection (requires a selection) |
| `⌘⌫` (Command-Delete) | Delete selection (requires a selection) |
| `⌘⇧N` | Expand Normalize panel (peak target) |
| `⌘⇧J` | Expand Trim Silence panel |
| `⌘⇧H` | Expand High-Pass panel |
| `⌘⇧E` | Expand Parametric EQ panel |
| `⌘⇧R` | Expand Reverb panel |

Notes:

- Shortcuts apply when the Editor has loaded audio.
- Selection-required effects do nothing useful without a waveform selection.
- Parameterized effects still need **Apply** after adjusting sliders.

## Generate / Training / Datasets

There are no special global shortcuts beyond standard text field editing. Use
the on-screen buttons for Generate, Cancel, Analyzer, dataset build, and
training controls.

## Accessibility Tips

- Prefer mouse/trackpad waveform selection for precise inpaint regions.
- Use fixed seeds in the inspector for keyboard-heavy A/B iteration without
  retyping prompts.
