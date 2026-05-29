# Anvil Patch Notes

This vendored copy is based on `kylehowells/demucs-mlx-swift` at commit
`c81c47178828db2d8bc66e64f80c745c64abdc94`.

Local changes:

- Keep only the `DemucsMLX` library target used by AnvilCore.
- Depend on the existing `swift-transformers` 0.1.x line via the `Transformers`
  product so AnvilApp does not link a second copy of the Hub runtime.
- Exclude the CLI/test/script targets because AnvilApp only needs in-process
  Swift/MLX separation.
