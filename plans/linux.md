# Linux: what is not tested yet

The Linux backend builds and passes `make smoke` in containers (arm64
native, x64 emulated). These items need a real Linux machine:

- pi in fullscreen over ssh: clicks, hover, and no output from ONNX
  Runtime on the terminal.
- `make e2e` on Linux.
- Latency per unit on a real CPU. The emulated x64 numbers mean nothing.
- int8 on x64. The x64 models are not released until this passes. In
  qemu, option F1 drops by .08 against PyTorch, outside the .02 limit
  (arm64: no drop); per-channel weights were worse (.24) and
  `reduce_range` a little better (.07). Without VNNI, x64 multiplies
  unsigned 8-bit activations by signed 8-bit weights and adds pairs of
  products into a saturating 16-bit value; qemu reports no VNNI, so it
  runs this path. With AVX-512 VNNI or AVX-VNNI, the sums are 32-bit,
  as on arm64. On the devbox: check the CPU flags, run `parity-onnx`.
  If it fails, choose between fp32 models for x64 and int8 with the
  sensitive layers kept in fp32.
  `data/golden-onnx-x64.json` is from the previous model; regenerate it
  with `golden-onnx` when the x64 model exists.
- glibc older than 2.35 (the build uses Ubuntu 22.04).
