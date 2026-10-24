# Linux: what is not tested yet

The Linux backend passes `make smoke` and `make e2e` in containers on
arm64 (Apple silicon) and x64 (AMD EPYC Zen 5, AVX-512 VNNI). These
items still need testing:

- pi in fullscreen over ssh in a real terminal: clicks, hover, and no
  output from ONNX Runtime on the terminal.
- int8 on x64 CPUs without AVX-512 VNNI (AVX2 only, or Intel's 256-bit
  AVX-VNNI). In qemu, option F1 dropped by .08.
- glibc older than 2.35 (the build uses Ubuntu 22.04).
