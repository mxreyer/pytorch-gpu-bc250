# GPU-accelerated PyTorch on the AsRock BC-250

The AsRock **BC-250** is a surplus crypto-mining board built around a
cut-down PlayStation 5 APU: RDNA2, up to 40 compute units, 16 GB of unified
memory. Its GPU reports as **gfx1013**, which ROCm does not support. This repo
shows how to run PyTorch on it anyway, and how fast it is once you do.

- **OpenCL instead of ROCm**: Mesa's **rusticl** driver plus the out-of-tree
  [`pytorch_ocl`](https://github.com/artyom-beilis/pytorch_dlprim) backend
  expose the GPU to PyTorch as `ocl:0`.
- **One rebuilt extension**: a patched `pt_ocl.so` from
  [pytorch-dlprim-gfx1013](https://github.com/mxreyer/pytorch-dlprim-gfx1013)
  makes training compile at all on rusticl, and makes it fast.
- **Benchmarked** against CUDA cloud GPUs with an identical ResNet-9 /
  CIFAR-10 workload: at 40 CUs / 2.0 GHz the BC-250 **trains ~2% faster than
  a cloud T4 and infers within 2% of it**, in fp32.

## The stack

```
  PyTorch 2.4 (CPU build)
        │  device "ocl:0"
        ▼
  pytorch_ocl  ── pt_ocl.so, patched (pytorch-dlprim-gfx1013)
        │  OpenCL
        ▼
  Mesa rusticl ── + Mesa's patched libclc (recommended)
        │
        ▼
  radeonsi (Gallium)  ──▶  amdgpu kernel driver  ──▶  /dev/dri/renderD128  ──▶  GPU (gfx1013)
```

Only the render node is needed, not ROCm's `/dev/kfd`, so passing the GPU into
a container is a plain `--device /dev/dri/renderD128`.

## Setup

See [SETUP.md](SETUP.md)

## Benchmarking results at a glance

ResNet-9 on CIFAR-10, 16 epochs, batch 128. Full tables and the protocol live in [BENCHMARK.md](BENCHMARK.md).
"Fixes-only" is `pytorch_ocl` with just the build and correctness fixes it needs to
run on rusticl at all (`pt_ocl-fixes-only.so`); "patched" adds the performance work
(`pt_ocl.so`). Both are attached to the
[pytorch-dlprim-gfx1013 releases](https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases)
from v1.3.0 on — see [SETUP.md](SETUP.md), Stage 2.

| Platform | Precision | Train img/s | Infer img/s |
| --- | --- | ---: | ---: |
| BC-250, 40 CU @ 1.5 GHz, fixes-only `pytorch_ocl` | fp32 | 786 | 3,892 |
| BC-250, 40 CU @ 2.0 GHz, fixes-only `pytorch_ocl` | fp32 | 919 | 4,489 |
| **BC-250, 40 CU @ 2.0 GHz, patched `pytorch_ocl`** | **fp32** | **2,337** | **7,130** |
| BC-250, same, `DLPRIM_CONV_FP16=1` | fp16 conv inner loop | 2,992 | 10,508 |
| T4 (Colab) | fp32 | 2,294 | 7,217 |
| A100 (Colab) | amp | 18,690 | 49,612 |

Test accuracy lands at 0.922–0.927 on every GPU platform: the OpenCL stack is
numerically correct.

## Repository layout

| Path | What it is |
| --- | --- |
| [SETUP.md](SETUP.md) | Bare-metal walkthrough: rusticl, PyTorch on `ocl:0`, the patched `pt_ocl.so`, Mesa's libclc, fp16 convolutions. |
| [BENCHMARK.md](BENCHMARK.md) | Workload rationale, protocol, results vs cloud GPUs, and how to run it anywhere. |
| `benchmark.py`, `compare.py` | Benchmark harness (ResNet-9 / CIFAR-10) and the table printer for its JSON output. Runs on `ocl:0`, `cuda` or `cpu`. |
| `results/` | Every result JSON behind the tables. |
| `gpu-check.py` | Quick proof that PyTorch is on the GPU, not silently on the CPU. |

## Related

- [bc250-jupyterhub-opencl-k3s](https://github.com/mxreyer/bc250-jupyterhub-opencl-k3s):
  serve this GPU to several users as JupyterHub notebooks on single-node k3s,
  with the GPU handed to unprivileged pods by a device plugin.
- [pytorch-dlprim-gfx1013](https://github.com/mxreyer/pytorch-dlprim-gfx1013):
  the patches, profiling and correctness work behind the patched `pt_ocl.so`.
- [duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock):
  unlocking 40 CUs and the higher GPU clock.
- The write-up: [Turning a mining board into a multi-user ML box](https://mxreyer.github.io/blog/bc250-ml-box/).

## License

[MIT](LICENSE) — Copyright (c) 2026 mxreyer.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
