# Setup: GPU-accelerated PyTorch on the BC-250

This walks through getting PyTorch to run on the AsRock BC-250's GPU, on
bare metal, and proving that it does. When you are done, the board trains
and infers on `ocl:0` at cloud-T4 speed (see [BENCHMARK.md](BENCHMARK.md)).

## OpenCL, not ROCm

The BC-250's GPU reports as **gfx1013**, which is not supported by ROCm.
Instead, this build uses the **OpenCL** path: specifically, **Mesa
`rusticl`**. The out-of-tree PyTorch backend **`pytorch_ocl`** (a.k.a.
`pytorch_dlprim`) exposes OpenCL devices to PyTorch as `ocl:0`.

A side benefit: the OpenCL path needs only the DRI render node
(`/dev/dri/renderD128`), not ROCm's `/dev/kfd`, which makes handing the GPU
to a container simple later on.

## Prerequisites

- **AsRock BC-250** on Fedora with recent Mesa (≥ 26.1, so rusticl on
  gfx1013 works). Community clock/fan daemon running and stable under
  sustained load.
- Optional but recommended: the 40 CU / 2.0 GHz unlock from
  [duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock).
  Everything below works at the stock 24 CUs, just slower.
- Python 3.12 (`sudo dnf install -y python3.12`).


## Table of Contents

| Stage | Result |
| ----- | ------ |
| 1 | `clinfo` sees the GPU through rusticl |
| 2 | PyTorch runs on the GPU, provably faster than CPU, and trains |
| 3 | *(recommended)* Mesa's patched libclc, so `torch.randn` and friends work on the GPU |
| 4 | *(optional)* fp16 convolutions for another ~1.3× |

## Stage 1 — OpenCL on bare metal

**Goal:** the GPU is visible to OpenCL through rusticl.

```bash
sudo dnf install -y mesa-libOpenCL clinfo

# rusticl only exposes the radeonsi gallium driver when explicitly told to.
# Without this the BC-250 does not appear as an OpenCL device at all. Put it
# in your shell profile — every OpenCL step below needs it.
export RUSTICL_ENABLE=radeonsi

clinfo -l
```

**Verify:** `clinfo -l` lists

```
Platform #0: rusticl
 `-- Device #0: AMD BC-250 (radeonsi, gfx1013, ...)
```

A `Rusticl warning: Patched Mesa libclc not detected` line on every run is
expected — [Stage 3](#stage-3--mesas-patched-libclc-recommended) deals with
what it breaks. Stage 2's rebuilt extension is needed either way.
Without `RUSTICL_ENABLE=radeonsi`, `clinfo -l` prints only `Platform #0:
rusticl` with no device under it.

## Stage 2 — PyTorch on the GPU

**Goal:** move a tensor to `ocl:0`, run a model, and **prove** the GPU is
doing the work. A silent fall-back to CPU looks exactly like success, so the
proof is a benchmark.

Use a dedicated virtualenv with the CPU build of PyTorch plus `pytorch_ocl`:

```bash
python3.12 -m venv bc250-ocl-venv
bc250-ocl-venv/bin/pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
bc250-ocl-venv/bin/pip install https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl
```

The prebuilt wheel uses the OpenCL 2.0 work-group reduction built-ins
(`work_group_reduce_add()`), which OpenCL 3.0 made optional and rusticl
does not provide, whichever libclc is installed (Stage 3 does not change
this). So `softmax`, `cross_entropy`, and bias gradients fail to compile.
`pytorch_dlprim` already contains a portable fallback for this; it's just
off by default, and it has a race that makes softmax come out wrong.
Rebuild the extension with the fallback on (`CUSTOM_REDUCE=1`) plus the
one-line barrier fix, and drop the resulting `pt_ocl.so` into the venv.
The patches and a build script live in a separate repository, [pytorch-dlprim-gfx1013](https://github.com/mxreyer/pytorch-dlprim-gfx1013), which also carries the
performance work on top (2.54× on training). Its
[releases](https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases) have a prebuilt `pt_ocl.so`, so the quick route is:

```bash
curl -fLo pt_ocl.so https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases/download/v1.2.0/pt_ocl.so
echo '0923e8f392f84f20647b4c927bac3baf122c5914d153e4708874cfe594a21d22  pt_ocl.so' | sha256sum -c
```

Copy the downloaded `pt_ocl.so` over the stock extension in the venv you made above, and
delete the compiled-kernel cache (it is keyed to the old kernel source):

```bash
cp pt_ocl.so \
   bc250-ocl-venv/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so
rm -f ~/.dlprimitives/cache.db
```

Then, verify (with `RUSTICL_ENABLE=radeonsi` in the environment, from Stage 1 —
without it `torch.rand(..., device="ocl:0")` raises `RuntimeError: Invalid
Device #0`):

```python
import time, torch, pytorch_ocl

x = torch.rand(4096, 4096, device="ocl:0")
_ = x @ x; torch.ocl.synchronize()
t = time.time()
for _ in range(10): y = x @ x
torch.ocl.synchronize()
print(f"{2*4096**3/((time.time()-t)/10)/1e9:.0f} GFLOP/s")
```

The GPU should report ~2000+ GFLOP/s versus ~300 for `x.cpu() @ x.cpu()`.

Then confirm a small classifier trains — this is what actually exercises the
rebuilt reduction kernels (conv/linear bias gradients, `log_softmax`,
`nll_loss`):

```python
import torch, pytorch_ocl
import torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
dev = "ocl:0"

net = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(8 * 32 * 32, 10),
).to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.1)

x = torch.randn(64, 3, 32, 32, device=dev)
y = torch.randint(0, 10, (64,), device=dev)

for step in range(50):
    opt.zero_grad()
    loss = F.cross_entropy(net(x), y)
    loss.backward()
    opt.step()
    if step == 0:
        first = loss.item()

print(f"loss {first:.3f} -> {loss.item():.3f}")
```

- **Without** the rebuilt `pt_ocl.so`, the first `backward()` throws an
  OpenCL *compile* error — `use of undeclared identifier
  'work_group_reduce_add'`.
- **With** it, the loss falls steadily (e.g. `2.29 -> 0.01`). Every op ran on
  the GPU.

A `Rusticl warning: Patched Mesa libclc not detected` line on startup is
expected until Stage 3 — the rebuild's reduction fallback does not need
that patched libclc, but a few other built-ins do.

For a quick standalone check, [`gpu-check.py`](gpu-check.py) runs the same
matmul comparison at three sizes:

```bash
RUSTICL_ENABLE=radeonsi bc250-ocl-venv/bin/python gpu-check.py
```

## Stage 3 — Mesa's patched libclc (recommended)

**Goal:** every OpenCL built-in works, including the transcendental ones.

Fedora's `libclc-spirv` (22.1.8, both Fedora 44 and 45) ships an empty
`__clc_flush_denormal_if_not_supported`, so any kernel that calls float
`sin`/`cos`/`tan`/`sincos`/`remquo` (plus `fma` on Mesa 26.1) fails to link
with `Internal compilation error: nir_shader not fully linked`. In PyTorch
terms that is `torch.randn(..., device="ocl:0")`, `.normal_()` and
`nn.init.normal_()` on the device. It is also the source of the
*"Patched Mesa libclc not detected"* warning. Training a model initialised on
the CPU and then moved to `ocl:0` (as in Stage 2) does not hit it, which is
why Stage 2 passes without this.

The fix is Mesa's maintained fork of libclc
([karolherbst/mesa-libclc](https://gitlab.freedesktop.org/karolherbst/mesa-libclc)),
which publishes drop-in binaries for the same LLVM version Fedora ships.
rusticl loads them from a path baked in at Mesa build time, so the two files
replace Fedora's in place. This overwrites files owned by the
`libclc-spirv` package, so keep the originals; a package update will put
Fedora's back and you need to redo this step:

```bash
V=22.1.8.3
URL=https://gitlab.freedesktop.org/api/v4/projects/27606/packages/generic/mesa-libclc/$V
mkdir -p ~/libclc-orig && cp /usr/lib64/clc/spirv*-mesa3d-.spv ~/libclc-orig/

curl -fLo /tmp/spirv64-mesa3d-.spv $URL/spirv64-mesa3d-.spv
curl -fLo /tmp/spirv-mesa3d-.spv   $URL/spirv-mesa3d-.spv
sha256sum -c <<'SUMS'
728cdfa747a1857829c104291984964971783a301a62e9468a3d5a4f1750c949  /tmp/spirv64-mesa3d-.spv
59552f9e22ec221061e9939fc90a46e4159d1400b2e1f85eff46924489d532aa  /tmp/spirv-mesa3d-.spv
SUMS

sudo install -m0644 /tmp/spirv64-mesa3d-.spv /tmp/spirv-mesa3d-.spv /usr/lib64/clc/
rm -f ~/.dlprimitives/cache.db
```

**Verify:** `clinfo -l` no longer prints the `Rusticl warning` line, and

```bash
RUSTICL_ENABLE=radeonsi bc250-ocl-venv/bin/python -c \
  'import torch, pytorch_ocl; print(torch.randn(64, device="ocl:0").std())'
```

prints a number near 1 instead of a compile error.
To undo, copy the files from `~/libclc-orig/` back, or
`sudo dnf reinstall libclc-spirv`.

## Stage 4 — fp16 convolutions (optional)

`DLPRIM_CONV_FP16=1` in the environment (or `os.environ[...]` at the very top
of a script, before the first convolution runs) switches the Winograd
convolution kernels to an fp16 inner loop: ~1.3× on ResNet-style training and
~1.5× on inference at ~0.5% numerical error, accuracy unchanged over 16
epochs. Tensors stay fp32 in memory; only the arithmetic inside the
convolution changes. It is off by default because it accumulates in fp16 and
is ~10× looser numerically than fp32. Details:
[the fp16 inner-loop write-up](https://github.com/mxreyer/pytorch-dlprim-gfx1013/blob/main/upstream/dlprimitives-winograd-fp16-inner-loop.md).

## Done!

The GPU works. From here:

- **Measure it** against cloud GPUs: [BENCHMARK.md](BENCHMARK.md).
- **Dig into why it is fast now**: the patches, profiling and correctness
  sweeps behind `pt_ocl.so` live in
  [pytorch-dlprim-gfx1013](https://github.com/mxreyer/pytorch-dlprim-gfx1013).

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
