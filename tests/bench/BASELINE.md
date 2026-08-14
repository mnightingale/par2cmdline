# CPU baseline (pre-GPU)

Reference numbers for the CPU GF16 path, captured before any GPU backend
existed. Re-measure with `parbench.py` on the same machine when comparing.

Two machines are recorded. **Never compare across them** — the whole point of
[the Vulkan work](../../docs/gpu-backend.md) is that the CPU:GPU bandwidth
ratio differs by platform, so each machine needs its own CPU baseline.

- [Machine A — Apple M2 Pro](#machine-a--apple-m2-pro-macos) (Metal backend)
- [Machine B — Ryzen 7 5800X + RTX 4070 Ti](#machine-b--ryzen-7-5800x--rtx-4070-ti-windows) (Vulkan target)

---

# Machine A — Apple M2 Pro (macOS)

## Machine

| | |
| --- | --- |
| CPU | Apple M2 Pro, 12 cores |
| GPU | Apple M2 Pro, 19 cores (unified memory) |
| Memory | 32 GiB unified, ~200 GB/s |
| OS | macOS (Darwin 25.5) |
| Build | `feature/gpu-gf16`, `./configure` defaults |

## Results

Settings: 20 files, `-b2000`, `-r15`, 10% damage, best of 3, **warm page cache**.
These are wall-clock totals including the scan phase.

| Corpus | Create | Repair (delete) | Repair (corrupt) |
| --- | --- | --- | --- |
| 1 GiB | 2.36 s | 2.06 s | 4.53 s |
| 10 GiB | 23.97 s | 20.77 s | 47.82 s |

At 10 GiB the block size is 5.1 MiB, with 2000 source and 300 recovery blocks;
10% damage means 200 blocks are reconstructed.

## GF16 throughput

Repair is two phases, and they must be separated. `par2 repair` first *scans*
every present file (reading and hashing it) and only then reconstructs the
missing blocks. Dividing total repair time by the GF16 work therefore charges
the scan to the compute and understates throughput badly.

Measured at 10 GiB by timing `par2 verify` (the scan alone) against a full repair:

| Phase | CPU | GPU (Metal) |
| --- | --- | --- |
| Scan | 9.72 s | 9.72 s |
| GF16 reconstruct | 11.45 s | 12.50 s (GPU busy 11.36 s) |
| Total | 21.17 s | 22.34 s |

Reconstructing 200 blocks from 2000 inputs at 5,368,712 B is 2147 GB of
multiply-add, giving:

- **CPU: 187 GB/s**
- **GPU: 189 GB/s** (from `MTLCommandBuffer` GPU busy time)

**The two are within 1% of each other.**

## What this means

Both paths saturate the M2 Pro's ~200 GB/s unified memory, which the CPU and
GPU share. The NEON path was already at ~94% of that ceiling, so there was
never headroom for a GPU to exploit — and the GPU additionally pays for staging
slices into device memory and reading results back, which is why it comes out
slightly *slower* overall:

| Mode | CPU | GPU | Ratio |
| --- | --- | --- | --- |
| Repair, delete | 21.34 s | 22.42 s | 0.95x |
| Repair, corrupt | 47.50 s | 49.57 s | 0.96x |
| Create | 24.80 s | 20.71 s | 1.20x |

Creation is the one case that improves, because it is not purely GF16-bound.

An earlier revision of this file claimed a 103 GB/s CPU baseline and a 1.84x
kernel speedup. That was wrong: the 103 GB/s figure included the scan phase in
the compute time, making the CPU look ~1.8x slower than it is. The Metal kernel
is not faster than the NEON path on this hardware; it is equal to it.

## Where a GPU backend does pay off

The conclusion is about *bandwidth ratio*, not about GPU versus CPU compute.
Apple Silicon is the worst case for this work because both processors draw on
one memory pool. A discrete GPU does not:

| | Memory bandwidth |
| --- | --- |
| M2 Pro (shared by CPU and GPU) | ~200 GB/s |
| RTX 4090 | ~1000 GB/s |
| RX 7900 XTX | ~960 GB/s |
| Typical desktop CPU | ~50–90 GB/s |

That is a 10x ratio rather than 1x, which is where the Vulkan backend is
expected to earn its keep. Nothing here has been measured on such hardware.

## Profiling

Set `PARPAR_GPU_STATS=1` to print a breakdown on the GPU path:

```
[GPU STATS] wall 12.50s | gpu 11.36s over 125 dispatches | stage 2.68s (10.0 GiB) | lut+encode 0.02s | readback 0.25s
```

This separates real GPU busy time from host-side staging, which is what makes
it possible to tell a slow kernel from lost overlap.

## Metal device capabilities (M2 Pro)

Probed via `MTLCopyAllDevices`; relevant to kernel design:

| Property | Value |
| --- | --- |
| Unified memory | yes — `MTLResourceStorageModeShared` buffers are zero-copy |
| Max threadgroup memory | 32,768 bytes |
| Max threads per threadgroup | 1024 |
| Max buffer length | 19,169 MiB |
| Recommended working set | 25,559 MiB |
| GPU families | Apple7, Apple8, Metal3 |

The 32 KiB threadgroup memory limit is the binding kernel constraint: the full
GF16 antilog table is 64K entries × 2 B = 128 KiB and does **not** fit. The
kernel must either cache a partial table (as the OpenCL backend does via
`LMEM_CACHE_SIZE`) or build small per-coefficient lookup tables in threadgroup
memory.

---

# Machine B — Ryzen 7 5800X + RTX 4070 Ti (Windows)

The Vulkan target. **CPU only** — measured before any Vulkan code existed, so
this is the number the GPU backend must beat on this machine.

## Machine

| | |
| --- | --- |
| CPU | AMD Ryzen 7 5800X, 8 cores / 16 threads, 3.8 GHz base |
| GPU | NVIDIA GeForce RTX 4070 Ti, 12 GB GDDR6X (~504 GB/s), driver 32.0.16.1088 |
| Memory | 32 GiB DDR4-3600, dual channel (~57.6 GB/s theoretical) |
| Storage | Samsung 980 PRO 2 TB NVMe |
| OS | Windows 11 Pro 26200 |
| Build | `feature/gpu-gf16`, MSVC `v145`, `Release\|x64`, `par2 --list-gpus` reports none |

## Results

Settings identical to Machine A: 10 GiB over 20 files, `-b2000`, `-r15`, 10%
damage, `-t` default (16), **warm page cache**, 3 repetitions. Block size
5,368,712 B; 2000 source and 300 recovery blocks; 200 blocks reconstructed.

Each repetition times `par2 verify -q` and `par2 repair -q` against the *same*
damage state, so the subtraction is valid. Medians of 3:

| Mode | Scan (`verify`) | Reconstruct (`repair − verify`) | Total (`repair`, best) |
| --- | --- | --- | --- |
| delete | 9.60 s | **20.12 s** | 29.72 s |
| corrupt | 27.22 s | 30.41 s | 56.97 s |

Creation was 34.54 s.

Per-run reconstruct, showing the spread:

| Mode | run 1 | run 2 | run 3 | median |
| --- | --- | --- | --- | --- |
| delete | 20.12 s | 20.44 s | 20.05 s | 20.12 s |
| corrupt | 30.41 s | 32.01 s | 28.56 s | 30.41 s |

## GF16 throughput

2000 × 200 × 5,368,712 = 2147 GB of multiply-add, so:

- **CPU: 107 GB/s** (2147 GB / 20.12 s median, delete mode)

**Use the delete-mode figure.** In corrupt mode `repair` does substantial work
that `verify` does not — rewriting repaired blocks back into the damaged files
in place — so the difference is not purely GF16 and the apparent 71 GB/s is an
underestimate. That mode is also visibly noisier (28.6–32.0 s across three runs,
against 20.1–20.4 s for delete).

## What this means for the Vulkan backend

For reference, Machine A's CPU measured 187 GB/s, so this CPU is ~0.57x of it —
but that comparison is not the useful one. What matters is the local ratio:

| | Bandwidth |
| --- | --- |
| RTX 4070 Ti (VRAM) | ~504 GB/s |
| This CPU, measured GF16 | 107 GB/s |
| This CPU, theoretical DRAM | ~57.6 GB/s |

The measured 107 GB/s exceeding DRAM bandwidth is expected and not an error:
the GF16 work metric counts multiply-add bytes, and the reconstruct loop re-reads
each input slice against many outputs out of cache rather than DRAM. That is
also precisely why the CPU is not as far behind as the DRAM figure suggests.

So the honest ceiling for the reconstruct phase is **~4.7x** (504/107), not the
5–7x [docs/gpu-backend.md](../../docs/gpu-backend.md) estimated from a generic
"50–90 GB/s desktop CPU" — that guess was low for this part. Real gains will be
below the ceiling once PCIe staging and readback are paid for.

End-to-end is bounded much harder, because the scan does not accelerate at all:

| Mode | Repair now | Reconstruct at 4.7x | Repair then | End-to-end |
| --- | --- | --- | --- | --- |
| delete | 29.72 s | 4.3 s | 13.9 s | 2.14x |
| corrupt | 56.97 s | 6.5 s | 33.7 s | 1.69x |

And with an *infinitely* fast GPU the delete-mode repair still cannot go below
the 9.60 s scan — a hard ceiling of 3.10x. **Any end-to-end claim above ~2x on
this machine deserves a second look.** Report the reconstruct phase separately;
it is the only number the backend actually controls.

## Reproducing

```bash
python3 tests/bench/parbench.py --par2 ./x64/Release/par2.exe \
        --size 10G --repeat 3 --phase-split --backend cpu
```

`--phase-split` is what produces the scan/reconstruct columns; without it the
harness reports only total repair wall time, which includes the scan and must
not be divided by the GF16 work.
