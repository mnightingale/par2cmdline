# CPU baseline (pre-GPU)

Reference numbers for the CPU GF16 path, captured before any GPU backend
existed. Re-measure with `parbench.py` on the same machine when comparing.

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
