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

| Corpus | Create | Repair (delete) | Repair (corrupt) |
| --- | --- | --- | --- |
| 1 GiB | 2.36 s | 2.06 s | 4.53 s |
| 10 GiB | 23.97 s | 20.77 s | 47.82 s |

At 10 GiB the block size is 5.1 MiB, with 2000 source and 300 recovery blocks;
10% damage means 200 blocks are reconstructed.

## Derived GF16 throughput

Reconstructing `N` blocks from `M` inputs costs `M × N × blocksize` multiply-adds:

- 1 GiB: 2000 × 200 × 536,872 B = **200 GiB** in 2.06 s → **104 GB/s**
- 10 GiB: 2000 × 200 × 5,368,712 B = **2.0 TiB** in 20.77 s → **103 GB/s**

The two agree closely, so the measurement scales linearly and the figure is
trustworthy as a baseline.

## What this implies for a GPU backend

**On this machine the ceiling is roughly 2x.** The M2 Pro's GPU shares the same
~200 GB/s unified memory as the CPU, and the CPU path already sustains ~103 GB/s
— about half of peak. Since the algorithm must stream inputs and outputs through
that same memory, no Metal kernel can exceed ~200 GB/s here. ParPar's NEON code
is genuinely well optimised; this is not headroom left on the table.

**The corrupt-mode floor is lower still.** The 27 s difference between delete
(20.77 s) and corrupt (47.82 s) at 10 GiB is the CPU-bound `FileCheckSummer`
scan, which GPU GF16 acceleration does not touch. Even an infinitely fast GF16
backend would only take corrupt-mode repair from 47.8 s to ~27 s — about 1.8x
end-to-end in the realistic damage case.

**Discrete GPUs are where the win is.** An RTX 4090 (~1000 GB/s) or RX 7900 XTX
(~960 GB/s) against a desktop CPU's ~50–90 GB/s is a 5–10x ratio rather than 2x.
The Metal backend is worth building for macOS coverage and because it proves the
whole integration path, but the headline performance result should be expected
to come from Vulkan on discrete hardware.

## Measured Metal kernel throughput

Raw `gf16_muladd` dispatch, 200 output slices, 1 MiB chunks, best of 5. This is
kernel time only — it excludes host transfer and checksum cost, so end-to-end
repair will be lower.

| Inputs per batch | Outputs per group | GF16 GB/s |
| --- | --- | --- |
| 8 | 8 | 182.5 |
| 16 | 4 | 185.7 |
| 16 | 8 | 186.9 |
| 32 | 4 | **189.3** |

**189 GB/s against the CPU's 103 GB/s — 1.84x**, which is essentially the
predicted ceiling: the kernel is saturating the machine's ~200 GB/s unified
memory. Spread across the tuning grid is only ~5%, confirming the kernel is
bandwidth-bound rather than ALU-bound, so there is little headroom left here.
Outputs-per-group of 4 or more is the sweet spot; below that, each input read
serves too few outputs.

Correctness was established first, against an independent scalar GF(2^16)
oracle (not ParPar's own implementation, which would only prove
self-consistency): 160 shape combinations covering input batches of 1–16,
output counts that do and do not divide the group size, slice lengths from 1 to
257 vectors, and both accumulate and overwrite modes — all bit-exact.

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
