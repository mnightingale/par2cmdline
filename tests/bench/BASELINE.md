# GF16 throughput baselines

Measured CPU and GPU numbers for the GF16 path. Re-measure with `parbench.py`
on the same machine when comparing.

Two machines are recorded. **Never compare across them** — the whole point of
[the Vulkan work](../../docs/gpu-backend.md) is that the CPU:GPU ratio depends
entirely on the platform, so each machine needs its own CPU baseline measured
alongside its GPU result.

- [Machine A — Apple M2 Pro](#machine-a--apple-m2-pro-macos) (Metal, unified memory)
- [Machine B — Ryzen 7 5800X + RTX 4070 Ti](#machine-b--ryzen-7-5800x--rtx-4070-ti-windows) (Vulkan, discrete)

The headline: on unified memory the GPU is **0.95x** on repair, on a discrete
card it is **5.15x** on the reconstruct phase. That difference is the entire
argument for the Vulkan backend, and neither number generalises to the other
platform.

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

CPU against the Vulkan backend, measured back to back in one run so the
speedup columns compare like with like.

## Machine

| | |
| --- | --- |
| CPU | AMD Ryzen 7 5800X, 8 cores / 16 threads, 3.8 GHz base |
| GPU | NVIDIA GeForce RTX 4070 Ti, 12 GB GDDR6X (~504 GB/s), driver 32.0.16.1088 |
| Memory | 32 GiB DDR4-3600, dual channel (~57.6 GB/s theoretical) |
| Storage | Samsung 980 PRO 2 TB NVMe |
| OS | Windows 11 Pro 26200 |
| Build | `feature/gpu-gf16`, MSVC `v145`, `Release\|x64`, Vulkan SDK 1.4.357.0 |

## Results

Settings identical to Machine A: 10 GiB over 20 files, `-b2000`, `-r15`, 10%
damage, `-t` default (16), **warm page cache**, 3 repetitions. Block size
5,368,712 B; 2000 source and 300 recovery blocks; 200 blocks reconstructed.

Each repetition times `par2 verify -q` and `par2 repair -q` against the *same*
damage state, so the subtraction is valid. Scan and reconstruct are medians of
3; total is the best of 3.

| Mode | Backend | Scan | Reconstruct | Total | Speedup |
| --- | --- | --- | --- | --- | --- |
| delete | CPU | 9.90 s | 18.97 s | 28.87 s | — |
| delete | **GPU** | 9.22 s | **3.68 s** | **12.79 s** | **5.15x** reconstruct, 2.26x total |
| corrupt | CPU | 28.18 s | 34.13 s | 57.74 s | — |
| corrupt | **GPU** | 30.39 s | **14.81 s** | **42.33 s** | 2.31x reconstruct, 1.36x total |

Creation: 31.31 s CPU, 17.95 s GPU — **1.74x**.

Per-run reconstruct, showing the spread. The GPU is markedly more consistent
than the CPU, which is what a dedicated processor with no other tenants looks
like:

| Mode | Backend | run 1 | run 2 | run 3 | median |
| --- | --- | --- | --- | --- | --- |
| delete | CPU | 18.97 s | 22.62 s | 18.09 s | 18.97 s |
| delete | GPU | 3.57 s | 3.68 s | 3.70 s | 3.68 s |
| corrupt | CPU | 29.56 s | 34.93 s | 34.13 s | 34.13 s |
| corrupt | GPU | 14.95 s | 14.75 s | 14.81 s | 14.81 s |

**The scan does not accelerate, and the measurement confirms it**: 9.90 s vs
9.22 s in delete mode, 28.18 s vs 30.39 s in corrupt. Any apparent scan
difference is run-to-run noise. This is the whole reason the phase split
exists.

## GF16 throughput

2000 × 200 × 5,368,712 = 2147 GB of multiply-add, so:

- **CPU: 113 GB/s** (2147 GB / 18.97 s median, delete mode)
- **GPU: 583 GB/s** (2147 GB / 3.68 s median, delete mode)

**Use the delete-mode figures.** In corrupt mode `repair` does substantial work
that `verify` does not — rewriting repaired blocks back into the damaged files
in place — so the difference is not purely GF16. That work does not accelerate,
which is why the corrupt-mode ratio (2.31x) is so much lower than delete's
5.15x rather than because the kernel is slower. It shows up directly: GPU
reconstruct is 3.68 s in delete mode and 14.81 s in corrupt, and the ~11 s
difference is file rewriting, not arithmetic.

### Where the GPU time actually goes

From `PARPAR_GPU_STATS=1` on a 10 GiB delete-mode repair, typical of three runs:

```
[GPU STATS] wall 3.40s | gpu 2.05s (copy 0.44s + kernel 1.61s) over 125 dispatches
            | host-stage 0.84s cpu (10.0 GiB) | lut+encode 0.06s | readback 0.20s
```

Read those fields carefully; an earlier revision of this file misread two of
them and drew the wrong conclusion:

- **`gpu` is the whole command buffer**, copies *and* dispatch, because both are
  recorded into it and the timestamps bracket the lot. It is now split. The
  kernel alone is 1.61 s — **1333 GB/s** — and the PCIe copies are 0.44 s.
- **`host-stage` is not the PCIe transfer.** It is the host-side `copy_cksum`
  memcpy into the mapped staging buffer, and it is summed across the transfer
  threads, so it is CPU time and an aggregate. It is not comparable to `wall`,
  and dividing 10 GiB by it does not give a transfer rate.

`wall` (3.40 s) is the backend's own view of the GF16 phase, and is not the
harness's 3.68 s reconstruct, which is `repair − verify` and also carries
writing the recovered files.

So the actual shape of the phase is:

| | | share of wall |
| --- | --- | --- |
| kernel | 1.61 s | 47% |
| PCIe copies | 0.44 s | 13% |
| readback | 0.20 s | 6% |
| everything else | ~1.15 s | 34% |

**The copies are 21% of GPU-busy time and 13% of the phase** — real, but not
the dominant cost, and that caps what moving them to a transfer queue can win.
The larger share is the 34% during which the GPU is idle, and the staging-depth
sweep in `docs/gpu-backend.md` §8 rules out "too few batches in flight" as the
cause, so that time is most likely spent waiting for par2 to hand over input
slices rather than anything the backend controls.

## Vulkan against OpenCL, in the same binary

ParPar's OpenCL backend is now built into par2cmdline as well
(`--gpu=<id>` selects it; `--list-gpus` reports the same card twice, once per
API). Both backends therefore run in the same host pipeline, on the same
corpus, against the same damage — which removes the confounds that make the
ParPar-hosted comparison further down hard to read.

10 GiB delete-mode repair, 2000 source, 200 reconstructed, two interleaved
rounds, best of each:

| backend | reconstruct | GF16 | vs CPU |
| --- | --- | --- | --- |
| CPU | 19.38 s | 110.8 GB/s | 1.00x |
| **Vulkan** | **3.86 s** | **556.1 GB/s** | **5.02x** |
| OpenCL | 6.14 s | 349.4 GB/s | 3.16x |

Both rounds agree (Vulkan 4.46/3.86, OpenCL 6.30/6.14) and the scan is
unchanged across all three, as it must be.

**Vulkan is ~1.59x faster than OpenCL here**, and the comparison is like for
like in the way that matters: OpenCL auto-selects its `Lookup` kernel — par2
reports `Multiply method: NVIDIA GeForce RTX 4070 Ti (OpenCL Lookup)`, the same
method ParPar picks — so this is two lookup-table GPU kernels, not two
different algorithms.

That is worth stating plainly because it inverts the expectation that prompted
the comparison: the concern was that `gf16_vulkan.comp` uses no permutation
operations and must therefore be leaving performance on the table. Against the
mature OpenCL implementation on the same hardware, it is ahead.

**Caveat: OpenCL here is not tuned.** It is constructed with `GF16OCL_AUTO` and
only the input batch size passed through; `targetIters` and `targetGrouping`
are left at their defaults, where ParPar's own CLI exposes them. Its reported
`max allocation` is also lower than Vulkan's (3070 MB against 4095 MB), which
may force different chunking. So read this as "OpenCL as auto-configured"
rather than "the best OpenCL can do".

## Cross-check against OpenCL (ParPar 0.4.5)

Upstream ParPar has its own GPU backend, reachable with `--opencl-process
100%`. Running it on the same card and corpus is the cheapest available check
that the Vulkan speedup is real rather than a measurement artefact, and it
answers a specific review question: the kernel uses no permutation/shuffle
operations, so is it leaving a lot on the table?

ParPar is a creator, not a repairer, so the comparable operation is creation.
Same corpus, 2000 input slices, 300 recovery slices, ~5.37 MB each — 3221 GB
of multiply-add:

| | metric | time | implied rate |
| --- | --- | --- | --- |
| par2 CPU (`--gpu=off`) | total wall | 31.07 s | 104 GB/s |
| par2 Vulkan (`--gpu=auto`) | total wall | 18.43 s | — |
| par2 Vulkan | GF16 phase wall | 18.18 s | 177 GB/s |
| par2 Vulkan | GPU busy | 4.61 s (copy 0.41 + kernel 4.20) | 767 GB/s kernel |
| ParPar CPU (Xor-JIT AVX2, 16 threads) | Processing time | 65.02 s | 50 GB/s |
| ParPar OpenCL (Lookup) | Processing time | 10.46 s | 308 GB/s |

**ParPar's OpenCL kernel is also a lookup-table method** — it reports
`Multiply method : Lookup, split into 8 * 8192 B workgroups`. The reference
GPU implementation made the same choice as `gf16_vulkan.comp`, so the absence
of shuffle operations is not an oversight peculiar to this backend. It is
consistent with the traffic estimate above: on a discrete GPU the multiply is
not the constraint, so the trick that dominates CPU SIMD has much less to give.

**The kernel is not what limits creation here.** It completes the whole
workload in 4.20 s of GPU time, yet the GF16 phase takes 18.18 s — the GPU is
busy **25%** of it, against 60% during repair. Creation folds reading and
hashing all 10 GiB into the same phase, and ParPar finishes the entire create
in 10.67 s wall, less than our GF16 phase alone. The gap is par2cmdline's
create pipeline overlapping I/O and hashing with compute, not the backend.

### Reading these numbers

**Trust the within-tool ratios, not the cross-tool absolutes.** ParPar's CPU
path measures 50 GB/s where par2's measures 104 GB/s on shared GF16 code, so
"Processing time" and the wall clocks here plainly do not span the same work.
Part of that is ParPar reporting `Input pass(es): 2` — which `-m 3G` did not
change, and which costs about 5% (11.02 s against 10.46 s) — but not a factor
of two. The discrepancy is unexplained.

On within-tool ratios: ParPar 6.2x (65.02 → 10.46), par2 1.69x end to end for
creation (31.07 → 18.43). The latter is diluted by the 75% idle, not by kernel
speed; the repair-phase 5.15x is the figure that reflects the backend itself.

Reproduce with the prebuilt Windows ParPar from its GitHub releases:

```bash
parpar -s 2000 -r 300 --opencl-process 100% -o out.par2 <corpus files>
parpar --opencl-list          # confirms the device is actually found
```

## Corrections to earlier estimates in this file

Two claims recorded here before the backend existed did not survive contact
with the measurement, and are worth stating explicitly rather than quietly
editing away:

**The "~4.7x ceiling" was not a real ceiling.** It came from dividing the GPU's
504 GB/s VRAM bandwidth by the CPU's 107 GB/s measured GF16 rate. That is not a
valid comparison: neither number is memory bandwidth. Both are multiply-add
metrics inflated by cache reuse — the same reason this CPU appears to exceed its
57.6 GB/s DRAM bandwidth. The measured 5.15x beats the supposed ceiling, and the
GPU's 1037 GB/s kernel rate likewise exceeds its own 504 GB/s VRAM bandwidth,
for exactly the same reason: the kernel re-reads each input from shared memory
and L2 across the output group rather than from VRAM per access. That is the
design working as intended.

**"Treat any end-to-end result much above 2x as suspect" was wrong.** The
measured 2.26x is legitimate. The correct check is not a threshold on the total
but whether the scan is unchanged between backends — it is, to within noise, so
the gain is coming from the phase the backend actually controls.

What does still hold is the shape of the constraint: the scan is untouched, so
end-to-end is bounded by it. Even an infinitely fast reconstruct leaves
delete-mode repair at the 9.22 s scan, a hard ceiling of 3.13x, and corrupt mode
at 1.90x. The backend has already captured most of the delete-mode headroom.

## Reproducing

```bash
python3 tests/bench/parbench.py --par2 ./x64/Release/par2.exe \
        --size 10G --repeat 3 --phase-split --backend cpu --backend gpu
```

`--phase-split` is what produces the scan/reconstruct columns; without it the
harness reports only total repair wall time, which includes the scan and must
not be divided by the GF16 work.
