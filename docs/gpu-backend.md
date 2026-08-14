# GPU GF16 backend — state and handoff

Working notes for the `feature/gpu-gf16` branch. Read this before continuing;
it records decisions, measured results, and several traps already paid for.

Branch is cut from `turbo` (not `master` — `parpar/` does not exist there).
Intended for upstreaming to animetosho/ParPar, so `parpar/` changes must stay
free of par2cmdline-specific assumptions and must not add hard build deps.

---

## 1. What exists

Metal backend, complete and working:

| Piece | File |
| --- | --- |
| Compute kernel | `parpar/gf16/gf16_metal.metal` |
| Backend | `parpar/gf16/controller_metal.{h,mm}` |
| Device enumeration | `parpar/gf16/gpu_device_metal.{h,mm}` |
| API-agnostic device layer + factory | `parpar/gf16/gpu_device.{h,cpp}` |
| Correctness test | `src/gpu_test.cpp` |
| Benchmark harness | `tests/bench/parbench.py` |
| Measured results | `tests/bench/BASELINE.md` |

CLI: `--gpu=auto|off|<id>` and `--list-gpus`. Default is `auto`, which falls
back to the CPU silently when no usable device exists. A GPU is never required.

---

## 2. The result so far, and the mistake that preceded it

**On the M2 Pro, CPU and GPU GF16 throughput are within 1% of each other**
(187 vs 189 GB/s), so the GPU is ~0.95x on repair once staging and readback are
added. Only creation improves (~1.20x). Both saturate the same ~200 GB/s
unified memory; the NEON path was already at ~94% of it.

**The trap:** an earlier revision claimed the CPU did 103 GB/s and the kernel
was 1.84x faster. That was wrong. `par2 repair` is **scan-then-reconstruct**,
and at 10 GiB the scan is 9.72 s of the 21.17 s total. Dividing *total* repair
time by GF16 work charges the scan to the compute and understates CPU
throughput by ~1.8x.

**Always** isolate the phases:

```bash
par2 verify -q bench.par2          # scan alone
par2 repair -q bench.par2          # scan + reconstruct
# GF16 time = repair - verify
```

or use the GPU-side counters:

```bash
PARPAR_GPU_STATS=1 par2 repair --gpu=auto -q bench.par2
# [GPU STATS] wall 12.50s | gpu 11.36s over 125 dispatches | stage 2.68s (10.0 GiB) | ...
```

`gpu` there is real GPU busy time, not wall time — the number to compare
against the CPU's reconstruct phase.

GF16 work for a repair is:

```
bytes = source_blocks × blocks_reconstructed × block_size
```

`par2 create` prints all three. At 10 GiB / `-b2000` / 10% damage that is
2000 × 200 × 5,368,712 = **2147 GB**.

---

## 3. Why Vulkan on a discrete GPU is the real test

The Apple result is about *bandwidth ratio*, not about GPU versus CPU compute.
Apple Silicon is the worst case: one memory pool shared by both processors, so
the ratio is 1:1 and there is nothing to win. A discrete GPU changes that:

| | Bandwidth |
| --- | --- |
| M2 Pro, shared | ~200 GB/s |
| **RTX 4070 Ti** | **~504 GB/s** |
| Typical desktop CPU | ~50–90 GB/s |

So expect the reconstruct phase to be roughly **5–7x** faster than that
machine's CPU — but *measure the CPU on that machine first*, do not reuse any
number from this document. The scan phase does not accelerate at all, so
end-to-end gains will be smaller, and smaller again in `corrupt` mode.

### What differs from Apple, and matters

1. **Staging crosses PCIe.** On unified memory, staging was a memcpy into a
   shared buffer. On a 4070 Ti every input slice must be transferred. Total
   PCIe traffic is inputs-in + outputs-back (~11 GB for a 10 GiB repair, ~0.5 s
   at PCIe 4.0), which is fine — but only because the kernel re-reads inputs
   from *device* memory, not host memory. Keep it that way.
2. **VRAM is finite.** 12 GB on a 4070 Ti. The output buffer is
   `numOutputs × sliceSizeAligned`, and par2's `-m` can make `chunksize` large.
   `reallocBuffers()` currently checks only max *allocation* size, which was
   sufficient on unified memory. **Vulkan must also check total device memory**
   and return `nullptr` so the caller falls back to the CPU rather than failing
   or thrashing. See `GPUDeviceInfo::memory`.
3. **Host-visible memory is slow.** Prefer a device-local buffer plus a
   host-visible staging ring, rather than the single shared buffer the Metal
   path uses.

---

## 4. What to implement

Add `GPU_API_VULKAN` support alongside Metal. The pattern is already in place;
follow it rather than inventing a new one.

### Files to create

| File | Mirrors |
| --- | --- |
| `parpar/gf16/gpu_device_vulkan.{h,cpp}` | `gpu_device_metal.{h,mm}` |
| `parpar/gf16/controller_vulkan.{h,cpp}` | `controller_metal.{h,mm}` |
| `parpar/gf16/gf16_vulkan.comp` | `gf16_metal.metal` |

### Files to edit

- `parpar/gf16/gpu_device.cpp` — call `gpu_vulkan_enumerate()` and add a
  `GPU_API_VULKAN` case to `gpu_create_backend()`. Both sites already have the
  `#ifdef PARPAR_VULKAN_SUPPORT` shape to copy from Metal.
- `configure.ac` — a `--disable-vulkan` block mirroring the Metal one.
- `Makefile.am` — a conditional archive mirroring `libparpar_gf16_metal.a`.
- `parpar/gf16.vcxproj` — add the new sources. **This is the build that
  matters on Windows**; autotools is not used there.

### The kernel

Port `gf16_metal.metal` to GLSL compute. It is short and the algorithm is
deliberately simple:

- GF(2^16), polynomial `0x1100b`.
- Multiplication uses **nibble lookup tables**, not log/antilog. Multiplication
  is linear over GF(2) in its second operand, so for a fixed coefficient `c` a
  value `v` splits into four nibbles and `c*v = XOR_j (c * (nibble_j << 4j))`.
  That is four 16-entry tables per coefficient — 128 bytes — versus 128 KiB for
  a full antilog table. Chosen because Apple exposes only 32 KiB of threadgroup
  memory; on NVIDIA you have 48–100 KiB of shared memory per block, so a
  log-table variant *might* win. Measure before switching.
- Tables are built **host-side** (`gf16_build_lut` in `controller_metal.mm`)
  and uploaded; building them on the GPU would repeat the work per workgroup.
- The kernel reads each input once per output group and applies it to every
  output in that group, which is what makes it compute- rather than
  bandwidth-bound. Tune `outputsPerGroup` upward on NVIDIA if shared memory
  allows — it directly sets arithmetic intensity.

Compile GLSL → SPIR-V with `glslangValidator` at build time and embed the
result as a `uint32_t[]`, exactly as the Metal build does for `.metallib`
(`Makefile.am`, target `parpar/gf16/gf16_metal_lib.h`). **Do not check the
binary into git** — the source and the shipped kernel must not be able to drift.

Load Vulkan with `dlopen`/`LoadLibrary` and resolve entry points manually, so
there is no link-time dependency and the binary still runs without a driver.

---

## 5. Gotchas already paid for

These cost real debugging time. Do not rediscover them.

1. **`outputExponents` defaults to 1, not 0.** `PAR2ProcCPU::setRecoverySlices`
   does `resize(numSlices, 1)`. Value-initialising to 0 makes
   `gfmat_coeff_from_log(log, 0)` return 1 for every output, so every output
   slice comes out identical. This made all 10 initial `gpu_test` shapes fail
   while the raw kernel was already correct.

2. **The accumulate flag must come from `processingAdd`, read inside
   `run_kernel`.** `ProcessData` runs inside the `while (blockoffset <
   blocksize)` chunk loop and calls `parpar.discardOutput()` every chunk, which
   resets `processingAdd`. Tracking "have I written the output yet" in a
   separate member silently corrupts any **multi-chunk** repair — which only
   happens when memory-constrained, so the default-settings tests all pass.
   Copy `controller_cpu.cpp:545`:
   ```cpp
   const bool accumulate = processingAdd;
   processingAdd = true;
   ```
   Regression test: `par2 repair -m4 ...` forces `chunksize < blocksize`.

3. **Dispatch must be serialised.** With parallel staging threads, `run_kernel`
   can be entered concurrently. The accumulate decision and the command
   submission must happen under one lock, or a batch can accumulate into the
   output and then a later "first" dispatch overwrites it, silently dropping
   that batch. See `dispatchMutex`.

4. **Batch completion is a token protocol.** Staging is spread across threads,
   so the batch is dispatched by whichever worker finishes last. Each area
   holds `pending` = outstanding slices + one "batch open" token released when
   the batch closes; otherwise a fast first slice dispatches a batch whose
   remaining slices have not been queued yet. `submitCount` is published with
   release / read with acquire.

5. **Drain before releasing buffers.** `MessageThread::end()` signals the
   worker without joining it. `_deinit()` sends a barrier request per thread
   and waits, then waits on an empty command buffer for the GPU, before freeing
   anything.

6. **Parallelising staging was not the win it looked like.** It took 10 GiB
   repair from 22.72 s to 22.34 s. The GPU was already 91% utilised; there was
   no host-side gap to close. Instrument before optimising.

---

## 6. Windows build

Windows uses MSVC project files, **not** autotools:

```bash
msbuild -property:PlatformToolset=ClangCL -property:Configuration=Release -property:Platform=x64 par2cmdline.sln
msbuild -property:Configuration=UnitTests-Release -property:Platform=x64 par2cmdline.sln
```

```bash
.\tests\run_tests.ps1 -Par2Binary ".\x64\Release\par2.exe"
```

Already done for you: `gpu_device.cpp/.h` added to `parpar/gf16.vcxproj`, and
`PARPAR_GPU_SUPPORT` added to its preprocessor definitions (all seven
`gf16_cksum_*.c` sources were already listed, so this links).

**Still needed:** `src/gpu_test.cpp` has no `tests/gpu_test.vcxproj` and is not
in `par2cmdline.sln`, so the correctness test will not run on Windows until one
is added. Copy `tests/galois_test.vcxproj` as the template. Do this early — it
is the test that catches GF16 errors.

---

## 7. Verification checklist

In order. Do not skip to benchmarking.

```bash
# 1. Correctness against the CPU backend, bit-exact.
#    Covers both coefficient paths and shapes that stress batching edges.
./tests/gpu_test

# 2. Full suite still green, and green on a machine with no GPU
#    (gpu_test skips rather than fails).
make check

# 3. Multi-chunk repair (catches gotcha #2).
par2 repair --gpu=auto -m4 -v <file>.par2

# 4. Backends interchangeable, both directions.
par2 create --gpu=off  ... && par2 repair --gpu=auto ...   # verify hashes
par2 create --gpu=auto ... && par2 repair --gpu=off  ...   # verify hashes

# 5. Only now, measure. CPU first, on this machine.
python3 tests/bench/parbench.py --size 10G --repeat 3 --backend cpu --backend gpu
```

The harness verifies every repair by SHA-256 against the original corpus and
reports a fast-but-wrong result as a failure. It runs both damage modes
separately — `delete` isolates GF16 throughput, `corrupt` includes the
CPU-bound scan. Report both; either alone misleads.

---

## 8. Open items

- Vulkan backend (this handoff).
- `tests/gpu_test.vcxproj` for the Windows build.
- Hybrid CPU+GPU split. `PAR2Proc` already supports it —
  `init()` takes `{backend, offset, size}` entries and
  `setCurrentSliceSize(size, sizeAlloc)` sets the split. Offsets must be
  **2-byte aligned** (`controller.cpp:77`). Only worth doing where the GPU is
  substantially faster, i.e. discrete — on unified memory it cannot help.
- `IGPUDevice` abstraction. Deliberately *not* written yet: designing it
  against a single backend is guesswork. Extract it once Vulkan exists and the
  common shape is visible.
