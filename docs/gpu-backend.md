# GPU GF16 backend — state and handoff

Working notes for the `feature/gpu-gf16` branch. Read this before continuing;
it records decisions, measured results, and several traps already paid for.

Branch is cut from `turbo` (not `master` — `parpar/` does not exist there).
Intended for upstreaming to animetosho/ParPar, so `parpar/` changes must stay
free of par2cmdline-specific assumptions and must not add hard build deps.

---

## 1. What exists

Metal and Vulkan backends, both complete and working:

| Piece | Metal | Vulkan |
| --- | --- | --- |
| Compute kernel | `parpar/gf16/gf16_metal.metal` | `parpar/gf16/gf16_vulkan.comp` |
| Backend | `parpar/gf16/controller_metal.{h,mm}` | `parpar/gf16/controller_vulkan.{h,cpp}` |
| Device enumeration | `parpar/gf16/gpu_device_metal.{h,mm}` | `parpar/gf16/gpu_device_vulkan.{h,cpp}` |
| Runtime API loader | — | `parpar/gf16/vulkan_loader.{h,cpp}` |

Shared between them:

| Piece | File |
| --- | --- |
| API-agnostic device layer + factory | `parpar/gf16/gpu_device.{h,cpp}` |
| Correctness test | `src/gpu_test.cpp` |
| Benchmark harness | `tests/bench/parbench.py` |
| Measured results | `tests/bench/BASELINE.md` |

`vulkan_loader` is a fourth file that this document's original plan did not
list. It exists because both the enumeration and the backend need the same
entry-point tables, and duplicating them would guarantee drift.

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

**Always** isolate the phases. The harness does it for you:

```bash
python3 tests/bench/parbench.py --size 10G --repeat 3 --phase-split
```

which is the automated form of:

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
the ratio is 1:1 and there is nothing to win. A discrete GPU changes that.

**This has now been measured on both machines.** Full numbers in
`tests/bench/BASELINE.md`; the 10 GiB delete-mode reconstruct phase:

| Machine | CPU | GPU | Ratio |
| --- | --- | --- | --- |
| M2 Pro, unified | 187 GB/s | 189 GB/s | 0.95x on repair |
| **5800X + RTX 4070 Ti** | **113 GB/s** | **583 GB/s** | **5.15x** |

End-to-end is bounded far harder, because the scan does not accelerate at all:

| Mode | Repair (CPU) | Repair (GPU) | of which scan | End-to-end |
| --- | --- | --- | --- | --- |
| delete | 28.87 s | 12.79 s | ~9.5 s | 2.26x |
| corrupt | 57.74 s | 42.33 s | ~29 s | 1.36x |

Creation is 1.74x. Even an infinitely fast reconstruct would only reach 3.13x
in delete mode and 1.90x in corrupt, so most of the available headroom is
already taken.

Two estimates in earlier revisions of this section were wrong, and the way they
were wrong is worth keeping:

- A "typical desktop CPU ~50–90 GB/s" guess, used to predict 5–7x. The CPU
  actually measures 113 GB/s, because the reconstruct loop re-reads inputs out
  of cache rather than DRAM.
- A "ceiling of ~4.7x", derived as 504 GB/s VRAM ÷ 107 GB/s CPU. **That is not
  a valid ceiling**: neither figure is memory bandwidth, both are multiply-add
  metrics inflated by cache reuse. The measured 5.15x exceeds it, and the
  kernel's own rate of 1037 GB/s exceeds the card's 504 GB/s VRAM bandwidth for
  the same reason.

Do not sanity-check a result against a bandwidth ratio. Check instead that the
**scan time is unchanged between backends** — if it is, the gain is coming from
the phase the backend actually controls.

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

## 4. How the Vulkan backend is built

`GPU_API_VULKAN` sits alongside Metal and follows the same pattern. Files as
listed in §1.

### Build wiring — done

- `parpar/gf16/gpu_device.cpp` — calls `gpu_vulkan_enumerate()` and has the
  `GPU_API_VULKAN` case in `gpu_create_backend()`.
- `parpar/gf16.vcxproj` — sources added, gated on a `ParParVulkan` property
  that auto-detects the Vulkan SDK. Without the SDK the rest of the project
  still builds and Vulkan simply reports no devices. Force it off with
  `-property:ParParVulkan=false`.
- `tests/gpu_test.vcxproj` — same detection, so the test picks the same
  backend the library was built with.

- `configure.ac` — a `--disable-vulkan` block mirroring the Metal one. Probes
  for `vulkan/vulkan.h` (honouring `$VULKAN_SDK`), for `glslangValidator` or
  `glslang` (recent SDKs renamed it; both accept the same flags), and for
  `dlopen`. Defaults to `auto`: absent any of those it disables itself with a
  warning, and only `--enable-vulkan` turns that into an error.
- `Makefile.am` — `libparpar_gf16_vulkan.a` under `if HAS_VULKAN`, mirroring
  `libparpar_gf16_metal.a`, plus a `gf16_vulkan_spv.h` rule. `glslangValidator
  --vn` emits the `uint32_t[]` directly, so unlike the Metal rule there is no
  `od`/`sed` step.

The autotools wiring has now been run, on macOS with `vulkan-headers` and
`glslang` from Homebrew:

```bash
./configure --enable-vulkan CPPFLAGS=-I/opt/homebrew/include
make && make check
make distcheck DISTCHECK_CONFIGURE_FLAGS="--enable-vulkan CPPFLAGS=-I/opt/homebrew/include"
```

`glslangValidator` compiled the shader to `gf16_vulkan_spv.h`,
`libparpar_gf16_vulkan.a` built, the suite stayed green, and the tarball built
the backend from its own sources — which is the check that matters, since a
plain `distcheck` auto-detects, finds no headers, silently builds CPU-only and
passes regardless. That is the same class of bug as the missing
`gf16_affine_avx10.h`.

`.github/workflows/build-check-linux.yml` now installs `libvulkan-dev` and
`glslang-tools` and configures with `--enable-vulkan`, so every push exercises
the enabled path rather than the disabled one. It passes
`DISTCHECK_CONFIGURE_FLAGS=--enable-vulkan` for the same reason as above. No
GPU or driver is needed on the runner: the backend resolves Vulkan through
`dlopen`, so headers and a shader compiler are enough to build and link it.

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

6. **Parallelising staging was not the win it looked like** — *on unified
   memory*. It took 10 GiB repair from 22.72 s to 22.34 s. The GPU was already
   91% utilised; there was no host-side gap to close. Instrument before
   optimising.

   **This does not carry over to discrete hardware.** There staging is a real
   PCIe transfer rather than a memcpy into memory the GPU already sees. On the
   4070 Ti a 10 GiB repair spends 0.92 s staging against 2.07 s of kernel, so
   the GPU is busy only ~63% of the GF16 phase. The lesson survives, the
   conclusion does not: instrument on the machine in front of you.

---

## 6. Windows build

Windows uses MSVC project files, **not** autotools:

```bash
msbuild -property:PlatformToolset=v145 -property:Configuration=Release -property:Platform=x64 par2cmdline.sln
msbuild -property:PlatformToolset=v145 -property:Configuration=UnitTests-Release -property:Platform=x64 par2cmdline.sln
```

```bash
.\tests\run_tests.ps1 -Par2Binary ".\x64\Release\par2.exe"
.\tests\unit_tests.ps1
```

**The `PlatformToolset` override is not optional.** `parpar/gf16.vcxproj` and
`parpar/hasher.vcxproj` are upstream ParPar files and pin `v143`; every
par2cmdline project pins `v145`. Whichever toolset the machine actually has,
both halves must be forced onto it or the build dies with MSB8020 on the two
parpar projects only. Do **not** "fix" this by editing the pin inside
`parpar/`: those files go upstream. `v145` compiles the SIMD sources fine; the
`-mavx2`-style `AdditionalOptions` in those projects are all conditioned on
`ClangCL` and are simply skipped.

On the benchmark machine (VS 18 Community) the registered toolset is `v145`.
Note that **installing the v143 compiler is not enough to drop the override** —
the compiler binaries land in `VC\Tools\MSVC\14.44.x`, but `PlatformToolset`
resolves against

```
MSBuild\Microsoft\VC\v180\Platforms\x64\PlatformToolsets\
```

which lists only `v145` unless the VS 2022 build tools register themselves
there. Check that directory before concluding a toolset is available; the
presence of a `VC\Tools\MSVC` directory says nothing about it.

Already done for you:

- `gpu_device.cpp/.h` added to `parpar/gf16.vcxproj`, and `PARPAR_GPU_SUPPORT`
  added to its preprocessor definitions (all seven `gf16_cksum_*.c` sources
  were already listed, so this links).
- `tests/gpu_test.vcxproj` exists, is in `par2cmdline.sln` under the `tests`
  folder, and is listed in `tests/build_unit_tests.ps1` and
  `tests/unit_tests.ps1`. It builds only in the `UnitTests-*` configurations,
  matching the other test projects. It compiles with
  `PARPAR_INVERT_SUPPORT;PARPAR_SLIM_GF16;PARPAR_GPU_SUPPORT` so that the
  headers it shares with `parpar/gf16.vcxproj` agree — `libpar2.vcxproj` sets
  none of these, which is safe there only because `PAR2ProcCPU` holds
  `Galois16Mul` behind a pointer. **Add `PARPAR_VULKAN_SUPPORT` there too when
  the backend lands**, and widen the `#ifdef PARPAR_METAL_SUPPORT` guards in
  `src/gpu_test.cpp` — as shipped it compiles and reports
  `SKIP: built without GPU backend support` on Windows.

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
#    --phase-split is required for any GB/s figure: without it the scan is
#    charged to the compute. See §2.
python3 tests/bench/parbench.py --size 10G --repeat 3 --phase-split \
        --backend cpu --backend gpu
```

On Windows, steps 1 and 2 are:

```bash
.\x64\Release\gpu_test.exe
.\tests\unit_tests.ps1
```

The harness verifies every repair by SHA-256 against the original corpus and
reports a fast-but-wrong result as a failure. It runs both damage modes
separately — `delete` isolates GF16 throughput, `corrupt` includes the
CPU-bound scan. Report both; either alone misleads.

---

## 8. Open items

- ~~Vulkan backend.~~ Done — see §1. Verified against the checklist in §7:
  `gpu_test` bit-exact across all 20 shapes and both coefficient paths, the
  full suite green, multi-chunk `-m4` repair correct, and both
  create/repair backend combinations hash-identical.
- ~~Verify the autotools build.~~ Done — §4. Both paths build, and Linux CI
  now covers the enabled one on every push. Still unproven: the **Metal**
  autotools path in CI, because GitHub's macOS runners may not ship the Metal
  shader compiler (recent Xcode makes it a separate download). The macOS
  workflow now reports whether it is present, so the next run will say.
- **Staging is now the bottleneck worth attacking.** On the 4070 Ti a 10 GiB
  repair is 2.07 s of kernel against 0.92 s of PCIe staging, leaving the GPU
  busy ~63% of the GF16 phase — see `BASELINE.md`. Note this contradicts
  gotcha #6, which was measured on unified memory where staging was a memcpy.
- ~~`tests/gpu_test.vcxproj` for the Windows build.~~ Done — see §6.
- ~~Windows CPU baseline.~~ Done — see §3 and `tests/bench/BASELINE.md`.
- ~~A `--phase-split` mode for `parbench.py`.~~ Done — the scan-subtraction in
  §2 is now in the harness rather than done by hand.
- Hybrid CPU+GPU split. `PAR2Proc` already supports it —
  `init()` takes `{backend, offset, size}` entries and
  `setCurrentSliceSize(size, sizeAlloc)` sets the split. Offsets must be
  **2-byte aligned** (`controller.cpp:77`). Only worth doing where the GPU is
  substantially faster, i.e. discrete — on unified memory it cannot help.
- `IGPUDevice` abstraction. Deliberately *not* written yet: designing it
  against a single backend is guesswork. Extract it once Vulkan exists and the
  common shape is visible.
