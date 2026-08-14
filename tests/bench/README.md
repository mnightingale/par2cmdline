# PAR2 repair benchmark harness

`parbench.py` measures PAR2 create and repair wall-clock time over a generated
corpus, so the GF16 backends (CPU vs GPU) can be compared on identical work.

It is **not** part of `make check` — the corpora are gigabytes and the runs take
minutes.

## Why repair is the interesting case

Reconstructing `N` missing blocks from `M` input blocks costs `M × N × blocksize`
GF16 multiply-adds. For a 10 GB corpus at the default 2000 blocks with 10%
damage that is roughly 2 TB of multiply-add against ~11 GB of disk I/O — an
arithmetic intensity high enough that repair is compute-bound, which is what
makes it a GPU-shaped problem in the first place.

Block count is therefore *the* dominant parameter, and results are meaningless
without it. Work scales as `blockcount × corpussize` for a fixed damage
percentage, so a run at `-b4000` does twice the GF16 work of one at `-b2000`
over the same bytes.

## Damage modes

Both are run by default and reported separately, because they exercise
different code paths and each one alone gives a misleading picture:

| Mode | What it does | What it measures |
| --- | --- | --- |
| `delete` | Removes whole files | Skips the sliding-window scan, so it isolates GF16 reconstruction throughput. **Overstates** the real-world win. |
| `corrupt` | Overwrites whole blocks in place | Realistic PAR2 damage, but repair first runs the CPU-bound `FileCheckSummer` scan across the entire corpus, which can dominate and **mask** the GF16 improvement. |

Corruption is applied at block granularity by sampling globally across every
block in the corpus. Scattering individual bytes would be wrong — PAR2 repairs
per block, so thinly-spread byte corruption destroys far more blocks than the
byte count implies.

## Correctness

Every repair is verified by comparing SHA-256 of each file against the manifest
recorded at corpus generation. A run that repairs quickly but incorrectly is
reported as a failure, not a fast result.

## Page cache

By default the page cache is left warm and the report says so. A 10 GB corpus
fits in RAM on most development machines, so timings include cache effects.

Since the GPU win is in compute rather than I/O, warm-cache measurement
actually isolates the thing under test — but it must be labelled honestly.

`--drop-caches` will drop it before each repair (`purge` on macOS,
`vm.drop_caches` on Linux; both typically need root). If the command fails the
harness **aborts** rather than reporting warm-cache numbers as cold.

## Usage

```bash
python3 tests/bench/parbench.py --size 10G --repeat 3
```

Comparing backends once the `--gpu` flag exists:

```bash
python3 tests/bench/parbench.py --size 10G --repeat 3 --backend cpu --backend gpu
```

Requesting a GPU backend from a binary built without `--gpu` support is a hard
error — the harness will not quietly measure the CPU path and label it as GPU.

Useful options:

| Option | Meaning |
| --- | --- |
| `--size` | Total corpus size (`1G`, `10G`, …), default `1G` |
| `--files` | Number of files, default 20. Sets the granularity of `delete` damage. |
| `--block-count` | par2 `-b`, default 2000 |
| `--redundancy` | par2 `-r` percent, default 15 |
| `--damage` | Percent damage, default 10. Must be below redundancy. |
| `--mode` | `delete`, `corrupt` or `both` (default) |
| `--repeat` | Repair repetitions, default 1 |
| `--threads` | par2 `-t`, default par2's own choice |
| `--json` | Write full results for later comparison |

The corpus is generated from a fixed seed and cached in `--workdir`, keyed by
size/file-count/seed, so repeated invocations reuse it. Damaged files are
regenerated from the seed rather than restored from a copy, which avoids
keeping a second multi-gigabyte pristine tree on disk.
