#!/usr/bin/env python3
"""PAR2 create/repair benchmark harness.

Measures wall-clock time for PAR2 creation and repair over a generated corpus,
so the GF16 backends (CPU vs GPU) can be compared on the same workload.

The point of interest is repair: reconstructing N missing blocks from M input
blocks costs M*N*blocksize GF16 multiply-adds, which dominates wall-clock time
once the corpus is large enough. Two damage modes are reported separately
because they exercise very different code paths:

  delete   whole files are removed. Repair skips the sliding-window scan and
           goes straight to Reed-Solomon reconstruction, isolating GF16
           throughput. Overstates the real-world win.

  corrupt  whole blocks are overwritten in place. Realistic, but repair first
           runs the CPU-bound FileCheckSummer scan over the entire corpus,
           which can dominate and mask the GF16 improvement.

Run both. Reporting only 'delete' would flatter the GPU; reporting only
'corrupt' would hide the speedup that is actually there.

--phase-split additionally times `par2 verify` against the same damage state
and subtracts it, isolating the reconstruct phase from the scan that precedes
it. Use it for any throughput claim: dividing *total* repair time by the GF16
work charges the scan to the compute, which is how an earlier revision of
BASELINE.md came to report a CPU baseline 1.8x too slow and a GPU speedup that
did not exist.

Not part of `make check` -- the corpora are far too large.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

CHUNK = 16 * 1024 * 1024

# par2 exit codes we care about (src/libpar2.h). `verify` on a damaged corpus
# is *expected* to return eRepairPossible; treating that as failure would make
# --phase-split unusable, and treating eRepairNotPossible as success would let
# us time a repair that never happened.
PAR2_SUCCESS = 0
PAR2_REPAIR_POSSIBLE = 1

# Maps a backend name to the extra par2 arguments that select it. The --gpu
# flag does not exist yet (it arrives with the GPU backend); until then only
# 'cpu' is runnable, and the others fail loudly rather than silently measuring
# the CPU path and labelling it as GPU.
BACKEND_ARGS = {
    "cpu": ["--gpu=off"],
    "gpu": ["--gpu=auto"],
    "hybrid": ["--gpu=auto", "--gpu-split=auto"],
}


def parse_size(s):
    """Parse '10G', '512M', '1024' into bytes."""
    s = s.strip().upper()
    mult = 1
    if s and s[-1] in "KMGT":
        mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[s[-1]]
        s = s[:-1]
    return int(float(s) * mult)


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def file_seed(seed, index):
    """Per-file seed, so a single damaged file can be regenerated without
    rebuilding the whole corpus."""
    return hashlib.sha256(f"{seed}:{index}".encode()).digest()


def write_file(path, nbytes, seed, index):
    """Write deterministic pseudorandom content, returning its sha256.

    Content must be genuinely varied: repetitive data produces many blocks
    sharing a CRC, which sends VerificationHashTable down its pathological
    collision path and would distort the scan timings in corrupt mode.
    """
    rng = random.Random(file_seed(seed, index))
    h = hashlib.sha256()
    remaining = nbytes
    with open(path, "wb") as f:
        while remaining > 0:
            n = min(CHUNK, remaining)
            buf = rng.randbytes(n)
            f.write(buf)
            h.update(buf)
            remaining -= n
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def ensure_corpus(cfg):
    """Create the corpus if absent or if its parameters changed. Returns a
    manifest mapping filename -> (size, sha256)."""
    corpus = cfg.corpus_dir
    manifest_path = os.path.join(cfg.workdir, "manifest.json")
    params = {
        "size": cfg.size,
        "files": cfg.files,
        "seed": cfg.seed,
    }

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            saved = json.load(f)
        if saved.get("params") == params:
            missing = [n for n in saved["files"] if not os.path.exists(os.path.join(corpus, n))]
            if not missing:
                print(f"Reusing corpus in {corpus}")
                return saved
            print(f"Corpus incomplete ({len(missing)} file(s) missing), regenerating those")
            for name in missing:
                idx = saved["files"][name]["index"]
                write_file(os.path.join(corpus, name), saved["files"][name]["size"], cfg.seed, idx)
            return saved

    print(f"Generating {human(cfg.size)} corpus across {cfg.files} files in {corpus}")
    shutil.rmtree(corpus, ignore_errors=True)
    os.makedirs(corpus, exist_ok=True)

    base = cfg.size // cfg.files
    manifest = {"params": params, "files": {}}
    t0 = time.monotonic()
    for i in range(cfg.files):
        # last file absorbs the remainder so the total is exact
        n = base if i < cfg.files - 1 else cfg.size - base * (cfg.files - 1)
        name = f"data{i:03d}.bin"
        digest = write_file(os.path.join(corpus, name), n, cfg.seed, i)
        manifest["files"][name] = {"size": n, "sha256": digest, "index": i}
        print(f"  {name}  {human(n)}", flush=True)
    print(f"Corpus generated in {time.monotonic() - t0:.1f}s")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def clean_artifacts(cfg):
    """Remove par2 files and repair backups left in the corpus directory."""
    for name in os.listdir(cfg.corpus_dir):
        if name.endswith(".par2") or re.search(r"\.\d+$", name):
            os.remove(os.path.join(cfg.corpus_dir, name))


def clean_backups(cfg):
    """par2 leaves .1/.2 backups of files it rewrote during repair."""
    for name in os.listdir(cfg.corpus_dir):
        if re.search(r"\.\d+$", name):
            os.remove(os.path.join(cfg.corpus_dir, name))


def run_par2(cfg, args, cwd):
    """Run par2, returning (seconds, stdout). Only the subprocess is timed."""
    cmd = [cfg.par2] + args
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    return elapsed, proc.returncode, proc.stdout + proc.stderr


def parse_create_stats(out):
    stats = {}
    for key, pat in (
        ("block_size", r"Block size:\s*(\d+)"),
        ("source_blocks", r"Source block count:\s*(\d+)"),
        ("recovery_blocks", r"Recovery block count:\s*(\d+)"),
    ):
        m = re.search(pat, out)
        if m:
            stats[key] = int(m.group(1))
    return stats


def create(cfg, backend, manifest):
    """Run par2 create, timed. Returns (seconds, stats)."""
    clean_artifacts(cfg)
    # no -q: the block size/count stats we need are printed at normal verbosity
    args = ["create", f"-b{cfg.block_count}", f"-r{cfg.redundancy}"]
    if cfg.threads:
        args.append(f"-t{cfg.threads}")
    args += backend_args(cfg, backend)
    args.append("bench.par2")
    args += sorted(manifest["files"])

    elapsed, rc, out = run_par2(cfg, args, cfg.corpus_dir)
    if rc != 0:
        sys.exit(f"par2 create failed (rc={rc}):\n{out}")
    stats = parse_create_stats(out)
    # Cache for --skip-create, which otherwise has no way to learn the block
    # size that the damage functions need.
    with open(os.path.join(cfg.workdir, "create-stats.json"), "w") as f:
        json.dump(stats, f)
    return elapsed, stats


def backend_args(cfg, backend):
    if backend == "cpu" and not cfg.have_gpu_flag:
        # --gpu doesn't exist yet; plain CPU build needs no flag
        return []
    return BACKEND_ARGS[backend]


def blocks_in(size, block_size):
    """Blocks a file of this size occupies. PAR2 blocks never span files, so
    every file rounds up and the last block is partially used."""
    return (size + block_size - 1) // block_size


def gf16_work(source_blocks, lost_blocks, block_size):
    """Multiply-add bytes to reconstruct lost_blocks outputs.

    Every source block is applied to every reconstructed block, so the work is
    the product. This is the only correct denominator for a throughput figure,
    and it must be paired with the reconstruct time alone -- see --phase-split.
    """
    return source_blocks * lost_blocks * block_size


def damage_delete(cfg, manifest, block_size):
    """Move whole files aside until ~damage% of total bytes is gone.

    Returns (names, bytes, lost_blocks, undo). Files are moved to a stash rather
    than deleted, so undo is a rename rather than a multi-gigabyte regeneration.

    Granularity is limited by file size: with few large files the achievable
    damage may overshoot the target, so we stop just short of it.
    """
    names = sorted(manifest["files"])
    rng = random.Random(cfg.seed ^ 0xD00D)
    rng.shuffle(names)
    target = cfg.size * cfg.damage / 100.0
    stash = os.path.join(cfg.workdir, "stash")
    os.makedirs(stash, exist_ok=True)

    removed, damaged, moved = 0, [], []
    for name in names:
        size = manifest["files"][name]["size"]
        # Never exceed the target once we have at least one file: overshooting
        # can push damage past the available redundancy and turn the benchmark
        # into a measurement of a failed repair. Under-damaging is the safe
        # direction, so we stop just short rather than just past.
        if damaged and removed + size > target:
            continue
        src = os.path.join(cfg.corpus_dir, name)
        dst = os.path.join(stash, name)
        os.replace(src, dst)
        moved.append((src, dst))
        removed += size
        damaged.append(name)
        if removed >= target * 0.95:
            break

    def undo():
        clean_backups(cfg)
        for src, dst in moved:
            os.replace(dst, src)  # overwrites whatever repair produced

    # Count blocks exactly rather than dividing bytes by block size: each file
    # rounds up to a whole block, so the byte estimate runs low by up to one
    # block per deleted file and would inflate any GB/s derived from it.
    lost_blocks = sum(blocks_in(manifest["files"][n]["size"], block_size)
                      for n in damaged)
    return damaged, removed, lost_blocks, undo


def damage_corrupt(cfg, manifest, block_size):
    """Overwrite whole blocks in place, so 'damage%' means percent of blocks.

    Returns (names, bytes, lost_blocks, undo).

    Scattering individual bytes would be wrong: PAR2 repairs at block
    granularity, so thinly-spread byte corruption destroys far more blocks
    than the byte count suggests.
    """
    rng = random.Random(cfg.seed ^ 0xC0DE)

    # Enumerate every block across the whole corpus and sample globally, so the
    # achieved damage matches the requested percentage exactly. Sampling
    # per-file and truncating loses a block per file and silently under-damages.
    blocks = []
    for name in sorted(manifest["files"]):
        size = manifest["files"][name]["size"]
        nblocks = blocks_in(size, block_size)
        for b in range(nblocks):
            off = b * block_size
            blocks.append((name, off, min(block_size, size - off)))

    ncorrupt = round(len(blocks) * cfg.damage / 100.0)
    picks = rng.sample(blocks, min(ncorrupt, len(blocks)))

    by_file = {}
    for name, off, length in picks:
        by_file.setdefault(name, []).append((off, length))

    # Stash the original bytes of exactly the regions being overwritten, so undo
    # is a small write-back rather than regenerating whole multi-gigabyte files.
    stash_path = os.path.join(cfg.workdir, "stash.bin")
    saved = []
    corrupted_bytes = 0
    with open(stash_path, "wb") as sf:
        for name, regions in sorted(by_file.items()):
            regions.sort()
            path = os.path.join(cfg.corpus_dir, name)
            with open(path, "r+b") as f:
                for off, length in regions:
                    f.seek(off)
                    saved.append((name, off, length, sf.tell()))
                    sf.write(f.read(length))
                    f.seek(off)
                    f.write(bytes([rng.randrange(256)]) * length)
                    corrupted_bytes += length

    def undo():
        clean_backups(cfg)
        with open(stash_path, "rb") as sf:
            current, fh = None, None
            try:
                for name, off, length, stash_off in saved:
                    if name != current:
                        if fh:
                            fh.close()
                        fh = open(os.path.join(cfg.corpus_dir, name), "r+b")
                        current = name
                    sf.seek(stash_off)
                    fh.seek(off)
                    fh.write(sf.read(length))
            finally:
                if fh:
                    fh.close()
        os.remove(stash_path)

    # picks are distinct blocks, so this is exact
    return sorted(by_file), corrupted_bytes, len(picks), undo


def verify(cfg, manifest):
    """Confirm every file matches the original. A fast wrong answer is worthless."""
    bad = []
    for name, info in sorted(manifest["files"].items()):
        path = os.path.join(cfg.corpus_dir, name)
        if not os.path.exists(path):
            bad.append(f"{name}: missing after repair")
        elif sha256_file(path) != info["sha256"]:
            bad.append(f"{name}: hash mismatch after repair")
    return bad


def drop_caches(cfg):
    if not cfg.drop_caches:
        return False
    subprocess.run(["sync"], check=False)
    cmd = cfg.drop_caches_cmd or (
        ["purge"] if platform.system() == "Darwin" else ["sysctl", "-w", "vm.drop_caches=3"]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(
            f"--drop-caches was requested but `{' '.join(cmd)}` failed "
            f"(rc={proc.returncode}): {proc.stderr.strip()}\n"
            "Refusing to continue: warm-cache numbers must not be reported as cold.\n"
            "Run this benchmark under a user that can drop caches, or pass "
            "--drop-caches-cmd, or drop the flag and accept warm-cache timings."
        )
    return True


def par2_args(cfg, backend, verb):
    args = [verb, "-q"]
    if cfg.threads:
        args.append(f"-t{cfg.threads}")
    args += backend_args(cfg, backend)
    args.append("bench.par2")
    return args


def repair_run(cfg, backend, manifest, mode, block_size, source_blocks):
    """One damage/repair/verify cycle. Returns a result dict."""
    if mode == "delete":
        damaged, dmg_bytes, lost_blocks, undo = damage_delete(cfg, manifest, block_size)
    else:
        damaged, dmg_bytes, lost_blocks, undo = damage_corrupt(cfg, manifest, block_size)

    scan_secs = None
    if cfg.phase_split:
        # Time the scan alone against this exact damage state. It has to run
        # before the repair, since afterwards there is nothing left to scan.
        # Caches are dropped again below so the repair is measured from the
        # same starting point rather than warmed by this pass.
        drop_caches(cfg)
        scan_secs, rc, out = run_par2(cfg, par2_args(cfg, backend, "verify"), cfg.corpus_dir)
        if rc != PAR2_REPAIR_POSSIBLE:
            undo()
            why = {
                PAR2_SUCCESS: "verify found nothing wrong -- the damage did not "
                              "take effect, so the repair below would measure nothing",
            }.get(rc, "verify says the damage exceeds the available recovery data")
            return {"ok": False, "error": f"{why} (rc={rc})", "output": out[-2000:]}

    cold = drop_caches(cfg)

    elapsed, rc, out = run_par2(cfg, par2_args(cfg, backend, "repair"), cfg.corpus_dir)

    if rc != 0:
        undo()
        return {"ok": False, "error": f"repair failed (rc={rc})", "output": out[-2000:]}

    # verify before undoing, so we are checking what repair actually produced
    bad = verify(cfg, manifest)
    undo()
    if bad:
        return {"ok": False, "error": "; ".join(bad)}

    r = {
        "ok": True,
        "seconds": elapsed,
        "damaged_files": len(damaged),
        "damaged_bytes": dmg_bytes,
        "lost_blocks": lost_blocks,
        "cold_cache": cold,
    }
    if scan_secs is not None:
        reconstruct = elapsed - scan_secs
        work = gf16_work(source_blocks, lost_blocks, block_size)
        r["scan_seconds"] = scan_secs
        r["reconstruct_seconds"] = reconstruct
        r["gf16_bytes"] = work
        # A negative or near-zero difference means the two phases were not
        # measured against comparable states (or the corpus is too small for
        # the subtraction to survive the noise); refuse to publish a number.
        r["gbps"] = work / reconstruct / 1e9 if reconstruct > 0.05 else None
    return r


def report_phase_split(results, block_size, source_blocks):
    """Scan and reconstruct separately, with throughput off the latter only."""
    print("\nPhases (median of runs; the difference of two timings is noisier")
    print("than either, so the median is more defensible than the best):")
    print(f"{'mode/backend':<24} {'scan':>9} {'reconst':>9} {'GB/s':>9} {'vs cpu':>9}")

    baselines, saw_corrupt = {}, False
    for key, runs in results["repair"].items():
        good = [r for r in runs if r.get("ok") and r.get("reconstruct_seconds") is not None]
        if not good:
            print(f"{key:<24} {'FAILED':>9}")
            continue
        mode = key.split("/")[0]
        saw_corrupt |= mode == "corrupt"
        scan = statistics.median(r["scan_seconds"] for r in good)
        recon = statistics.median(r["reconstruct_seconds"] for r in good)
        # Derive the rate from the median time rather than taking the median of
        # the per-run rates: with an even number of runs those disagree, since
        # the mean of two ratios is not the ratio of their means.
        work = good[0]["gf16_bytes"]
        rate = f"{work / recon / 1e9:.1f}" if recon > 0.05 else "-"
        if key.endswith("/cpu"):
            baselines[mode] = recon
        speedup = f"{baselines[mode] / recon:.2f}x" if mode in baselines and recon > 0 else "-"
        print(f"{key:<24} {scan:>8.2f}s {recon:>8.2f}s {rate:>9} {speedup:>9}")

    per_block = gf16_work(source_blocks, 1, block_size)
    print(f"\nGF16 work: {source_blocks} source blocks x blocks reconstructed x "
          f"{block_size} B  ({human(per_block)} per reconstructed block)")
    if saw_corrupt:
        print("\nNOTE: in corrupt mode the subtraction is not purely GF16. `repair`\n"
              "      rewrites recovered blocks back into the damaged files in place,\n"
              "      work `verify` never does, so the reconstruct column is inflated\n"
              "      and its GB/s understated. Quote the delete-mode figure.")


def main():
    # long runs are usually redirected to a log; keep progress observable
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--par2", default="./par2", help="path to par2 binary (default ./par2)")
    ap.add_argument("--workdir", default="./bench-work", help="scratch directory for corpus")
    ap.add_argument("--size", default="1G", help="total corpus size, e.g. 1G, 10G (default 1G)")
    ap.add_argument("--files", type=int, default=20, help="number of corpus files (default 20)")
    ap.add_argument("--seed", type=int, default=1, help="corpus PRNG seed (default 1)")
    ap.add_argument("--block-count", type=int, default=2000,
                    help="par2 -b block count (default 2000). This is the dominant "
                         "performance parameter: GF16 work scales with it.")
    ap.add_argument("--redundancy", type=int, default=15, help="par2 -r%% (default 15)")
    ap.add_argument("--damage", type=float, default=10.0, help="percent damage (default 10)")
    ap.add_argument("--mode", choices=["delete", "corrupt", "both"], default="both")
    ap.add_argument("--backend", action="append", choices=list(BACKEND_ARGS),
                    help="backend to test; repeatable (default cpu)")
    ap.add_argument("--repeat", type=int, default=1, help="repair repetitions (default 1)")
    ap.add_argument("--threads", type=int, default=0, help="par2 -t (0 = par2 default)")
    ap.add_argument("--phase-split", action="store_true",
                    help="also time `par2 verify` against the same damage state and "
                         "subtract it, isolating the reconstruct phase from the scan. "
                         "Required for any GB/s claim; roughly doubles runtime.")
    ap.add_argument("--drop-caches", action="store_true",
                    help="drop the page cache before each timed par2 run (both phases "
                         "under --phase-split); aborts if it cannot")
    ap.add_argument("--drop-caches-cmd", nargs="+", help="override the cache-drop command")
    ap.add_argument("--skip-create", action="store_true",
                    help="reuse existing bench.par2 instead of timing creation")
    ap.add_argument("--json", help="write results to this path as JSON")
    cfg = ap.parse_args()

    cfg.size = parse_size(cfg.size)
    cfg.par2 = os.path.abspath(cfg.par2)
    cfg.workdir = os.path.abspath(cfg.workdir)
    cfg.corpus_dir = os.path.join(cfg.workdir, "corpus")
    cfg.backend = cfg.backend or ["cpu"]
    os.makedirs(cfg.workdir, exist_ok=True)

    if not os.path.exists(cfg.par2):
        sys.exit(f"par2 binary not found: {cfg.par2}")

    help_out = subprocess.run([cfg.par2, "--help"], capture_output=True, text=True).stdout
    cfg.have_gpu_flag = "--gpu" in help_out
    needs_gpu = [b for b in cfg.backend if b != "cpu"]
    if needs_gpu and not cfg.have_gpu_flag:
        sys.exit(
            f"backend(s) {', '.join(needs_gpu)} requested, but this par2 build has no "
            "--gpu flag. Build the GPU-enabled binary first; refusing to run the CPU "
            "path and report it as GPU."
        )

    if cfg.damage >= cfg.redundancy:
        sys.exit(
            f"--damage {cfg.damage}%% >= --redundancy {cfg.redundancy}%%: repair would be "
            "impossible. Raise redundancy or lower damage."
        )

    manifest = ensure_corpus(cfg)
    modes = ["delete", "corrupt"] if cfg.mode == "both" else [cfg.mode]

    results = {
        "config": {
            "size": cfg.size, "files": cfg.files, "block_count": cfg.block_count,
            "redundancy": cfg.redundancy, "damage": cfg.damage, "threads": cfg.threads,
            "repeat": cfg.repeat, "host": platform.platform(), "machine": platform.machine(),
        },
        "create": {}, "repair": {},
    }

    # Create once per backend (also a useful measurement in its own right).
    stats = None
    for backend in cfg.backend:
        if cfg.skip_create and os.path.exists(os.path.join(cfg.corpus_dir, "bench.par2")):
            print(f"Skipping create for {backend} (--skip-create)")
            cached = os.path.join(cfg.workdir, "create-stats.json")
            if stats is None and os.path.exists(cached):
                with open(cached) as f:
                    stats = json.load(f)
            continue
        print(f"\n=== create [{backend}] ===")
        secs, st = create(cfg, backend, manifest)
        stats = st or stats
        results["create"][backend] = {"seconds": secs, **st}
        print(f"  {secs:.2f}s   block size {st.get('block_size')}  "
              f"source blocks {st.get('source_blocks')}  "
              f"recovery blocks {st.get('recovery_blocks')}")

    if stats is None:
        sys.exit("could not determine block size (create was skipped and no stats cached)")

    block_size = stats["block_size"]
    source_blocks = stats["source_blocks"]
    recovery_blocks = stats["recovery_blocks"]

    for mode in modes:
        for backend in cfg.backend:
            key = f"{mode}/{backend}"
            print(f"\n=== repair [{key}] ===")
            runs = []
            for i in range(cfg.repeat):
                r = repair_run(cfg, backend, manifest, mode, block_size, source_blocks)
                if not r["ok"]:
                    print(f"  run {i + 1}: FAILED - {r['error']}")
                    runs.append(r)
                    break
                # blocks actually lost, checked against available recovery
                if r["lost_blocks"] > recovery_blocks:
                    print(f"  WARNING: {r['lost_blocks']} blocks damaged but only "
                          f"{recovery_blocks} recovery blocks exist")
                pct = 100.0 * r["damaged_bytes"] / cfg.size
                line = (f"  run {i + 1}: {r['seconds']:.2f}s  "
                        f"({r['damaged_files']} files, {r['lost_blocks']} blocks, "
                        f"{human(r['damaged_bytes'])} = {pct:.1f}% damaged, cache "
                        f"{'cold' if r['cold_cache'] else 'warm'})")
                if cfg.phase_split:
                    gbps = r["gbps"]
                    line += (f"\n           scan {r['scan_seconds']:.2f}s  "
                             f"reconstruct {r['reconstruct_seconds']:.2f}s"
                             + (f"  = {gbps:.1f} GB/s" if gbps else
                                "  (too short to derive a rate)"))
                print(line)
                runs.append(r)
            results["repair"][key] = runs

    # ---- report ----
    print("\n" + "=" * 72)
    print(f"Corpus {human(cfg.size)} in {cfg.files} files | block size {human(block_size)} | "
          f"{source_blocks} source + {recovery_blocks} recovery blocks")
    print(f"Redundancy {cfg.redundancy}% | damage {cfg.damage}% | "
          f"cache {'cold' if cfg.drop_caches else 'WARM (not dropped)'}")
    print("=" * 72)

    # Total repair wall time. This is what a user experiences, but it includes
    # the scan, so it is NOT a GF16 throughput measurement -- see below.
    print("Total repair (scan + reconstruct):")
    print(f"{'mode/backend':<24} {'best':>9} {'mean':>9} {'vs cpu':>9}")
    baselines = {}
    for key, runs in results["repair"].items():
        good = [r["seconds"] for r in runs if r.get("ok")]
        if not good:
            print(f"{key:<24} {'FAILED':>9}")
            continue
        mode = key.split("/")[0]
        best, mean = min(good), sum(good) / len(good)
        if key.endswith("/cpu"):
            baselines[mode] = best
        speedup = f"{baselines[mode] / best:.2f}x" if mode in baselines else "-"
        print(f"{key:<24} {best:>8.2f}s {mean:>8.2f}s {speedup:>9}")

    if cfg.phase_split:
        report_phase_split(results, block_size, source_blocks)

    if cfg.json:
        with open(cfg.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {cfg.json}")

    failed = any(not r.get("ok") for runs in results["repair"].values() for r in runs)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
