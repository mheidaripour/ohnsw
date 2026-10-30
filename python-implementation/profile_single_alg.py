"""
profile_single_alg.py
=====================

Resource and performance profiler for benchmark_single_alg.py runs.

Two modes:

monitor (default)
    Runs benchmark_single_alg.py as a subprocess and samples system resource
    usage (RSS/USS memory, CPU%, disk I/O, system available memory) with
    psutil at a fixed interval. The benchmark's stdout phase markers are
    timestamped so the time series is segmented into build vs. traversal.
    Low overhead: safe to use with full trial counts.

instrumented
    Runs the same workload in-process with cProfile wrapped separately
    around the graph-build and traversal phases, and tracemalloc snapshots
    taken after build, after the first trial, and at the end (to attribute
    memory to allocation sites and detect per-trial growth). Adds large
    overhead: use with a reduced trial count.

Artifacts are written to <outdir>/<run-name>/ using the same naming
convention as the logs in local-tests/logs/.

Example:
    python3 profile_single_alg.py --n 131072 --alg cd --pt multiwrite
    python3 profile_single_alg.py --n 1024 --alg rw --pt original \
        --mode instrumented --trials 5
"""

import argparse
import cProfile
import csv
import io
import json
import math
import os
import pstats
import resource
import subprocess
import sys
import threading
import time
import tracemalloc

# defaults from the experiment guidance
DEFAULTS = {
    "d": 20,
    "bs": 64,
    "trials": 50,
    "pt": "multiwrite",
    "alg": "cd",
    "move": True,
    "copies": False,
    "prime": True,
    "staticinsertion": True,
    "wl": 50,
    "df": 0.9,
}

MB = 1024 * 1024

# stdout markers printed by build_from_networkx / benchmark, used to
# segment the resource time series into phases
PHASE_MARKERS = [
    ("Graph from NetworkX:", "build"),
    ("Benchmarking Graph", "traversal"),
    # benchmark_single_alg flushes this line as soon as the algorithm finishes;
    # "----- Finished test:" often arrives in the same stdout burst at exit
    ("Post algorithm", "post_traversal"),
    ("----- Finished test:", "done"),
]


def make_run_name(args) -> str:
    """Reproduce the log-file naming convention of launch_local_tests.sh"""
    def b(v: bool) -> str:
        return "true" if v else "false"

    name = (
        f"ER_n-{args.n}_d-{args.d}_bs-{args.bs}_trials-{args.trials}"
        f"_pt-{args.pt}_alg-{args.alg}_move-{b(args.move)}_copies-{b(args.copies)}"
        f"_prime-{b(args.prime)}_staticinsertion-{b(args.staticinsertion)}"
    )
    if args.alg == "rw":
        name += f"_wl-{args.wl}"
    elif args.alg == "pr":
        name += f"_wl-{args.wl}_df-{args.df}"
    if args.branching_factor is not None:
        name += f"_bf-{args.branching_factor}"
    if args.path_oram_rary_read_evictions is not None:
        name += f"_raryevict-{args.path_oram_rary_read_evictions}"
    return name


def benchmark_cli_args(args) -> list:
    """Build the CLI argument list for benchmark_single_alg.py"""
    cli_args = [
        "--n", str(args.n),
        "--d", str(args.d),
        "--bs", str(args.bs),
        "--trials", str(args.trials),
        "--pt", args.pt,
        "--alg", args.alg,
        "--move" if args.move else "--no_move",
        "--copies" if args.copies else "--no_copies",
        "--prime" if args.prime else "--no_prime",
        "--staticinsertion" if args.staticinsertion else "--no_staticinsertion",
        "--wl", str(args.wl),
        "--df", str(args.df),
    ]
    if args.branching_factor is not None:
        cli_args += ["--branching_factor", str(args.branching_factor)]
    if args.path_oram_rary_read_evictions is not None:
        cli_args += [
            "--path_oram_rary_read_evictions",
            str(args.path_oram_rary_read_evictions),
        ]
    return cli_args


# ---------------------------------------------------------------------------
# monitor mode
# ---------------------------------------------------------------------------

def normalize_phase_durations(durations: dict[str, float]) -> dict[str, float]:
    """Merge burst-read stdout phases into a single algorithm duration.

    When the benchmark subprocess exits, remaining stdout is often delivered
    in one burst, so traversal/post_traversal/done markers share nearly the
    same timestamp even though the algorithm ran for several seconds.
    """
    traversal = durations.get("traversal", 0.0)
    post_traversal = durations.get("post_traversal", 0.0)
    done = durations.get("done", 0.0)
    if traversal < 0.5 and (post_traversal + done) > 0.5:
        normalized = {
            k: v for k, v in durations.items()
            if k not in ("traversal", "post_traversal", "done")
        }
        normalized["algorithm"] = round(traversal + post_traversal + done, 2)
        return normalized
    return durations


class PhaseTracker:
    """Tracks the current benchmark phase based on stdout markers"""

    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "startup"
        self.events = []  # (elapsed_seconds, phase)

    def update(self, line: str, elapsed: float) -> None:
        for marker, phase in PHASE_MARKERS:
            if marker in line:
                with self.lock:
                    if phase != self.phase:
                        self.phase = phase
                        self.events.append((elapsed, phase))
                return

    def current(self) -> str:
        with self.lock:
            return self.phase


def run_monitor(args, run_dir: str, run_name: str) -> None:
    import psutil

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_single_alg.py")
    cmd = [sys.executable, script] + benchmark_cli_args(args)

    log_path = os.path.join(run_dir, "benchmark.log")
    csv_path = os.path.join(run_dir, "resources.csv")

    print(f"[profiler] launching: {' '.join(cmd)}")
    print(f"[profiler] artifacts: {run_dir}")

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    tracker = PhaseTracker()
    tracker.events.append((0.0, "startup"))

    def read_stdout():
        with open(log_path, "w") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                elapsed = time.monotonic() - t0
                log.write(line)
                log.flush()
                tracker.update(line, elapsed)

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()

    ps_proc = psutil.Process(proc.pid)
    ps_proc.cpu_percent()  # prime the CPU counter

    samples = []
    aborted = False
    peak_by_phase = {}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_s", "phase", "rss_mb", "uss_mb", "cpu_percent",
            "sys_available_mb", "io_read_mb", "io_write_mb",
        ])
        while proc.poll() is None:
            time.sleep(args.interval)
            elapsed = time.monotonic() - t0
            try:
                with ps_proc.oneshot():
                    mem = ps_proc.memory_full_info()
                    cpu = ps_proc.cpu_percent()
                    try:
                        ioc = ps_proc.io_counters()
                        io_read, io_write = ioc.read_bytes / MB, ioc.write_bytes / MB
                    except (psutil.AccessDenied, AttributeError):
                        io_read = io_write = 0.0
            except psutil.NoSuchProcess:
                break

            sys_avail = psutil.virtual_memory().available / MB
            phase = tracker.current()
            rss = mem.rss / MB
            uss = getattr(mem, "uss", 0) / MB
            row = (round(elapsed, 2), phase, round(rss, 1), round(uss, 1),
                   cpu, round(sys_avail, 1), round(io_read, 1), round(io_write, 1))
            writer.writerow(row)
            f.flush()
            samples.append(row)
            peak_by_phase[phase] = max(peak_by_phase.get(phase, 0.0), rss)

            if sys_avail < args.min_free_mb:
                print(f"[profiler] ABORT: system available memory "
                      f"{sys_avail:.0f} MB < {args.min_free_mb} MB threshold, killing benchmark")
                proc.kill()
                aborted = True
                break

    proc.wait()
    reader.join(timeout=10)
    wall = time.monotonic() - t0

    # per-phase durations from phase transition events
    events = tracker.events + [(wall, "end")]
    durations = {}
    for (start, phase), (end, _) in zip(events, events[1:]):
        durations[phase] = durations.get(phase, 0.0) + (end - start)

    summary = {
        "run": run_name,
        "mode": "monitor",
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "aborted_low_memory": aborted,
        "wall_time_s": round(wall, 2),
        "phase_durations_s": normalize_phase_durations(
            {k: round(v, 2) for k, v in durations.items()}),
        "peak_rss_mb": round(max((s[2] for s in samples), default=0.0), 1),
        "peak_rss_by_phase_mb": {k: round(v, 1) for k, v in peak_by_phase.items()},
        "avg_cpu_percent": round(
            sum(s[4] for s in samples) / len(samples), 1) if samples else 0.0,
        "samples": len(samples),
    }
    write_summary(run_dir, summary)
    plot_monitor(run_dir, run_name, samples, tracker.events)


def plot_monitor(run_dir: str, run_name: str, samples, events) -> None:
    """Plot memory/CPU over time with phase boundaries; skip if matplotlib missing"""
    if not samples:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[profiler] matplotlib not available, skipping plot")
        return

    t = [s[0] for s in samples]
    rss = [s[2] for s in samples]
    cpu = [s[4] for s in samples]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(t, rss, color="tab:blue", label="RSS (MB)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("memory (MB)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(t, cpu, color="tab:orange", alpha=0.5, label="CPU %")
    ax2.set_ylabel("CPU %", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    for elapsed, phase in events:
        if phase in ("build", "traversal", "post_traversal", "done"):
            ax1.axvline(elapsed, color="gray", linestyle="--", alpha=0.7)
            ax1.text(elapsed, max(rss) * 0.98, f" {phase}", rotation=90,
                     va="top", fontsize=8, color="gray")

    ax1.set_title(run_name, fontsize=9)
    fig.tight_layout()
    out = os.path.join(run_dir, "resource_usage.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[profiler] plot written to {out}")


# ---------------------------------------------------------------------------
# instrumented mode
# ---------------------------------------------------------------------------

def run_instrumented(args, run_dir: str, run_name: str) -> None:
    # imported lazily so monitor mode does not load the project modules
    from oblivious_graph import build_from_networkx, benchmark
    from smart_pointer import SmartPointer
    from networkx import erdos_renyi_graph

    p = args.d / args.n
    if p > 1.0:
        print(f"Skipping test: p={p} > 1.0")
        return

    print(f"[profiler] instrumented run, artifacts: {run_dir}")
    SmartPointer.set_smart_pointer(args.pt)
    bf = (
        args.branching_factor
        if args.branching_factor is not None
        else math.floor((args.bs - 2 * 8) / 8)
    )
    if args.pt == "multiwriterary":
        print("Setting branching factor for multiwriterary pointer to", bf)
        SmartPointer.set_branching_factor(bf)

    tee = TeeLogger(os.path.join(run_dir, "benchmark.log"))
    tracemalloc.start()
    timings = {}

    # ---- build phase ----
    prof_build = cProfile.Profile()
    t0 = time.monotonic()
    prof_build.enable()
    ograph = build_from_networkx(
        function=erdos_renyi_graph,
        name=run_name,
        graph=None,
        args=(args.n, p, 1),
        branching_factor=bf,
        static_insertion=args.staticinsertion,
        move_semantics=args.move,
        make_copies=args.copies,
        prime_graph=args.prime,
        seed=1,
        connect_all=True,
        queue_osam=True,
        stack_osam=True,
        avl_tree_osam=True,
        random_weight_min=None,
        random_weight_max=None,
        sort_names=False,
        ordered_dynamic_vertices=False,
    )
    prof_build.disable()
    timings["build_s"] = time.monotonic() - t0
    snap_build = tracemalloc.take_snapshot()
    traced_after_build = tracemalloc.get_traced_memory()

    def run_trials(profiler, trials):
        profiler.enable()
        benchmark(
            ograph=ograph, name=run_name, trials=trials, algorithm=args.alg,
            entry_point=None, next_entry_point=None,
            pick_random_entry_points=True,
            move_semantics=args.move, make_copies=args.copies,
            walk_length=args.wl, damping_factor=args.df,
            queue_osam=True, stack_osam=True, avl_tree_osam=True,
            print_trace=False, print_each_trial=False,
        )
        profiler.disable()

    # ---- traversal phase: first trial, then the rest ----
    prof_trav = cProfile.Profile()
    t0 = time.monotonic()
    run_trials(prof_trav, 1)
    snap_first = tracemalloc.take_snapshot()

    remaining = args.trials - 1
    if remaining > 0:
        run_trials(prof_trav, remaining)
    timings["traversal_s"] = time.monotonic() - t0
    snap_final = tracemalloc.take_snapshot()
    traced_final = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tee.close()

    # ---- cProfile reports ----
    prof_build.dump_stats(os.path.join(run_dir, "build.prof"))
    prof_trav.dump_stats(os.path.join(run_dir, "traversal.prof"))
    write_pstats_report(prof_build, os.path.join(run_dir, "cprofile_build.txt"))
    write_pstats_report(prof_trav, os.path.join(run_dir, "cprofile_traversal.txt"))

    # ---- tracemalloc report ----
    per_trial_growth = None
    if remaining > 0:
        size_first = sum(st.size for st in snap_first.statistics("filename"))
        size_final = sum(st.size for st in snap_final.statistics("filename"))
        per_trial_growth = (size_final - size_first) / remaining

    with open(os.path.join(run_dir, "tracemalloc.txt"), "w") as f:
        f.write(f"traced memory after build: current={traced_after_build[0]/MB:.1f} MB, "
                f"peak={traced_after_build[1]/MB:.1f} MB\n")
        f.write(f"traced memory at end:      current={traced_final[0]/MB:.1f} MB, "
                f"peak={traced_final[1]/MB:.1f} MB\n")
        if per_trial_growth is not None:
            f.write(f"memory growth per trial (trial 2..{args.trials}): "
                    f"{per_trial_growth/1024:.1f} KB/trial\n")
        f.write("\n=== top 25 allocation sites after build ===\n")
        for st in snap_build.statistics("lineno")[:25]:
            f.write(f"{st}\n")
        f.write("\n=== top 25 allocation sites at end ===\n")
        for st in snap_final.statistics("lineno")[:25]:
            f.write(f"{st}\n")
        f.write("\n=== top 25 growth sites: end vs after first trial ===\n")
        for st in snap_final.compare_to(snap_first, "lineno")[:25]:
            f.write(f"{st}\n")

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB
    summary = {
        "run": run_name,
        "mode": "instrumented",
        "wall_time_s": round(timings["build_s"] + timings["traversal_s"], 2),
        "phase_durations_s": {
            "build": round(timings["build_s"], 2),
            "traversal": round(timings["traversal_s"], 2),
        },
        "peak_rss_mb": round(max_rss, 1),
        "tracemalloc_peak_mb": round(traced_final[1] / MB, 1),
        "tracemalloc_after_build_mb": round(traced_after_build[0] / MB, 1),
        "tracemalloc_end_mb": round(traced_final[0] / MB, 1),
        "memory_growth_per_trial_kb": (
            round(per_trial_growth / 1024, 1) if per_trial_growth is not None else None),
        "note": "timings include cProfile+tracemalloc overhead; "
                "use monitor mode for clean timings",
    }
    write_summary(run_dir, summary)


class TeeLogger:
    """Duplicate stdout to a log file during the in-process run"""

    def __init__(self, path: str):
        self.file = open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


def write_pstats_report(profiler: cProfile.Profile, path: str) -> None:
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.strip_dirs()
    buf.write("=== sorted by cumulative time (top 40) ===\n")
    stats.sort_stats("cumulative").print_stats(40)
    buf.write("\n=== sorted by internal time (top 40) ===\n")
    stats.sort_stats("tottime").print_stats(40)
    with open(path, "w") as f:
        f.write(buf.getvalue())


def write_summary(run_dir: str, summary: dict) -> None:
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[profiler] ===== summary =====")
    for key, value in summary.items():
        print(f"[profiler] {key}: {value}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, default=DEFAULTS["d"])
    parser.add_argument("--bs", type=int, default=DEFAULTS["bs"])
    parser.add_argument("--trials", type=int, default=DEFAULTS["trials"])
    parser.add_argument("--pt", type=str, default=DEFAULTS["pt"],
                        choices=["original", "multiwrite", "multiwriterary", "recursive"])
    parser.add_argument("--alg", type=str, default=DEFAULTS["alg"])

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--move", dest="move", action="store_true")
    group1.add_argument("--no_move", dest="move", action="store_false")
    parser.set_defaults(move=DEFAULTS["move"])

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--copies", dest="copies", action="store_true")
    group2.add_argument("--no_copies", dest="copies", action="store_false")
    parser.set_defaults(copies=DEFAULTS["copies"])

    group3 = parser.add_mutually_exclusive_group()
    group3.add_argument("--prime", dest="prime", action="store_true")
    group3.add_argument("--no_prime", dest="prime", action="store_false")
    parser.set_defaults(prime=DEFAULTS["prime"])

    group4 = parser.add_mutually_exclusive_group()
    group4.add_argument("--staticinsertion", dest="staticinsertion", action="store_true")
    group4.add_argument("--no_staticinsertion", dest="staticinsertion", action="store_false")
    parser.set_defaults(staticinsertion=DEFAULTS["staticinsertion"])

    parser.add_argument("--wl", type=int, default=DEFAULTS["wl"])
    parser.add_argument("--df", type=float, default=DEFAULTS["df"])
    parser.add_argument(
        "--branching_factor",
        type=int,
        default=None,
        help=(
            "Branching factor used for graph fanout trees and multiwriterary "
            "pointers. Defaults to floor((bs - 16) / 8)."
        ),
    )
    parser.add_argument(
        "--path_oram_rary_read_evictions",
        type=int,
        default=None,
        help=(
            "Number of full PathORAM eviction paths to run after each r-ary "
            "read when monitor mode delegates to benchmark_single_alg.py."
        ),
    )

    parser.add_argument("--mode", type=str, default="monitor",
                        choices=["monitor", "instrumented"])
    parser.add_argument("--interval", type=float, default=0.5,
                        help="psutil sampling interval in seconds (monitor mode)")
    parser.add_argument("--min_free_mb", type=float, default=500.0,
                        help="abort the run if system available memory drops below this")
    parser.add_argument("--outdir", type=str, default="local-tests/profiles")

    args = parser.parse_args()

    run_name = make_run_name(args) + f"_mode-{args.mode}"
    run_dir = os.path.join(args.outdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.mode == "monitor":
        run_monitor(args, run_dir, run_name)
    else:
        run_instrumented(args, run_dir, run_name)


if __name__ == "__main__":
    main()
