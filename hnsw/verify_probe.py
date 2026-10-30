#!/usr/bin/env python3
"""Check probe_partitions.py predictions against runs that actually happened.

``probe_partitions.py`` predicts a block size by partitioning the base layer
and measuring the largest partition, without running any of the OSAM
machinery. That prediction is only worth anything if the partition it measured
is the partition ``o_hnsw.py`` goes on to build. This script closes that loop:
it joins each ``sweep/partitions_<dataset>.csv`` row to the sweep logs in
``logs/<dataset>_bssweep/`` by partition count and compares the four quantities
both sides report.

    probe CSV column      o_hnsw log line
    --------------------  -----------------------------
    max_partition_nodes   " max cluster size:   N nodes"
    intra_edges           "Intra-cluster edges: N"
    cross_edges           "Cross-cluster edges: N"
    max_incoming_cross    " Max copies / cluster: N"

All four must match exactly. They are counts over the same partition of the
same graph, so there is no tolerance to allow: a mismatch means the probe and
the run partitioned differently, and every block size the probe predicted for
that dataset is suspect.

Usage:
    python verify_probe.py                       # every dataset with a sweep
    python verify_probe.py --dataset sift1m laion1m
    python verify_probe.py --csv sweep/partitions_sift1m.csv --logs logs/sift1m_bssweep
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "plots"))

from logparse import load_dir  # noqa: E402

HERE = Path(__file__).resolve().parent

# CSV column -> (RunLog attribute, human label). The probe writes one row per
# requested partition count; the log prints these once per run.
COMPARED = (
    ("max_partition_nodes", "max_cluster_size", "max partition nodes"),
    ("intra_edges", "intra_cluster_edges", "intra edges"),
    ("cross_edges", "cross_cluster_edges", "cross edges"),
    ("max_incoming_cross", "max_copies_per_cluster", "max incoming cross"),
)


def default_pairs() -> list[tuple[str, Path, Path]]:
    """Every (dataset, csv, log_dir) triple that exists on disk.

    Log directories are named after the dataset with a ``_bssweep`` suffix,
    except sift1m, which has both an ip and an l2 sweep. The l2 one is the
    correct-metric run and the one the paper's tables are built from, but the
    partition layout is identical either way -- METIS never sees the metric --
    so either directory verifies the same prediction. Both are checked when
    present; agreeing twice is not a problem.
    """
    pairs: list[tuple[str, Path, Path]] = []
    for csv_path in sorted((HERE / "sweep").glob("partitions_*.csv")):
        dataset = csv_path.stem[len("partitions_") :]
        for suffix in ("", "-l2"):
            log_dir = HERE / "logs" / f"{dataset}{suffix}_bssweep"
            if log_dir.is_dir():
                pairs.append((f"{dataset}{suffix}", csv_path, log_dir))
    return pairs


def load_csv(csv_path: Path) -> dict[int, dict]:
    """Probe rows keyed by partition count, both as requested and as realized.

    METIS does not always return the count it was asked for -- sift1m's 32000
    rung comes back as 31998 -- and the log prints what was built, not what was
    requested. Indexing under both keys lets either number find the row. They
    are equal for every other rung, so the second insert is usually a no-op.
    """
    rows: dict[int, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["n_clusters_requested"])] = row
            rows[int(row["num_partitions"])] = row
    return rows


def verify(dataset: str, csv_path: Path, log_dir: Path) -> tuple[int, int, int]:
    """Compare one dataset. Returns (checked, mismatched, unmatched_logs)."""
    rows = load_csv(csv_path)
    runs, dropped = load_dir(log_dir, partitioning="metis")

    print(f"\n{dataset}: {csv_path.relative_to(HERE)} vs {log_dir.relative_to(HERE)}")
    if dropped:
        # Incomplete logs are the norm here -- the sweep's RSS watchdog kills
        # runs that grow too large. They carry no cluster block, so they cannot
        # be compared, but they should not vanish silently either.
        print(f"  ({len(dropped)} incomplete log(s) skipped)")

    checked = mismatched = unmatched = 0
    seen: set[int] = set()

    for run in sorted(runs, key=lambda r: (r.num_clusters or 0, r.path.name)):
        nc = run.num_clusters
        if nc is None:
            continue
        row = rows.get(nc)
        if row is None:
            # A log whose partition count the probe never predicted -- usually
            # a hand-launched run outside the sweep ladder.
            print(f"  ?  nc={nc}: no probe row ({run.path.name})")
            unmatched += 1
            continue

        # Every run at a given nc builds the same partition, so re-checking
        # each pointer type is redundant; report the first and note the rest.
        repeat = nc in seen
        seen.add(nc)

        bad = []
        for col, attr, label in COMPARED:
            predicted = int(row[col])
            actual = getattr(run, attr)
            if actual is None:
                continue
            if actual != predicted:
                bad.append(f"{label}: probe {predicted:,} vs run {actual:,}")

        checked += 1
        if bad:
            mismatched += 1
            print(f"  FAIL nc={nc} ({run.path.name})")
            for line in bad:
                print(f"       {line}")
        elif not repeat:
            print(
                f"  ok   nc={nc:>6}  |P|max={int(row['max_partition_nodes']):,}"
                f"  intra={int(row['intra_edges']):,}"
                f"  cross={int(row['cross_edges']):,}"
                f"  copies={int(row['max_incoming_cross']):,}"
            )

    return checked, mismatched, unmatched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", nargs="+", help="restrict to these dataset names")
    ap.add_argument("--csv", type=Path, help="probe CSV (with --logs, checks one pair)")
    ap.add_argument("--logs", type=Path, help="sweep log directory (with --csv)")
    args = ap.parse_args()

    if bool(args.csv) != bool(args.logs):
        ap.error("--csv and --logs must be given together")

    if args.csv:
        pairs = [(args.csv.stem, args.csv, args.logs)]
    else:
        pairs = default_pairs()
        if args.dataset:
            wanted = set(args.dataset)
            pairs = [p for p in pairs if p[0] in wanted or p[0].rsplit("-", 1)[0] in wanted]

    if not pairs:
        print("no (probe CSV, sweep log directory) pairs found", file=sys.stderr)
        return 2

    totals = [0, 0, 0]
    for dataset, csv_path, log_dir in pairs:
        counts = verify(dataset, csv_path, log_dir)
        totals = [a + b for a, b in zip(totals, counts)]

    checked, mismatched, unmatched = totals
    print(f"\n{checked} run(s) checked, {mismatched} mismatch(es), {unmatched} unmatched log(s)")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
