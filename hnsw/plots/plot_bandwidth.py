#!/usr/bin/env python
"""Bandwidth vs. pointer block size, from o_hnsw.py run logs.

Bandwidth model
---------------
Each ORAM access moves

    per_read = 4 * (b + 1) * r * block_size * log2(total_allocs)  [bytes, r-ary]
    per_read = 4 * (b + 1) * block_size * log2(total_allocs)      [bytes, multiwrite]

and a run performs ``total_reads`` of them, so

    total = total_reads * per_read                                [bytes]

``b`` is the eviction/bucket parameter; the two curves plotted are b = 0 and
b = 1.  ``total_allocs`` defaults to the whole-run allocation count -- the same
number the log reports as ``log2(whole-run allocs)`` -- and can be switched to
the build-phase or search-phase counts with ``--allocs``.

Examples
--------
    python plots/plot_bandwidth.py                       # r-ary, sift1m, defaults
    python plots/plot_bandwidth.py --mode per-read
    python plots/plot_bandwidth.py --pointer-type multiwrite --theme both
    python plots/plot_bandwidth.py --csv out/bandwidth.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import style
from bandwidth import oram_cost, fmt_bytes, fmt_bytes_short, unit_for
from logparse import dedupe_by, load_dir

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def collect(args) -> list[dict]:
    """Parse the logs and turn them into one row per block size."""
    runs, dropped = load_dir(
        args.logs,
        pointer_type=args.pointer_type,
        partitioning=args.partitioning,
    )
    for run in dropped:
        print(f"  ! skipping incomplete log (no memory-access summary): {run.path.name}",
              file=sys.stderr)
    if not runs:
        raise SystemExit(
            f"no complete {args.pointer_type} logs found in {args.logs}"
        )

    # One point per block size; if a config was rerun, keep the newest log.
    runs = dedupe_by(runs, key=lambda r: r.block_size)
    runs.sort(key=lambda r: r.block_size)

    rows = []
    for run in runs:
        log2_allocs = run.log2_allocs(args.allocs)
        row = {
            "log": run.path.name,
            "pointer_type": run.pointer_type,
            "pointer_label": run.pointer_label,
            "block_size": run.block_size,
            "reads": run.total_reads,
            "allocs": {"whole-run": run.whole_run_allocs,
                       "build": run.build_allocs,
                       "search": run.total_allocs}[args.allocs],
            "log2_allocs": log2_allocs,
            "recall_at_10": run.recall_at_10,
        }
        for b in args.b:
            per_read = oram_cost(run.pointer_type, run.block_size, log2_allocs,
                                 b, run.branching_factor)
            row[f"per_read_b{b}"] = per_read
            row[f"total_b{b}"] = per_read * run.total_reads
        rows.append(row)
    return rows


def value_key(mode: str, b: int) -> str:
    return f"{'total' if mode == 'total' else 'per_read'}_b{b}"


def figure_size(args) -> tuple[float, float]:
    """Figure size in inches: a LaTeX column by default in paper mode."""
    if args.width:
        w = {"column": style.COLUMN_WIDTH_IN, "text": style.TEXT_WIDTH_IN}.get(args.width)
        w = float(args.width) if w is None else w
    elif args.paper:
        w = style.COLUMN_WIDTH_IN
    else:
        w = 7.6
    return (w, w * (0.72 if args.paper else 0.605))


def plot(rows: list[dict], args, theme_name: str) -> list[Path]:
    theme = style.THEMES[theme_name]
    style.apply(theme, paper=args.paper)

    xs = [r["block_size"] for r in rows]
    series = {b: [r[value_key(args.mode, b)] for r in rows] for b in args.b}

    unit, scale = unit_for(args.pointer_type, max(max(v) for v in series.values()))

    fig, ax = plt.subplots(figsize=figure_size(args))
    style.style_axes(ax, theme)
    lbl_size = 7 if args.paper else 10

    for i, b in enumerate(args.b):
        color = theme.series[i % len(theme.series)]
        ys = [v / scale for v in series[b]]
        ax.plot(
            xs, ys,
            color=color,
            marker=style.series_marker(i),
            markeredgecolor=theme.surface,   # surface ring where marks overlap
            label=f"b = {b}",
            zorder=3 - i * 0.1,
        )
        # Direct label at the right end, so identity is not colour-alone.
        ax.annotate(
            f"b = {b}",
            xy=(xs[-1], ys[-1]),
            xytext=(6 if args.paper else 8, 0),
            textcoords="offset points",
            va="center", ha="left",
            color=theme.text_secondary,
            fontsize=lbl_size,
        )

    ax.set_xscale("log", base=2)
    if not args.linear_y:
        ax.set_yscale("log")

    ax.xaxis.set_major_locator(FixedLocator(xs))
    # Full "512 KiB" labels overlap once the figure is down at column width.
    tick = fmt_bytes_short if args.paper else fmt_bytes
    ax.xaxis.set_major_formatter(FixedFormatter([tick(x) for x in xs]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.set_xlabel("Pointer block size")
    what = "Total bandwidth" if args.mode == "total" else "Bandwidth per read"
    ax.set_ylabel(f"{what} ({unit})")

    label = rows[0]["pointer_label"]
    dataset = args.dataset or Path(args.logs).name

    # Title and subtitle both live above the axes, stacked with explicit gaps so
    # they never collide with the tallest y tick label.  The title is off by
    # default -- the LaTeX \caption carries it -- and opt-in via --title.
    show_title = bool(args.title)
    if show_title:
        ax.text(0.0, 1.13, f"{what} vs. pointer block size", transform=ax.transAxes,
                color=theme.text_primary, fontsize=13, fontweight="bold",
                va="bottom", ha="left")
    ax.text(0.0, 1.035 if show_title else 1.02, dataset,
            transform=ax.transAxes, color=theme.muted, fontsize=lbl_size,
            va="bottom", ha="left")

    # One series is already named by its direct label; a legend would be noise.
    if len(args.b) > 1:
        ax.legend(loc="upper left", title=None)
    # Extra room on the right for the direct labels.
    ax.set_xlim(xs[0] / 1.6, xs[-1] * 2.6)

    out_base = Path(args.out) if args.out else (
        REPO / "plots" / "out" / f"bandwidth_{dataset}_{rows[0]['pointer_type']}_{args.mode}"
    )
    if theme_name == "dark":
        out_base = out_base.with_name(out_base.name + "_dark")
    return style.save(fig, out_base, formats=args.formats)


def print_table(rows: list[dict], args) -> None:
    unit, scale = unit_for(args.pointer_type,
                           max(r[value_key(args.mode, b)] for r in rows for b in args.b))
    head = f"{'block size':>11}  {'reads':>9}  {'log2(allocs)':>12}"
    head += "".join(f"  {'b=' + str(b) + ' (' + unit + ')':>16}" for b in args.b)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{fmt_bytes(r['block_size']):>11}  {r['reads']:>9,}  {r['log2_allocs']:>12.3f}"
        line += "".join(f"  {r[value_key(args.mode, b)] / scale:>16,.2f}" for b in args.b)
        print(line)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--logs", default=str(REPO / "logs" / "sift1m"),
                   help="directory of o_hnsw.py log files (default: logs/sift1m)")
    p.add_argument("--pointer-type", default="rary",
                   help="rary (= multiwriterary), multiwrite, ... (default: rary)")
    p.add_argument("--partitioning", default=None,
                   help="only use runs with this partitioning, e.g. metis")
    p.add_argument("--b", type=int, nargs="+", default=[1],
                   help="values of b to plot (default: 1)")
    p.add_argument("--mode", choices=["per-read", "total"], default="per-read",
                   help="per-read = the formula alone (default); "
                        "total = per-read x total reads")
    p.add_argument("--allocs", choices=["whole-run", "build", "search"], default="whole-run",
                   help="which allocation count feeds log2 (default: whole-run)")
    p.add_argument("--dataset", default=None, help="label for the title")
    p.add_argument("--paper", action="store_true",
                   help="publication preset: column width, Times-compatible serif at "
                        "8pt, thin strokes, TrueType (not Type 3) embedding, no title")
    p.add_argument("--width", default=None,
                   help="figure width: 'column', 'text', or inches (default: "
                        "column under --paper, 7.6in otherwise)")
    p.add_argument("--title", dest="title", action="store_true", default=None,
                   help="draw the in-figure title (default: off)")
    p.add_argument("--no-title", dest="title", action="store_false",
                   help="drop the in-figure title; the LaTeX caption carries it")
    p.add_argument("--linear-y", action="store_true", help="linear y axis instead of log")
    p.add_argument("--theme", choices=["light", "dark", "both"], default="light")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--out", default=None, help="output path without extension")
    p.add_argument("--csv", default=None, help="also write the underlying numbers here")
    args = p.parse_args(argv)

    rows = collect(args)
    print_table(rows, args)
    if args.csv:
        write_csv(rows, Path(args.csv))

    themes = ["light", "dark"] if args.theme == "both" else [args.theme]
    for t in themes:
        for path in plot(rows, args, t):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
