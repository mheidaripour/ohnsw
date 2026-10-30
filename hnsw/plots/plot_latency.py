#!/usr/bin/env python
"""End-to-end latency vs. block size, one line per pointer scheme.

The intro figure: at a fixed accuracy level, what does a block size cost you
end to end, and how far apart do the two pointer schemes sit along that axis?

Latency model
-------------
A query pays for the round trips it serializes on, plus the bytes it drags
over the wire. Under ``--bandwidth-basis query`` (the correct reading)::

    latency = reads_per_query
            * (RTT + oram_cost(pointer_type, block_size, log2 allocs, b, r) / BW)

with ``RTT = 100 ms`` and ``BW = 100 MB/s`` by default. Each of the
``reads_per_query`` accesses costs one round trip plus one ORAM path's worth of
transfer, per ``bandwidth.oram_cost``:

    r-ary:      4 * (b + 1) * r * block_size * log2(N_alloc)
    multiwrite: 4 * (b + 1) * block_size * log2(N_alloc)

Both branches are in bytes, so bw_term below divides bytes by a byte rate and
the latency axis reads as real seconds for both schemes. The r-ary branch
carries r as well as the block size: --pointer-branching-factor pins r while
--bs sweeps, so the two are independent and an r-ary access no longer costs
what a multiwrite access at the same block size costs.

The legacy ``--bandwidth-basis access`` charges the bytes of a *single* access
per query while still charging every access a round trip. It is dimensionally
inconsistent -- a query making R accesses moves R paths, not one -- and is kept
only to reproduce tables published before the fix. Pipelining cannot rescue it:
overlap hides latency, not bytes, and this search is a sequential pointer chase
with nothing to overlap.

Both terms respond to block size, in opposite directions:

* Bigger blocks hold a whole partition, so ``get_neighbors_in_cluster`` fetches
  it in one read instead of many. Reads per query falls, and with it the RTT
  term.
* But bytes per access grow linearly in block size while reads per query only
  decays, so the bandwidth term rises.

Under the per-query basis both terms carry the ``reads_per_query`` factor, so
total bytes go as roughly ``B**0.8`` and the curve is monotone increasing across
every block size these sweeps cover: each scheme's cheapest measured point is
the smallest block swept, and the true optimum lies below it. Expect
``(AT AN ENDPOINT -- widen the sweep)`` on every series.

The figure rings and labels the lowest measured point on every curve either way.
Whether that point is an interior optimum or just where the sweep stopped is a
question ``print_table`` answers in words; the figure does not try to, because
marking one curve's minimum and silently skipping the other's -- which is what
an interior-only rule does on monotone data -- reads as an oversight rather than
as a caveat.

Fixed accuracy
--------------
Nothing here sweeps recall. Partitioning changes where data sits, not what the
search returns, so recall is expected to be identical at every point -- and the
table prints R@10 on every row so that claim is visible rather than asserted.
This script warns if it is not.

Reading the two series
----------------------
``--bs`` only reaches the r-ary branching factor (o_hnsw.py:2126-2130), so a
multiwrite run does not change when block size does. The OSAM+ series moves
because the *partition count* moves underneath it. Charging multiwrite for the
block size is still right -- the partition payload occupies a block of that
size whichever pointer scheme addresses it; multiwrite simply needs more
accesses to reach it. That is the comparison.

Examples
--------
    python plots/plot_latency.py --logs logs/1k_bssweep --dataset 1k
    python plots/plot_latency.py --logs logs/sift1m_bssweep --linear-y
    python plots/plot_latency.py --bandwidth 10e6      # a WAN link
    python plots/plot_latency.py --paper --out plots/out/fig_latency
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter

import style
from bandwidth import oram_cost, fmt_bytes, fmt_bytes_short
from logparse import POINTER_LABELS, POINTER_PAPER_LABELS, dedupe_by, load_dir

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_RTT = 0.1        # seconds
DEFAULT_BW = 100e6       # bytes/second

# Labelled tick ladder for the log y axis: a doubling sequence anchored on 10,
# i.e. ... 2.5, 5, 10, 20, 40 ... A latency axis is read as "twice as slow",
# not as an order of magnitude, so equal ratios between labels beat a per-decade
# 1-2-3-5-7 ladder, whose 3 and 7 steps land on no round number and read as an
# arbitrary place to put a gridline.
Y_ANCHOR = 10.0
Y_MAX_LABELS = 9        # past this the labels collide at column width


def _snapped(lo: float, hi: float, base: float, step: int) -> list[float]:
    """Ticks at ``Y_ANCHOR * base ** (step*k)`` bracketing ``[lo, hi]``."""
    eps = 1e-9
    k_lo = math.floor(math.log(lo / Y_ANCHOR, base) / step + eps)
    k_hi = math.ceil(math.log(hi / Y_ANCHOR, base) / step - eps)
    return [Y_ANCHOR * base ** (step * k) for k in range(k_lo, k_hi + 1)]


def ladder_ticks(lo: float, hi: float) -> list[float]:
    """Doubling ticks bracketing ``[lo, hi]``, anchored on 10.

    The range is snapped outward to ladder values: autoscale stops wherever the
    data does, which routinely leaves the topmost point sitting above the
    highest labelled tick with nothing to read its value against.

    Past about three decades a doubling ladder needs more labels than fit, and
    thinning it (x4, x8, ...) walks off the round numbers -- 0.0024, 0.15625.
    There the ladder falls back to decades, thinned by decades if need be, which
    stay round however far the range stretches. No figure in this directory is
    anywhere near that wide; the fallback is only so a WAN ``--bandwidth`` or a
    stray outlier degrades into a readable axis rather than an unreadable one.
    """
    ticks = _snapped(lo, hi, 2.0, 1)
    if len(ticks) <= Y_MAX_LABELS:
        return ticks
    step = 1
    while len(ticks := _snapped(lo, hi, 10.0, step)) > Y_MAX_LABELS:
        step += 1
    return ticks


def _widest_px(renderer, artists) -> float:
    """Widest rendered extent among these text artists, in display pixels."""
    if not artists:
        return 0.0
    return max(a.get_window_extent(renderer).width for a in artists)


def _pt_to_px(fig, pts: float) -> float:
    return pts * fig.dpi / 72.0


def _px_to_pt(fig, px: float) -> float:
    return px * 72.0 / fig.dpi


def series_label(pointer_type: str, label_style: str) -> str:
    if label_style == "paper":
        return POINTER_PAPER_LABELS.get(pointer_type, pointer_type)
    return POINTER_LABELS.get(pointer_type, pointer_type)


def rows_for(runs, args) -> list[dict]:
    """One row per run, with both latency terms broken out."""
    rows = []
    for run in runs:
        rpq = run.reads_per_query
        if rpq is None:
            print(f"  ! skipping {run.path.name}: no reads/query in log", file=sys.stderr)
            continue

        log2_allocs = run.log2_allocs(args.allocs)
        per_read = oram_cost(run.pointer_type, run.block_size, log2_allocs,
                             args.b, run.branching_factor)

        # `query` charges every access a query makes, which is the correct
        # reading: the accesses are a sequential pointer chase, and bandwidth is
        # a rate, so pipelining could hide round trips but never bytes.
        # `access` charges one access per query and only reproduces the earlier
        # tables. Keep in step with make_tables.row_for.
        bytes_moved = per_read * (rpq if args.bandwidth_basis == "query" else 1)
        rtt_count = rpq if args.rtt_basis == "read" else run.avg_rounds_per_query

        if rtt_count is None:
            print(f"  ! skipping {run.path.name}: no rounds/query in log", file=sys.stderr)
            continue

        bw_term = (bytes_moved / args.bandwidth)
        rtt_term = rtt_count * args.rtt

        rows.append({
            "log": run.path.name,
            "pointer_type": run.pointer_type,
            "n_clusters": run.num_clusters,
            "max_cluster_size": run.max_cluster_size,
            "block_size": run.block_size,
            "reads_per_query": rpq,
            "log2_allocs": log2_allocs,
            "per_read_bytes": per_read,
            "bytes_per_query": bytes_moved,
            "rtt_term": rtt_term,
            "bw_term": bw_term,
            "latency": rtt_term + bw_term,
            "recall_at_10": run.recall_at_10,
        })
    return rows


def collect(args) -> dict[str, list[dict]]:
    """``{pointer_type: rows sorted by block size}``, one entry per series."""
    out = {}
    for pointer_type in args.pointer_types:
        runs, dropped = load_dir(
            args.logs, pointer_type=pointer_type, partitioning=args.partitioning
        )
        for run in dropped:
            print(f"  ! skipping incomplete log (no memory-access summary): "
                  f"{run.path.name}", file=sys.stderr)
        if not runs:
            print(f"  ! no complete {pointer_type} logs in {args.logs} -- "
                  f"series omitted", file=sys.stderr)
            continue

        # A config is (partition count, block size). Deduping on block size
        # alone would silently drop one of two partition counts that rounded
        # to the same block, which is a real possibility -- see the run=0 rows
        # probe_partitions.py emits.
        runs = dedupe_by(runs, key=lambda r: (r.num_clusters, r.block_size))
        runs.sort(key=lambda r: r.block_size)

        rows = rows_for(runs, args)
        if not rows:
            continue

        seen = {}
        for row in rows:
            bs = row["block_size"]
            if bs in seen:
                raise SystemExit(
                    f"{pointer_type}: two runs share block size {fmt_bytes(bs)} "
                    f"(n_clusters {seen[bs]} and {row['n_clusters']}). Re-run "
                    f"probe_partitions.py, which marks such duplicates run=0, "
                    f"and drop the extra log."
                )
            seen[bs] = row["n_clusters"]

        out[pointer_type] = rows

    if not out:
        raise SystemExit(f"no usable logs found in {args.logs}")
    return out


def check_fixed_accuracy(series: dict[str, list[dict]]) -> None:
    """The premise of the figure: cost varies, accuracy does not."""
    recalls = {r["recall_at_10"] for rows in series.values() for r in rows
               if r["recall_at_10"] is not None}
    if len(recalls) > 1:
        lo, hi = min(recalls), max(recalls)
        print(f"  ! R@10 varies from {lo:.4f} to {hi:.4f} across these runs, so "
              f"this is NOT a fixed-accuracy comparison", file=sys.stderr)


def interpolated_minimum(rows: list[dict]):
    """Refine the argmin by fitting a parabola in log2(block size).

    The sampled minimum is only as precise as the partition-count grid. Fitting
    the argmin and its two neighbours gives a better estimate of where the
    curve actually bottoms out -- reported in the table only, since drawing it
    would imply a resolution the data does not have.
    """
    import math

    i = min(range(len(rows)), key=lambda j: rows[j]["latency"])
    if i == 0 or i == len(rows) - 1:
        return None  # at an endpoint: the true minimum is outside the sweep

    x = [math.log2(rows[j]["block_size"]) for j in (i - 1, i, i + 1)]
    y = [rows[j]["latency"] for j in (i - 1, i, i + 1)]

    d1, d2 = x[1] - x[0], x[2] - x[1]
    denom = (y[0] - y[1]) * d2 + (y[2] - y[1]) * d1
    if denom == 0:
        return None
    # Vertex of the parabola through the three points.
    offset = 0.5 * ((y[0] - y[1]) * d2 * d2 - (y[2] - y[1]) * d1 * d1) / denom
    return 2 ** (x[1] + offset)


def figure_size(args) -> tuple[float, float]:
    if args.width:
        w = {"column": style.COLUMN_WIDTH_IN, "text": style.TEXT_WIDTH_IN}.get(args.width)
        w = float(args.width) if w is None else w
    elif args.paper:
        w = style.COLUMN_WIDTH_IN
    else:
        w = 7.6
    return (w, w * (0.72 if args.paper else 0.605))


def plot(series: dict[str, list[dict]], args, theme_name: str) -> list[Path]:
    theme = style.THEMES[theme_name]
    style.apply(theme, paper=args.paper)

    fig, ax = plt.subplots(figsize=figure_size(args))
    style.style_axes(ax, theme)
    lbl_size = 7 if args.paper else 10

    # Union of both series' block sizes: a pointer type missing one config
    # leaves a gap in its own line rather than moving the other's ticks.
    all_x = sorted({r["block_size"] for rows in series.values() for r in rows})

    # Full "512 KiB" labels collide once there are more than about six of them,
    # and always at column width. fmt_bytes_short gives "512K" instead, roughly
    # halving the label width. Picked here rather than down at the axis so a
    # "min @ ..." label quotes its block size exactly the way its tick does.
    tick = fmt_bytes_short if (args.paper or len(all_x) > 6) else fmt_bytes

    # Slot order is fixed across every figure in this directory: 0 = r-ary,
    # 1 = multiwrite. Keeping it stable is what lets a reader carry colour
    # meaning between figures.
    slot_of = {"rary": 0, "multiwrite": 1}

    # Every y value actually drawn, component overlays included, so the axis
    # can be sized to cover all of them rather than to autoscale's margins.
    y_all: list[float] = []

    # Text hanging off the right end of a curve. The right x limit is solved
    # from their measured widths so the box encloses them rather than cutting
    # through them -- see the measurement pass below.
    edge_labels: list = []
    # One record per ringed minimum, placed after the first draw once the two
    # labels' rendered extents are known.
    min_marks: list[dict] = []

    for pointer_type, rows in series.items():
        i = slot_of.get(pointer_type, len(slot_of) + list(series).index(pointer_type))
        color = theme.series[i % len(theme.series)]
        label = series_label(pointer_type, args.label_style)

        xs = [r["block_size"] for r in rows]
        ys = [r["latency"] for r in rows]
        y_all += ys

        ax.plot(
            xs, ys,
            color=color,
            marker=style.series_marker(i),
            markeredgecolor=theme.surface,   # surface ring where marks overlap
            label=label,
            zorder=3 - i * 0.1,
        )
        # Direct label at the right end, so identity is not colour-alone. It
        # sits inside the box (the x limit is widened to make room), so it needs
        # a surface-coloured patch behind it to stay legible over a gridline.
        edge_labels.append(ax.annotate(
            label,
            xy=(xs[-1], ys[-1]),
            xytext=(6 if args.paper else 8, 0),
            textcoords="offset points",
            va="center", ha="left",
            color=theme.text_primary,
            fontsize=lbl_size,
            fontweight="bold",
            bbox=dict(facecolor=theme.surface, edgecolor="none", pad=0.8),
            zorder=5,
        ))

        # Ring the lowest measured point on every curve, wherever it falls.
        # Whether that point is an interior optimum or just where the sweep
        # stopped is a question the figure does not try to answer -- print_table
        # says "(AT AN ENDPOINT -- widen the sweep)" in words instead. Leaving
        # an endpoint minimum unmarked only made the figure look inconsistent,
        # labelling one curve and not the other.
        j = min(range(len(rows)), key=lambda n: rows[n]["latency"])
        if args.annotate_min:
            ax.plot(
                [xs[j]], [ys[j]],
                marker="o",
                markersize=matplotlib.rcParams["lines.markersize"] * style.MIN_RING_SCALE,
                markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=1.0 if args.paper else 1.4,
                linestyle="none", zorder=4,
            )
            min_marks.append({
                "text": f"min @ {tick(xs[j])}",
                "xy": (xs[j], ys[j]),
                "y": ys[j],
                "color": color,
                # Left/centre/right by where the point falls along the sweep, so
                # a label on either edge hangs inward rather than off the box.
                "frac": all_x.index(xs[j]) / max(1, len(all_x) - 1),
            })

        if args.show_components and pointer_type in args.components_for:
            for key, tag, dash in (("rtt_term", "RTT", (0, (1, 1.6))),
                                   ("bw_term", "BW", (0, (4, 1.6)))):
                cs = [r[key] for r in rows]
                y_all += [c for c in cs if c > 0]
                ax.plot(xs, cs, color=color, alpha=0.4, linestyle=dash,
                        marker="none", linewidth=0.9, zorder=1.5)
                edge_labels.append(ax.annotate(
                    tag, xy=(xs[-1], cs[-1]),
                    xytext=(6 if args.paper else 8, 0),
                    textcoords="offset points", va="center", ha="left",
                    color=color, alpha=0.7, fontsize=lbl_size - 1))

    ax.set_xscale("log", base=2)

    # A gridline at every measured block size, whether or not it ends up
    # labelled: the thinning pass below blanks label text, never ticks.
    ax.xaxis.set_major_locator(FixedLocator(all_x))
    ax.xaxis.set_major_formatter(FixedFormatter([tick(x) for x in all_x]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    if not args.linear_y:
        ax.set_yscale("log")
        # These are seconds, not orders of magnitude, so matplotlib's default
        # LogFormatterSciNotation ("6 x 10^0") is actively misleading -- nobody
        # reads latency that way. Plain numbers on the doubling ladder instead.
        plain = FuncFormatter(lambda v, _pos: f"{v:g}")
        ticks = ladder_ticks(min(y_all), max(y_all))
        # Every step is a doubling, so no step is more structural than another:
        # one set of majors, one gridline weight. (The decade split the 1-2-3-5-7
        # ladder needed only existed because 1 and 10 *were* special there.)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(plain)
        ax.set_ylim(ticks[0], ticks[-1])

    ax.set_xlabel("Block Size")
    ax.set_ylabel("End-to-end latency per query (s)")

    dataset = args.dataset or Path(args.logs).name
    net = (f"{args.rtt * 1e3:g} ms RTT, {args.bandwidth / 1e6:g} MB/s")
    subtitle = f"{dataset}  ·  {net}"
    # The two bandwidth bases differ by a factor of reads/query and are
    # otherwise visually identical, so record which one drew this figure.
    if args.bandwidth_basis != "query":
        subtitle += "  ·  BW/access"

    recalls = {r["recall_at_10"] for rows in series.values() for r in rows
               if r["recall_at_10"] is not None}
    if len(recalls) == 1:
        subtitle += f"  ·  R@10 = {recalls.pop():.4f}"

    # Title and subtitle both live above the axes, stacked with explicit gaps so
    # they never collide with the tallest y tick label. Both are off under
    # --paper -- the LaTeX \caption carries the dataset, the network and R@10,
    # and the space is better spent on the plot -- and stay on for screen runs,
    # where they are what tells two output files apart.
    show_title = bool(args.title)
    show_subtitle = (not args.paper) if args.subtitle is None else args.subtitle
    if show_title:
        ax.text(0.0, 1.13, "End-to-end latency vs. block size",
                transform=ax.transAxes, color=theme.text_primary,
                fontsize=13, fontweight="bold", va="bottom", ha="left")
    if show_subtitle:
        ax.text(0.0, 1.035 if show_title else 1.02, subtitle,
                transform=ax.transAxes, color=theme.muted, fontsize=lbl_size,
                va="bottom", ha="left")

    # --- measurement pass ------------------------------------------------
    # Three things below need rendered text extents, which only exist after a
    # draw: the right margin that has to hold the direct series labels, how many
    # x tick labels fit without touching, and where the "min @ ..." labels can
    # go. The axes rect is the matplotlib default (no tight/constrained layout,
    # and savefig's bbox="tight" only crops the saved region afterwards), so the
    # axes geometry measured here is the geometry that gets written -- one draw
    # per adjustment, no iterating to a fixed point.
    left = all_x[0] / 1.6
    ax.set_xlim(left, all_x[-1] * 2.6)     # provisional, just to have a scale

    # Both series carry direct labels; a legend would repeat them. Solve the
    # right limit so those labels land *inside* the box: with the axis log2, a
    # label occupying fraction f of the axes width needs the drawn span to be
    # the data span over (1 - f).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_w = ax.get_window_extent().width
    # get_window_extent on a Text is the glyphs only -- the bbox patch drawn
    # behind it sits outside that, so reserve the offset from the point plus a
    # few points of slack for the patch, or the label lands *on* the spine.
    label_px = _widest_px(renderer, edge_labels) + _pt_to_px(
        fig, (6 if args.paper else 8) + 4)
    if axes_w > 0 and label_px > 0:
        span = math.log2(all_x[-1] / left)
        # Capped so a pathological label cannot squash the data into a sliver.
        frac = min(label_px / axes_w, 0.45)
        ax.set_xlim(left, left * 2 ** (span / (1.0 - frac)))
        fig.canvas.draw()

    # Same problem in the other direction: a curve whose last point sits just
    # under the top tick gets a direct label that rides the top spine. These
    # labels are already the rightmost thing on their curve, so nudging one
    # clear of the frame costs no clarity -- there is nothing it could be
    # confused with.
    box = ax.get_window_extent()
    for ann in edge_labels:
        ext = ann.get_window_extent(renderer)
        over = max(ext.y1 - box.y1, 0.0) or -max(box.y0 - ext.y0, 0.0)
        if over:
            dx, dy = ann.get_position()
            ann.set_position((dx, dy - _px_to_pt(fig, over) - 1.0))
    fig.canvas.draw()

    # Thin the x tick *labels* until no two touch, keeping both ends labelled so
    # the reader can always see where the sweep starts and stops. The ticks and
    # their gridlines stay at every measured block size.
    labels = [tick(x) for x in all_x]
    if len(all_x) > 1:
        x_px = [ax.transData.transform((x, ax.get_ylim()[0]))[0] for x in all_x]
        step_px = (x_px[-1] - x_px[0]) / (len(all_x) - 1)
        wide_px = _widest_px(renderer, ax.get_xticklabels()) + _pt_to_px(fig, 3)
        keep = 1
        while keep < 5 and step_px * keep < wide_px:
            keep += 1
        if keep > 1:
            last = len(all_x) - 1
            shown = set(range(0, len(all_x), keep)) | {last}
            # Forcing the last index in can leave it a partial step from its
            # neighbour; drop that neighbour rather than let the two touch.
            if last % keep:
                shown.discard(last - (last % keep))
            labels = [labels[i] if i in shown else "" for i in range(len(all_x))]
            ax.xaxis.set_major_formatter(FixedFormatter(labels))
            fig.canvas.draw()

    # Place the minimum labels. With the curves monotone under the per-query
    # basis every series bottoms out at the same leftmost block, so stacking all
    # the labels on one side would pile them onto each other and onto the rings.
    # Instead they fan out from their own points: the highest minimum labels
    # upward into the empty region above its curve, the rest downward into the
    # empty region under theirs. Two labels leaving two different points in two
    # different directions cannot collide.
    if min_marks:
        top = max(m["y"] for m in min_marks)
        # Both gaps are measured from the ring's edge, not from the point, so
        # they hold if the mark or MIN_RING_SCALE is ever retuned.
        ring_r = matplotlib.rcParams["lines.markersize"] * style.MIN_RING_SCALE / 2
        # Asymmetric on purpose. Above the point the label also has to clear the
        # curve climbing away to the right, so it needs the room. Below there is
        # nothing to clear -- the curve rises away from its own minimum -- so it
        # tucks in just under the ring, which reads as attached to the mark
        # rather than adrift, and keeps the axis extension below small.
        gap_up = ring_r + (7.0 if args.paper else 9.0)
        gap_down = ring_r + (2.5 if args.paper else 3.5)
        below = []
        for m in min_marks:
            up = m["y"] >= top
            ha, dx = ("left", -4) if m["frac"] <= 1 / 3 else (
                ("right", 4) if m["frac"] >= 2 / 3 else ("center", 0))
            ann = ax.annotate(
                m["text"], xy=m["xy"],
                xytext=(dx, gap_up if up else -gap_down),
                textcoords="offset points",
                ha=ha, va="bottom" if up else "top",
                color=m["color"], fontsize=lbl_size, zorder=5,
            )
            if not up:
                below.append((ann, m["xy"]))
        fig.canvas.draw()

        # A downward label can hang below the lowest ladder tick, which is where
        # the y axis stops. Give it the room rather than clipping it: the tick
        # ladder is unchanged, the bottom tick just no longer sits on the spine.
        #
        # Extending the axis also *compresses* the data, so the point the label
        # hangs from does not rise by the full extension -- growing the range by
        # a fraction g lifts a point sitting u of the way up to (u+g)/(1+g).
        # Solve that for the g which puts the label's own gap above the spine
        # instead of stepping by the shortfall and undershooting.
        box = ax.get_window_extent()
        h = box.height
        want = 0.0
        for ann, xy in below:
            bottom = ann.get_window_extent(renderer).y0
            point = ax.transData.transform(xy)[1]
            gap = (point - bottom + _pt_to_px(fig, 2)) / h   # label's own reach
            need = (box.y0 + _pt_to_px(fig, 2) - bottom) / h  # shortfall
            if need > 0 and gap < 1:
                want = max(want, need / (1.0 - gap))
        if want > 0:
            y0, y1 = ax.get_ylim()
            if args.linear_y:
                ax.set_ylim(y0 - (y1 - y0) * want, y1)
            else:
                ax.set_ylim(y0 / (y1 / y0) ** want, y1)

    tag = "-".join(sorted(series))
    out_base = Path(args.out) if args.out else (
        REPO / "plots" / "out" / f"latency_{dataset}_{tag}_{args.rtt_basis}"
    )
    if theme_name == "dark":
        out_base = out_base.with_name(out_base.name + "_dark")
    return style.save(fig, out_base, formats=args.formats)


def print_table(series: dict[str, list[dict]], args) -> None:
    head = (f"{'pointer':>11}  {'n_clust':>8}  {'block size':>11}  {'reads/q':>9}  "
            f"{'log2(allocs)':>12}  {'RTT (s)':>9}  {'BW (s)':>9}  "
            f"{'latency (s)':>12}  {'R@10':>7}")
    print(head)
    print("-" * len(head))
    for pointer_type, rows in series.items():
        label = series_label(pointer_type, args.label_style)
        for r in rows:
            recall = f"{r['recall_at_10']:.4f}" if r["recall_at_10"] is not None else "-"
            # Unpartitioned runs print no cluster block, so there is no count
            # to report -- logparse.py:66-67.
            nc = "none" if r["n_clusters"] is None else f"{r['n_clusters']:,}"
            print(f"{label:>11}  {nc:>8}  "
                  f"{fmt_bytes(r['block_size']):>11}  {r['reads_per_query']:>9.1f}  "
                  f"{r['log2_allocs']:>12.3f}  {r['rtt_term']:>9.3f}  "
                  f"{r['bw_term']:>9.3f}  {r['latency']:>12.3f}  {recall:>7}")

    print()
    for pointer_type, rows in series.items():
        label = series_label(pointer_type, args.label_style)
        j = min(range(len(rows)), key=lambda n: rows[n]["latency"])
        best = rows[j]
        note = f"  minimum {best['latency']:.3f}s at {fmt_bytes(best['block_size'])}"
        if j in (0, len(rows) - 1):
            note += "  (AT AN ENDPOINT -- widen the sweep)"
        else:
            fit = interpolated_minimum(rows)
            if fit:
                note += f", interpolated optimum ~{fmt_bytes(fit)}"
        print(f"{label:>11}{note}")


def write_csv(series: dict[str, list[dict]], path: Path) -> None:
    rows = [r for rows in series.values() for r in rows]
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
    p.add_argument("--logs", default=str(REPO / "logs" / "sift1m_bssweep"),
                   help="directory of o_hnsw.py log files")
    p.add_argument("--pointer-types", nargs="+", default=["rary", "multiwrite"],
                   help="one series per pointer type (default: rary multiwrite)")
    p.add_argument("--partitioning", default=None,
                   help="only use runs with this partitioning, e.g. metis")
    p.add_argument("--b", type=int, default=1,
                   help="eviction/bucket parameter in the bandwidth formula (default: 1)")
    p.add_argument("--allocs", choices=["whole-run", "build", "search"], default="whole-run",
                   help="which allocation count feeds log2 (default: whole-run)")
    p.add_argument("--rtt", type=float, default=DEFAULT_RTT,
                   help="seconds per round trip (default: 0.1)")
    p.add_argument("--bandwidth", type=float, default=DEFAULT_BW,
                   help="network bandwidth in bytes/s (default: 100e6)")
    p.add_argument("--bandwidth-basis", choices=["access", "query"], default="query",
                   help="query = the bytes a whole query moves, reads/query "
                        "times one access (default); this is the paper's "
                        "T = R(RTT + S/BW). access = one access worth of bytes "
                        "-- kept only to reproduce pre-fix figures. Must match "
                        "make_tables.py")
    p.add_argument("--rtt-basis", choices=["read", "round"], default="read",
                   help="read = one round trip per ORAM read (default); "
                        "round = per logical HNSW hop. DIAGNOSTIC ONLY: logical "
                        "rounds do not respond to block size or pointer type, so "
                        "'round' turns the first term into a constant offset and "
                        "the curve monotone by construction")
    p.add_argument("--label-style", choices=["paper", "canonical"], default="paper",
                   help="paper = BOSAM/OSAM+ (default); canonical = r-ary/multiwrite")
    p.add_argument("--annotate-min", dest="annotate_min", action="store_true", default=True,
                   help="ring each curve's minimum (default: on)")
    p.add_argument("--no-annotate-min", dest="annotate_min", action="store_false",
                   help="drop the minimum annotation")
    p.add_argument("--show-components", action="store_true",
                   help="overlay the RTT and bandwidth terms separately")
    p.add_argument("--components-for", nargs="+", default=["rary"],
                   help="which series get component overlays (default: rary). "
                        "Six lines at column width are unreadable, hence one series")
    p.add_argument("--dataset", default=None, help="label for the subtitle")
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
    p.add_argument("--subtitle", dest="subtitle", action="store_true", default=None,
                   help="draw the dataset/network/R@10 line above the axes "
                        "(default: on, but off under --paper, where the LaTeX "
                        "caption carries it)")
    p.add_argument("--no-subtitle", dest="subtitle", action="store_false",
                   help="drop the in-figure subtitle")
    p.add_argument("--linear-y", action="store_true", help="linear y axis instead of log")
    p.add_argument("--theme", choices=["light", "dark", "both"], default="light")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--out", default=None, help="output path without extension")
    p.add_argument("--csv", default=None, help="also write the underlying numbers here")
    args = p.parse_args(argv)

    series = collect(args)
    check_fixed_accuracy(series)

    missing = set()
    for pointer_type, rows in series.items():
        xs = {r["block_size"] for r in rows}
        for other, other_rows in series.items():
            if other == pointer_type:
                continue
            for bs in {r["block_size"] for r in other_rows} - xs:
                missing.add((pointer_type, bs, other))
    for pointer_type, bs, other in sorted(missing, key=lambda t: t[1]):
        print(f"  ! {pointer_type} has no run at {fmt_bytes(bs)} (but {other} does)",
              file=sys.stderr)

    print_table(series, args)
    if args.csv:
        write_csv(series, Path(args.csv))

    themes = ["light", "dark"] if args.theme == "both" else [args.theme]
    for t in themes:
        for path in plot(series, args, t):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
