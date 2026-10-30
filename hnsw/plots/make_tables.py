"""Emit the experiments-section LaTeX tables from the block-size sweep logs.

Three tables, written to ``hnsw/paper/tables/`` as standalone ``.tex`` files
plus a ``tables.tex`` that ``\\input``s all three:

  1. ``tab_datasets.tex``  -- dataset configuration (size, dim, efC, efS, M, metric)
  2. ``tab_cost.tex``      -- oblivious memory cost per query at the best block size
  3. ``tab_network.tex``   -- modelled bandwidth/latency and recall

The numbers come from the same parsers the figures use (``logparse``,
``bandwidth``), and the latency model is the one in ``plot_latency.py`` -- same
defaults, same formula -- so a table can never disagree with the plot beside it.

Each scheme is reported at *its own* latency-minimising block size.  A dataset
with no sweep logs still contributes its configuration row to table 1 and is
simply absent from tables 2 and 3.

``--bandwidth-basis query`` charges every access a query makes, which is the
correct reading: the search is sequential pointer chasing, so the accesses
cannot overlap on the wire, and bandwidth is a rate -- pipelining could hide
round trips but never bytes.  Under it both terms scale with reads/query, every
curve is monotone in the block size over the measured range, and each scheme's
optimum pins to the smallest block swept.  It is the default.  ``access``
charges one access per query while still charging every access a round trip,
and is kept only to reproduce earlier tables.

Usage::

    python plots/make_tables.py --bandwidth-basis query
    python plots/make_tables.py --bandwidth 10e6 --rtt 0.05   # a different link
    python plots/make_tables.py --datasets sift1m laion1m 1k 10k
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bandwidth import oram_cost, fmt_bytes
from logparse import POINTER_PAPER_LABELS, dedupe_by, load_dir

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_RTT = 0.1        # seconds        -- keep in step with plot_latency.py
DEFAULT_BW = 100e6       # bytes/second

POINTER_ORDER = ["rary", "multiwrite"]


@dataclass(frozen=True)
class DatasetSpec:
    """The build-time configuration of one dataset.

    Mirrors the dataset case block in ``exp_blocksize_sweep.sh``; M and
    efConstruction are the two numbers in the graph file name
    (``hnsw_<M>_<efConstruction>[_<out-tree branching>]_<metric>.json``).
    """

    label: str
    size: int
    dim: int
    ef_construction: int
    ef_search: int
    m: int
    metric: str
    queries: int
    footnote: str = ""      # LaTeX marker, e.g. r"\dagger"
    log_dir: str = ""       # sweep log directory, default ``<key>_bssweep``


DATASETS: dict[str, DatasetSpec] = {
    "1k": DatasetSpec("LAION-1k", 1_000, 512, 40, 64, 32, "IP", 100),
    "10k": DatasetSpec("LAION-10k", 10_000, 512, 40, 64, 32, "IP", 100),
    "100k": DatasetSpec("LAION-100k", 100_000, 512, 40, 64, 32, "IP", 100),
    # SIFT is swept under L2, not IP. Its ground truth is L2 -- exact L2
    # reproduces gt's top-10 at 100% where exact IP manages 91% -- so the older
    # IP sweep in logs/sift1m_bssweep measured a metric mismatch as much as it
    # measured the system, and plateaued at R@10 = 0.921. The L2 sweep reaches
    # 0.9921. Those logs live in their own directory so both survive; point
    # log_dir back at "sift1m_bssweep" to reproduce the IP tables.
    "sift1m": DatasetSpec("SIFT-1M", 1_000_000, 128, 80, 64, 64, "L2", 1000,
                          log_dir="sift1m-l2_bssweep"),
    "laion1m": DatasetSpec("LAION-1M", 1_000_448, 512, 40, 64, 32, "IP", 1000),
    "tripclick": DatasetSpec("TripClick", 1_523_871, 768, 160, 64, 128, "IP", 1000,
                             footnote=r"\ddagger"),
    "msmarco1m": DatasetSpec("MSMARCO-1M", 1_000_000, 768, 200, 64, 128, "IP", 1000,
                             footnote=r"\S"),
}

DEFAULT_DATASETS = ["sift1m", "laion1m", "tripclick", "msmarco1m"]

# Footnote bodies, keyed by the marker they hang off. Only the markers actually
# used by the emitted rows are printed.
FOOTNOTES = {
    # \dagger used to explain SIFT's IP/L2 mismatch and the R@10 = 0.921 ceiling
    # it imposed. The sweep now runs under L2, so the Metric column states it
    # plainly and no footnote is needed. Restore this entry if sift1m is ever
    # pointed back at the IP logs.
    r"\ddagger": (
        "Configuration listed for completeness; the TripClick sweep has not "
        "been run."
    ),
    r"\S": (
        "The first 1M of MSMARCO's 8,841,823 passages, embedded with "
        "TCT-ColBERT-v2-HN+ (768-d, dot product). The graph parameters are "
        "Compass's own for this dataset ($M{=}128$, efConstruction $200$); the "
        "corpus is truncated to 1M so the oblivious sweep fits in memory, and "
        "to match the scale of the other three rows."
    ),
}


# --------------------------------------------------------------- formatting ---

def tex_int(n: float) -> str:
    """Thousands-separated integer that survives both text and math mode."""
    return f"{round(n):,}".replace(",", "{,}")


def tex_num(x: float) -> str:
    """Per-query counts: separators once they get large, one decimal below."""
    return tex_int(x) if abs(x) >= 1000 else f"{x:.1f}"


def tex_sci(n: float, digits: int = 2) -> str:
    """``$8.36\\times10^{7}$`` -- allocation counts are only read as magnitudes."""
    if n <= 0:
        return "--"
    exp = math.floor(math.log10(n))
    return rf"${n / 10 ** exp:.{digits}f}\times10^{{{exp}}}$"


def tex_bytes(n: int) -> str:
    """Binary units with a thin space: ``512\\,KiB``. Block sizes are powers of 2."""
    return fmt_bytes(n).replace(" ", r"\,")


def tex_si_bytes(n: float) -> str:
    """Decimal units: ``110.4\\,MB``.

    Deliberately decimal, unlike :func:`tex_bytes`. The bandwidth is quoted in
    decimal MB/s, so decimal bytes let a reader divide the two columns in their
    head and land on the latency column. That only works if the mantissa keeps
    enough digits to be worth dividing, so small ones get an extra decimal --
    ``3.7\\,GB`` implies 37 s of transfer where ``3.67\\,GB`` implies the 36.7 s
    the latency column actually reports.
    """
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            mantissa = n / scale
            return rf"{mantissa:.2f}\,{unit}" if mantissa < 10 else rf"{mantissa:.1f}\,{unit}"
    return rf"{n:.0f}\,B"


# ------------------------------------------------------------------- loading ---

def row_for(run, args) -> dict | None:
    """One sweep run, reduced to the quantities the tables report.

    The latency model is ``plot_latency.rows_for``: one round trip per ORAM
    read, plus the bytes those reads move, per ``bandwidth.oram_cost`` --
    every access under ``--bandwidth-basis query``, one under ``access``.
    """
    rpq = run.reads_per_query
    if rpq is None:
        print(f"  ! skipping {run.path.name}: no reads/query in log", file=sys.stderr)
        return None

    log2_allocs = run.log2_allocs(args.allocs)
    per_access = oram_cost(run.pointer_type, run.block_size, log2_allocs,
                           args.b, run.branching_factor)

    # `query` charges every access a query makes; `access` charges one. The
    # search is sequential pointer chasing, so the accesses cannot overlap and
    # `query` is the honest reading -- `access` is kept only to reproduce
    # earlier tables. Keep in step with plot_latency.rows_for.
    per_query = per_access * (rpq if args.bandwidth_basis == "query" else 1)

    queries = run.num_queries or 0
    rounds = (run.total_rounds / queries) if (run.total_rounds and queries) \
        else run.avg_rounds_per_query
    writes = (run.total_writes / queries) if (run.total_writes is not None and queries) \
        else run.avg_writes_per_query

    return {
        "log": run.path.name,
        "pointer_type": run.pointer_type,
        "n_clusters": run.num_clusters,
        "max_cluster_size": run.max_cluster_size,
        "branching_factor": run.branching_factor,
        "block_size": run.block_size,
        "rounds_per_query": rounds,
        "reads_per_query": rpq,
        "writes_per_query": writes,
        "whole_run_allocs": run.whole_run_allocs,
        "log2_allocs": log2_allocs,
        "per_access_bytes": per_access,
        "bytes_per_query": per_query,
        "latency": per_query / args.bandwidth + rpq * args.rtt,
        "recall_at_1": run.recall_at_1,
        "recall_at_10": run.recall_at_10,
        "ef_search": run.ef_search,
        "num_queries": queries,
    }


def best_rows(dataset: str, args) -> list[dict]:
    """The latency-minimising run per pointer scheme, in POINTER_ORDER."""
    spec = DATASETS[dataset]
    log_dir = Path(args.logs_root) / (spec.log_dir or f"{dataset}_bssweep")
    if not log_dir.is_dir():
        return []

    out = []
    for pointer_type in POINTER_ORDER:
        runs, dropped = load_dir(log_dir, pointer_type=pointer_type,
                                 partitioning=args.partitioning)
        for run in dropped:
            print(f"  ! {dataset}: skipping incomplete log {run.path.name}",
                  file=sys.stderr)
        if not runs:
            continue

        runs = dedupe_by(runs, key=lambda r: (r.num_clusters, r.block_size))
        runs.sort(key=lambda r: r.block_size)
        rows = [r for r in (row_for(r, args) for r in runs) if r]
        if not rows:
            continue

        i = min(range(len(rows)), key=lambda j: rows[j]["latency"])
        if i in (0, len(rows) - 1):
            print(f"  ! {dataset}/{pointer_type}: the minimum sits at an endpoint "
                  f"of the sweep ({tex_bytes(rows[i]['block_size'])}) -- the true "
                  f"optimum may lie outside it", file=sys.stderr)
        out.append(rows[i])
    return out


def check_against_spec(dataset: str, rows: list[dict]) -> None:
    """The config table is hand-written; make sure the runs agree with it."""
    spec = DATASETS[dataset]
    for r in rows:
        if r["ef_search"] not in (None, spec.ef_search):
            print(f"  ! {dataset}: {r['log']} ran ef_search={r['ef_search']}, but the "
                  f"dataset table says {spec.ef_search}", file=sys.stderr)
        if r["num_queries"] not in (0, spec.queries):
            print(f"  ! {dataset}: {r['log']} ran {r['num_queries']} queries, but the "
                  f"dataset table says {spec.queries}", file=sys.stderr)


# -------------------------------------------------------------------- tables ---

def wrap(body: str, colspec: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def footnote_text(markers: list[str]) -> str:
    # Ordered by FOOTNOTES rather than by a hardcoded marker list, so adding a
    # dataset with a new marker does not silently drop its footnote body -- the
    # marker still renders on the row, which is the failure that looks fine.
    used = [m for m in FOOTNOTES if m in markers]
    unknown = [m for m in markers if m not in FOOTNOTES]
    if unknown:
        raise KeyError(f"dataset footnote markers with no FOOTNOTES entry: "
                       f"{sorted(set(unknown))}")
    return "".join(f"\n${m}$~{FOOTNOTES[m]}" for m in used)


def table_datasets(datasets: list[str], args) -> str:
    header = (r"\textsf{Dataset} & \textsf{Size} & \textsf{Dim} & "
              r"$\mathsf{efConst.}$ & $\mathsf{efSearch}$ & \textsf{M} & "
              r"\textsf{Metric} \\")

    lines, markers = [], []
    for name in datasets:
        s = DATASETS[name]
        metric = s.metric + (f"$^{{{s.footnote}}}$" if s.footnote else "")
        if s.footnote:
            markers.append(s.footnote)
        lines.append(f"{s.label} & {tex_int(s.size)} & {s.dim} & {s.ef_construction} & "
                     f"{s.ef_search} & {s.m} & {metric} \\\\")

    body = header + "\n\\midrule\n" + "\n".join(lines)
    caption = (
        "Dataset configuration. Abbrev.: Dim=dimensions, "
        r"$\mathsf{efConst.}$=$\mathsf{efConstruction}$, M=max connections per "
        "node, Metric=similarity metric (L2 or IP). All runs retrieve $k=10$ "
        "neighbours." + footnote_text(markers)
    )
    return wrap(body, "l | r | r | r | r | r | c", caption, "tab:datasets")


def table_cost(results: dict[str, list[dict]], args) -> str:
    header = (r"\textsf{Dataset} & \textsf{Scheme} & \textsf{Block} & "
              r"$|P|_{\max}$ & $r$ & \textsf{Rounds/q} & \textsf{Reads/q} & "
              r"\textsf{Writes/q} & \textsf{Allocs} \\")

    blocks = []
    for name, rows in results.items():
        out = []
        for i, r in enumerate(rows):
            head = (rf"\multirow{{{len(rows)}}}{{*}}{{{DATASETS[name].label}}}"
                    if i == 0 else "")
            rary = r["pointer_type"] == "rary"
            out.append(
                f"{head} & {POINTER_PAPER_LABELS[r['pointer_type']]} & "
                # max_cluster_size is None for unpartitioned (and louvain) runs,
                # which print no cluster block -- logparse.py:66-67. Those runs
                # have no partition for the column to describe, so it reads "--"
                # rather than crashing tex_int.
                f"{tex_bytes(r['block_size'])} & "
                f"{tex_int(r['max_cluster_size']) if r['max_cluster_size'] is not None else '--'} & "
                # r, as the run reported it. This used to be derived from the
                # block size, which was the same number until
                # --pointer-branching-factor let the two diverge -- on a
                # decoupled log the derivation reads 1022 where the run used 64.
                f"{tex_int(r['branching_factor']) if rary and r['branching_factor'] is not None else '--'} & "
                f"{r['rounds_per_query']:.1f} & {r['reads_per_query']:.1f} & "
                f"{tex_num(r['writes_per_query'])} & "
                f"{tex_sci(r['whole_run_allocs'])} \\\\"
            )
        blocks.append("\n".join(out))

    body = header + "\n\\midrule\n" + "\n\\midrule\n".join(blocks)
    # Under the per-query bandwidth basis every series is monotone across the
    # sweep, so "its own latency-minimising block size" resolves to the smallest
    # block measured, for both schemes and every dataset. Saying so in the
    # caption keeps the table from reading as if a turn had been located.
    coincide = len({r["block_size"] for rows in results.values() for r in rows}) == 1
    where = (
        "both schemes at the smallest block size swept, which is where the "
        "modelled latency of Table~\\ref{tab:network} is lowest for every "
        "dataset. The curves are still descending there, so this is the "
        "cheapest measured point rather than a located optimum"
        if coincide else
        "each scheme at its own latency-minimising block size"
    )
    caption = (
        f"Per-query oblivious memory cost, {where}. "
        "$|P|_{\\max}$=largest METIS partition "
        "(nodes); $r$=r-ary pointer branching factor, derived from the block "
        "size as $\\lfloor (B-16)/8 \\rfloor$ and not applicable to OSAM+; "
        "Allocs=allocations over the whole run, build and search. BOSAM trades "
        "writes for reads: it moves an order of magnitude less data per round "
        "trip at the cost of a far larger write volume, which the "
        "multi-write memory absorbs in one logical step. Logical rounds are "
        "graph hops and depend only on the partition count, not on the pointer "
        "scheme."
    )
    return wrap(body, "l | l | r | r | r | r | r | r | r", caption, "tab:cost")


def table_network(results: dict[str, list[dict]], args) -> str:
    per_query = args.bandwidth_basis == "query"
    bw_head = r"\textsf{BW/query}" if per_query else r"\textsf{BW/access}"
    header = (r"\textsf{Dataset} & \textsf{Scheme} & \textsf{Block} & "
              rf"$\log_2 N_{{\mathsf{{alloc}}}}$ & {bw_head} & "
              r"\textsf{Latency (s)} & \textsf{R@1} & \textsf{R@10} \\")

    blocks = []
    for name, rows in results.items():
        best = min(r["latency"] for r in rows)
        out = []
        for i, r in enumerate(rows):
            head = (rf"\multirow{{{len(rows)}}}{{*}}{{{DATASETS[name].label}}}"
                    if i == 0 else "")
            lat = f"{r['latency']:.2f}"
            if r["latency"] == best:
                lat = rf"\textbf{{{lat}}}"
            recall = [f"{r[k]:.3f}" if r[k] is not None else "--"
                      for k in ("recall_at_1", "recall_at_10")]
            out.append(
                f"{head} & {POINTER_PAPER_LABELS[r['pointer_type']]} & "
                f"{tex_bytes(r['block_size'])} & {r['log2_allocs']:.2f} & "
                f"{tex_si_bytes(r['bytes_per_query'])} & {lat} & "
                f"{recall[0]} & {recall[1]} \\\\"
            )
        blocks.append("\n".join(out))

    body = header + "\n\\midrule\n" + "\n\\midrule\n".join(blocks)
    if per_query:
        model = (
            "BW/query is the bytes a whole query moves: one ORAM access costs "
            f"$4(b{{+}}1)B\\log_2 N_{{\\mathsf{{alloc}}}}$ with $b={args.b}$, and a query "
            "serializes on reads/query of them. Latency $=$ reads/query "
            "$\\times\\,(\\mathrm{RTT} + 4(b{+}1)B\\log_2 "
            "N_{\\mathsf{alloc}}/\\mathrm{BW})$, "
        )
    else:
        model = (
            "BW/access is the bytes one ORAM access moves, "
            f"$4(b{{+}}1)B\\log_2 N_{{\\mathsf{{alloc}}}}$ with $b={args.b}$. "
            "Latency $=$ BW/access$\\,/\\,\\mathrm{BW} + \\mathrm{RTT} \\times$ reads "
            "per query, "
        )
    caption = (
        "Modelled end-to-end cost and retrieval quality, at the same operating "
        "points as Table~\\ref{tab:cost}. " + model +
        f"with $\\mathrm{{RTT}}={args.rtt * 1e3:g}$\\,ms and "
        f"$\\mathrm{{BW}}={args.bandwidth / 1e6:g}$\\,MB/s. Recall is invariant "
        "across block sizes and pointer schemes: obliviousness changes what a "
        "query costs, not what it returns."
    )
    return wrap(body, "l | l | r | r | r | r | r | r", caption, "tab:network")


# ---------------------------------------------------------------------- main ---

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   help="datasets to report; one with no sweep logs appears in "
                        "the configuration table only (default: %(default)s)")
    p.add_argument("--logs-root", default=str(REPO / "logs"),
                   help="directory holding the <dataset>_bssweep log directories")
    p.add_argument("--partitioning", default=None,
                   help="only use runs with this partitioning, e.g. metis")
    p.add_argument("--b", type=int, default=1,
                   help="eviction/bucket parameter in the bandwidth formula (default: 1)")
    p.add_argument("--allocs", choices=["whole-run", "build", "search"],
                   default="whole-run",
                   help="which allocation count feeds log2 (default: whole-run)")
    p.add_argument("--rtt", type=float, default=DEFAULT_RTT,
                   help="seconds per round trip (default: 0.1)")
    p.add_argument("--bandwidth", type=float, default=DEFAULT_BW,
                   help="network bandwidth in bytes/s (default: 100e6)")
    p.add_argument("--bandwidth-basis", choices=["access", "query"], default="query",
                   help="query = the bytes a whole query moves, reads/query "
                        "times one access (default); this is the paper's "
                        "T = R(RTT + S/BW). access = one access worth of bytes, "
                        "charging the transfer term once per query while "
                        "charging RTT per read -- kept only to reproduce "
                        "pre-fix tables. Must match plot_latency.py")
    p.add_argument("--out", default=str(REPO / "paper" / "tables"),
                   help="output directory (default: %(default)s)")
    args = p.parse_args(argv)

    unknown = [d for d in args.datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {', '.join(unknown)}; "
                         f"known: {', '.join(DATASETS)}")

    results: dict[str, list[dict]] = {}
    for name in args.datasets:
        rows = best_rows(name, args)
        if rows:
            check_against_spec(name, rows)
            results[name] = rows
        else:
            print(f"  . {name}: no sweep logs -- configuration table only")

    if not results:
        raise SystemExit(f"no sweep logs found under {args.logs_root}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "tab_datasets.tex": table_datasets(args.datasets, args),
        "tab_cost.tex": table_cost(results, args),
        "tab_network.tex": table_network(results, args),
    }
    for name, text in files.items():
        (out_dir / name).write_text(text)
        print(f"wrote {out_dir / name}")

    inputs = "".join(f"\\input{{{Path(n).stem}}}\n" for n in files)
    (out_dir / "tables.tex").write_text(
        "% Generated by plots/make_tables.py -- do not edit by hand.\n"
        "% Preamble: \\usepackage{booktabs} \\usepackage{multirow}\n" + inputs
    )
    print(f"wrote {out_dir / 'tables.tex'}")

    # A wrapper to eyeball the tables on their own before pasting them into the
    # paper -- \input{tables} rather than a copy, so it cannot drift.
    (out_dir / "standalone_preview.tex").write_text(
        "% Generated by plots/make_tables.py. Preview only, not part of the paper:\n"
        "%     pdflatex standalone_preview.tex\n"
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in,landscape]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{multirow}\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        "\\input{tables}\n"
        "\\end{document}\n"
    )
    print(f"wrote {out_dir / 'standalone_preview.tex'}")

    for name, rows in results.items():
        for r in rows:
            # Unpartitioned runs have no partition count to report.
            nc = "none" if r["n_clusters"] is None else str(r["n_clusters"])
            print(f"  {name:9} {POINTER_PAPER_LABELS[r['pointer_type']]:6} "
                  f"{tex_bytes(r['block_size']):>10}  nc={nc:<6} "
                  f"latency {r['latency']:.3f}s  ({r['log']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
