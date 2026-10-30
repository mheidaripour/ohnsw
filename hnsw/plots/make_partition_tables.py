#!/usr/bin/env python3
r"""Emit the appendix LaTeX tables comparing METIS and K-means partitioning.

Three tables, written to ``hnsw/paper/tables/`` as standalone ``.tex`` files
plus a ``tables_appendix.tex`` that ``\input``s all three:

  1. ``tab_part_quality.tex``   -- partition quality (balance and cut)
  2. ``tab_part_query.tex``     -- query behaviour and recall
  3. ``tab_part_blocksize.tex`` -- the block size each partitioning implies

The input is the JSON that ``cluster/plaintext/hnsw-metis-partitioned.py`` and
``cluster/plaintext/hnsw-clustered.py`` write, produced by
``exp_plaintext_partitioning.sh``. Nothing here reads the oblivious sweep logs;
these are plaintext measurements of the partitioning itself.

Unlike ``make_tables.py``, the tables are transposed -- metrics run down the
rows and the four dataset/method combinations across the columns. The natural
orientation needs thirteen columns, which no single-column USENIX table can
hold; transposed it is five, and rows can be added without reflowing anything.

``tables.tex`` is deliberately left alone: ``make_tables.py`` regenerates it and
would clobber any edit made here.

Usage::

    python plots/make_partition_tables.py
    python plots/make_partition_tables.py --k 2000
    python plots/make_partition_tables.py --datasets sift1m
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_tables import tex_bytes, tex_int, wrap  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_DATASETS = ["sift1m", "laion1m"]

DATASET_LABELS = {
    "sift1m": "SIFT-1M",
    "laion1m": "LAION-1M",
    "1k": "LAION-1k",
    "10k": "LAION-10k",
}

# Each method's script writes the same numbers under different key names and to
# a different directory. Everything downstream reads through this table, so the
# asymmetry is confined to one place.
METHODS = {
    "metis": {
        "label": "METIS",
        "log_dir": "hnsw_metis_partitioned",
        "prefix": "hnsw_metis_partitioned",
        "metrics_key": "partition_metrics",
        "detail_key": "cluster_detail_stats",
        "time_key": "partition_time_s",
        "count_key": "n_partitions",
    },
    "kmeans": {
        "label": "K-means",
        "log_dir": "hnsw_clustered",
        "prefix": "hnsw_clustered",
        "metrics_key": "cluster_metrics",
        "detail_key": "cluster_detail_summary",
        "time_key": "clustering_time_s",
        "count_key": "n_clusters",
    },
}

METHOD_ORDER = ["metis", "kmeans"]

BYTES_PER_INT = 4  # both scripts charge 4 bytes per stored integer


# ------------------------------------------------------------------ loading ---

def next_pow2(n: int) -> int:
    """Smallest power of two >= n. Same function probe_partitions.py applies."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def find_json(log_root: Path, method: str, dataset: str, k: int) -> Path | None:
    """Newest results JSON for one (method, dataset, k), or None.

    The run name ``exp_plaintext_partitioning.sh`` assigns is
    ``<dataset>_k<k>``, and the script appends its own timestamp, so a rerun
    leaves both files behind. Newest by mtime wins, matching ``dedupe_by`` in
    ``logparse``.
    """
    spec = METHODS[method]
    pattern = f"{spec['prefix']}_{dataset}_k{k}_*.json"
    hits = sorted(
        (log_root / spec["log_dir"]).glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    return hits[-1] if hits else None


def find_stdout_log(log_root: Path, method: str, dataset: str, k: int) -> Path | None:
    """Newest stdout log for a run, or None.

    A few quantities the wide tables need are printed but never written to the
    JSON -- notably the per-query minimum and maximum partition counts. Named
    ``<dataset>_k<k>_<timestamp>.log`` by exp_plaintext_partitioning.sh, which
    stamps the launch time, so this cannot be matched to the JSON's own
    (later) timestamp; newest wins, as with the JSON.
    """
    hits = sorted(
        (log_root / METHODS[method]["log_dir"]).glob(f"{dataset}_k{k}_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    return hits[-1] if hits else None


# The two scripts print the same three numbers under different nouns: METIS
# calls a partition a "community", K-means calls it a "cluster". One alternation
# per quantity rather than one pattern per method, so a wording change on either
# side fails loudly instead of silently dropping a column.
_LOG_PATTERNS = {
    # METIS: "Total communities loaded:"    K-means: "Total clusters accessed:"
    "accessed_total": r"^\s*Total (?:clusters accessed|communities loaded):\s*([\d,]+)",
    # METIS: "Min communities searched per query:"   K-means: "Min clusters/query:"
    "accessed_min": r"^\s*Min (?:clusters/query|communities searched per query):\s*([\d,]+)",
    "accessed_max": r"^\s*Max (?:clusters/query|communities searched per query):\s*([\d,]+)",
}


def parse_stdout(path: Path | None) -> dict:
    """The printed-only quantities. Missing keys are left absent, not zeroed."""
    if path is None:
        return {}
    text = path.read_text(errors="replace")
    out = {}
    for name, pattern in _LOG_PATTERNS.items():
        m = re.search(pattern, text, re.M)
        if m:
            out[name] = int(m.group(1).replace(",", ""))
    return out


def load_run(path: Path, method: str, stdout_path: Path | None = None) -> dict:
    """One results JSON, flattened to the quantities the tables report."""
    spec = METHODS[method]
    with open(path) as f:
        d = json.load(f)

    m = d[spec["metrics_key"]]
    detail = d[spec["detail_key"]]

    # graph_bytes is the payload a block must hold: 4 bytes per node id, per
    # intra adjacency entry, and two per cross edge. Vectors are excluded --
    # o_hnsw keeps them in plaintext outside the ORAM.
    max_bytes = int(detail["graph_bytes"]["max"])
    avg_bytes = float(detail["graph_bytes"]["avg"])
    block = next_pow2(max_bytes)

    return {
        "path": path,
        "n_parts": int(m[spec["count_key"]]),
        "size_min": int(m["size_min"]),
        "size_max": int(m["size_max"]),
        "size_mean": float(m["size_mean"]),
        "size_median": float(m["size_median"]),
        "size_cv": float(m["size_cv"]),
        "size_gini": float(m["size_gini"]),
        "size_max_over_median": float(m["size_max_over_median"]),
        "edge_cuts": int(m["edge_cuts"]),
        "l0_undirected_edges": int(m["l0_undirected_edges"]),
        "cut_ratio": float(m["cut_ratio"]),
        "intra_fraction": float(m["intra_fraction"]),
        "time_s": float(m[spec["time_key"]]),
        "neighbor_avg": float(
            d["neighbor_cluster_stats"]["neighbor_clusters_per_cluster"]["avg"]
        ),
        "max_nodes": int(detail["n_nodes"]["max"]),
        "max_bytes": max_bytes,
        "avg_bytes": avg_bytes,
        "block_size": block,
        # What a block wastes: it is padded to a power of two above the largest
        # partition, but the average partition is what actually occupies it.
        "padding_waste": 1.0 - avg_bytes / block,
        "num_queries": int(d["num_queries"]),
        "clusters_accessed": float(d["avg_clusters_accessed"]),
        "cache_hits": float(d["avg_cache_hits"]),
        "cache_hit_rate": float(d["cache_hit_rate"]),
        "hops": float(d["avg_hops"]),
        "nodes_visited": float(d["avg_nodes_visited"]),
        "recall_at_1": float(d["recall"]["R@1"]),
        "recall_at_10": float(d["recall"]["R@10"]),
        "ms_per_query": 1000.0 * float(d["avg_time_per_query"]),
        # Per-partition distributions, for the wide stats table. Each summary
        # entry carries min/max/avg/total over partitions; the maxima are taken
        # independently per quantity, so the largest partition by node count
        # need not be the largest by bytes.
        "detail": {
            key: detail[key]
            for key in ("n_nodes", "intra_edges", "cross_edges",
                        "neighbor_clusters_count", "graph_bytes",
                        "graph_plus_vectors_bytes")
        },
        **parse_stdout(stdout_path),
    }


def load_all(log_root: Path, datasets: list[str], k: int) -> dict[tuple[str, str], dict]:
    """Every available (dataset, method) run. Missing runs are reported, not faked."""
    runs: dict[tuple[str, str], dict] = {}
    for dataset in datasets:
        for method in METHOD_ORDER:
            path = find_json(log_root, method, dataset, k)
            if path is None:
                print(
                    f"missing: {dataset}/{method} at k={k} "
                    f"(run ./exp_plaintext_partitioning.sh --dataset {dataset} "
                    f"--method {method})",
                    file=sys.stderr,
                )
                continue
            runs[(dataset, method)] = load_run(
                path, method, find_stdout_log(log_root, method, dataset, k)
            )
    return runs


# --------------------------------------------------------------- formatting ---

def tex_float(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def tex_pct(x: float, digits: int = 1) -> str:
    return rf"{100 * x:.{digits}f}\%"


def tex_bytes_approx(n: float) -> str:
    r"""Measured byte counts, to one decimal: ``10.5\,KiB``.

    ``tex_bytes`` renders with ``%g``, which is exactly right for block sizes --
    they are powers of two and print as ``512\,KiB`` -- and wrong for measured
    payloads, where it spills six significant digits (``10.4922\,KiB``). One
    decimal is all the precision a reader can use when the point of the column
    is which power of two it lands under.
    """
    for unit, scale in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= scale:
            return rf"{n / scale:.1f}\,{unit}"
    return rf"{n:.0f}\,B"


def tex_seconds(x: float) -> str:
    r"""Whole seconds, with a floor so sub-second runs do not print as ``0``."""
    return r"$<$1" if x < 1 else f"{x:.0f}"


def present(datasets: list[str], runs: dict) -> list[tuple[str, str]]:
    """The (dataset, method) columns that actually have data, in report order."""
    return [
        (ds, m)
        for ds in datasets
        for m in METHOD_ORDER
        if (ds, m) in runs
    ]


def header_rows(columns: list[tuple[str, str]]) -> str:
    r"""Two header lines: dataset groups over method names.

    ``\cmidrule(lr)`` under each group keeps the two datasets visually apart
    without a vertical rule cutting through the row labels.
    """
    groups: list[tuple[str, int]] = []
    for ds, _ in columns:
        if groups and groups[-1][0] == ds:
            groups[-1] = (ds, groups[-1][1] + 1)
        else:
            groups.append((ds, 1))

    top = [""]
    rules = []
    col = 2
    for ds, span in groups:
        label = DATASET_LABELS.get(ds, ds)
        top.append(rf"\multicolumn{{{span}}}{{c}}{{\textsf{{{label}}}}}")
        rules.append(rf"\cmidrule(lr){{{col}-{col + span - 1}}}")
        col += span

    methods = [""] + [rf"\textsf{{{METHODS[m]['label']}}}" for _, m in columns]
    return (
        " & ".join(top) + r" \\" + "\n"
        + "".join(rules) + "\n"
        + " & ".join(methods) + r" \\"
    )


def metric_rows(columns: list[tuple[str, str]], runs: dict, rows: list) -> str:
    """Render ``(label, key_or_fn, formatter, better)`` specs into table lines.

    ``better`` is ``"lo"``, ``"hi"`` or ``None``; when set, the winning cell
    within each dataset group is bolded. Rows where the two methods agree to
    the rendered precision are left unbolded -- bolding a tie asserts a
    difference that is not there, which matters here because the recall and hop
    rows are supposed to be identical.
    """
    lines = []
    for spec in rows:
        if spec is None:
            lines.append(r"\midrule")
            continue
        label, key, fmt, better = spec

        cells = {}
        for ds, method in columns:
            run = runs[(ds, method)]
            cells[(ds, method)] = key(run) if callable(key) else run[key]

        winners: set[tuple[str, str]] = set()
        if better:
            by_ds: dict[str, list[tuple[str, str]]] = {}
            for col in columns:
                by_ds.setdefault(col[0], []).append(col)
            for group in by_ds.values():
                if len(group) < 2:
                    continue
                rendered = {c: fmt(cells[c]) for c in group}
                if len(set(rendered.values())) == 1:
                    continue  # a tie -- see the docstring
                pick = min if better == "lo" else max
                winners.add(pick(group, key=lambda c: cells[c]))

        rendered_cells = [
            rf"\textbf{{{fmt(cells[c])}}}" if c in winners else fmt(cells[c])
            for c in columns
        ]
        lines.append(f"{label} & " + " & ".join(rendered_cells) + r" \\")
    return "\n".join(lines)


def colspec(columns: list[tuple[str, str]]) -> str:
    return "l | " + " ".join(["r"] * len(columns))


# ------------------------------------------------------------------- tables ---

def table_quality(columns, runs, k: int) -> str:
    rows = [
        ("Partitions", "n_parts", tex_int, None),
        ("Size, min", "size_min", tex_int, None),
        ("Size, max", "size_max", tex_int, "lo"),
        ("Size, mean", "size_mean", lambda x: f"{x:.1f}", None),
        ("Size, max/median", "size_max_over_median", lambda x: tex_float(x, 2), "lo"),
        (r"Balance, CV", "size_cv", lambda x: tex_float(x, 4), "lo"),
        (r"Balance, Gini", "size_gini", lambda x: tex_float(x, 4), "lo"),
        None,
        ("Edge cuts", "edge_cuts", tex_int, "lo"),
        ("Cut ratio", "cut_ratio", lambda x: tex_float(x, 4), "lo"),
        ("Intra-edge fraction", "intra_fraction", lambda x: tex_float(x, 4), "hi"),
        ("Neighbour parts/part", "neighbor_avg", lambda x: f"{x:.1f}", "lo"),
        ("Partitioning time (s)", "time_s", tex_seconds, "lo"),
    ]
    body = header_rows(columns) + "\n\\midrule\n" + metric_rows(columns, runs, rows)
    caption = (
        rf"Plaintext partition quality at $n_c={tex_int(k)}$. METIS partitions the "
        "symmetrised base-layer graph under a balance constraint; K-means partitions "
        "the vector space and is blind to the graph. Both are measured from the same "
        "definitions over the same index, so the columns are directly comparable. "
        "Abbrev.: CV=coefficient of variation of partition size, Gini=Gini "
        "coefficient of the size distribution, cut ratio=cut edges over all "
        "undirected base-layer edges, intra-edge fraction=directed edges that stay "
        "inside a partition, neighbour parts/part=distinct partitions a partition "
        "has a cross edge to. Bold marks the better value; lower is better except "
        "for the intra-edge fraction."
    )
    return wrap(body, colspec(columns), caption, "tab:part-quality")


def table_query(columns, runs, k: int) -> str:
    rows = [
        ("Partitions touched/query", "clusters_accessed", lambda x: f"{x:.1f}", "lo"),
        ("Cache hits/query", "cache_hits", lambda x: f"{x:.1f}", "hi"),
        ("Cache hit rate", "cache_hit_rate", lambda x: tex_pct(x, 1), "hi"),
        None,
        ("Hops/query", "hops", lambda x: f"{x:.1f}", None),
        ("Nodes visited/query", "nodes_visited", lambda x: f"{x:.1f}", None),
        ("R@1", "recall_at_1", lambda x: tex_float(x, 4), None),
        ("R@10", "recall_at_10", lambda x: tex_float(x, 4), None),
    ]
    body = header_rows(columns) + "\n\\midrule\n" + metric_rows(columns, runs, rows)
    caption = (
        rf"Plaintext query behaviour at $n_c={tex_int(k)}$, $k=10$, "
        r"$\mathsf{efSearch}=64$. The lower block is invariant to the partitioning: "
        "both methods traverse the same HNSW graph with the same search algorithm, "
        "so partitioning decides which subgraph is fetched, never which candidates "
        "are expanded. Recall is identical on both datasets, as are the hop and node "
        "counts on SIFT-1M; on LAION-1M they differ by $0.005\\%$ "
        "($1501.93$ against $1502.00$ hops), which is tie-breaking among equidistant "
        "candidates resolved in enumeration order, not a difference in what the "
        "search explores. Only the upper block -- how many partitions a query touches "
        "and how often the per-partition subgraph cache hits -- separates the two "
        "methods. Wall-clock throughput is omitted; it is sensitive to machine load "
        "and says nothing about the layout. Bold marks the better value."
    )
    return wrap(body, colspec(columns), caption, "tab:part-query")


def table_blocksize(columns, runs, k: int) -> str:
    rows = [
        (r"$|P|_{\max}$ (nodes)", "max_nodes", tex_int, "lo"),
        ("Max payload", "max_bytes", tex_bytes_approx, "lo"),
        ("Mean payload", "avg_bytes", tex_bytes_approx, None),
        None,
        ("Block size $B$", "block_size", tex_bytes, "lo"),
        ("Padding waste", "padding_waste", lambda x: tex_pct(x, 1), "lo"),
    ]
    body = header_rows(columns) + "\n\\midrule\n" + metric_rows(columns, runs, rows)
    caption = (
        rf"Block size implied by each partitioning at $n_c={tex_int(k)}$. A block is "
        "padded to hold the largest partition, so $B$ is the next power of two above "
        "the maximum payload, and every block in the ORAM is that size regardless of "
        "how little the average partition needs. Payload counts four bytes per stored "
        "integer -- one per node id, one per intra-partition adjacency entry, two per "
        "cross edge (neighbour id and its owning partition) -- and excludes vectors, "
        "which are kept in plaintext outside the ORAM. Maxima are taken independently "
        "per column, so the largest partition by node count need not be the largest by "
        "bytes. Padding waste is the fraction of a block the mean partition leaves "
        "unused. Lower is better throughout."
    )
    return wrap(body, colspec(columns), caption, "tab:part-blocksize")


# -------------------------------------------------------------- wide tables ---
#
# A second pair of tables in the full-width \table* format: one row per
# (dataset, method) instead of one column, which is what a per-partition
# distribution needs -- each cell is a min/mean/max triple and would not fit a
# single column. They report the distribution behind the summary tables above,
# not different measurements.

MX = r"\mx{%s}"

# The preamble line the two wide tables need. Emitted into the preview and
# printed at the end of a run, because a \mx that is not defined fails at
# compile time in the main document, where this file cannot put it.
MX_DEFINITION = (
    r"\newcommand{\mx}[1]{{\setlength{\fboxsep}{1.5pt}"
    r"\textbf{\colorbox{yellow}{#1}}}}" "\n"
)


def wrap_wide(body: str, colspec: str, header: str, caption: str, label: str) -> str:
    r"""Full-width float with \resizebox, matching the paper's wide-table style.

    Caption above the tabular here, unlike :func:`wrap` -- with \resizebox the
    caption has to be outside the box or it is scaled along with the table.
    """
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table*}\n"
    )


def wide_rows(datasets: list[str], runs: dict) -> list[tuple[str, str, dict]]:
    return [(ds, m, runs[(ds, m)]) for ds in datasets for m in METHOD_ORDER
            if (ds, m) in runs]


def highlight_worse_max(rows, getter) -> set[int]:
    r"""Indices whose max is the larger of the two methods within a dataset.

    ``\mx`` marks the value that binds: the block is padded to the largest
    partition, so between two methods at the same partition count it is the
    bigger maximum that sets the cost. Ties are left unmarked -- highlighting
    both would assert a difference that is not there.
    """
    worst: set[int] = set()
    by_ds: dict[str, list[int]] = {}
    for i, (ds, _, _) in enumerate(rows):
        by_ds.setdefault(ds, []).append(i)
    for idxs in by_ds.values():
        if len(idxs) < 2:
            continue
        vals = {i: getter(rows[i][2]) for i in idxs}
        hi = max(vals.values())
        if len({round(v, 6) for v in vals.values()}) > 1:
            worst |= {i for i, v in vals.items() if v == hi}
    return worst


def triple(summary: dict, fmt, mark_max: bool = False) -> str:
    r"""``min / mean / max`` in one cell, optionally highlighting the max.

    Only the max component is wrapped, never the whole triple: \mx sets a
    yellow \colorbox, and boxing a three-number cell in every K-means row
    turns half the table yellow, which draws the eye to the method rather than
    to the quantity that matters.
    """
    hi = fmt(summary["max"])
    return (f"{fmt(summary['min'])} / {fmt(summary['avg'])} / "
            f"{MX % hi if mark_max else hi}")


def table_partition_stats(datasets: list[str], runs: dict, k: int) -> str:
    rows = wide_rows(datasets, runs)
    header = (
        r"\multicolumn{2}{c|}{} & \multicolumn{4}{c|}{\textbf{Per partition "
        r"(min / mean / max)}}" "\n"
        r"& \multicolumn{2}{c}{\textbf{Size, MiB (max / block)}} \\" "\n"
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-8}" "\n"
        r"\textbf{Dataset} & \textbf{Method} & \textbf{Nodes} & \textbf{Intra edges}"
        "\n"
        r"& \textbf{Cross edges} & \textbf{Neighbors} & \textbf{Graph} & "
        r"\textbf{Graph+vec} \\"
    )

    # One highlight set per quantity: the method whose maximum is larger.
    marks = {
        key: highlight_worse_max(rows, lambda r, key=key: r["detail"][key]["max"])
        for key in ("n_nodes", "intra_edges", "cross_edges",
                    "neighbor_clusters_count", "graph_bytes",
                    "graph_plus_vectors_bytes")
    }

    MIB = 1 << 20

    def pair(summary: dict, marked: bool) -> str:
        """``max / next_pow2(max)`` in MiB, highlighting the max when it binds.

        Binary units, not the decimal MB the other columns use: the second
        value is a power of two by construction, and in decimal MB a 1 MiB
        block prints as 1.05, which hides exactly the property that matters.
        """
        hi = f"{summary['max'] / MIB:.2f}"
        block = next_pow2(int(summary["max"])) / MIB
        return (f"{MX % hi if marked else hi} / "
                f"{block:.2f}")

    lines = []
    prev_ds = None
    for i, (ds, method, r) in enumerate(rows):
        if prev_ds is not None and ds != prev_ds:
            lines.append(r"\midrule")
        label = DATASET_LABELS.get(ds, ds) if ds != prev_ds else ""
        prev_ds = ds
        d = r["detail"]
        cells = [
            label,
            METHODS[method]["label"],
            triple(d["n_nodes"], tex_int, i in marks["n_nodes"]),
            triple(d["intra_edges"], tex_int, i in marks["intra_edges"]),
            triple(d["cross_edges"], tex_int, i in marks["cross_edges"]),
            triple(d["neighbor_clusters_count"], tex_int,
                   i in marks["neighbor_clusters_count"]),
            pair(d["graph_bytes"], i in marks["graph_bytes"]),
            pair(d["graph_plus_vectors_bytes"],
                 i in marks["graph_plus_vectors_bytes"]),
        ]
        lines.append(" & ".join(cells) + r" \\")

    caption = (
        rf"Per-partition statistics ($n_c{{=}}{tex_int(k)}$). Highlighted is the "
        "larger of the two methods' maxima within each dataset --- the value that "
        "binds, since a block is padded to the largest partition and every block is "
        "then that size. The block column is the next power of two above the "
        "maximum. Only the Graph figure is a block size the system uses: vectors "
        "are kept in plaintext outside the ORAM, so the Graph+vec column is what "
        "the block would have to be if they were stored obliviously too. Byte "
        "estimates assume 4 bytes per integer and \\texttt{float32} vectors "
        "($d{=}512$ for LAION, $d{=}128$ for SIFT); maxima are taken independently "
        "per column, so the largest partition by node count need not be the largest "
        "by bytes."
    )
    return wrap_wide(body="\n".join(lines),
                     colspec="ll|cccc|cc",
                     header=header, caption=caption, label="tab:partition-stats")


def table_query_stats(datasets: list[str], runs: dict, k: int) -> str:
    rows = wide_rows(datasets, runs)
    header = (
        r"\multicolumn{2}{c|}{}&\multicolumn{1}{c|}{\textbf{Latency}} & "
        r"\multicolumn{2}{c|}{\textbf{Partition access}}" "\n"
        r"& \multicolumn{2}{c|}{\textbf{Traversal}} & "
        r"\multicolumn{2}{c}{\textbf{Recall}} \\" "\n"
        r"\cmidrule(lr){3-3}\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}"
        "\n"
        r"\textbf{Dataset} & \textbf{Method} & \textbf{ms/query}" "\n"
        r"& \textbf{Avg/query} & \textbf{Min--max} & "
        r"\textbf{Hops} & \textbf{Nodes} & \textbf{R@1} & \textbf{R@10} \\"
    )

    lines = []
    prev_ds = None
    for ds, method, r in rows:
        if prev_ds is not None and ds != prev_ds:
            lines.append(r"\midrule")
        label = DATASET_LABELS.get(ds, ds) if ds != prev_ds else ""
        prev_ds = ds
        # Printed-only fields; an em dash rather than a zero if a stdout log was
        # not found, so a missing log cannot read as a measurement of zero.
        # The per-run total is deliberately not shown: it is exactly
        # num_queries x Avg/query, so it adds a column without adding a fact.
        if "accessed_min" in r and "accessed_max" in r:
            span = f"{tex_int(r['accessed_min'])}--{tex_int(r['accessed_max'])}"
        else:
            span = "---"
        lines.append(" & ".join([
            label,
            METHODS[method]["label"],
            f"{r['ms_per_query']:.2f}",
            f"{r['clusters_accessed']:.2f}",
            span,
            f"{r['hops']:.1f}",
            f"{r['nodes_visited']:.1f}",
            tex_float(r["recall_at_1"], 4),
            tex_float(r["recall_at_10"], 4),
        ]) + r" \\")

    caption = (
        rf"Query and recall statistics ({tex_int(next(iter(runs.values()))['num_queries'])} "
        r"queries, $k{=}10$, $\mathit{ef}{=}64$, "
        rf"{tex_int(k)} partitions). Both partition-access columns are per query: "
        "the mean, and the range over the query set. Latency is plaintext Python "
        "traversal and is reported for completeness only --- it is sensitive to "
        "machine load and is not a claim about either method's speed. Traversal "
        "and recall are invariant to the partitioning: both methods walk the same "
        "graph, and the residual difference on LAION-1M is tie-breaking between "
        "equidistant candidates."
    )
    return wrap_wide(body="\n".join(lines),
                     colspec="ll|r|rc|rr|rr",
                     header=header, caption=caption, label="tab:query-stats")


# --------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--k", type=int, default=1000,
                    help="partition count the runs used (default: 1000)")
    ap.add_argument("--log-root", type=Path,
                    default=REPO / "cluster" / "plaintext" / "log",
                    help="directory holding hnsw_{metis_partitioned,clustered}/")
    ap.add_argument("--out", type=Path, default=REPO / "paper" / "tables",
                    help="output directory for the .tex files")
    args = ap.parse_args()

    runs = load_all(args.log_root, args.datasets, args.k)
    columns = present(args.datasets, runs)
    if not columns:
        print("no runs found -- nothing to write", file=sys.stderr)
        return 1

    # A one-sided dataset would render as a group of one with nothing to
    # compare against, and every "better value" bold would be vacuous.
    for ds in args.datasets:
        have = [m for m in METHOD_ORDER if (ds, m) in runs]
        if have and len(have) < len(METHOD_ORDER):
            print(f"warning: {ds} has only {have[0]} -- comparison is one-sided",
                  file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    tables = {
        "tab_part_quality.tex": table_quality(columns, runs, args.k),
        "tab_part_query.tex": table_query(columns, runs, args.k),
        "tab_part_blocksize.tex": table_blocksize(columns, runs, args.k),
        "tab_partition_stats.tex": table_partition_stats(args.datasets, runs, args.k),
        "tab_query_stats.tex": table_query_stats(args.datasets, runs, args.k),
    }
    for name, text in tables.items():
        (args.out / name).write_text(text)
        print(f"wrote {args.out / name}")

    # Bare names, matching tables.tex: both are read from inside this
    # directory, so a "tables/" prefix would resolve to tables/tables/.
    # appendix_partitioning.tex sits one level up and spells the prefix itself.
    index = (
        "% Generated by plots/make_partition_tables.py -- do not edit.\n"
        "% Requires: \\usepackage{booktabs}\n"
        "% Convenience bundle for previewing all three at once. The paper\n"
        "% \\inputs them individually from paper/appendix_partitioning.tex, so\n"
        "% that each float is declared where it is first discussed.\n"
        + "".join(f"\\input{{{name[:-4]}}}\n" for name in tables)
    )
    (args.out / "tables_appendix.tex").write_text(index)
    print(f"wrote {args.out / 'tables_appendix.tex'}")

    # Its own preview file rather than an edit to standalone_preview.tex, which
    # make_tables.py regenerates and would clobber. Single-column geometry, not
    # the landscape one the other preview uses: these tables are meant to fit a
    # USENIX column, and a preview that gives them a full page would hide the
    # one failure mode worth catching.
    preview = (
        "% Generated by plots/make_partition_tables.py. Preview only, not part\n"
        "% of the paper:\n"
        "%     pdflatex standalone_preview_appendix.tex\n"
        "\\documentclass[10pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}   % \\resizebox, for the two wide tables\n"
        "\\usepackage{xcolor}     % \\colorbox, for \\mx\n"
        + MX_DEFINITION
        + "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        + "".join(f"\\input{{{name[:-4]}}}\n" for name in tables)
        + "\\end{document}\n"
    )
    (args.out / "standalone_preview_appendix.tex").write_text(preview)
    print(f"wrote {args.out / 'standalone_preview_appendix.tex'}")

    for (ds, method), run in runs.items():
        print(f"  {ds}/{method}: {run['path'].name} "
              f"({run['n_parts']} parts, {run['num_queries']} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
