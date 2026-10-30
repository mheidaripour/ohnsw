"""Parse o_hnsw.py run logs into structured records.

The log files are the raw stdout of ``o_hnsw.py`` runs (see ``exp.sh``).  Every
value is read out of the log *body*, never out of the file name -- some of the
existing file names carry a stale ``bsXXXk`` tag that disagrees with the block
size the run actually used.

Typical use::

    from logparse import load_dir
    runs = load_dir("logs/sift1m", pointer_type="rary")
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The r-fallback lives with the bandwidth model rather than here, so the
# derivation has one definition. bandwidth imports nothing, so this is not a
# cycle.
from bandwidth import branching_factor

# Canonical short names for the --pointer-type values o_hnsw.py accepts.
POINTER_ALIASES = {
    "multiwriterary": "rary",
    "rary": "rary",
    "multiwrite": "multiwrite",
    "singlewrite": "singlewrite",
    "plain": "plain",
}

POINTER_LABELS = {
    "rary": "r-ary",
    "multiwrite": "multiwrite",
    "singlewrite": "single-write",
    "plain": "plain",
}

# The names the paper gives these schemes. Kept apart from POINTER_LABELS so a
# figure can say either what the implementation is or what the paper calls it.
POINTER_PAPER_LABELS = {
    "rary": "BOSAM",
    "multiwrite": "OSAM+",
}


@dataclass
class RunLog:
    """One o_hnsw.py run, as recovered from its log file."""

    path: Path
    dataset: str = ""

    # --- configuration -------------------------------------------------
    partitioning: str = ""
    pointer_type: str = ""          # canonical short name, e.g. "rary"
    pointer_type_raw: str = ""      # exactly as printed, e.g. "multiwriterary"
    block_size: int | None = None   # bytes
    # r-ary fan-out as the run reported it. Printed only for multiwriterary,
    # and only since --pointer-branching-factor existed, so it is None on every
    # log from before the decoupling -- see the branching_factor property.
    pointer_branching_factor: int | None = None
    num_queries: int | None = None
    k: int | None = None
    ef_search: int | None = None
    num_clusters: int | None = None

    # --- realized partition shape --------------------------------------
    # What the run actually built, as opposed to what was requested. The
    # block-size sweep derives --bs from a predicted max partition size, so
    # max_cluster_size is how a figure checks that prediction against reality.
    # Absent from louvain and unpartitioned runs, which print no cluster block.
    avg_cluster_size: float | None = None
    max_cluster_size: int | None = None
    max_copies_per_cluster: int | None = None
    intra_cluster_edges: int | None = None
    cross_cluster_edges: int | None = None

    # --- memory access statistics --------------------------------------
    total_reads: int | None = None
    total_writes: int | None = None
    total_allocs: int | None = None         # search phase
    build_allocs: int | None = None
    whole_run_allocs: int | None = None
    log2_whole_run_allocs: float | None = None   # as printed by the log

    # Per-query averages exactly as printed (1 decimal place); prefer the
    # reads_per_query property, which recovers the full precision.
    avg_reads_per_query: float | None = None
    avg_writes_per_query: float | None = None
    avg_allocs_per_query: float | None = None

    # --- search quality / cost -----------------------------------------
    recall_at_1: float | None = None
    recall_at_10: float | None = None
    faiss_avg_time: float | None = None
    oblivious_avg_time: float | None = None
    total_rounds: int | None = None
    avg_rounds_per_query: float | None = None

    @property
    def complete(self) -> bool:
        """True when the run reached the memory-access summary block."""
        return None not in (self.block_size, self.total_reads, self.whole_run_allocs)

    @property
    def pointer_label(self) -> str:
        return POINTER_LABELS.get(self.pointer_type, self.pointer_type)

    @property
    def reads_per_query(self) -> float | None:
        """Memory reads per query, at full precision where possible.

        The log prints this rounded to one decimal; dividing the totals
        recovers the exact value.  Falls back to the printed average.
        """
        if self.total_reads is not None and self.num_queries:
            return self.total_reads / self.num_queries
        return self.avg_reads_per_query

    @property
    def branching_factor(self) -> int | None:
        """r for this run: what the log reports, else what --bs implied.

        The fallback is exact rather than approximate. Before
        --pointer-branching-factor existed, o_hnsw.py derived r from the block
        size as ``floor((bs - 2 * 8) / 8)`` and printed nothing, so recomputing
        it recovers the value that run was actually configured with -- not an
        estimate of it. Runs since the decoupling print the line and are read
        from it, which is the whole point: there r no longer follows from bs.
        """
        if self.pointer_branching_factor is not None:
            return self.pointer_branching_factor
        if self.block_size is None:
            return None
        return branching_factor(self.block_size)

    def log2_allocs(self, which: str = "whole-run") -> float:
        """log2 of an allocation count, recomputed from the exact integer.

        ``which`` is one of ``whole-run`` (default), ``build`` or ``search``.
        The log already prints ``log2(whole-run allocs)`` rounded to 2 decimals;
        recomputing keeps full precision and gives access to the other two.
        """
        counts = {
            "whole-run": self.whole_run_allocs,
            "build": self.build_allocs,
            "search": self.total_allocs,
        }
        if which not in counts:
            raise ValueError(f"unknown alloc scope {which!r}; expected one of {sorted(counts)}")
        n = counts[which]
        if n is None:
            if which == "whole-run" and self.log2_whole_run_allocs is not None:
                return self.log2_whole_run_allocs
            raise ValueError(f"{self.path.name}: no {which} allocation count in log")
        if n <= 0:
            raise ValueError(f"{self.path.name}: non-positive {which} allocation count {n}")
        return math.log2(n)


def _int(text: str) -> int:
    return int(text.replace(",", ""))


# field name -> (regex, converter).  All patterns are matched with re.M.
_PATTERNS: dict[str, tuple[str, object]] = {
    "partitioning": (r"^Clustering:\s+(\S+)", str),
    "pointer_type_raw": (r"^Pointer type:\s+(\S+)", str),
    "block_size": (r"^Pointer block size:\s+([\d,]+)\s*bytes", _int),
    "num_queries": (r"^Queries:\s+([\d,]+)", _int),
    "k": (r"^k:\s+([\d,]+)", _int),
    "ef_search": (r"^ef_search:\s+([\d,]+)", _int),
    "num_clusters": (r"^\s*Number of clusters:\s+([\d,]+)", _int),
    # The cluster block indents by one space and lowercases "max cluster size"
    # while capitalising its neighbours -- match leniently rather than relying
    # on that. "Number of clusters:" above is distinct from the header's
    # "Num clusters:", so the two do not collide.
    "avg_cluster_size": (r"^\s*Avg cluster size:\s+([\d.]+)\s*nodes", float),
    "max_cluster_size": (r"^\s*max cluster size:\s+([\d,]+)\s*nodes", _int),
    "max_copies_per_cluster": (r"^\s*Max copies / cluster:\s+([\d,]+)", _int),
    "intra_cluster_edges": (r"^Intra-cluster edges:\s+([\d,]+)", _int),
    "cross_cluster_edges": (r"^Cross-cluster edges:\s+([\d,]+)", _int),
    "total_reads": (r"^Total reads:\s+([\d,]+)", _int),
    "total_writes": (r"^Total writes:\s+([\d,]+)", _int),
    "total_allocs": (r"^Total allocations:\s+([\d,]+)", _int),
    "build_allocs": (r"^Build-phase allocations:\s*([\d,]+)", _int),
    "whole_run_allocs": (r"^Whole-run allocations:\s+([\d,]+)", _int),
    "log2_whole_run_allocs": (r"^log2\(whole-run allocs\):\s+([\d.]+)", float),
    "pointer_branching_factor": (r"^Pointer branching:\s+([\d,]+)", _int),
    "avg_reads_per_query": (r"^Average reads/query:\s+([\d.]+)", float),
    "avg_writes_per_query": (r"^Average writes/query:\s+([\d.]+)", float),
    "avg_allocs_per_query": (r"^Average allocs/query:\s+([\d.]+)", float),
    "recall_at_1": (r"^R@1\s*=\s*([\d.]+)", float),
    "recall_at_10": (r"^R@10\s*=\s*([\d.]+)", float),
    "faiss_avg_time": (r"^FAISS average time:\s+([\d.]+)s", float),
    "oblivious_avg_time": (r"^Oblivious average time:\s+([\d.]+)s", float),
    "total_rounds": (r"^Total logical rounds:\s+([\d,]+)", _int),
    "avg_rounds_per_query": (r"^Average rounds/query:\s+([\d.]+)", float),
}

_DATASET_RE = re.compile(r"(sift\d+[a-z]*|laion[\w-]*?|gist\d*[a-z]*|deep\d*[a-z]*)", re.I)


def _infer_dataset(path: Path) -> str:
    m = _DATASET_RE.search(path.stem)
    if m:
        return m.group(1).lower()
    return path.parent.name


def parse_log(path: str | Path) -> RunLog:
    """Read one log file into a :class:`RunLog`.

    Truncated logs (a run still in flight, or one that died) parse fine; they
    simply come back with ``complete == False``.
    """
    path = Path(path)
    text = path.read_text(errors="replace")

    values: dict[str, object] = {}
    for name, (pattern, convert) in _PATTERNS.items():
        m = re.search(pattern, text, re.M)
        if m:
            try:
                values[name] = convert(m.group(1))
            except ValueError:
                pass

    raw = str(values.get("pointer_type_raw", "")).lower()
    values["pointer_type"] = POINTER_ALIASES.get(raw, raw)
    values["dataset"] = _infer_dataset(path)

    known = {f.name for f in fields(RunLog)}
    return RunLog(path=path, **{k: v for k, v in values.items() if k in known})


def load_dir(
    log_dir: str | Path,
    pointer_type: str | None = None,
    partitioning: str | None = None,
    pattern: str = "*.log",
    complete_only: bool = True,
) -> tuple[list[RunLog], list[RunLog]]:
    """Parse every log in ``log_dir``.

    Returns ``(kept, dropped)``.  ``dropped`` holds the logs that matched the
    filters but were incomplete -- callers should report them rather than let
    them vanish silently.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise FileNotFoundError(f"log directory not found: {log_dir}")

    wanted_pointer = POINTER_ALIASES.get((pointer_type or "").lower(), pointer_type)

    kept: list[RunLog] = []
    dropped: list[RunLog] = []
    for f in sorted(log_dir.glob(pattern)):
        run = parse_log(f)
        if wanted_pointer and run.pointer_type != wanted_pointer:
            continue
        if partitioning and run.partitioning != partitioning:
            continue
        if complete_only and not run.complete:
            dropped.append(run)
            continue
        kept.append(run)
    return kept, dropped


def dedupe_by(runs: list[RunLog], key) -> list[RunLog]:
    """Keep the newest run per ``key`` (by file mtime), preserving key order."""
    best: dict[object, RunLog] = {}
    for run in runs:
        k = key(run)
        prev = best.get(k)
        if prev is None or run.path.stat().st_mtime > prev.path.stat().st_mtime:
            best[k] = run
    return list(best.values())
