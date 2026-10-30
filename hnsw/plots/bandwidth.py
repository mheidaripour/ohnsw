"""The ORAM bandwidth model, in one place.

Both plots derive their numbers from here, so the formula has a single
definition and cannot drift between figures.
"""

from __future__ import annotations

BYTE_UNITS = [("B", 1.0), ("KiB", 2 ** 10), ("MiB", 2 ** 20),
              ("GiB", 2 ** 30), ("TiB", 2 ** 40), ("PiB", 2 ** 50)]

# An r-ary NODE cell is ``parent + group[r] + idx``: two 8-byte fields and r
# slots of one address each. o_hnsw.py inverts exactly this to derive r from a
# block size, as ``floor((bs - 2 * 8) / 8)``.
POINTER_SLOT_BYTES = 8
POINTER_HEADER_BYTES = 16

BUCKET_CAPACITY = 4
EVICTION_PATHS = {
    "rary": 2,          # BOSAM  ->  S_B = 24 * B * log2(N_alloc)
    "multiwrite": 1,    # OSAM+  ->  S_O = 16 * B * log2(N_alloc)
}




def branching_factor(block_size: int) -> int:
    """The r-ary pointer fan-out implied by a block size.

    Matches o_hnsw.py's ``floor((bs - 2 * 8) / 8)``. This is the value a run
    got when r was not set explicitly; since --pointer-branching-factor exists
    it is a fallback for older logs, not the definition of r.
    """
    return (block_size - POINTER_HEADER_BYTES) // POINTER_SLOT_BYTES


def oram_cost(pointer_type: str, block_size: int, log2_allocs: float, b: int,
              r: int | None = None) -> float:
    """Bytes moved by a single ORAM access, per the paper's network model.

        S = 2 * Z * (1 + e) * B * log2(N_alloc)

    ``Z = 4`` is the bucket capacity, ``B = block_size`` the bytes in one
    block, and ``log2(N_alloc)`` the tree height, so ``Z * B * height`` is one
    PathORAM path. ``(1 + e)`` counts the paths an access moves -- one read
    path plus ``e`` eviction paths -- and the leading ``2`` is the path
    travelling in both directions between client and server.

    ``e`` is a property of the scheme, not of the run:

        OSAM+ (multiwrite)  e_O = 1  ->  S_O = 16 * B * log2(N_alloc)
        BOSAM (r-ary)       e_B = 2  ->  S_B = 24 * B * log2(N_alloc)

    single_access_machine.py:397-401 is where a run makes that distinction:
    the eviction count applies only to reads tagged RARY_POINTER_STRUCTURE,
    and every other read evicts exactly once.

    CAVEAT: e_B = 2 is therefore an upper bound for BOSAM rather than a
    measurement -- the same run's Vertex, FanOut and AVL reads take the
    single-eviction branch. Nothing in the logs separates the two access
    classes, so the model charges e_B to all of them.

    ``b`` was the eviction count back when it came from ``--b`` on the
    command line and applied to both schemes alike; e_O/e_B above now fix it
    per pointer type, so ``--b`` in make_tables.py, plot_latency.py and
    plot_tradeoff.py NO LONGER CHANGES ANY NUMBER. plot_tradeoff.py draws one
    series per value in its ``--b`` list, and those series are now identical.

    ``r`` stopped mattering when the r-ary branch below was folded away: the
    fan-out scaling it applied is not part of the paper's model.

    Both stay in the signature because all three callers pass them
    positionally; dropping them means editing every call site.

    Raises ValueError for a pointer type absent from EVICTION_PATHS.
    """
    try:
        e = EVICTION_PATHS[pointer_type]
    except KeyError:
        raise ValueError(
            f"no eviction count for pointer type {pointer_type!r}; known: "
            f"{', '.join(sorted(EVICTION_PATHS))}. The paper's network model "
            "prices only OSAM+ and BOSAM -- add the scheme to EVICTION_PATHS "
            "deliberately rather than letting it be reported as OSAM+."
        ) from None
    return 2 * BUCKET_CAPACITY * (1 + e) * log2_allocs * block_size



def pick_unit(max_value: float) -> tuple[str, float]:
    """One byte unit for a whole axis, chosen from the largest value on it."""
    name, scale = BYTE_UNITS[0]
    for n, s in BYTE_UNITS:
        if max_value >= s:
            name, scale = n, s
    return name, scale


def unit_for(pointer_type: str, max_value: float) -> tuple[str, float]:
    """Axis unit and scale for a series, chosen by what oram_cost returned.

    Every branch of oram_cost is in bytes, so this is pick_unit for all of
    them. It kept a separate decimal "slots" ladder for r-ary while that branch
    returned a slot count; the pointer_type argument is retained so callers do
    not have to change if a non-byte cost is ever added back.
    """
    return pick_unit(max_value)


def fmt_bytes(n: float) -> str:
    """Render a byte count with its own natural unit, e.g. ``64 KiB``."""
    name, scale = pick_unit(n)
    return f"{n / scale:g} {name}"


def fmt_bytes_short(n: float) -> str:
    """Compact byte count for crowded axes, e.g. ``64K``, ``1M``.

    At LaTeX column width a run of ``512 KiB`` / ``1 MiB`` tick labels overlaps;
    these fit.
    """
    name, scale = pick_unit(n)
    return f"{n / scale:g}{name[0].upper() if name != 'B' else ''}"
