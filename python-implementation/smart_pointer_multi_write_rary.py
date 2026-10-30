"""
SmartPointerMultiWriteRary
==========================

General inverted r-ary pointer tree on a multi-write / single-read SAM, with
splay-on-dereference performed on the *virtual* binary expansion of each r-ary
group.  Works for any even branching factor and any number
of pointers n >= 1 at any tree depth (e.g. BF=4 with n=64 -> 3 group levels).

Physical layout (BF=4, n=16 example, drawn downward)
----------------------------------------------------

                ROOT(value)
               /  |   |   \\
             N    N   N    N          internal group (4 members)
            /|\\\\  ...          ...
          N N N N                     leaf group (pointer positions)

Cell types
----------

ROOT(value)                  the unique cell holding the shared payload.
NODE(parent, group, idx)     member of an r-ary *group*:
    parent : SAM address of the cell this group hangs under (ROOT or NODE).
    group  : tuple of the addresses of ALL members of this group
             (length <= BF, None marks a deleted member).
    idx    : this cell's position in group.

Why members store the group (and there is no separate HUB cell)
---------------------------------------------------------------

The SAM allows each address to be READ exactly once (writes are unlimited and
do NOT reset the read budget).  Splay must relink sibling subtrees without
reading them.  A NODE cell contains only *upward* information (parent + group)
-- its children point at its address and are unaffected by rewrites -- so any
known sibling address can be blindly rewritten into a new group.  A HUB cell
holding downward slot pointers cannot be blindly relinked (its slot pointers
would be lost), which is why the previous HUB/LEAF design could not support multi-level
trees or shared put across 3+ pointers.

Virtual binary expansion
------------------------

A group of m <= BF members under parent A is viewed as a nearly balanced
virtual binary tree over its BF slots.  The virtual internal nodes are never
stored.  For a member at index i, the sibling sets are the slot intervals
hanging off the path from i to the root of this virtual tree.

When BF is a power of two this is the same as the bit-flip description.  For
non-power even BF, the virtual tree is simply not perfect, but its height is
ceil(log2(BF)).

Dereference / splay
-------------------

deref(head) reads each cell on the path head -> ROOT exactly once (depth + 1
reads total) and rebuilds the touched structure:

  * sibling slot-blocks of every consumed cell become "pending" subtree sets,
  * pending sets are pair-merged into fresh groups bottom-up (one merge round
    per ~2 consumed cells),
  * at the ROOT a single fresh root cell is allocated and the remaining
    pending members are linked under it as one group (<= BF members).

Operation costs
---------------

copy : one read -- read the copied head once and replace it with all requested
        fresh live heads.  They stay in the current sibling group when there is
        room; otherwise they are installed below a fresh internal member.
get  : depth+1 reads, amortized O(log_2 d) with groups re-merged up to BF
        members, so balanced fan-out shortens physical depth by log2(BF).
put  : one deref + one write to the unique root.
"""

from __future__ import annotations

import copy
import math
import random
from sys import maxsize
from typing import Any, Dict, List, Optional, Tuple

from attribute_at_positions import AttributeAtPositions
from single_access_machine import (
    alloc,
    get_global_counter,
    path_oram_evict,
    read,
    reset as sam_reset,
    return_available_address,
    set_max_reads,
    set_max_writes,
    write,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BRANCHING_FACTOR = 256


def _check_bf(bf: int) -> int:
    if bf < 2 or bf % 2 != 0:
        raise ValueError(f"Branching factor must be an even integer >= 2, got {bf}")
    return bf


def _ceil_log2_bf(bf: int) -> int:
    return math.ceil(math.log2(bf))

# ---------------------------------------------------------------------------
# SAM cell tags
# ---------------------------------------------------------------------------

ROOT_TAG = "ROOT"
NODE_TAG = "NODE"

RootTuple = Tuple[str, None, Any] # (tag, parent, value)
NodeTuple = Tuple[str, int, Tuple[Optional[int], ...], int] # (tag, parent, group, idx)


def sibling_blocks(idx: int, group_len: int, bf: int) -> List[List[int]]:
    """
    Slot-index blocks of the sibling subtrees of `idx` in a balanced virtual
    binary expansion of one group, ordered bottom-up.

    This works for any even BF >= 2.  The virtual tree is built over the
    interval [0, group_len) by recursively splitting intervals as evenly as
    possible.  We use the upper midpoint so odd-size intervals put the extra
    slot on the left, matching the old power-of-two partial-group convention
    more closely.
    """
    if group_len < 1:
        return []

    if bf < 2 or bf % 2 != 0:
        raise ValueError(f"Branching factor must be an even integer >= 2, got {bf}")

    if not 0 <= idx < group_len:
        raise ValueError(f"idx {idx} out of range for group_len {group_len}")

    blocks_top_down: List[List[int]] = []

    lo, hi = 0, group_len
    while hi - lo > 1:
        mid = (lo + hi + 1) // 2

        if idx < mid:
            # Path goes left; right interval is the sibling subtree.
            blocks_top_down.append(list(range(mid, hi)))
            hi = mid
        else:
            # Path goes right; left interval is the sibling subtree.
            blocks_top_down.append(list(range(lo, mid)))
            lo = mid

    return list(reversed(blocks_top_down))


class SmartPointerMultiWriteRary:
    label = "multiwrite_rary"
    pointer_osam = True
    structure = "SmartPointerMultiWriteRary"

    # class-level BF; call set_branching_factor() before use
    branching_factor: int = DEFAULT_BRANCHING_FACTOR

    # path depth (NODE cells from head to ROOT, ROOT excluded) of the most
    # recent deref; reads consumed by that deref == _last_depth + 1.
    _last_depth: int = 0
    _new_since_read: int = 0 
    _bulk_copy_write_count: Optional[int] = None
    def __init__(
        self,
        head: int | SmartPointerMultiWriteRary | None,
    ) -> None:
        """Wrap a SAM head address, or smart-copy another pointer's head."""
        if isinstance(head, int) or head is None:
            self.head = head
        elif isinstance(head, SmartPointerMultiWriteRary):
            new_copies, new_orig = SmartPointerMultiWriteRary.copy_raw_address(
                head.head
            )
            self.head = new_copies[0]
            head.head = new_orig
        else:
            raise ValueError(f"Invalid head type: {type(head)}")

    def __repr__(self) -> str:
        """Show the underlying SAM head address."""
        return str(self.head)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @classmethod
    def set_branching_factor(cls, bf: int) -> None:
        """Set the r-ary branching factor (must be an even integer >= 2)."""
        cls.branching_factor = _check_bf(bf)

    @classmethod
    def bf(cls) -> int:
        """Return the current branching factor BF."""
        return cls.branching_factor

    @classmethod
    def k(cls) -> int:
        """Return ceil(log2(BF)), the height of the virtual binary expansion."""
        return _ceil_log2_bf(cls.branching_factor)

    @classmethod
    def _write_cell(cls, address: int, value: Any) -> None:
        write(address, value, cls.structure)
        if cls._bulk_copy_write_count is None:
            return

        cls._bulk_copy_write_count += 1
        while cls._bulk_copy_write_count >= cls.bf():
            path_oram_evict(cls.bf())
            cls._bulk_copy_write_count -= cls.bf()

    # ------------------------------------------------------------------
    # Group primitives (write-only; never read)
    # ------------------------------------------------------------------

    @classmethod
    def normalize_group(cls, members: List[int], parent: int) -> None:
        """
        Normalize one sibling group under `parent`.

        The stored group has exactly BF slots, padded with None.  Each live member
        stores the same parent and the same padded group tuple, together with its
        own slot index.
        """
        if not 1 <= len(members) <= cls.bf():
            raise ValueError(
                f"group size must be 1..{cls.bf()}, got {len(members)}"
            )

        if len(set(members)) != len(members):
            raise ValueError("normalize_group: duplicate address in sibling group")

        padded: Tuple[Optional[int], ...] = tuple(
            members + [None] * (cls.bf() - len(members))
        )

        for i, m in enumerate(members):
            cls._write_cell(m, (NODE_TAG, parent, padded, i))


    @classmethod
    def _sibling_sets(
        cls, group: Tuple[Optional[int], ...], idx: int
    ) -> List[List[int]]:
        """Occupied sibling subtree sets of slot `idx`, bottom-up."""
        sets: List[List[int]] = []
        for block in sibling_blocks(idx, len(group), cls.bf()):
            members = [group[i] for i in block if group[i] is not None]
            if members:
                sets.append(members)  # type: ignore[arg-type]
        return sets
    
    @classmethod
    def _min_dense(cls) -> int:
        """Minimum size for a non-leaf sibling group."""
        return cls.bf() // 2


    @classmethod
    def _live_members(
        cls,
        group: Tuple[Optional[int], ...],
    ) -> List[int]:
        """Return the live members of a sibling group, compacting away holes."""
        return [m for m in group if m is not None]


    @classmethod
    def _live_group_without_index(
        cls,
        group: Tuple[Optional[int], ...],
        remove_idx: int,
    ) -> List[int]:
        """Return live group members except the member at remove_idx."""
        survivors: List[int] = []
        for i, m in enumerate(group):
            if i == remove_idx:
                continue
            if m is not None:
                survivors.append(m)
        return survivors


    @classmethod
    def _delete_raw_address(cls, addr: int) -> None:
        """
        Delete one live pointer-tree address.

        The address is read exactly once and removed from the inverted tree.
        Underfull groups that existed before the deletion are treated as leaf
        groups and are not rebalanced.  Dense groups that become underfull are
        repaired with their parent group.
        """
        node = read(addr, cls.structure)
        cls._check_node_tuple(addr, node)
        return_available_address(addr)

        if node[0] == ROOT_TAG:
            # Last live reference to the value.
            return

        if node[0] != NODE_TAG:
            raise ValueError(f"delete: bad tag {node[0]!r}")

        _, parent, group, idx = node
        cls._remove_from_sibling_group(parent, group, idx)


    @classmethod
    def _remove_from_sibling_group(
        cls,
        parent: int,
        group: Tuple[Optional[int], ...],
        idx: int,
    ) -> None:
        """
        Remove one member from a sibling group under parent.

        If the group was already underfull, it is treated as a leaf group.
        If the group was dense and becomes underfull, repair with the parent group.
        """
        min_dense = cls._min_dense()

        old_live = cls._live_members(group)
        survivors = cls._live_group_without_index(group, idx)

        # If no children remain below parent, remove parent from the tree as well.
        if len(survivors) == 0:
            cls._delete_raw_address(parent)
            return

        # Existing underfull groups are exactly the leaf groups.  They do not need
        # density repair; just compact away the deleted member.
        if len(old_live) < min_dense:
            cls.normalize_group(survivors, parent)
            return

        # If the group remains dense, compact and stop.
        if len(survivors) >= min_dense:
            cls.normalize_group(survivors, parent)
            return

        # The group was dense before deletion and is now underfull.  Repair it.
        cls._repair_underfull_group(parent, survivors)


    @classmethod
    def _repair_underfull_group(
        cls,
        parent: int,
        survivors: List[int],
    ) -> None:
        """
        Repair an underfull sibling group under `parent`.

        The group became underfull only because it was dense before the deletion.
        We repair by borrowing one member from the parent group when possible;
        otherwise we merge the underfull group into the parent group.
        """
        min_dense = cls._min_dense()

        if len(survivors) >= min_dense:
            cls.normalize_group(survivors, parent)
            return

        parent_node = read(parent, cls.structure)
        cls._check_node_tuple(parent, parent_node)
        return_available_address(parent)

        # Root exception: the top group under the root may be underfull.
        if parent_node[0] == ROOT_TAG:
            _, _, value = parent_node

            if len(survivors) == 1:
                # Contract the only survivor into the root.
                write(survivors[0], (ROOT_TAG, None, value), cls.structure)
                return

            # Keep an underfull top group below a fresh root.
            new_root = alloc(cls.pointer_osam, cls.structure)
            write(new_root, (ROOT_TAG, None, value), cls.structure)
            cls.normalize_group(survivors, new_root)
            return

        if parent_node[0] != NODE_TAG:
            raise ValueError(f"delete repair: bad tag {parent_node[0]!r}")

        _, grandparent, parent_group, parent_idx = parent_node
        upper_live = cls._live_members(parent_group)

        # If the parent group is underfull, then by the invariant it should be a
        # leaf group.  But it has a child group below `parent`, so this indicates
        # structural corruption.
        if len(upper_live) < min_dense:
            raise ValueError(
                "delete repair found an underfull non-leaf parent group"
            )

        # Borrow case: parent group has slack above the density threshold.
        # Move one sibling of `parent` down into the underfull group.
        if len(upper_live) > min_dense:
            borrowed: Optional[int] = None
            for i, m in enumerate(parent_group):
                if i != parent_idx and m is not None:
                    borrowed = m
                    break

            if borrowed is None:
                raise ValueError("delete repair expected a borrowable sibling")

            # Since `parent` was read, replace it by a fresh address.
            new_parent = alloc(cls.pointer_osam, cls.structure)

            lower_group = survivors + [borrowed]

            upper_group: List[int] = []
            for i, m in enumerate(parent_group):
                if m is None:
                    continue
                if i == parent_idx:
                    upper_group.append(new_parent)
                elif m == borrowed:
                    continue
                else:
                    upper_group.append(m)

            cls.normalize_group(lower_group, new_parent)
            cls.normalize_group(upper_group, grandparent)
            return

        # Merge case: the parent group is exactly minimally dense.  Remove the
        # consumed parent and splice the survivors into its position.
        merged_group: List[int] = []
        for i, m in enumerate(parent_group):
            if m is None:
                continue
            if i == parent_idx:
                merged_group.extend(survivors)
            else:
                merged_group.append(m)

        if len(merged_group) == 0:
            cls._delete_raw_address(grandparent)
            return

        if len(merged_group) == 1:
            cls._contract_single_child_to_parent(merged_group[0], grandparent)
            return

        cls.normalize_group(merged_group, grandparent)


    @classmethod
    def _contract_single_child_to_parent(cls, child: int, parent: int) -> None:
        """
        Contract a singleton group upward through parent.

        If parent is the root, the child becomes the root.  Otherwise, child
        replaces parent in the parent's sibling group.
        """
        parent_node = read(parent, cls.structure)
        cls._check_node_tuple(parent, parent_node)
        return_available_address(parent)

        if parent_node[0] == ROOT_TAG:
            _, _, value = parent_node
            write(child, (ROOT_TAG, None, value), cls.structure)
            return

        if parent_node[0] != NODE_TAG:
            raise ValueError(f"contract: bad tag {parent_node[0]!r}")

        _, grandparent, parent_group, parent_idx = parent_node

        new_group: List[int] = []
        for i, m in enumerate(parent_group):
            if m is None:
                continue
            if i == parent_idx:
                new_group.append(child)
            else:
                new_group.append(m)

        if len(new_group) == 1:
            cls._contract_single_child_to_parent(new_group[0], grandparent)
        else:
            cls.normalize_group(new_group, grandparent)

    @classmethod
    def _check_node_tuple(cls, addr: int, node: Any) -> None:
        if node[0] == ROOT_TAG:
            return

        if node[0] != NODE_TAG:
            raise ValueError(f"bad node tag at {addr}: {node[0]!r}")

        _, parent, group, idx = node

        if len(group) != cls.bf():
            raise ValueError(f"group at {addr} has length {len(group)}, expected {cls.bf()}")

        if not (0 <= idx < len(group)):
            raise ValueError(f"bad idx {idx} at {addr}")

        if group[idx] != addr:
            raise ValueError(f"group/index mismatch at {addr}")

        live = [m for m in group if m is not None]
        if len(live) != len(set(live)):
            raise ValueError(f"duplicate live address in group at {addr}")

        if len(live) > cls.bf():
            raise ValueError(f"oversized group at {addr}")
    # copy -- read the copied head once and install the refreshed original plus
    #     every requested copy together.  This avoids the repeated reads and
    #     sibling-group rewrites incurred by copying one at a time.
    @classmethod
    def copy_raw_address(
        cls, old_head: int, num_copies: int = 1
    ) -> Tuple[List[int], int]:
        started_counting = cls._bulk_copy_write_count is None
        if started_counting:
            cls._bulk_copy_write_count = 0
        try:
            result = cls._copy_raw_address_with_bounded_writes(old_head, num_copies)
            if started_counting and cls._bulk_copy_write_count:
                path_oram_evict(cls.bf())
                cls._bulk_copy_write_count = 0
            return result
        finally:
            if started_counting:
                cls._bulk_copy_write_count = None

    @classmethod
    def _copy_raw_address_with_bounded_writes(
        cls, old_head: int, num_copies: int = 1
    ) -> Tuple[List[int], int]:
        """
        Create `num_copies` aliases of `old_head` in one tree update.

        The copied address is READ and therefore consumed.  The operation returns
        fresh live heads:

            ([copy_head, ...], new_original_head)

        If the current sibling group has room, the old head is replaced by the
        refreshed original and copies in the same group.  Otherwise, the old head
        is replaced by a fresh internal member and a balanced subtree containing
        all returned heads is installed beneath it.
        """
        if isinstance(num_copies, bool) or not isinstance(num_copies, int):
            raise TypeError("num_copies must be an integer")
        if num_copies < 1:
            raise ValueError(f"num_copies must be >= 1, got {num_copies}")

        node = read(old_head, cls.structure)
        cls._check_node_tuple(old_head, node)
        return_available_address(old_head)

        total_heads = num_copies + 1

        # Case 1: copying the unique root pointer.
        if node[0] == ROOT_TAG:
            _, _, value = node

            # The old root address was read, so create a fresh root cell.
            new_root = alloc(cls.pointer_osam, cls.structure)
            cls._write_cell(new_root, (ROOT_TAG, None, value))

            # Install every alias together below the fresh root.
            returned_heads: List[int] = []
            cls._build_subtree(new_root, total_heads, returned_heads)
            return returned_heads[1:], returned_heads[0]

        if node[0] != NODE_TAG:
            raise ValueError(f"copy: bad tag {node[0]!r}")

        _, parent, group, idx = node

        live_members = [m for m in group if m is not None]

        # Case 2: the sibling group has room for every returned head.
        if len(live_members) - 1 + total_heads <= cls.bf():
            returned_heads = [
                alloc(cls.pointer_osam, cls.structure) for _ in range(total_heads)
            ]
            new_group: List[int] = []
            for i, m in enumerate(group):
                if m is None:
                    continue
                if i == idx:
                    new_group.extend(returned_heads)
                else:
                    new_group.append(m)

            cls.normalize_group(new_group, parent)
            return returned_heads[1:], returned_heads[0]

        # Case 3: all returned heads do not fit.  Replace old_head by a fresh
        # internal member and install them together in a balanced subtree.
        internal = alloc(cls.pointer_osam, cls.structure)

        upper_group: List[int] = []
        for i, m in enumerate(group):
            if m is None:
                continue
            if i == idx:
                upper_group.append(internal)
            else:
                upper_group.append(m)

        cls.normalize_group(upper_group, parent)
        returned_heads = []
        cls._build_subtree(internal, total_heads, returned_heads)

        return returned_heads[1:], returned_heads[0]

    @classmethod
    def __smart_copy_pointer(
        cls, p0: SmartPointerMultiWriteRary, num_copies: int = 1
    ) -> SmartPointerMultiWriteRary | List[SmartPointerMultiWriteRary]:
        if p0.head is None:
            if isinstance(num_copies, bool) or not isinstance(num_copies, int):
                raise TypeError("num_copies must be an integer")
            if num_copies < 1:
                raise ValueError(f"num_copies must be >= 1, got {num_copies}")
            copies = [SmartPointerMultiWriteRary(None) for _ in range(num_copies)]
            return copies[0] if num_copies == 1 else copies

        new_copies, new_orig = cls.copy_raw_address(p0.head, num_copies)
        p0.head = new_orig
        copies = [SmartPointerMultiWriteRary(head) for head in new_copies]
        return copies[0] if num_copies == 1 else copies

    @classmethod
    def copy(
        cls, content: Any, temp_copy: bool = False, num_copies: int = 1
    ) -> Any:
        """Smart-copy an object, returning a list when `num_copies` is above 1."""
        if isinstance(content, SmartPointerMultiWriteRary):
            return cls.__smart_copy_pointer(content, num_copies)
        if isinstance(num_copies, bool) or not isinstance(num_copies, int):
            raise TypeError("num_copies must be an integer")
        if num_copies < 1:
            raise ValueError(f"num_copies must be >= 1, got {num_copies}")
        if hasattr(content, "smart_copy") and callable(content.smart_copy):
            if num_copies > 1:
                return content.smart_copy(content, temp_copy, num_copies)
            return content.smart_copy(content, temp_copy)
        if num_copies > 1:
            return [cls.copy(content, temp_copy) for _ in range(num_copies)]
        if isinstance(content, list):
            return [cls.copy(x, temp_copy) for x in content]
        if isinstance(content, tuple):
            return tuple(cls.copy(x, temp_copy) for x in content)
        if isinstance(content, set):
            return {cls.copy(x, temp_copy) for x in content}
        if isinstance(content, frozenset):
            return frozenset(cls.copy(x, temp_copy) for x in content)
        if isinstance(content, dict):
            return {k: cls.copy(v, temp_copy) for k, v in content.items()}
        if content is not None:
            return copy.deepcopy(content)
        return content

    # ------------------------------------------------------------------
    # deref + generalized splay
    # ------------------------------------------------------------------

    @classmethod
    def deref(cls, head: int) -> Tuple[int, int, Any]:
        """
        Dereference `head`: read every cell on the path to ROOT exactly once,
        rebuild the touched structure with the accessed pointer near the new
        root, and return (new_head, new_root_addr, value).

        The new root cell is allocated but NOT written -- the caller writes
        the (possibly updated) value into it, preserving the single-root
        invariant.
        """
        ## node =(tag, parent, group, idx)
        node = read(head, cls.structure) #
        cls._check_node_tuple(head, node)
        return_available_address(head)
        p_new = alloc(cls.pointer_osam, cls.structure)

        if node[0] == ROOT_TAG:
            cls._last_depth = 0
            return p_new, p_new, node[2]
        if node[0] != NODE_TAG:
            raise ValueError(f"deref: bad tag {node[0]!r}")

        _, parent, group, idx = node

        # pending: subtree-root sets awaiting placement above the consumed path.
        # Initially this contains the fresh replacement for the accessed pointer
        # together with the sibling blocks of the consumed head.  During splay, pending
        # is repeatedly packed into fresh sibling groups before the next path read.
        pending: List[List[int]] = [[p_new]]
        pending.extend(cls._sibling_sets(group, idx))

        new_root, value = cls.splay(pending, parent)
        return p_new, new_root, value

    @classmethod
    def _flatten_pending(cls, pending: List[List[int]]) -> List[int]:
        """
        Flatten pending subtree-root sets, preserving order.

        Each element of `pending` is a set/list of subtree roots that have not yet
        been attached above the consumed path.  The flattened list is the local
        affected region to be packed into one or more sibling groups.
        """
        out: List[int] = []
        seen = set()

        for part in pending:
            for x in part:
                if x in seen:
                    raise ValueError(f"duplicate pending root {x}")
                seen.add(x)
                out.append(x)

        return out


    @classmethod
    def _dense_chunks(cls, members: List[int]) -> List[List[int]]:
        """
        Partition members into chunks of size at most BF and, when more than one
        chunk is needed, each chunk has size at least BF/2.
        """
        n = len(members)
        bf = cls.bf()
        min_dense = cls._min_dense()

        if n <= bf:
            return [members]

        num_chunks = math.ceil(n / bf)
        base, rem = divmod(n, num_chunks)

        if base < min_dense:
            raise ValueError(
                f"cannot densely partition {n} members with bf={bf}, "
                f"min_dense={min_dense}"
            )

        sizes = [base + 1] * rem + [base] * (num_chunks - rem)

        chunks: List[List[int]] = []
        i = 0
        for size in sizes:
            chunks.append(members[i : i + size])
            i += size

        return chunks


    @classmethod
    def _pack_members_as_groups(cls, members: List[int]) -> List[List[int]]:
        """
        Pack `members` into fresh sibling groups.  If more than one output group is
        needed, every output group is dense.
        """
        bf = cls.bf()

        if len(members) <= bf:
            return [members]

        packed: List[List[int]] = []
        for group in cls._dense_chunks(members):
            parent = alloc(cls.pointer_osam, cls.structure)
            cls.normalize_group(group, parent)
            packed.append([parent])

        return packed


    @classmethod
    def _pack_pending_after_read(cls, pending: List[List[int]]) -> List[List[int]]:
        """
        Pack the pending region after one path read.

        Invariant before a path read:
            pending contains at most BF live roots, unless it is already represented
            by a constant number of packed singleton roots.

        A path read adds at most BF-1 sibling roots.  Therefore this packing step
        writes at most O(BF) addresses before the next read.
        """
        members = cls._flatten_pending(pending)
        return cls._pack_members_as_groups(members)


    @classmethod
    def _collapse_pending_for_root(cls, pending: List[List[int]]) -> List[int]:
        """
        Collapse the remaining pending region into one group under the new root.
        Repeatedly pack all pending roots until the root has at most BF children.
        """
        bf = cls.bf()

        while True:
            members = cls._flatten_pending(pending)
            if len(members) <= bf:
                return members
            pending = cls._pack_members_as_groups(members)


    @classmethod
    def splay(cls, pending: List[List[int]], x: int) -> Tuple[int, Any]:
        """
        Climb from address `x` to the ROOT, rebuilding the consumed path with
        bounded write bursts.

        The input `pending` represents the local subtree roots already exposed
        below `x`; initially it contains the fresh replacement for the accessed
        pointer together with the sibling blocks of the consumed head.  At each
        step, the algorithm reads one path node, adds that node's sibling blocks to
        the pending region, and immediately packs the affected region if it no
        longer fits in one sibling group.

        Thus between two reads, the algorithm writes back only a constant-size
        affected region.  More concretely, each packing phase writes at most
        2*BF-1 NODE cells, because before packing the pending region
        contains at most one stored group plus the sibling roots exposed by one
        newly read group.
        """
        depth = 1

        while True:
            xnode = read(x, cls.structure)
            cls._check_node_tuple(x, xnode)

            if xnode[0] == ROOT_TAG:
                cls._last_depth = depth

                # Allocate a fresh root address.  The caller writes the ROOT cell
                # with the value; splay only attaches the rebuilt child group below it.
                new_root = alloc(cls.pointer_osam, cls.structure)

                root_children = cls._collapse_pending_for_root(pending)
                cls.normalize_group(root_children, new_root)

                return new_root, xnode[2]

            if xnode[0] != NODE_TAG:
                raise ValueError(f"splay: bad path tag {xnode[0]!r}")

            _, xparent, xgroup, xidx = xnode

            # Add the sibling subtrees of the consumed path node.
            pending.extend(cls._sibling_sets(xgroup, xidx))

            # Write back the constant-size affected region before the next read.
            pending = cls._pack_pending_after_read(pending)

            depth += 1
            x = xparent

    # ------------------------------------------------------------------
    # Public API (mirrors smart_pointer_multi_write)
    # ------------------------------------------------------------------

    @classmethod
    def new(cls, value: Any = None) -> int:
        """Create a fresh ROOT cell holding `value`; return its head address."""
        root = alloc(cls.pointer_osam, cls.structure)
        write(root, (ROOT_TAG, None, value), cls.structure)
        cls._new_since_read +=1
        if cls._new_since_read >= cls.branching_factor:
            cls._new_since_read = 0
            path_oram_evict(cls.branching_factor)
        return root

    @classmethod
    def get(cls, p: SmartPointerMultiWriteRary) -> Any:
        """Dereference `p` (splaying its path) and return the shared value."""
        assert isinstance(p.head, int)
        new_head, root, value = cls.deref(p.head)
        p.head = new_head
        write(root, (ROOT_TAG, None, value), cls.structure)
        return value

    @classmethod
    def get_and_copy(cls, p: SmartPointerMultiWriteRary) -> Any:
        """Get `p`'s value and return an independent smart-copy of it."""
        value = cls.get(p)
        return cls.copy(value)

    @classmethod
    def is_single_reference(cls, p: SmartPointerMultiWriteRary) -> bool:
        """True if `p` is the sole pointer to its value (head becomes the root)."""
        assert isinstance(p.head, int)
        new_head, root, value = cls.deref(p.head)
        write(root, (ROOT_TAG, None, value), cls.structure)
        p.head = new_head
        return new_head == root

    @classmethod
    def get_attr(
        cls,
        p: SmartPointerMultiWriteRary,
        attributes: AttributeAtPositions | List[str] | str,
        make_copies: bool = True,
    ) -> Any:
        """Get selected attribute(s) of `p`'s value, optionally smart-copied."""
        value = cls.get(p)
        content: Dict[str, Any] | str | None

        if isinstance(attributes, AttributeAtPositions):
            content = {}
            try:
                list_attr = getattr(value, attributes.attribute)
                sublist = []
                for pos in attributes.positions:
                    elem = list_attr[pos]
                    if make_copies:
                        elem = cls.copy(elem, temp_copy=make_copies)
                    sublist.append(elem)
                content[attributes.attribute] = sublist
            except Exception:
                content[attributes.attribute] = None
        elif isinstance(attributes, list):
            content = {}
            for attr in attributes:
                try:
                    elem = getattr(value, attr)
                    if make_copies:
                        elem = cls.copy(elem, temp_copy=make_copies)
                    content[attr] = elem
                except Exception:
                    content[attr] = None
        elif isinstance(attributes, str):
            try:
                content = getattr(value, attributes)
                if make_copies:
                    content = cls.copy(content, temp_copy=make_copies)
            except Exception:
                content = None
        else:
            raise TypeError(
                "Attributes must be a list, a string, or AttributeAtPositions"
            )
        return content

    @classmethod
    def put(
        cls,
        p: SmartPointerMultiWriteRary,
        value: Any,
        delete_old: bool = True,
    ) -> None:
        """Set the shared value via a single root write, visible to all aliases."""
        assert isinstance(p.head, int)
        new_head, root, old_value = cls.deref(p.head)
        p.head = new_head
        if delete_old:
            cls.delete(old_value)
        # single root cell: one write makes the new value visible to ALL
        # aliases, regardless of how many pointers share it or tree depth
        write(root, (ROOT_TAG, None, cls.copy(value)), cls.structure)

    @classmethod
    def put_attr(
        cls,
        p: SmartPointerMultiWriteRary,
        attributes: Dict[str, Any],
        delete_old: bool = False,
    ) -> None:
        """Update named attributes of `p`'s shared value in place."""
        assert isinstance(p.head, int)
        new_head, root, value = cls.deref(p.head)
        for attr in attributes:
            if delete_old and hasattr(value, attr):
                cls.delete(getattr(value, attr))
            setattr(value, attr, attributes[attr])
        p.head = new_head
        write(root, (ROOT_TAG, None, value), cls.structure)

    @classmethod
    def delete(cls, p: Any) -> None:
        """Drop a pointer or recursively delete smart pointers inside containers."""
        if isinstance(p, SmartPointerMultiWriteRary):
            if p.head is None:
                return
            cls._delete_raw_address(p.head)
            p.head = None
            return

        if isinstance(p, list):
            for elem in p:
                cls.delete(elem)
            return

        if isinstance(p, tuple):
            for elem in p:
                cls.delete(elem)
            return

        if isinstance(p, set):
            for elem in list(p):
                cls.delete(elem)
            return

        if isinstance(p, frozenset):
            for elem in p:
                cls.delete(elem)
            return

        if isinstance(p, dict):
            for key, value in list(p.items()):
                cls.delete(key)
                cls.delete(value)
            return

        if hasattr(p, "smart_delete") and callable(p.smart_delete):
            p.smart_delete()
            return

    # ------------------------------------------------------------------
    # General r-ary installation (any n >= 1, any BF = 2^k)
    # ------------------------------------------------------------------

    @classmethod
    def install_pointers(
        cls, value: Any, count: int
    ) -> List[SmartPointerMultiWriteRary]:
        """
        Build `count` pointers sharing one value in a balanced r-ary tree of
        depth ceil(log_BF(count)): e.g. BF=4, count=64 -> ROOT and 3 levels
        of 4-member groups (64 leaf pointer cells).
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        root = cls.new(value)
        leaves: List[int] = []
        cls._build_subtree(root, count, leaves)
        return [SmartPointerMultiWriteRary(addr) for addr in leaves]

    @classmethod
    def _build_subtree(cls, parent: int, n: int, leaves: List[int]) -> None:
        bf = cls.bf()
        if n <= bf:
            members = [alloc(cls.pointer_osam, cls.structure) for _ in range(n)]
            cls.normalize_group(members, parent)
            leaves.extend(members)
            return
        base, rem = divmod(n, bf)
        quotas = [base + 1] * rem + [base] * (bf - rem)
        members = [alloc(cls.pointer_osam, cls.structure) for _ in range(bf)]
        cls.normalize_group(members, parent)
        for m, q in zip(members, quotas):
            if q == 1:
                leaves.append(m)  # group slot doubles as a leaf pointer cell
            else:
                cls._build_subtree(m, q, leaves)


# ---------------------------------------------------------------------------
# Tests / demos
# ---------------------------------------------------------------------------

def _reset_sam() -> None:
    sam_reset()
    set_max_reads(1)
    set_max_writes(maxsize)


SP = SmartPointerMultiWriteRary


def test_sibling_blocks() -> None:
    SP.set_branching_factor(4)
    assert sibling_blocks(0, 4, 4) == [[1], [2, 3]]
    assert sibling_blocks(2, 4, 4) == [[3], [0, 1]]
    assert sibling_blocks(5, 8, 8) == [[4], [6, 7], [0, 1, 2, 3]]
    assert sibling_blocks(0, 2, 2) == [[1]]
    assert sibling_blocks(0, 1, 4) == []          # singleton group
    assert sibling_blocks(1, 3, 4) == [[0], [2]]
    print("test_sibling_blocks: OK")

def test_even_non_power_branching_factors() -> None:
    for bf in (6, 10, 14):
        _reset_sam()
        SP.set_branching_factor(bf)

        ptrs = SP.install_pointers(f"v{bf}", 3 * bf + 2)
        for p in ptrs:
            assert SP.get(p) == f"v{bf}"

        SP.put(ptrs[len(ptrs) // 2], f"w{bf}")
        for p in ptrs:
            assert SP.get(p) == f"w{bf}"

    print("test_even_non_power_branching_factors: OK")


def test_copy_and_get_bf4() -> None:
    _reset_sam()
    SP.set_branching_factor(4)
    p1 = SP(SP.new("cat"))
    p2 = SP.copy(p1)
    p3 = SP.copy(p1)
    assert SP.get(p1) == "cat"
    assert SP.get(p2) == "cat"
    assert SP.get(p3) == "cat"
    assert SP.get(p2) == "cat"
    print("test_copy_and_get_bf4: OK, counters:", get_global_counter())


def test_copy_expands_nonfull_group() -> None:
    _reset_sam()
    SP.set_branching_factor(4)

    p1 = SP(SP.new("x"))
    p2 = SP.copy(p1)

    # Copy one of the two siblings.  The group has room, so this should expand
    # the same group rather than creating a lower group.
    p3 = SP.copy(p1)

    assert SP.get(p1) == "x"
    assert SP.get(p2) == "x"
    assert SP.get(p3) == "x"
    print("test_copy_extend_current_group: OK")


def test_copy_full_group_creates_lower_group() -> None:
    _reset_sam()
    SP.set_branching_factor(4)

    p1 = SP(SP.new("x"))
    p2 = SP.copy(p1)
    p3 = SP.copy(p1)
    p4 = SP.copy(p1)

    # At this point a group of size 4 can be full.  One more copy should create
    # a fresh lower group below a fresh internal replacement.
    p5 = SP.copy(p1)

    for p in (p1, p2, p3, p4, p5):
        assert SP.get(p) == "x"
    print("test_copy_creates_lower_group: OK")


def test_bulk_copy() -> None:
    """Bulk copy returns all aliases while preserving scalar single-copy calls."""
    _reset_sam()
    SP.set_branching_factor(4)

    original = SP(SP.new("bulk"))
    copies = SP.copy(original, num_copies=10)

    assert isinstance(copies, list)
    assert len(copies) == 10
    assert all(isinstance(p, SP) for p in copies)
    for p in [original, *copies]:
        assert SP.get(p) == "bulk"

    single_copy = SP.copy(original)
    assert isinstance(single_copy, SP)
    assert SP.get(single_copy) == "bulk"
    print("test_bulk_copy: OK")


def test_copy_raw_address_bulk_contract() -> None:
    _reset_sam()
    SP.set_branching_factor(4)

    copy_heads, original_head = SP.copy_raw_address(
        SP.new("raw"), num_copies=6
    )

    assert isinstance(copy_heads, list)
    assert len(copy_heads) == 6
    pointers = [SP(original_head), *(SP(head) for head in copy_heads)]
    for p in pointers:
        assert SP.get(p) == "raw"
    print("test_copy_raw_address_bulk_contract: OK")


def test_bulk_copy_existing_group() -> None:
    """Bulk copy can expand a non-root group directly when all heads fit."""
    _reset_sam()
    SP.set_branching_factor(8)

    original = SP(SP.new("group"))
    first_copy = SP.copy(original)
    copies = SP.copy(original, num_copies=3)

    assert len(copies) == 3
    for p in [original, first_copy, *copies]:
        assert SP.get(p) == "group"
    print("test_bulk_copy_existing_group: OK")


def test_bulk_copy_reduces_operations() -> None:
    """One bulk update uses fewer reads and writes than repeated copies."""
    num_copies = 10

    _reset_sam()
    SP.set_branching_factor(4)
    bulk_original = SP(SP.new("value"))
    before = get_global_counter()
    SP.copy(bulk_original, num_copies=num_copies)
    bulk_cost = _delta(before, get_global_counter())

    _reset_sam()
    SP.set_branching_factor(4)
    sequential_original = SP(SP.new("value"))
    before = get_global_counter()
    for _ in range(num_copies):
        SP.copy(sequential_original)
    sequential_cost = _delta(before, get_global_counter())

    assert bulk_cost[1] == 1
    assert bulk_cost[1] < sequential_cost[1]
    assert bulk_cost[2] < sequential_cost[2]
    print(
        "test_bulk_copy_reduces_operations: OK,",
        f"bulk={bulk_cost}, sequential={sequential_cost}",
    )


def test_bulk_copy_validation() -> None:
    _reset_sam()
    SP.set_branching_factor(4)
    original = SP(SP.new("value"))

    for invalid in (0, -1):
        try:
            SP.copy(original, num_copies=invalid)
            raise AssertionError(f"accepted invalid num_copies={invalid!r}")
        except ValueError:
            pass

    for invalid in (True, 1.5, "2"):
        try:
            SP.copy(original, num_copies=invalid)
            raise AssertionError(f"accepted invalid num_copies={invalid!r}")
        except TypeError:
            pass
    print("test_bulk_copy_validation: OK")


def test_smart_pointer_bulk_copy_interface() -> None:
    from smart_pointer import SmartPointer

    _reset_sam()
    SmartPointer.set_smart_pointer("multiwriterary")
    SmartPointer.set_branching_factor(4)

    original = SmartPointer(SmartPointer.new("wrapped"))
    copies = SmartPointer.smart_copy(original, num_copies=5)

    assert isinstance(copies, list)
    assert len(copies) == 5
    assert all(isinstance(p, SmartPointer) for p in copies)
    for p in [original, *copies]:
        assert SmartPointer.get(p) == "wrapped"
    print("test_smart_pointer_bulk_copy_interface: OK")


def test_other_smart_pointer_copy_contract_unchanged() -> None:
    from smart_pointer import SmartPointer

    _reset_sam()
    SmartPointer.set_smart_pointer("original")
    original = SmartPointer(SmartPointer.new("single-write"))

    single_copy = SmartPointer.copy(original)
    assert isinstance(single_copy, SmartPointer)
    assert not isinstance(single_copy, list)

    try:
        SmartPointer.copy(original, num_copies=2)
        raise AssertionError("single-write pointer accepted num_copies=2")
    except NotImplementedError:
        pass
    print("test_other_smart_pointer_copy_contract_unchanged: OK")

def test_delete_compacts_group() -> None:
    _reset_sam()
    SP.set_branching_factor(4)

    ptrs = SP.install_pointers("v", 4)
    SP.delete(ptrs[1])

    for i in (0, 2, 3):
        assert SP.get(ptrs[i]) == "v"
    print("test_delete_compacts_group: OK")


def test_delete_contracts_unary_group() -> None:
    _reset_sam()
    SP.set_branching_factor(4)

    p1 = SP(SP.new("v"))
    p2 = SP.copy(p1)

    SP.delete(p2)

    assert SP.get(p1) == "v"
    assert SP.is_single_reference(p1)
    print("test_delete_contracts_unary_group: OK")

def test_install_bf4_n64() -> None:
    """The headline case: BF=4, n=64 -> 3 group levels under ROOT."""
    _reset_sam()
    SP.set_branching_factor(4)
    ptrs = SP.install_pointers("shared", 64)
    assert len(ptrs) == 64
    for i, p in enumerate(ptrs):
        assert SP.get(p) == "shared", f"first get, ptr {i}"
    for i, p in enumerate(ptrs):
        assert SP.get(p) == "shared", f"second get, ptr {i}"
    SP.put(ptrs[5], "updated")
    for i, p in enumerate(ptrs):
        assert SP.get(p) == "updated", f"after put, ptr {i}"
    print("test_install_bf4_n64: OK, counters:", get_global_counter())


def test_install_non_power_n() -> None:
    """n need not be a power of 2 (only BF must)."""
    _reset_sam()
    SP.set_branching_factor(4)
    for n in (2, 3, 5, 10, 17, 50):
        ptrs = SP.install_pointers(f"v{n}", n)
        assert len(ptrs) == n
        for p in ptrs:
            assert SP.get(p) == f"v{n}"
        SP.put(ptrs[n // 2], f"w{n}")
        for p in ptrs:
            assert SP.get(p) == f"w{n}"
    print("test_install_non_power_n: OK")


def test_put_shared_copychain_bf4() -> None:
    """Shared put across 16 copy-chain pointers after individual gets
    (the case that was broken in the HUB/LEAF design)."""
    _reset_sam()
    SP.set_branching_factor(4)
    ptrs = [SP(SP.new("x"))]
    for _ in range(15):
        ptrs.append(SP.copy(ptrs[-1]))
    for p in ptrs:
        assert SP.get(p) == "x"
    SP.put(ptrs[7], "y")
    for i, p in enumerate(ptrs):
        assert SP.get(p) == "y", f"ptr {i}"
    print("test_put_shared_copychain_bf4: OK")


def test_put_shared_hub_after_gets() -> None:
    """Shared put across 3+ installed pointers after each was get separately."""
    _reset_sam()
    SP.set_branching_factor(4)
    ptrs = SP.install_pointers("shared", 4)
    for p in ptrs:
        SP.get(p)
    SP.put(ptrs[0], "new")
    for i, p in enumerate(ptrs):
        assert SP.get(p) == "new", f"ptr {i}"
    print("test_put_shared_hub_after_gets: OK")


def test_bf2_parity() -> None:
    """BF=2 behaves like the binary multi-write pointer."""
    _reset_sam()
    SP.set_branching_factor(2)
    p1 = SP(SP.new("dog"))
    p2 = SP.copy(p1)
    assert SP.get(p1) == "dog"
    SP.put(p1, "wolf")
    assert SP.get(p2) == "wolf"
    assert SP.get(p1) == "wolf"
    print("test_bf2_parity: OK")


def test_is_single_reference() -> None:
    _reset_sam()
    SP.set_branching_factor(4)
    p1 = SP(SP.new("solo"))
    assert SP.is_single_reference(p1)
    p2 = SP.copy(p1)
    assert not SP.is_single_reference(p1)
    assert not SP.is_single_reference(p2)
    print("test_is_single_reference: OK")


def test_delete() -> None:
    _reset_sam()
    SP.set_branching_factor(4)
    ptrs = SP.install_pointers("v", 4)
    SP.delete(ptrs[1])
    assert ptrs[1].head is None
    for i in (0, 2, 3):
        assert SP.get(ptrs[i]) == "v", f"ptr {i} after delete"
    SP.put(ptrs[0], "w")
    for i in (0, 2, 3):
        assert SP.get(ptrs[i]) == "w", f"ptr {i} after put"
    print("test_delete: OK")


def test_stress_mixed() -> None:
    """Installed tree + extra copies + interleaved gets/puts."""
    _reset_sam()
    SP.set_branching_factor(4)
    ptrs = SP.install_pointers(0, 20)
    ptrs.append(SP.copy(ptrs[3]))
    ptrs.append(SP.copy(ptrs[-1]))
    for round_no in range(1, 4):
        SP.put(ptrs[round_no * 5 % len(ptrs)], round_no)
        for i, p in enumerate(ptrs):
            assert SP.get(p) == round_no, f"round {round_no}, ptr {i}"
    print("test_stress_mixed: OK")


# ---------------------------------------------------------------------------
# Shared benchmark helpers (used by all benchmark_* functions)
# ---------------------------------------------------------------------------

def _build_copy_chain(make_ptr, copy_fn, new_fn, n: int):
    ptrs = [make_ptr(new_fn("v"))]
    for _ in range(n - 1):
        ptrs.append(copy_fn(ptrs[-1]))
    return ptrs


def _delta(
    before: Tuple[int, int, int], after: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    return (after[0] - before[0], after[1] - before[1], after[2] - before[2])


def _fmt_row(vals: List[int]) -> str:
    return f"{min(vals):>3d} /{sum(vals)/len(vals):6.2f} /{max(vals):>4d} /{sum(vals):>5d}"


# depth is a per-get path length; a summed total is not meaningful, so
# report only min/avg/max for it.
def _fmt_depth(vals: List[int]) -> str:
    return f"{min(vals):>3d} /{sum(vals)/len(vals):6.2f} /{max(vals):>4d}"


def _summary_line(name: str, samples: List[Tuple[int, int, int, int]]) -> str:
    allocs = [s[0] for s in samples]
    reads = [s[1] for s in samples]
    writes = [s[2] for s in samples]
    depths = [s[3] for s in samples]
    atot, rtot, wtot = sum(allocs), sum(reads), sum(writes)
    return (
        f"  {name:<18}"
        f"{_fmt_depth(depths):<28}"
        f"{_fmt_row(allocs):<28}"
        f"{_fmt_row(reads):<28}"
        f"{_fmt_row(writes):<28}"
        f"{atot:>6d},{rtot:>5d},{wtot:>5d}"
    )


_BENCH_HEADER = (
    f"  {'config':<18}"
    f"{'depth min/avg/max':<28}"
    f"{'allocs min/avg/max/total':<28}"
    f"{'reads min/avg/max/total':<28}"
    f"{'writes min/avg/max/total':<28}"
    f"{'total (a,r,w)':>18}"
)


# One measured get sweep: returns (allocs, reads, writes, depth) per get.
# depth_getter maps the per-get read delta to a path depth so binary and
# r-ary share the same sweep logic.
def _measured_sweep(ptrs, get_fn, depth_getter):
    samples: List[Tuple[int, int, int, int]] = []
    for p in ptrs:
        a0, r0, w0 = get_global_counter()
        get_fn(p)
        a1, r1, w1 = get_global_counter()
        dr = r1 - r0
        samples.append((a1 - a0, dr, w1 - w0, depth_getter(dr)))
    return samples


# reads consumed by the get == path cells incl ROOT; depth excludes ROOT.
def _binary_depth(dr: int) -> int:
    return max(dr - 1, 0)


# exact path depth recorded by the r-ary deref (NODE cells head -> ROOT).
def _rary_depth(dr: int) -> int:
    return SP._last_depth


def _print_per_get_log(
    title: str,
    samples: List[Tuple[int, int, int, int]],
    labels: Optional[List[int]] = None,
) -> None:
    # labels lets callers report the actual pointer index gotten (e.g. random
    # priming); defaults to the sequential position in the sweep.
    tags = labels if labels is not None else list(range(len(samples)))
    label_word = "ptr" if labels is not None else "get"
    print(f"\n  {title} ({label_word}#: depth, allocs, reads, writes):")
    for tag, (a, r, w, d) in zip(tags, samples):
        print(f"    {label_word} {tag:>3d}: depth={d:>2d}  a={a:>2d}  r={r:>2d}  w={w:>3d}")


def benchmark_prime_then_get(
    num_pointers: int = 64,
    prime_sweeps: int = 2,
    bfs: Tuple[int, ...] = (4, 8, 16),
    log_each_get: bool = True,
) -> None:
    """
    Three-phase priming benchmark on a shared copy chain:

      Phase 1 (build)     : build a copy chain of `num_pointers` aliases.
      Phase 2 (prime)     : `prime_sweeps` full get sweeps over all aliases,
                            rebalancing the inverted pointer tree via splay.
      Phase 3 (benchmark) : one more get sweep over all aliases, snapshotting
                            (allocs, reads, writes, depth) per get.

    `depth` is the physical path length of the accessed pointer: the number of
    NODE cells read from the pointer head up to ROOT (ROOT excluded).  For the
    r-ary tree, deref consumes exactly `depth + 1` reads (the +1 is ROOT), so
    reads per get and depth track each other directly.  For the binary baseline
    depth is derived the same way: reads consumed by the get, minus the ROOT
    read.

    Compares the binary multi-write baseline against r-ary BF in `bfs`.

    Note: sam_reset() is NOT called between phases because it would wipe SAM
    cells and invalidate live pointer heads.
    """
    from smart_pointer_multi_write import SmartPointerMultiWrite

    # Each config returns (name, build_cost, prime_sweep_samples, bench_samples)
    # where prime_sweep_samples is a list (one entry per prime sweep) of per-get
    # sample lists, all measured the same way as the Phase 3 benchmark sweep.
    def run_config(name, make_ptr, copy_fn, new_fn, get_fn, depth_getter):
        t0 = get_global_counter()
        ptrs = _build_copy_chain(make_ptr, copy_fn, new_fn, num_pointers)
        build_cost = _delta(t0, get_global_counter())
        prime_sweep_samples = [
            _measured_sweep(ptrs, get_fn, depth_getter) for _ in range(prime_sweeps)
        ]
        bench_samples = _measured_sweep(ptrs, get_fn, depth_getter)
        return name, build_cost, prime_sweep_samples, bench_samples

    def run_binary():
        _reset_sam()
        return run_config(
            "binary chain",
            lambda h: SmartPointerMultiWrite(h),
            SmartPointerMultiWrite.copy,
            SmartPointerMultiWrite.new,
            SmartPointerMultiWrite.get,
            _binary_depth,
        )

    def run_rary(bf: int):
        _reset_sam()
        SP.set_branching_factor(bf)
        return run_config(
            f"rary BF={bf} chain",
            lambda h: SP(h),
            SP.copy,
            SP.new,
            SP.get,
            _rary_depth,
        )

    rows = [run_binary()]
    for bf in bfs:
        rows.append(run_rary(bf))

    gets = num_pointers
    print(
        f"\n=== benchmark_prime_then_get "
        f"(n={num_pointers}, prime_sweeps={prime_sweeps}, copy chain) ==="
    )

    # ---- Phase 1 (build) cost --------------------------------------------
    print(f"\n  Phase 1 (build) cost:")
    print(f"  {'config':<18}{'build (a,r,w)':>22}")
    for name, build, _, _ in rows:
        ba, br, bw = build
        print(f"  {name:<18}{ba:>8d},{br:>6d},{bw:>6d}")

    # ---- Phase 2 (prime) per-get log -------------------------------------
    if log_each_get:
        for name, _, prime_sweep_samples, _ in rows:
            for s, samples in enumerate(prime_sweep_samples, 1):
                _print_per_get_log(f"Phase 2 prime sweep {s} per-get log [{name}]", samples)

    # ---- Phase 2 (prime) summary, per sweep ------------------------------
    print(f"\n  Phase 2 (prime): per-get over {gets} gets, one row per sweep")
    print(_BENCH_HEADER)
    for name, _, prime_sweep_samples, _ in rows:
        for s, samples in enumerate(prime_sweep_samples, 1):
            print(_summary_line(f"{name} s{s}", samples))

    # ---- Phase 3 (benchmark) per-get log ---------------------------------
    if log_each_get:
        for name, _, _, samples in rows:
            _print_per_get_log(f"Phase 3 per-get log [{name}]", samples)

    # ---- Phase 3 (benchmark) summary -------------------------------------
    print(f"\n  Phase 3 (benchmark, after {prime_sweeps} prime sweeps): per-get over {gets} gets")
    print(_BENCH_HEADER)
    for name, _, _, samples in rows:
        print(_summary_line(name, samples))


def benchmark_random_prime_then_get(
    num_pointers: int = 64,
    prime_fraction: float = 0.25,
    bfs: Tuple[int, ...] = (4, 8, 16),
    log_each_get: bool = True,
    seed: int = 0,
) -> None:
    """
    Partial-priming benchmark on a shared copy chain: instead of full sweeps,
    the prime phase gets only a random subset of the aliases.

      Phase 1 (build)     : build a copy chain of `num_pointers` aliases.
      Phase 2 (prime)     : one get over a random `prime_fraction` of the
                            aliases (default 25%), sampled without
                            replacement; only these gets splay the tree.
      Phase 3 (benchmark) : one full get sweep over ALL aliases, snapshotting
                            (allocs, reads, writes, depth) per get.

    The same random indices (fixed by `seed`) are used for every config, so
    binary and all r-ary BFs prime exactly the same pointers.  Metrics have
    the same meaning as in benchmark_prime_then_get.

    Note: sam_reset() is NOT called between phases because it would wipe SAM
    cells and invalidate live pointer heads.
    """
    from smart_pointer_multi_write import SmartPointerMultiWrite

    if not 0.0 < prime_fraction <= 1.0:
        raise ValueError(f"prime_fraction must be in (0, 1], got {prime_fraction}")

    n_prime = max(1, round(num_pointers * prime_fraction))
    prime_indices = sorted(random.Random(seed).sample(range(num_pointers), n_prime))

    # Each config returns (name, build_cost, prime_samples, bench_samples).
    def run_config(name, make_ptr, copy_fn, new_fn, get_fn, depth_getter):
        t0 = get_global_counter()
        ptrs = _build_copy_chain(make_ptr, copy_fn, new_fn, num_pointers)
        build_cost = _delta(t0, get_global_counter())
        prime_samples = _measured_sweep(
            [ptrs[i] for i in prime_indices], get_fn, depth_getter
        )
        bench_samples = _measured_sweep(ptrs, get_fn, depth_getter)
        return name, build_cost, prime_samples, bench_samples

    def run_binary():
        _reset_sam()
        return run_config(
            "binary chain",
            lambda h: SmartPointerMultiWrite(h),
            SmartPointerMultiWrite.copy,
            SmartPointerMultiWrite.new,
            SmartPointerMultiWrite.get,
            _binary_depth,
        )

    def run_rary(bf: int):
        _reset_sam()
        SP.set_branching_factor(bf)
        return run_config(
            f"rary BF={bf} chain",
            lambda h: SP(h),
            SP.copy,
            SP.new,
            SP.get,
            _rary_depth,
        )

    rows = [run_binary()]
    for bf in bfs:
        rows.append(run_rary(bf))

    print(
        f"\n=== benchmark_random_prime_then_get "
        f"(n={num_pointers}, prime={n_prime} random gets "
        f"[{prime_fraction:.0%}], seed={seed}, copy chain) ==="
    )
    print(f"  primed pointer indices: {prime_indices}")

    # ---- Phase 1 (build) cost --------------------------------------------
    print(f"\n  Phase 1 (build) cost:")
    print(f"  {'config':<18}{'build (a,r,w)':>22}")
    for name, build, _, _ in rows:
        ba, br, bw = build
        print(f"  {name:<18}{ba:>8d},{br:>6d},{bw:>6d}")

    # ---- Phase 2 (random prime) per-get log ------------------------------
    if log_each_get:
        for name, _, prime_samples, _ in rows:
            _print_per_get_log(
                f"Phase 2 random-prime per-get log [{name}]",
                prime_samples,
                labels=prime_indices,
            )

    # ---- Phase 2 (random prime) summary ----------------------------------
    print(f"\n  Phase 2 (random prime): per-get over {n_prime} gets")
    print(_BENCH_HEADER)
    for name, _, prime_samples, _ in rows:
        print(_summary_line(name, prime_samples))

    # ---- Phase 3 (benchmark) per-get log ---------------------------------
    if log_each_get:
        for name, _, _, samples in rows:
            _print_per_get_log(f"Phase 3 per-get log [{name}]", samples)

    # ---- Phase 3 (benchmark) summary -------------------------------------
    print(
        f"\n  Phase 3 (benchmark, after {n_prime} random prime gets): "
        f"per-get over {num_pointers} gets"
    )
    print(_BENCH_HEADER)
    for name, _, _, samples in rows:
        print(_summary_line(name, samples))


def run_all_tests() -> None:
    # correctness tests
    test_sibling_blocks()
    test_copy_and_get_bf4()
    test_copy_expands_nonfull_group()
    test_copy_full_group_creates_lower_group()
    test_bulk_copy()
    test_copy_raw_address_bulk_contract()
    test_bulk_copy_existing_group()
    test_bulk_copy_reduces_operations()
    test_bulk_copy_validation()
    test_smart_pointer_bulk_copy_interface()
    test_other_smart_pointer_copy_contract_unchanged()
    test_delete_compacts_group()
    test_delete_contracts_unary_group()
    test_install_bf4_n64()
    test_install_non_power_n()
    test_put_shared_copychain_bf4()
    test_put_shared_hub_after_gets()
    test_bf2_parity()
    test_is_single_reference()
    test_delete()
    test_stress_mixed()
    test_even_non_power_branching_factors()
    print("\nAll tests passed.")

    # benchmarks (grouped by n; per-get log off for the bulk run)
    benchmark_prime_then_get(64, log_each_get=False)
    benchmark_random_prime_then_get(64, log_each_get=False)
    benchmark_prime_then_get(16, log_each_get=False)
    benchmark_random_prime_then_get(16, log_each_get=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="r-ary multi-write smart pointer: tests and priming benchmark."
    )
    sub = parser.add_subparsers(dest="command")

    bench = sub.add_parser(
        "benchmark_prime",
        help="run the three-phase priming benchmark (build / prime / benchmark)",
    )
    bench.add_argument("-n", "--num-pointers", type=int, default=64,
                       help="number of aliases in the copy chain (default: 64)")
    bench.add_argument("-s", "--sweeps", type=int, default=2,
                       help="number of priming sweeps before the benchmark (default: 2)")
    bench.add_argument("--bf", type=int, nargs="+", default=[4, 8, 16],
                       help="r-ary branching factors to compare (default: 4 8 16)")
    bench.add_argument("--log-each-get", action="store_true",
                       help="print per-get metrics for every phase")

    rand = sub.add_parser(
        "benchmark_random_prime",
        help="prime by getting a random fraction of the pointers, then benchmark",
    )
    rand.add_argument("-n", "--num-pointers", type=int, default=64,
                      help="number of aliases in the copy chain (default: 64)")
    rand.add_argument("-f", "--fraction", type=float, default=0.25,
                      help="fraction of pointers to prime with random gets (default: 0.25)")
    rand.add_argument("--seed", type=int, default=0,
                      help="RNG seed for the random pointer sample (default: 0)")
    rand.add_argument("--bf", type=int, nargs="+", default=[4, 8, 16],
                      help="r-ary branching factors to compare (default: 4 8 16)")
    rand.add_argument("--log-each-get", action="store_true",
                      help="print per-get metrics for every phase")

    args = parser.parse_args()
    if args.command == "benchmark_prime":
        benchmark_prime_then_get(
            args.num_pointers,
            prime_sweeps=args.sweeps,
            bfs=tuple(args.bf),
            log_each_get=args.log_each_get,
        )
    elif args.command == "benchmark_random_prime":
        benchmark_random_prime_then_get(
            args.num_pointers,
            prime_fraction=args.fraction,
            bfs=tuple(args.bf),
            log_each_get=args.log_each_get,
            seed=args.seed,
        )
    else:
        run_all_tests()
