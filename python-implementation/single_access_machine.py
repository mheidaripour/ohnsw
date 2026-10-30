import bisect
import random
from sys import maxsize
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Any

try:
    from _path_oram_replay import replay as native_path_oram_replay
    _native_signature = (
        getattr(native_path_oram_replay, "__text_signature__", None)
        or getattr(native_path_oram_replay, "__doc__", "")
    )
    _native_supports_explicit_evictions = "explicit evict" in _native_signature
    if "rary_read_evictions" not in _native_signature:
        native_path_oram_replay = None
except ImportError:
    native_path_oram_replay = None
    _native_supports_explicit_evictions = False

"""
SingleAccessMachine - manages address allocating, reading, and writing
"""

PathORAMTraceEntry = Tuple[str, int, bool, str | None]
RARY_POINTER_STRUCTURE = "SmartPointerMultiWriteRary"
_REVERSE_BYTE = tuple(
    int(f"{value:08b}"[::-1], 2)
    for value in range(256)
)

@dataclass(slots=True)
class PathORAMBlock:
    """One logical value stored in the PathORAM replay."""
    address: int
    leaf: int
    version: int
    level: int = 0


@dataclass
class PathORAMReplaySnapshot:
    """Replay state saved at a trace index so later phases can resume there."""
    trace_index: int
    tree: Dict[int, List[PathORAMBlock]]
    stash: Dict[int, PathORAMBlock]
    address_leaves: Dict[int, int]
    current_versions: Dict[int, int]
    write_versions: Dict[int, int]
    evict_counter: int
    rng_state: Any


class PathORAMStashReplay:
    """
    Replay a SAM trace through Path OSAM/PathORAM-style storage.

    The replay uses a fixed binary tree with a caller-provided number of leaves,
    deterministic reverse-lexicographic eviction, and stale-version merging for
    repeated writes to the same logical SAM address.
    """
    def __init__(
            self,
            trace: List[PathORAMTraceEntry],
            phase_start: int = 0,
            tree_size: int = 1,
            bucket_capacity: int = 4,
            rary_read_evictions: int | None = None,
            rary_read_eviction_blocks: int | None = None,
            seed: int = 1,
            snapshot: PathORAMReplaySnapshot | None = None,
        ) -> None:
        if tree_size <= 0 or tree_size & (tree_size - 1) != 0:
            raise ValueError("PathORAM tree size must be a positive power of two")
        if bucket_capacity <= 0:
            raise ValueError("PathORAM bucket capacity must be positive")
        if rary_read_evictions is None:
            rary_read_evictions = rary_read_eviction_blocks
        if (
            rary_read_evictions is not None
            and rary_read_evictions <= 0
        ):
            raise ValueError("r-ary read eviction count must be positive")

        self.trace = trace
        self.phase_start = phase_start
        self.bucket_capacity = bucket_capacity
        self.rary_read_evictions = (
            rary_read_evictions
            if rary_read_evictions is not None
            else 1
        )
        self.rng = random.Random(seed)
        self.num_leaves = tree_size
        self.tree_height = self.num_leaves.bit_length() - 1
        self.tree: Dict[int, List[PathORAMBlock]] = dict()
        self.stash: Dict[int, PathORAMBlock] = dict()
        self.stash_by_leaf: Dict[int, Set[int]] = dict()
        self.stash_leaves: List[int] = []
        self.address_leaves: Dict[int, int] = dict()
        self.current_versions: Dict[int, int] = dict()
        self.write_versions: Dict[int, int] = dict()
        self.evict_counter = 0
        self.max_stash_size = 0
        self._monitor_stash = False
        self.replay_start = 0

        if snapshot is not None:
            self.tree = self._copy_tree(snapshot.tree)
            self.stash = self._copy_stash(snapshot.stash)
            self._rebuild_stash_index()
            self.address_leaves = dict(snapshot.address_leaves)
            self.current_versions = dict(snapshot.current_versions)
            self.write_versions = dict(snapshot.write_versions)
            self.evict_counter = snapshot.evict_counter
            self.rng.setstate(snapshot.rng_state)
            self.replay_start = snapshot.trace_index

    @staticmethod
    def _copy_block(block: PathORAMBlock) -> PathORAMBlock:
        return PathORAMBlock(
            address=block.address,
            leaf=block.leaf,
            version=block.version,
            level=block.level,
        )

    @classmethod
    def _copy_blocks(cls, blocks: List[PathORAMBlock]) -> List[PathORAMBlock]:
        return [cls._copy_block(block) for block in blocks]

    @classmethod
    def _copy_stash(
            cls,
            stash: Dict[int, PathORAMBlock],
        ) -> Dict[int, PathORAMBlock]:
        return {
            address: cls._copy_block(block)
            for address, block in stash.items()
        }

    @classmethod
    def _copy_tree(
            cls,
            tree: Dict[int, List[PathORAMBlock]],
        ) -> Dict[int, List[PathORAMBlock]]:
        return {
            bucket: cls._copy_blocks(blocks)
            for bucket, blocks in tree.items()
        }

    def snapshot(self, trace_index: int) -> PathORAMReplaySnapshot:
        """Return a copy of the current replay state."""
        return PathORAMReplaySnapshot(
            trace_index=trace_index,
            tree=self._copy_tree(self.tree),
            stash=self._copy_stash(self.stash),
            address_leaves=dict(self.address_leaves),
            current_versions=dict(self.current_versions),
            write_versions=dict(self.write_versions),
            evict_counter=self.evict_counter,
            rng_state=self.rng.getstate(),
        )

    def _sample_leaf(self) -> int:
        return self.rng.randrange(self.num_leaves)

    def _leaf_prefix(self, leaf: int, level: int) -> int:
        return leaf >> (self.tree_height - level)

    def _bucket_index(self, leaf: int, level: int) -> int:
        prefix = self._leaf_prefix(leaf, level)
        return (1 << level) + prefix

    def _path_indices(self, leaf: int) -> List[int]:
        path = [1]
        bucket = 1
        for level in range(self.tree_height):
            bucket = (bucket << 1) | ((leaf >> (self.tree_height - level - 1)) & 1)
            path.append(bucket)
        return path

    @staticmethod
    def _bit_reverse(value: int, bits: int) -> int:
        out = 0
        while bits >= 8:
            out = (out << 8) | _REVERSE_BYTE[value & 0xFF]
            value >>= 8
            bits -= 8
        if bits:
            out = (out << bits) | (_REVERSE_BYTE[value & 0xFF] >> (8 - bits))
        return out

    def _update_max_stash_size(self) -> None:
        if self._monitor_stash:
            self.max_stash_size = max(self.max_stash_size, len(self.stash))

    def _is_live(self, address: int) -> bool:
        return address in self.current_versions

    def _index_block(self, block: PathORAMBlock) -> None:
        addresses = self.stash_by_leaf.get(block.leaf)
        if addresses is None:
            self.stash_by_leaf[block.leaf] = {block.address}
            bisect.insort(self.stash_leaves, block.leaf)
        else:
            addresses.add(block.address)

    def _unindex_block(self, block: PathORAMBlock) -> None:
        addresses = self.stash_by_leaf.get(block.leaf)
        if addresses is None:
            return
        addresses.discard(block.address)
        if not addresses:
            del self.stash_by_leaf[block.leaf]
            index = bisect.bisect_left(self.stash_leaves, block.leaf)
            if index < len(self.stash_leaves) and self.stash_leaves[index] == block.leaf:
                self.stash_leaves.pop(index)

    def _rebuild_stash_index(self) -> None:
        self.stash_by_leaf.clear()
        self.stash_leaves.clear()
        for block in self.stash.values():
            self._index_block(block)

    def _prefix_leaf_range(self, level: int, prefix: int) -> Tuple[int, int]:
        shift = self.tree_height - level
        start = prefix << shift
        end = (prefix + 1) << shift
        return start, end

    def _candidate_blocks_from_prefix(
            self,
            level: int,
            prefix: int,
            limit: int,
        ) -> List[PathORAMBlock]:
        if limit <= 0:
            return []

        start_leaf, end_leaf = self._prefix_leaf_range(level, prefix)
        start = bisect.bisect_left(self.stash_leaves, start_leaf)
        end = bisect.bisect_left(self.stash_leaves, end_leaf, start)

        blocks: List[PathORAMBlock] = []
        for leaf in self.stash_leaves[start:end]:
            for address in self.stash_by_leaf.get(leaf, set()):
                block = self.stash.get(address)
                if block is None:
                    continue
                blocks.append(block)
                if len(blocks) == limit:
                    return blocks
        return blocks

    def _add_to_stash(self, block: PathORAMBlock) -> None:
        current_version = self.current_versions.get(block.address)
        if current_version is None or block.version != current_version:
            return

        prev = self.stash.get(block.address)
        if (
            prev is None
            or block.version > prev.version
            or (
                block.version == prev.version
                and block.level < prev.level
            )
        ):
            if prev is not None:
                self._unindex_block(prev)
            self.stash[block.address] = block
            self._index_block(block)
            self._update_max_stash_size()

    def _read_path(self, leaf: int) -> None:
        for level, bucket in enumerate(self._path_indices(leaf)):
            for block in self.tree.pop(bucket, []):
                block.level = level
                self._add_to_stash(block)

    def _remove_from_stash(self, address: int) -> None:
        block = self.stash.pop(address, None)
        if block is not None:
            self._unindex_block(block)
        self.current_versions.pop(address, None)

    def _write_path_from_stash(
            self,
            leaf: int,
            path: List[int],
            max_blocks: int | None = None,
        ) -> None:
        """
        Greedily write one eviction path from leaf to root.

        Each stash block is classified once by the deepest bucket on this path
        that can hold it. Processing levels bottom-up then lets unplaced blocks
        flow toward ancestors without rescanning the whole stash at each level.
        """
        remaining: List[PathORAMBlock] = []
        placed = 0
        if max_blocks is None:
            max_blocks = (self.tree_height + 1) * self.bucket_capacity

        for level in range(self.tree_height, -1, -1):
            block_budget_left = max_blocks - placed
            if block_budget_left <= 0:
                break

            capacity_left = min(
                (level + 1) * self.bucket_capacity,
                block_budget_left,
            )
            if len(remaining) < capacity_left:
                needed = capacity_left - len(remaining)
                if level == self.tree_height:
                    prefix_level = self.tree_height
                    prefix = leaf
                else:
                    prefix_level = level + 1
                    prefix = self._leaf_prefix(leaf, prefix_level) ^ 1

                remaining.extend(
                    self._candidate_blocks_from_prefix(
                        prefix_level,
                        prefix,
                        needed,
                    )
                )

            select_count = min(self.bucket_capacity, block_budget_left)
            selected = remaining[:select_count]
            remaining = remaining[select_count:]
            for block in selected:
                block.level = level
                self._remove_from_stash(block.address)
            placed += len(selected)

            bucket = path[level]
            if selected:
                self.tree[bucket] = selected

            self._update_max_stash_size()

    def _evict(self, max_blocks: int | None = None) -> None:
        leaf = self._bit_reverse(
            self.evict_counter % self.num_leaves,
            self.tree_height,
        )
        self.evict_counter += 1
        path = self._path_indices(leaf)

        for level, bucket in enumerate(path):
            for block in self.tree.pop(bucket, []):
                block.level = level
                self._add_to_stash(block)
        self._write_path_from_stash(leaf, path, max_blocks)

    def _alloc(self, address: int) -> None:
        if not self._is_live(address):
            self.address_leaves[address] = self._sample_leaf()

    def _write(self, address: int, structure: str | None) -> None:
        if address not in self.address_leaves:
            self._alloc(address)

        if structure != RARY_POINTER_STRUCTURE:
            # Path OSAM write first performs a dummy ReadAndRm(Alloc()).
            self._read_path(self._sample_leaf())

        old_block = self.stash.pop(address, None)
        if old_block is not None:
            self._unindex_block(old_block)
        version = self.write_versions.get(address, 0) + 1
        self.write_versions[address] = version
        self.current_versions[address] = version
        self._add_to_stash(
            PathORAMBlock(
                address=address,
                leaf=self.address_leaves[address],
                version=version,
                level=0,
            )
        )
        self._update_max_stash_size()
        if structure != RARY_POINTER_STRUCTURE:
            self._evict()

    def _read(self, address: int, remove: bool, structure: str | None) -> None:
        leaf = self.address_leaves.get(address)
        if leaf is None:
            leaf = self._sample_leaf()
        self._read_path(leaf)
        if remove:
            self._remove_from_stash(address)
            self._update_max_stash_size()
        if structure == RARY_POINTER_STRUCTURE:
            for _ in range(self.rary_read_evictions):
                self._evict()
        else:
            self._evict()

    def _evict_many(self, count: int) -> None:
        if count <= 0:
            raise ValueError("PathORAM eviction count must be positive")
        for _ in range(count):
            self._evict()

    def replay(self) -> Dict[str, int]:
        for i in range(self.replay_start, len(self.trace)):
            entry = self.trace[i]
            if len(entry) == 3:
                op, address, remove = entry
                structure = None
            else:
                op, address, remove, structure = entry
            replayed = i - self.replay_start + 1
            if replayed % 100000 == 0:
                print(
                    "PathORAM stash replay progress: "
                    f"{replayed}/{len(self.trace) - self.replay_start} ops "
                    f"(trace_ops={i + 1}/{len(self.trace)}, "
                    f"phase_ops={max(i + 1 - self.phase_start, 0)})",
                    flush=True,
                )

            if i >= self.phase_start and not self._monitor_stash:
                self._monitor_stash = True
                self._update_max_stash_size()

            match op:
                case "alloc":
                    self._alloc(address)
                case "write":
                    self._write(address, structure)
                case "read":
                    self._read(address, remove, structure)
                case "evict":
                    self._evict_many(address)
                case _:
                    raise ValueError(f"Unknown PathORAM trace operation {op}")

        return {
            "max_stash_size": self.max_stash_size,
            "final_stash_size": len(self.stash),
            "bucket_capacity": self.bucket_capacity,
            "rary_read_evictions": self.rary_read_evictions,
            "num_leaves": self.num_leaves,
            "tree_height": self.tree_height,
            "trace_operations": len(self.trace),
            "phase_operations": max(len(self.trace) - self.phase_start, 0),
        }


class SingleAccessMachine:
    def __init__(self, max_reads: int = 1, max_writes: int = maxsize) -> None:
        """Initialize OSAM stats"""
        # counts SAM operations
        self.alloc_counter = 0 # also controls the addresses being given out
        self.read_counter = 0
        self.write_counter = 0

        # writes count - reads count
        self.write_batches = 0

        # the largest number of writes made at one point
        self.max_write_batches = 0

        # basic address storage
        self.position_map: Dict[int, List[int]] = dict() # key: address, value: [current reads, current writes]
        self.storage: Dict[int, Any] = dict() # key: address, value: value

        # reuse previously allocated addresses no longer being used during multi-read
        self.available_addresses: List[int] = []

        # storage limits
        self.max_reads = max_reads 
        self.max_writes = max_writes
       
        # plaintext storage - simulate storage that does not use OSAM
        self.plaintext_counter = -1
        self.plaintext_storage: Dict[int, Any] = dict() # key: address, value: value

        # reuse previously allocated plaintext addresses no longer being used
        self.plaintext_available_addresses: List[int] = []

        # flag for printing traces
        self.traces = False

        # when True, retired single-read addresses (returned via
        # return_available_address) have their bookkeeping dropped.
        # self.free_retired = True

        # map SAM-based structure to its number of SAM operations
        self.structures: Set = set()
        self.structure_allocs: Dict[str, int] = dict()
        self.structure_reads: Dict[str, int] = dict()
        self.structure_writes: Dict[str, int] = dict()

        # Logical operation trace used to replay PathORAM stash occupancy.
        self.path_oram_trace: List[PathORAMTraceEntry] = []
        self.path_oram_phase_start = 0
        self.path_oram_tree_size: int | None = None
        self.path_oram_bucket_capacity = 4
        self.path_oram_rary_read_evictions: int | None = None
        self.path_oram_seed = 1
        self.path_oram_replay_snapshots: Dict[int, PathORAMReplaySnapshot] = dict()
        self.native_path_oram_replay_snapshots: Dict[int, Any] = dict()
        
    def multiple_reads_enabled(self) -> bool:
        """Output boolean if multiple reads are enabled"""
        return self.max_reads > 1

    def alloc(self, use_osam: bool = True, structure: str | None = None) -> int:
        """Generate an fresh address to be written to and read from"""
        # simulate plaintext / non-oblivious storage if OSAM is disabled
        if not use_osam:
            # reuse previously assigned address that is not in current use
            if self.plaintext_available_addresses:
                address = self.plaintext_available_addresses.pop()

            # assign new plaintext address
            else:
                address = self.plaintext_counter
                self.plaintext_counter -= 1
            
            return address

        # recycle used addresses in multi-read settings
        elif self.multiple_reads_enabled() and self.available_addresses:
            address = self.available_addresses.pop()
            self.path_oram_trace.append(("alloc", address, False, structure))
            return address

        # assign new address
        address = self.alloc_counter
        self.position_map[address] = [0, 0] # current reads and writes
        self.alloc_counter += 1
        self.path_oram_trace.append(("alloc", address, False, structure))

        # track allocs by structure
        if structure is not None:
            self.structures.add(structure)
            if structure in self.structure_allocs:
                self.structure_allocs[structure] += 1
            else:
                self.structure_allocs[structure] = 1
        
        # print traces
        if self.traces: 
            print("alloc counter:", self.alloc_counter)
    
        return address

    def write(self, address: int, value: Any, structure: str | None = None):
        """Writes value to address in storage"""
        # a negative address indicates an plaintext operation
        if address < 0:
            self.plaintext_storage[address] = value
            return

        # ensure address exists
        try:
            stats = self.position_map[address]
        except:
            raise KeyError(f"No address {address}!")
        
        # ensure write limit is not exceeded
        if stats[1] >= self.max_writes:
            raise RuntimeError(f"Address {address} cannot be written to more than {self.max_writes} time(s)!")
        else:
            stats[1] += 1
       
        # store value at address
        self.storage[address] = value
        self.path_oram_trace.append(("write", address, False, structure))

        # increment counter
        self.write_counter += 1

        # track writes by structure
        if structure is not None:
            self.structures.add(structure)
            if structure in self.structure_writes:
                self.structure_writes[structure] += 1
            else:
                self.structure_writes[structure] = 1

        # print traces
        if self.traces:
            print("write counter:", self.write_counter)

        # track max write batches
        self.write_batches += 1
        if self.write_batches > self.max_write_batches:
            self.max_write_batches = self.write_batches

    def read(self, address: int , structure: str | None = None) -> Any:
        """Get value associated with address"""
        # a negative address indicates an plaintext operation
        if address < 0:
            try:
                value = self.plaintext_storage[address]
            except:
                raise KeyError(f"Address {address} not found in plaintext storage!")
            return value

        # ensure address exists
        try:
            stats = self.position_map[address]
        except:
            raise KeyError(f"No address {address}!")
            
        # ensure address is not written to more than the max
        if stats[0] >= self.max_reads:
            raise RuntimeError(f"Address {address} cannot be read from more than {self.max_reads} time(s)!")
        else:
            stats[0] += 1

        value = None
        storage = None
        
        # retrieve value from storage
        try:
            value = self.storage[address]
            storage = self.storage
        except:
            assert address in self.position_map

        # only delete entry from storage and position_map if it cannot be read from again
        remove = stats[0] >= self.max_reads
        if remove:
            self.position_map.pop(address, None)
            if storage is not None:
                del storage[address]
        self.path_oram_trace.append(("read", address, remove, structure))

        # if storage is not None and stats[0] >= self.max_reads:
        #     del storage[address]

        # increment counter
        self.read_counter += 1

        # track reads by structure
        if structure is not None:
            self.structures.add(structure)
            if structure in self.structure_reads:
                self.structure_reads[structure] += 1
            else:
                self.structure_reads[structure] = 1

        # print traces
        if self.traces:
            print("read counter:", self.read_counter)

        # decrement write batches
        if self.write_batches > 0:
            self.write_batches -= 1

        return value

    def path_oram_evict(self, count: int) -> None:
        """Append PathORAM eviction-only operations to the replay trace."""
        if count <= 0:
            raise ValueError("PathORAM eviction count must be positive")
        self.path_oram_trace.append(("evict", count, False, None))

    def return_available_address(self, address: int | None) -> None:
        """
        Return previously allocated address not in use to be allocated again later
        Trusts the caller the given address is not currently in use
        """
        if address is not None:
            # a negative address indicates an plaintext operation
            if address < 0:
                if address not in self.plaintext_available_addresses:
                    self.plaintext_available_addresses.append(address)

            # can only return addresses in the multi-read setting (simulate recursive pointer)
            elif self.multiple_reads_enabled():
                if address not in self.available_addresses:
                    self.available_addresses.append(address)

            # single-read setting: the address is retired and will not be
            # reused, so drop its bookkeeping so position_map/storage track the
            # live working set instead of every address ever allocated
            # elif self.free_retired:
            #     self.position_map.pop(address, None)
            #     self.storage.pop(address, None)

    def reset(self) -> None:
        """Reset all SAM counters / stats / storage"""
        self.alloc_counter = 0
        self.read_counter = 0
        self.write_counter = 0
        self.position_map.clear()
        self.storage.clear()
        self.available_addresses.clear()
        self.plaintext_counter = -1
        self.plaintext_storage.clear()
        self.plaintext_available_addresses.clear()
        self.structure_allocs.clear()
        self.structure_reads.clear()
        self.structure_writes.clear()
        self.structures.clear()
        self.write_batches = 0
        self.max_write_batches = 0
        self.path_oram_trace.clear()
        self.path_oram_phase_start = 0
        self.path_oram_replay_snapshots.clear()
        self.native_path_oram_replay_snapshots.clear()

    def set_max_reads(self, max_reads: int) -> None:
        """Set max reads"""
        assert max_reads > 0
        self.max_reads = max_reads
        
    def set_max_writes(self, max_writes: int) -> None:
        """Set max writes"""
        assert max_writes > 0
        self.max_writes = max_writes

    def get_global_counter(self) -> Tuple[int, int, int]:
        """Output SAM stats (allocs, reads, writes)"""
        return self.alloc_counter, self.read_counter, self.write_counter

    def print_stats_by_structure(self, flush: bool = True) -> None:
        """Print each SAM-based structure to its number of SAM operations"""
        if not self.structure_allocs:
            return 
            
        for structure in sorted(list(self.structures)):
            try:
                a = self.structure_allocs[structure]
            except:
                a = 0

            try:
                r = self.structure_reads[structure]
            except:
                r = 0

            try:
                w = self.structure_writes[structure]
            except:
                w = 0

            print(f"Structure stats: {structure} ({a}, {r}, {w})", flush=flush)

    def print_max_write_batches(self, flush: bool = True) -> None:    
        print(f"\nMax write batches: {self.max_write_batches}", flush=flush)

    def reset_write_batches(self) -> None:    
        """Reset write batches outside of reset()"""
        self.write_batches = 0
        self.max_write_batches = 0
        self.path_oram_phase_start = len(self.path_oram_trace)

    def _path_oram_snapshot_for_phase(self) -> PathORAMReplaySnapshot | None:
        """Return the best cached replay state at or before the phase start."""
        candidates = [
            trace_index
            for trace_index in self.path_oram_replay_snapshots
            if trace_index <= self.path_oram_phase_start
        ]
        if not candidates:
            return None
        return self.path_oram_replay_snapshots[max(candidates)]

    def get_path_oram_stash_stats(self) -> Dict[str, int]:
        """Compute PathORAM stash stats for the current trace phase."""
        if self.path_oram_tree_size is None:
            raise RuntimeError("PathORAM tree size is not configured")
        has_explicit_evictions = any(
            entry[0] == "evict"
            for entry in self.path_oram_trace
        )
        if (
            native_path_oram_replay is not None
            and (
                not has_explicit_evictions
                or _native_supports_explicit_evictions
            )
        ):
            snapshot = self.native_path_oram_replay_snapshots.get(
                self.path_oram_phase_start
            )
            (
                max_stash_size,
                final_stash_size,
                phase_operations,
                trace_operations,
                snapshot,
            ) = native_path_oram_replay(
                self.path_oram_trace,
                self.path_oram_phase_start,
                self.path_oram_tree_size,
                self.path_oram_bucket_capacity,
                self.path_oram_rary_read_evictions,
                self.path_oram_seed,
                snapshot,
            )
            self.native_path_oram_replay_snapshots[len(self.path_oram_trace)] = snapshot
            return {
                "max_stash_size": max_stash_size,
                "final_stash_size": final_stash_size,
                "bucket_capacity": self.path_oram_bucket_capacity,
                "rary_read_evictions": self.path_oram_rary_read_evictions or 1,
                "num_leaves": self.path_oram_tree_size,
                "tree_height": self.path_oram_tree_size.bit_length() - 1,
                "trace_operations": trace_operations,
                "phase_operations": phase_operations,
            }
        snapshot = self._path_oram_snapshot_for_phase()
        replay = PathORAMStashReplay(
            trace=self.path_oram_trace,
            phase_start=self.path_oram_phase_start,
            tree_size=self.path_oram_tree_size,
            bucket_capacity=self.path_oram_bucket_capacity,
            rary_read_evictions=self.path_oram_rary_read_evictions,
            seed=self.path_oram_seed,
            snapshot=snapshot,
        )
        stats = replay.replay()
        self.path_oram_replay_snapshots[len(self.path_oram_trace)] = (
            replay.snapshot(len(self.path_oram_trace))
        )
        return stats

    def set_path_oram_tree_size(self, tree_size: int | None) -> None:
        """Set the fixed PathORAM leaf count used by stash replay."""
        if tree_size is not None and (
            tree_size <= 0 or tree_size & (tree_size - 1) != 0
        ):
            raise ValueError("PathORAM tree size must be a positive power of two")
        if self.path_oram_tree_size != tree_size:
            self.path_oram_replay_snapshots.clear()
            self.native_path_oram_replay_snapshots.clear()
            self.path_oram_tree_size = tree_size

    def set_path_oram_bucket_capacity(self, bucket_capacity: int) -> None:
        """Set the bucket capacity used by stash replay."""
        if bucket_capacity <= 0:
            raise ValueError("PathORAM bucket capacity must be positive")
        if self.path_oram_bucket_capacity != bucket_capacity:
            self.path_oram_replay_snapshots.clear()
            self.native_path_oram_replay_snapshots.clear()
            self.path_oram_bucket_capacity = bucket_capacity

    def set_path_oram_rary_read_evictions(
            self,
            evictions: int | None,
        ) -> None:
        """Set how many full eviction paths a r-ary read performs."""
        if evictions is not None and evictions <= 0:
            raise ValueError("r-ary read eviction count must be positive")
        if self.path_oram_rary_read_evictions != evictions:
            self.path_oram_replay_snapshots.clear()
            self.native_path_oram_replay_snapshots.clear()
            self.path_oram_rary_read_evictions = evictions

    def set_path_oram_rary_read_eviction_blocks(
            self,
            blocks: int | None,
        ) -> None:
        """Compatibility alias for set_path_oram_rary_read_evictions."""
        self.set_path_oram_rary_read_evictions(blocks)

    def path_oram_stash_enabled(self) -> bool:
        """Return whether PathORAM stash replay is configured."""
        return self.path_oram_tree_size is not None

    def print_path_oram_stash_stats(self, flush: bool = True) -> None:
        """Print maximum stash size from a PathORAM replay."""
        if not self.path_oram_stash_enabled():
            return
        stats = self.get_path_oram_stash_stats()
        print(
            "PathORAM stash stats: "
            f"max={stats['max_stash_size']}, "
            f"final={stats['final_stash_size']}, "
            f"bucket_capacity={stats['bucket_capacity']}, "
            f"rary_read_evictions={stats['rary_read_evictions']}, "
            f"leaves={stats['num_leaves']}, "
            f"height={stats['tree_height']}, "
            f"phase_ops={stats['phase_operations']}",
            flush=flush,
        )

    def clear_available_addresses(self) -> None:
        """Clear saved addresses outside of reset()"""
        self.available_addresses.clear()
        self.plaintext_available_addresses.clear()


"""
Global SAM with global functions to be imported
"""

# global SAM
sam = SingleAccessMachine()

def multiple_reads_enabled() -> int:
    return sam.multiple_reads_enabled()
    
def get_global_counter() -> Tuple[int, int, int]:
    return sam.get_global_counter()

def print_stats_by_structure(flush: bool = True) -> None:
    sam.print_stats_by_structure(flush)

def print_max_write_batches(flush: bool = True) -> None: 
    sam.print_max_write_batches(flush)

def get_path_oram_stash_stats() -> Dict[str, int]:
    return sam.get_path_oram_stash_stats()

def print_path_oram_stash_stats(flush: bool = True) -> None:
    sam.print_path_oram_stash_stats(flush)

def set_path_oram_tree_size(tree_size: int | None) -> None:
    sam.set_path_oram_tree_size(tree_size)

def set_path_oram_bucket_capacity(bucket_capacity: int) -> None:
    sam.set_path_oram_bucket_capacity(bucket_capacity)

def set_path_oram_rary_read_eviction_blocks(blocks: int | None) -> None:
    sam.set_path_oram_rary_read_eviction_blocks(blocks)

def set_path_oram_rary_read_evictions(evictions: int | None) -> None:
    sam.set_path_oram_rary_read_evictions(evictions)

def path_oram_stash_enabled() -> bool:
    return sam.path_oram_stash_enabled()

def path_oram_evict(count: int) -> None:
    sam.path_oram_evict(count)

def reset_write_batches() -> None:
    sam.reset_write_batches()

def clear_available_addresses() -> None:
    sam.clear_available_addresses()

def alloc(use_osam: bool = True, structure: str | None = None) -> int:
    address = sam.alloc(use_osam, structure)
    if sam.traces:
        print("alloc output: ", address)
    
    return address

def return_available_address(address: int | None) -> None:
    sam.return_available_address(address)
    
def write(address: int, value: Any, structure: str | None) -> None:
    if sam.traces:
        print("write input: ", address, value)

    sam.write(address, value, structure)

def read(address: int, structure: str | None = None) -> Any:
    value = sam.read(address, structure)
    if sam.traces:
        print("read input: ", address, "  value:", value)

    return value

def reset() -> None:
    if sam.traces:
        print("sam reset")

    sam.reset()

def set_max_reads(max_reads: int) -> None:
    sam.set_max_reads(max_reads)

def set_max_writes(max_writes: int) -> None:
    sam.set_max_writes(max_writes)

def set_traces(on: bool) -> None:
    sam.traces = on
