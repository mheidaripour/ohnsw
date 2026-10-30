use std::collections::{BTreeMap, HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const RARY_STRUCTURE: &str = "SmartPointerMultiWriteRary";

#[derive(Clone, Copy)]
struct Block {
    address: i64,
    leaf: u64,
    version: u64,
    level: u8,
}

#[derive(Clone)]
struct ReplayState {
    tree: HashMap<u64, Vec<Block>>,
    stash: HashMap<i64, Block>,
    stash_by_leaf: BTreeMap<u64, HashSet<i64>>,
    address_leaves: HashMap<i64, u64>,
    current_versions: HashMap<i64, u64>,
    write_versions: HashMap<i64, u64>,
    evict_counter: u64,
    rng: StdRng,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct ReplaySnapshot {
    trace_index: usize,
    state: ReplayState,
}

#[derive(Clone, Copy)]
struct Config {
    tree_size: u64,
    tree_height: u32,
    bucket_capacity: usize,
    rary_read_evictions: usize,
}

impl ReplayState {
    fn new(seed: u64) -> Self {
        Self {
            tree: HashMap::new(),
            stash: HashMap::new(),
            stash_by_leaf: BTreeMap::new(),
            address_leaves: HashMap::new(),
            current_versions: HashMap::new(),
            write_versions: HashMap::new(),
            evict_counter: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn sample_leaf(&mut self, config: Config) -> u64 {
        self.rng.gen_range(0..config.tree_size)
    }

    fn path_indices(leaf: u64, config: Config) -> Vec<u64> {
        let mut path = Vec::with_capacity(config.tree_height as usize + 1);
        let mut bucket = 1_u64;
        path.push(bucket);
        for level in 0..config.tree_height {
            bucket = (bucket << 1) | ((leaf >> (config.tree_height - level - 1)) & 1);
            path.push(bucket);
        }
        path
    }

    fn leaf_prefix(leaf: u64, level: u32, config: Config) -> u64 {
        leaf >> (config.tree_height - level)
    }

    fn index_block(&mut self, block: Block) {
        self.stash_by_leaf
            .entry(block.leaf)
            .or_default()
            .insert(block.address);
    }

    fn unindex_block(&mut self, block: Block) {
        let remove_leaf = if let Some(addresses) = self.stash_by_leaf.get_mut(&block.leaf) {
            addresses.remove(&block.address);
            addresses.is_empty()
        } else {
            false
        };
        if remove_leaf {
            self.stash_by_leaf.remove(&block.leaf);
        }
    }

    fn add_to_stash(&mut self, block: Block) {
        if self.current_versions.get(&block.address) != Some(&block.version) {
            return;
        }
        let replace = match self.stash.get(&block.address) {
            None => true,
            Some(previous) => {
                block.version > previous.version
                    || (block.version == previous.version && block.level < previous.level)
            }
        };
        if replace {
            if let Some(previous) = self.stash.insert(block.address, block) {
                self.unindex_block(previous);
            }
            self.index_block(block);
        }
    }

    fn read_path(&mut self, leaf: u64, config: Config) {
        for (level, bucket) in Self::path_indices(leaf, config).into_iter().enumerate() {
            if let Some(blocks) = self.tree.remove(&bucket) {
                for mut block in blocks {
                    block.level = level as u8;
                    self.add_to_stash(block);
                }
            }
        }
    }

    fn remove_from_stash(&mut self, address: i64) {
        if let Some(block) = self.stash.remove(&address) {
            self.unindex_block(block);
        }
        self.current_versions.remove(&address);
    }

    fn candidates(&self, level: u32, prefix: u64, limit: usize, config: Config) -> Vec<Block> {
        if limit == 0 {
            return Vec::new();
        }
        let shift = config.tree_height - level;
        let start_leaf = prefix << shift;
        let end_leaf = (prefix + 1) << shift;
        let mut blocks = Vec::with_capacity(limit);
        for (_, addresses) in self.stash_by_leaf.range(start_leaf..end_leaf) {
            for address in addresses {
                if let Some(block) = self.stash.get(address) {
                    blocks.push(*block);
                    if blocks.len() == limit {
                        return blocks;
                    }
                }
            }
        }
        blocks
    }

    fn write_path(&mut self, leaf: u64, path: &[u64], max_blocks: usize, config: Config) {
        let mut remaining = Vec::new();
        let mut placed = 0;
        for level in (0..=config.tree_height).rev() {
            let budget = max_blocks.saturating_sub(placed);
            if budget == 0 {
                break;
            }
            let capacity = ((level as usize + 1) * config.bucket_capacity).min(budget);
            if remaining.len() < capacity {
                let needed = capacity - remaining.len();
                let (prefix_level, prefix) = if level == config.tree_height {
                    (config.tree_height, leaf)
                } else {
                    (level + 1, Self::leaf_prefix(leaf, level + 1, config) ^ 1)
                };
                remaining.extend(self.candidates(prefix_level, prefix, needed, config));
            }
            let select_count = config.bucket_capacity.min(budget);
            let selected: Vec<Block> = remaining.drain(..select_count.min(remaining.len())).collect();
            for block in &selected {
                self.remove_from_stash(block.address);
            }
            placed += selected.len();
            if !selected.is_empty() {
                self.tree.insert(path[level as usize], selected);
            }
        }
    }

    fn bit_reverse(mut value: u64, bits: u32) -> u64 {
        let mut result = 0;
        for _ in 0..bits {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }
        result
    }

    fn evict(&mut self, max_blocks: Option<usize>, config: Config) {
        let leaf = Self::bit_reverse(self.evict_counter % config.tree_size, config.tree_height);
        self.evict_counter += 1;
        let path = Self::path_indices(leaf, config);
        for (level, bucket) in path.iter().enumerate() {
            if let Some(blocks) = self.tree.remove(bucket) {
                for mut block in blocks {
                    block.level = level as u8;
                    self.add_to_stash(block);
                }
            }
        }
        let limit = max_blocks.unwrap_or((config.tree_height as usize + 1) * config.bucket_capacity);
        self.write_path(leaf, &path, limit, config);
    }

    fn alloc(&mut self, address: i64, config: Config) {
        if !self.current_versions.contains_key(&address) {
            if !self.address_leaves.contains_key(&address) {
                let leaf = self.sample_leaf(config);
                self.address_leaves.insert(address, leaf);
            }
        }
    }

    fn write(&mut self, address: i64, structure: Option<&str>, config: Config) {
        if !self.address_leaves.contains_key(&address) {
            self.alloc(address, config);
        }
        if structure != Some(RARY_STRUCTURE) {
            let leaf = self.sample_leaf(config);
            self.read_path(leaf, config);
        }
        if let Some(block) = self.stash.remove(&address) {
            self.unindex_block(block);
        }
        let version = self.write_versions.get(&address).copied().unwrap_or(0) + 1;
        self.write_versions.insert(address, version);
        self.current_versions.insert(address, version);
        let block = Block { address, leaf: self.address_leaves[&address], version, level: 0 };
        self.add_to_stash(block);
        if structure != Some(RARY_STRUCTURE) {
            self.evict(None, config);
        }
    }

    fn read(&mut self, address: i64, remove: bool, structure: Option<&str>, config: Config) {
        let leaf = self.address_leaves.get(&address).copied().unwrap_or_else(|| self.sample_leaf(config));
        self.read_path(leaf, config);
        if remove {
            self.remove_from_stash(address);
        }
        if structure == Some(RARY_STRUCTURE) {
            for _ in 0..config.rary_read_evictions {
                self.evict(None, config);
            }
        } else {
            self.evict(None, config);
        }
    }

    fn evict_many(&mut self, count: i64, config: Config) -> PyResult<()> {
        if count <= 0 {
            return Err(PyValueError::new_err("evict count must be positive"));
        }
        for _ in 0..count {
            self.evict(None, config);
        }
        Ok(())
    }
}

fn parse_config(tree_size: u64, bucket_capacity: usize, rary_evictions: Option<usize>) -> PyResult<Config> {
    if tree_size == 0 || !tree_size.is_power_of_two() {
        return Err(PyValueError::new_err("tree_size must be a positive power of two"));
    }
    if bucket_capacity == 0 {
        return Err(PyValueError::new_err("bucket_capacity must be positive"));
    }
    let rary_read_evictions = rary_evictions.unwrap_or(1);
    if rary_read_evictions == 0 {
        return Err(PyValueError::new_err("rary_read_evictions must be positive"));
    }
    Ok(Config {
        tree_size,
        tree_height: tree_size.trailing_zeros(),
        bucket_capacity,
        rary_read_evictions,
    })
}

/// Replay a PathORAM trace; supports explicit evict trace entries.
#[pyfunction]
#[pyo3(signature = (trace, phase_start=0, tree_size=1, bucket_capacity=4, rary_read_evictions=None, seed=1, snapshot=None))]
fn replay(
    py: Python<'_>,
    trace: Vec<(String, i64, bool, Option<String>)>,
    phase_start: usize,
    tree_size: u64,
    bucket_capacity: usize,
    rary_read_evictions: Option<usize>,
    seed: u64,
    snapshot: Option<PyRef<'_, ReplaySnapshot>>,
) -> PyResult<(usize, usize, usize, usize, Py<ReplaySnapshot>)> {
    let config = parse_config(tree_size, bucket_capacity, rary_read_evictions)?;
    let (mut state, start) = match snapshot {
        Some(snapshot) => (snapshot.state.clone(), snapshot.trace_index),
        None => (ReplayState::new(seed), 0),
    };
    let mut max_stash_size = 0;
    if start <= phase_start {
        max_stash_size = state.stash.len();
    }
    for (index, (op, address, remove, structure)) in trace.iter().enumerate().skip(start) {
        match op.as_str() {
            "alloc" => state.alloc(*address, config),
            "write" => state.write(*address, structure.as_deref(), config),
            "read" => state.read(*address, *remove, structure.as_deref(), config),
            "evict" => state.evict_many(*address, config)?,
            _ => return Err(PyValueError::new_err(format!("unknown replay operation: {op}"))),
        }
        if index >= phase_start {
            max_stash_size = max_stash_size.max(state.stash.len());
        }
    }
    let final_stash_size = state.stash.len();
    let snapshot = Py::new(py, ReplaySnapshot { trace_index: trace.len(), state })?;
    Ok((max_stash_size, final_stash_size, trace.len().saturating_sub(phase_start), trace.len(), snapshot))
}

#[pymodule]
fn _path_oram_replay(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ReplaySnapshot>()?;
    module.add_function(wrap_pyfunction!(replay, module)?)?;
    Ok(())
}
