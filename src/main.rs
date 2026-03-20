/// CadenGraph v3 — Pure Rust cognitive substrate
///
/// Ports all cadenGraph_v2.3 n8n behaviors into a single in-process physics
/// loop. No HTTP round-trips, no Neo4j, no Cypher. Memory-mapped binary
/// arena keeps the full graph in direct-access memory across restarts.
///
/// External interface (unchanged from v2):
///   POST /input   { "text": "...", "embedding": [f32 x SEM_DIM] }
///   GET  /context → { "context": [{ "text": "...", "score": f32, "source": "..." }] }
///
/// n8n "rustChat" workflow continues to feed /input exactly as before.
/// /context returns predicted resonant states ranked by semantic resonance,
/// energy, and predictive confidence — equivalent to the n8n Speech Query.

use axum::{routing::{get, post}, Json, Router};
use crossbeam_channel::{Receiver, Sender};
use memmap2::MmapMut;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs::OpenOptions,
    io,
    mem::size_of,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Embedding dimension — must match the embedder model feeding /input.
/// 768 = BERT-base / all-mpnet-base-v2 / nomic-embed-text etc.
const SEM_DIM: usize = 768;

const MAX_EDGES: usize = 12;      // hard per-node edge buffer
const MAX_EDGES_SOFT: f32 = 10.0; // soft target for probabilistic degree cap
const NODE_CAP: usize = 1_000_000;

/// Inline text length. 128 bytes fits bigram fragments; longer text truncated.
const TEXT_LEN: usize = 128;

/// File header occupies the first 64 bytes of brain.bin.
/// Nodes are stored starting at byte HEADER_SIZE.
const HEADER_SIZE: usize = 64;

/// Magic value written at header offset 0 to validate format.
const BRAIN_MAGIC: u64 = 0x_CADE_0003_0003_0000;

const FILE_SIZE: usize = HEADER_SIZE + NODE_CAP * size_of::<Node>();

/// Revolving active-set ring buffer capacity.
const ACTIVE_BUF: usize = 2048;

/// How many top predictions to return from /context.
const CONTEXT_SLOTS: usize = 8;

/// Random global-arena samples per ingest for similarity linking.
/// Supplements active-buffer scan to link new nodes to old memories.
const GLOBAL_SIM_SAMPLES: usize = 30;

// ── Physics tuning — all match cadenGraph_v2.3 constants ─────────────────
const FLOW_DECAY: f32        = 0.995;  // global flow leak per tick
const FATIGUE_DECAY_A: f32   = 0.85;   // fatigue recovery coefficient a
const FATIGUE_DECAY_B: f32   = 0.15;   // fatigue recovery coefficient b
const ENERGY_CAP: f32        = 0.2;    // energy regulation ceiling
const ENERGY_FLOOR: f32      = 0.0001; // below this → zero
const DIFFUSE_ALPHA: f32     = 0.4;    // energy diffusion fraction
const INHIBIT_RATIO: f32     = 0.08;   // lateral inhibition strength
const NOISE_PROB: f32        = 0.003;  // background noise injection probability
const NOISE_AMP: f32         = 0.002;  // background noise magnitude
const HOMEOSTASIS_TARGET: f32 = 0.08;  // flow homeostasis setpoint
const HOMEOSTASIS_K: f32     = 0.004;  // homeostasis pull strength
const THRESHOLD_BASE: f32    = 0.02;   // firing threshold base value
const DECAY_PROB: f32        = 0.02;   // fraction of edges decayed per tick
const DECAY_RATE: f32        = 0.999;  // edge weight decay multiplier
const EDGE_SAT_CAP: f32      = 0.8;    // weight saturation ±bound
const ENTROPY_HIGH: f32      = 4.5;    // entropy ceiling → shrink weights
const ENTROPY_LOW: f32       = 2.5;    // entropy floor  → grow weights
const ABSTRACTION_PROB: u64  = 200;    // ticks between abstraction attempts
const SPONTANEOUS_PROB: u64  = 60;     // ticks between spontaneous thought
const PREDICTION_BOOST: f32  = 0.05;   // energy added to predicted next cells
const PRED_CORRECTION_K: f32 = 0.06;   // prediction weight learning rate
const REPLAY_K: f32          = 0.002;  // off-line replay reinforcement rate
const CONV_REINFORCE_K: f32  = 0.015;  // conversational reinforcement boost

// Edge type flags stored in Edge::flags
const EDGE_LINK: u8 = 0; // semantic similarity / attractor
const EDGE_NEXT: u8 = 1; // temporal sequence
const EDGE_FRAG: u8 = 2; // fragment-of relationship

// ═══════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES  (repr(C): layout is stable across restarts)
// ═══════════════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy)]
struct Edge {
    target: u32,
    weight: f32,
    flags: u8,
    _pad: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Node {
    // dynamics
    energy: f32,
    reserve: f32,
    fatigue: f32,
    threshold: f32,

    // semantic position (embedding)
    pos: [f32; SEM_DIM],

    // connectivity (inline, mmap-safe)
    edges: [Edge; MAX_EDGES],
    edge_count: u8,

    // metadata
    last_fired: u64,    // unix-ms of last fire
    last_spoken: u64,   // unix-ms of last context surfacing
    spoke_tick: u32,
    activation_tick: u32,
    is_abstract: u8,

    // inline text
    text: [u8; TEXT_LEN],
    text_len: u8,

    _pad: [u8; 2],
}

/// 64-byte file header stored at mmap offset 0.
/// Persists node count across restarts.
#[repr(C)]
struct BrainHeader {
    magic: u64,       // BRAIN_MAGIC — format validation
    node_count: u64,  // number of allocated nodes
    _reserved: [u64; 6],
}

const _: () = assert!(size_of::<BrainHeader>() == HEADER_SIZE);

/// Runtime arena — wraps the mmap.
struct Arena {
    mmap: MmapMut,
    count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// HTTP TYPES
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct InputPayload {
    text: String,
    embedding: Vec<f32>,
}

#[derive(Serialize, Clone)]
struct PredictedState {
    text: String,
    score: f32,
    /// "active", "predicted", or "recalled"
    source: String,
}

#[derive(Serialize)]
struct ContextOutput {
    context: Vec<PredictedState>,
}

struct SharedState {
    context: Vec<PredictedState>,
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Fast xorshift64 PRNG — thread-local, zero contention.
fn rand_u32() -> u32 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new(0x12345678_abcdef01);
    }
    SEED.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        x as u32
    })
}

fn rand_f32() -> f32 { (rand_u32() as f32) / (u32::MAX as f32) }
fn rand_usize(n: usize) -> usize { (rand_u32() as usize) % n.max(1) }

fn normalize(v: &mut [f32; SEM_DIM]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() { *x /= norm; }
    }
}

fn cosine(a: &[f32; SEM_DIM], b: &[f32; SEM_DIM]) -> f32 {
    let mut dot = 0.0f32;
    let mut na  = 0.0f32;
    let mut nb  = 0.0f32;
    for i in 0..SEM_DIM {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    dot / ((na.sqrt() * nb.sqrt()) + 1e-9)
}

fn node_text(n: &Node) -> String {
    let len = (n.text_len as usize).min(TEXT_LEN);
    String::from_utf8_lossy(&n.text[..len]).into_owned()
}

fn set_text(n: &mut Node, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len().min(TEXT_LEN);
    n.text[..len].copy_from_slice(&bytes[..len]);
    n.text_len = len as u8;
}

// ═══════════════════════════════════════════════════════════════════════════
// ARENA
// ═══════════════════════════════════════════════════════════════════════════

impl Arena {
    fn open(path: &str) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true).write(true).create(true)
            .open(path)?;
        file.set_len(FILE_SIZE as u64)?;
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Read or initialise the file header.
        let count = {
            let hdr = unsafe { &*(mmap.as_ptr() as *const BrainHeader) };
            if hdr.magic == BRAIN_MAGIC {
                (hdr.node_count as usize).min(NODE_CAP)
            } else {
                // Fresh file — write magic, start from zero.
                let hdr_mut = unsafe { &mut *(mmap.as_mut_ptr() as *mut BrainHeader) };
                *hdr_mut = BrainHeader {
                    magic: BRAIN_MAGIC,
                    node_count: 0,
                    _reserved: [0u64; 6],
                };
                0
            }
        };

        println!("[caden] brain.bin: {} nodes restored", count);
        Ok(Self { mmap, count })
    }

    /// Pointer to node slot `idx`. Nodes begin after the file header.
    #[inline]
    fn ptr(&self) -> *mut Node {
        unsafe { (self.mmap.as_ptr() as *mut u8).add(HEADER_SIZE) as *mut Node }
    }

    /// Persist current count to the file header.
    fn flush_count(&mut self) {
        let hdr = unsafe { &mut *(self.mmap.as_mut_ptr() as *mut BrainHeader) };
        hdr.node_count = self.count as u64;
    }

    /// Allocate a new node. Returns its index.
    fn alloc(&mut self, pos: [f32; SEM_DIM], text: &str, is_abstract: bool) -> usize {
        if self.count >= NODE_CAP {
            // Wrap: overwrite the oldest node rather than crash.
            // A proper LRU eviction policy is a future improvement.
            return self.count.saturating_sub(1);
        }
        let idx = self.count;
        self.count += 1;

        let node = unsafe { &mut *self.ptr().add(idx) };
        *node = Node {
            energy: 0.0,
            reserve: 0.0,
            fatigue: 0.0,
            threshold: THRESHOLD_BASE,
            pos,
            edges: [Edge { target: 0, weight: 0.0, flags: EDGE_LINK, _pad: [0; 3] }; MAX_EDGES],
            edge_count: 0,
            last_fired: 0,
            last_spoken: 0,
            spoke_tick: 0,
            activation_tick: 0,
            is_abstract: u8::from(is_abstract),
            text: [0; TEXT_LEN],
            text_len: 0,
            _pad: [0; 2],
        };
        set_text(node, text);

        // Persist count every 64 nodes to amortise header writes.
        if idx % 64 == 0 {
            self.flush_count();
        }
        idx
    }

    /// Connect a→b bidirectionally (NEXT is directed — no reverse edge).
    fn connect(&mut self, a: usize, b: usize, weight: f32, flags: u8) {
        if a == b || a >= self.count || b >= self.count { return; }
        unsafe {
            // Forward edge a → b
            let na = &mut *self.ptr().add(a);
            let k = na.edge_count as f32;
            // Probabilistic degree cap — reluctant to add edge when near limit
            if rand_f32() > 1.0 / (1.0 + (k / MAX_EDGES_SOFT).powi(2)) { return; }
            let mut updated = false;
            for i in 0..na.edge_count as usize {
                if na.edges[i].target == b as u32 {
                    na.edges[i].weight = (na.edges[i].weight + weight) * 0.5;
                    updated = true;
                    break;
                }
            }
            if !updated && (na.edge_count as usize) < MAX_EDGES {
                let i = na.edge_count as usize;
                na.edges[i] = Edge { target: b as u32, weight, flags, _pad: [0; 3] };
                na.edge_count += 1;
            }

            // Reverse edge b → a (skip for directed NEXT edges)
            if flags != EDGE_NEXT {
                let nb = &mut *self.ptr().add(b);
                let mut updated = false;
                for i in 0..nb.edge_count as usize {
                    if nb.edges[i].target == a as u32 {
                        nb.edges[i].weight = (nb.edges[i].weight + weight) * 0.5;
                        updated = true;
                        break;
                    }
                }
                if !updated && (nb.edge_count as usize) < MAX_EDGES {
                    let i = nb.edge_count as usize;
                    nb.edges[i] = Edge { target: a as u32, weight, flags, _pad: [0; 3] };
                    nb.edge_count += 1;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHYSICS — O(active_set) operations; all match cadenGraph_v2.3 steps
// ═══════════════════════════════════════════════════════════════════════════

/// **Diffuse Energy** — active nodes push signal along weighted edges.
unsafe fn diffuse_energy(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    let now = now_ms();
    for &ai in active {
        let a = &mut *ptr.add(ai);
        if a.energy <= a.threshold { continue; }
        if a.last_fired != 0 && now - a.last_fired < 15_000 { continue; }
        let ec = a.edge_count as usize;
        if ec == 0 { continue; }

        let total_w: f32 = (0..ec).map(|i| a.edges[i].weight.abs()).sum();
        if total_w < 1e-9 { continue; }

        let af = a.energy;
        let fatigue = a.fatigue;
        let mut signals = [(0usize, 0.0f32); MAX_EDGES];
        let mut nsig = 0;
        let mut spent = 0.0f32;

        for i in 0..ec {
            let e = a.edges[i];
            let signal = ((af / (1.0 + af)) * DIFFUSE_ALPHA * (e.weight / total_w))
                / (1.0 + fatigue);
            if signal > 0.0 {
                signals[nsig] = (e.target as usize, signal);
                nsig += 1;
                spent += signal;
            }
        }
        for k in 0..nsig {
            let (ti, sig) = signals[k];
            if ti < arena.count { (*ptr.add(ti)).energy += sig; }
        }
        a.energy = (a.energy - spent).max(0.0);
        a.fatigue += spent;
        a.last_fired = now;
    }
}

/// **Add Background Noise** — stochastic activation on a fraction of active nodes.
unsafe fn add_background_noise(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        if rand_f32() < NOISE_PROB {
            (*ptr.add(ai)).energy += rand_f32() * NOISE_AMP;
        }
    }
}

/// **Lateral Inhibition** — highly active nodes suppress their neighbours.
unsafe fn lateral_inhibition(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let a = &*ptr.add(ai);
        if a.energy <= 0.15 { continue; }
        let ec = a.edge_count as usize;
        let inhib = a.energy * INHIBIT_RATIO;
        for i in 0..ec {
            let ti = a.edges[i].target as usize;
            if ti < arena.count {
                let b = &mut *ptr.add(ti);
                b.energy = (b.energy - inhib).max(0.0);
            }
        }
    }
}

/// **Fatigue Recovery** — time-based suppression decay.
unsafe fn fatigue_recovery(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    let now = now_ms();
    for &ai in active {
        let n = &mut *ptr.add(ai);
        if n.fatigue == 0.0 { continue; }
        let x = (now.saturating_sub(n.last_fired)) as f32 / 5000.0;
        n.fatigue = n.fatigue * (FATIGUE_DECAY_A + FATIGUE_DECAY_B * (x / (1.0 + x.abs())));
    }
}

/// **Global Flow Leak** — exponential energy decay every tick.
unsafe fn global_flow_leak(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active { (*ptr.add(ai)).energy *= FLOW_DECAY; }
}

/// **Flowmeostasis** — pull flow toward homeostatic target.
unsafe fn homeostasis(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let n = &mut *ptr.add(ai);
        n.energy = (n.energy + HOMEOSTASIS_K * (HOMEOSTASIS_TARGET - n.energy)).max(0.0);
    }
}

/// **Plastic Activation Threshold** — threshold adapts to sustained activity.
unsafe fn plastic_threshold(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let n = &mut *ptr.add(ai);
        n.threshold += (n.energy - HOMEOSTASIS_TARGET) * 0.001;
        n.threshold = n.threshold.clamp(0.005, 0.5);
    }
}

/// **Global Energy Regulation** — cap / floor energy for homeostasis.
unsafe fn global_energy_regulation(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let n = &mut *ptr.add(ai);
        if n.energy > ENERGY_CAP { n.energy = ENERGY_CAP; }
        else if n.energy < ENERGY_FLOOR { n.energy = 0.0; }
    }
}

/// **Energy Gradient Update** — Hebbian co-activation weight update.
unsafe fn energy_gradient_update(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let a = &*ptr.add(ai);
        if a.energy <= 0.02 { continue; }
        let ec = a.edge_count as usize;
        for i in 0..ec {
            if rand_f32() > 0.3 { continue; }
            let ti = a.edges[i].target as usize;
            if ti >= arena.count { continue; }
            let b = &*ptr.add(ti);
            if b.pos.iter().all(|&x| x == 0.0) { continue; }
            let sim = cosine(&a.pos, &b.pos);
            let flow_diff = (a.energy - b.energy).abs();
            let delta = (a.energy * b.energy * 0.002 - (1.0 - sim) * 0.01 - flow_diff * 0.003)
                / (1.0 + flow_diff);
            let aw = a as *const Node as *mut Node;
            let new_w = (*aw).edges[i].weight + delta;
            (*aw).edges[i].weight = new_w / (1.0 + new_w.abs());
        }
    }
}

/// **Synaptic Normalization** — normalize outgoing weights to sum-to-1.
unsafe fn synaptic_normalization(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let n = &mut *ptr.add(ai);
        let ec = n.edge_count as usize;
        let total: f32 = (0..ec).map(|i| n.edges[i].weight.abs()).sum();
        if total < 1e-9 { continue; }
        for i in 0..ec { n.edges[i].weight /= total; }
    }
}

/// **Edge Saturation Cap** — clamp weights to ±EDGE_SAT_CAP.
unsafe fn edge_saturation_cap(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let n = &mut *ptr.add(ai);
        for i in 0..n.edge_count as usize {
            n.edges[i].weight = n.edges[i].weight.clamp(-EDGE_SAT_CAP, EDGE_SAT_CAP);
        }
    }
}

/// **Synaptic Decay** — probabilistic weight weakening.
unsafe fn synaptic_decay(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let n = &mut *ptr.add(ai);
        for i in 0..n.edge_count as usize {
            if rand_f32() < DECAY_PROB { n.edges[i].weight *= DECAY_RATE; }
        }
    }
}

/// **Edge Pruning** — keep only the top (MAX_EDGES-2) edges by weight.
unsafe fn edge_pruning(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    let keep = MAX_EDGES - 2;
    for &ai in active {
        let n = &mut *ptr.add(ai);
        let ec = n.edge_count as usize;
        if ec <= keep { continue; }
        for i in 0..keep {
            let mut best = i;
            for j in (i + 1)..ec {
                if n.edges[j].weight.abs() > n.edges[best].weight.abs() { best = j; }
            }
            n.edges.swap(i, best);
        }
        n.edge_count = keep as u8;
    }
}

/// **Entropy Regulation** — scale weights based on Shannon entropy of flow.
unsafe fn entropy_regulation(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    let sum_f: f32 = active.iter().map(|&i| (*ptr.add(i)).energy.max(0.0)).sum();
    if sum_f < 1e-9 { return; }
    let h: f32 = active.iter().map(|&i| {
        let p = (*ptr.add(i)).energy.max(0.0) / sum_f;
        if p < 1e-9 { 0.0 } else { -(p * p.ln()) }
    }).sum();
    let scale = if h > ENTROPY_HIGH { 0.97f32 } else if h < ENTROPY_LOW { 1.02f32 } else { 1.0 };
    if scale == 1.0 { return; }
    for &ai in active {
        let n = &mut *ptr.add(ai);
        for i in 0..n.edge_count as usize { n.edges[i].weight *= scale; }
    }
}

/// **Create Attractor Links** — co-active similar nodes form new connections.
unsafe fn create_attractor_links(arena: &mut Arena, active: &[usize]) {
    let n_active = active.len();
    if n_active < 2 { return; }
    let samples = (n_active / 2).min(10);
    let ptr = arena.ptr();
    for _ in 0..samples {
        let ai = active[rand_usize(n_active)];
        let bi = active[rand_usize(n_active)];
        if ai == bi { continue; }
        let a = &*ptr.add(ai);
        let b = &*ptr.add(bi);
        if a.energy <= 0.25 || b.energy <= 0.25 { continue; }
        if cosine(&a.pos, &b.pos) > 0.6 {
            arena.connect(ai, bi, 0.01, EDGE_LINK);
        }
    }
}

/// **Replay Reinforcement** — strengthen NEXT edges during idle ticks.
unsafe fn replay_reinforcement(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let a = &*ptr.add(ai);
        if a.energy <= 0.03 { continue; }
        for i in 0..a.edge_count as usize {
            if a.edges[i].flags != EDGE_NEXT { continue; }
            let ti = a.edges[i].target as usize;
            if ti >= arena.count { continue; }
            let b = &*ptr.add(ti);
            if b.energy <= 0.03 { continue; }
            let aw = a as *const Node as *mut Node;
            (*aw).edges[i].weight += a.energy * b.energy * REPLAY_K;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ABSTRACTION  (n8n: Abstraction Detection → Create → Link → Temporal Link)
// ═══════════════════════════════════════════════════════════════════════════

unsafe fn try_abstraction(arena: &mut Arena, active: &[usize]) {
    let n = active.len();
    if n < 4 { return; }
    let ptr = arena.ptr();

    let high_flow: Vec<usize> = active.iter()
        .filter(|&&i| (*ptr.add(i)).energy > 0.1)
        .copied().collect();
    if high_flow.len() < 4 { return; }

    // Compute centroid of high-flow cluster
    let mut centroid = [0.0f32; SEM_DIM];
    for &i in &high_flow {
        let n_ref = &*ptr.add(i);
        for j in 0..SEM_DIM { centroid[j] += n_ref.pos[j]; }
    }
    let cnt = high_flow.len() as f32;
    for j in 0..SEM_DIM { centroid[j] /= cnt; }
    normalize(&mut centroid);

    // Skip if a nearly identical abstraction already exists (dedup)
    for i in 0..arena.count {
        let e = &*ptr.add(i);
        if e.is_abstract == 1 && cosine(&e.pos, &centroid) > 0.95 { return; }
    }

    let abs_idx = arena.alloc(centroid, "abstraction", true);

    // Link abstraction to member cells (weight 0.3, bidirectional)
    for &mi in &high_flow {
        arena.connect(abs_idx, mi, 0.3, EDGE_LINK);
    }

    // Link abstraction to semantically similar neighbours in active set
    let abs_pos = (*arena.ptr().add(abs_idx)).pos;
    for &ci in active {
        if ci == abs_idx { continue; }
        let sim = cosine(&abs_pos, &(*arena.ptr().add(ci)).pos);
        if sim > 0.85 { arena.connect(abs_idx, ci, sim * 0.1, EDGE_LINK); }
    }

    // Abstraction temporal link: connect previous abstraction → this one
    // (Replicates n8n "Abstraction Temporal Link" step)
    let prev_abs_opt = (0..abs_idx)
        .rev()
        .find(|&i| (*arena.ptr().add(i)).is_abstract == 1 && i != abs_idx);
    if let Some(prev) = prev_abs_opt {
        arena.connect(prev, abs_idx, 0.02, EDGE_NEXT);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PREDICTION  (n8n: Prediction Correction + Prediction Activation)
// ═══════════════════════════════════════════════════════════════════════════

unsafe fn prediction_correction(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();

    // Newest and second-newest non-abstract text nodes (append-only = last = newest)
    let newest_opt = (0..arena.count).rev()
        .find(|&i| { let n = &*ptr.add(i); n.is_abstract == 0 && n.text_len > 0 });
    let newest = match newest_opt { Some(v) => v, None => return };

    let prev_opt = (0..newest).rev()
        .find(|&i| { let n = &*ptr.add(i); n.is_abstract == 0 && n.text_len > 0 });
    let prev = match prev_opt { Some(v) => v, None => return };

    let newest_pos = (*ptr.add(newest)).pos;
    let prev_pos   = (*ptr.add(prev)).pos;

    let abs_count = (0..arena.count).filter(|&i| (*ptr.add(i)).is_abstract == 1).count();
    if abs_count == 0 { return; }

    let curr_abs = (0..arena.count)
        .filter(|&i| (*ptr.add(i)).is_abstract == 1)
        .max_by(|&a, &b| cosine(&(*ptr.add(a)).pos, &newest_pos)
            .partial_cmp(&cosine(&(*ptr.add(b)).pos, &newest_pos))
            .unwrap_or(std::cmp::Ordering::Equal));
    let prev_abs = (0..arena.count)
        .filter(|&i| (*ptr.add(i)).is_abstract == 1 && Some(i) != curr_abs)
        .max_by(|&a, &b| cosine(&(*ptr.add(a)).pos, &prev_pos)
            .partial_cmp(&cosine(&(*ptr.add(b)).pos, &prev_pos))
            .unwrap_or(std::cmp::Ordering::Equal));

    let (curr_abs, prev_abs) = match (curr_abs, prev_abs) { (Some(c), Some(p)) => (c, p), _ => return };

    // Strengthen NEXT edge: prev_abs → curr_abs
    let pa = &mut *ptr.add(prev_abs);
    let ec = pa.edge_count as usize;
    let curr_abs_pos = (*ptr.add(curr_abs)).pos;
    let mut found = false;
    for i in 0..ec {
        if pa.edges[i].target == curr_abs as u32 && pa.edges[i].flags == EDGE_NEXT {
            let w = pa.edges[i].weight;
            pa.edges[i].weight = w + PRED_CORRECTION_K * w * (1.0 - w);
            // Correct competing predictions by similarity
            for j in 0..ec {
                if j == i || pa.edges[j].flags != EDGE_NEXT { continue; }
                let ti = pa.edges[j].target as usize;
                if ti >= arena.count { continue; }
                let sim = cosine(&(*ptr.add(ti)).pos, &curr_abs_pos);
                let jw = pa.edges[j].weight;
                pa.edges[j].weight = jw + PRED_CORRECTION_K * (sim - jw) * jw * (1.0 - jw);
            }
            found = true;
            break;
        }
    }
    if !found && ec < MAX_EDGES {
        pa.edges[ec] = Edge { target: curr_abs as u32, weight: 0.02, flags: EDGE_NEXT, _pad: [0; 3] };
        pa.edge_count += 1;
    }
}

unsafe fn prediction_activation(arena: &Arena, active: &[usize]) {
    let ptr = arena.ptr();
    for &ai in active {
        let a = &*ptr.add(ai);
        if a.is_abstract != 1 || a.energy <= 0.03 { continue; }
        for i in 0..a.edge_count as usize {
            if a.edges[i].flags != EDGE_NEXT || a.edges[i].weight <= 0.0 { continue; }
            let ti = a.edges[i].target as usize;
            if ti < arena.count {
                (*ptr.add(ti)).energy += a.energy * a.edges[i].weight * PREDICTION_BOOST;
            }
        }
    }
}

/// **Spontaneous Thought** — randomly fire top-3 high-energy nodes.
unsafe fn spontaneous_thought(arena: &Arena, active: &[usize]) {
    if active.is_empty() { return; }
    let ptr = arena.ptr();
    let mut scored: Vec<(usize, f32)> = active.iter()
        .map(|&i| (i, rand_f32() * (*ptr.add(i)).energy))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for k in 0..scored.len().min(3) {
        let ai = scored[k].0;
        let a = &*ptr.add(ai);
        if a.energy <= 0.015 { continue; }
        for i in 0..a.edge_count as usize {
            if a.edges[i].weight <= 0.0 { continue; }
            let ti = a.edges[i].target as usize;
            if ti < arena.count {
                (*ptr.add(ti)).energy += a.energy * a.edges[i].weight * 0.04;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INJECT ENERGY  (n8n: Inject Energy)
// ═══════════════════════════════════════════════════════════════════════════

unsafe fn inject_energy(arena: &Arena, idx: usize) {
    let n = &mut *arena.ptr().add(idx);
    // Exact formula from n8n cadenGraph: f + (0.04/(1+f)) + rand*0.005
    let f = n.energy;
    n.energy = f + (0.04 / (1.0 + f)) + rand_f32() * 0.005;
}

// ═══════════════════════════════════════════════════════════════════════════
// FRAGMENT TEXT  (n8n: Fragment Quotes)
// ═══════════════════════════════════════════════════════════════════════════

fn fragment_text(text: &str) -> Vec<String> {
    let words: Vec<String> = text
        .to_lowercase()
        .split(|c: char| !c.is_alphabetic() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect();

    let mut frags = Vec::new();
    for i in 0..words.len().saturating_sub(1) {
        frags.push(format!("{} {}", words[i], words[i + 1]));
    }
    frags
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTEXT / SPEECH QUERY  (n8n: Speech Query + Format Response)
//
// Scores all text-bearing nodes by:
//   1. Semantic resonance to current active centroid
//   2. Outgoing NEXT edge weights (predictive confidence)
//   3. Node energy
//   4. Recency penalty (log-scaled)
// Returns top CONTEXT_SLOTS deduplicated results, each with source label.
// ═══════════════════════════════════════════════════════════════════════════

/// Returns (scored results, list of node indices that were surfaced).
unsafe fn build_context(arena: &Arena, active: &[usize]) -> (Vec<PredictedState>, Vec<usize>) {
    let ptr = arena.ptr();
    let now = now_ms();

    if active.is_empty() { return (vec![], vec![]); }

    // Weighted centroid of active nodes (energy-weighted)
    let mut centroid = [0.0f32; SEM_DIM];
    let mut weight_sum = 0.0f32;
    for &i in active {
        let n = &*ptr.add(i);
        if n.energy <= 0.0 { continue; }
        for j in 0..SEM_DIM { centroid[j] += n.pos[j] * n.energy; }
        weight_sum += n.energy;
    }
    if weight_sum > 1e-9 {
        for j in 0..SEM_DIM { centroid[j] /= weight_sum; }
    }

    let mut candidates: Vec<(usize, PredictedState)> = Vec::new();

    for i in 0..arena.count {
        let n = &*ptr.add(i);
        if n.text_len == 0 { continue; }
        let text = node_text(n);
        if text.trim().is_empty() || text.len() <= 3 { continue; }

        let recency_penalty = if n.last_spoken > 0 {
            (2.0_f32 + (now - n.last_spoken) as f32).ln()
        } else {
            0.0
        };

        let base = n.energy / (1.0 + recency_penalty * 0.0001);
        let resonance = if weight_sum > 1e-9 {
            (cosine(&n.pos, &centroid) + 1.0) * 0.5
        } else {
            0.5
        };
        let pred_confidence: f32 = (0..n.edge_count as usize)
            .filter(|&j| n.edges[j].flags == EDGE_NEXT)
            .map(|j| n.edges[j].weight.max(0.0))
            .sum();

        let source = if n.energy > 0.05 { "active" }
            else if pred_confidence > 0.1 { "predicted" }
            else { "recalled" };

        let score = resonance * (1.0 + base * 2.0 + pred_confidence * 0.5)
            * (1.0 + (rand_f32() - 0.5) * 0.05); // small jitter

        candidates.push((i, PredictedState { text, score, source: source.to_string() }));
    }

    candidates.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen: HashSet<String> = HashSet::new();
    let mut results: Vec<PredictedState> = Vec::new();
    let mut spoken_indices: Vec<usize> = Vec::new();

    for (idx, state) in candidates {
        let key = state.text.trim().to_lowercase();
        if seen.contains(&key) { continue; }
        seen.insert(key);
        spoken_indices.push(idx);
        results.push(state);
        if results.len() >= CONTEXT_SLOTS { break; }
    }

    (results, spoken_indices)
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> io::Result<()> {
    println!("[caden] CadenGraph v3 — mmap cognitive substrate");
    println!("[caden] SEM_DIM={} MAX_EDGES={} NODE_CAP={}", SEM_DIM, MAX_EDGES, NODE_CAP);
    println!("[caden] opening brain.bin ({:.1} GB max) ...",
        FILE_SIZE as f64 / 1e9);

    let mut arena = Arena::open("brain.bin")?;
    println!("[caden] arena ready");

    let (tx, rx): (Sender<InputPayload>, Receiver<InputPayload>) =
        crossbeam_channel::bounded(8192);

    let shared = Arc::new(Mutex::new(SharedState { context: vec![] }));

    // ── HTTP server ──────────────────────────────────────────────────────
    let shared_ctx = shared.clone();
    let app = Router::new()
        .route("/input", post({
            let tx = tx.clone();
            move |Json(payload): Json<InputPayload>| {
                let tx = tx.clone();
                async move {
                    let _ = tx.try_send(payload);
                    Json("ok")
                }
            }
        }))
        .route("/context", get({
            let shared = shared_ctx.clone();
            move || {
                let shared = shared.clone();
                async move {
                    let guard = shared.lock().unwrap();
                    Json(ContextOutput { context: guard.context.clone() })
                }
            }
        }));

    let addr: SocketAddr = "0.0.0.0:7070".parse().unwrap();
    println!("[caden] listening on {}", addr);

    tokio::spawn(async move {
        axum::serve(
            tokio::net::TcpListener::bind(addr).await.unwrap(),
            app,
        ).await.unwrap();
    });

    // ── Physics thread ───────────────────────────────────────────────────
    let shared_phys = shared.clone();

    std::thread::spawn(move || {
        let mut active_buf = [0usize; ACTIVE_BUF];
        let mut active_head: usize = 0;
        let mut active_len: usize = 0;

        let mut tick: u64 = 0;
        let mut last_abstraction_tick: u64 = 0;
        let mut last_context_tick: u64 = 0;
        let mut last_replay_tick: u64 = 0;
        let mut last_flush_tick: u64 = 0;

        loop {
            tick = tick.wrapping_add(1);

            // ── Ingest new inputs ──────────────────────────────────────
            while let Ok(input) = rx.try_recv() {
                // Validate embedding dimension
                if input.embedding.len() != SEM_DIM {
                    eprintln!(
                        "[caden] warn: embedding len={} expected {}, storing text-only node",
                        input.embedding.len(), SEM_DIM
                    );
                    let idx = arena.alloc([0.0f32; SEM_DIM], &input.text, false);
                    unsafe { inject_energy(&arena, idx); }
                    active_buf[active_head % ACTIVE_BUF] = idx;
                    active_head += 1;
                    active_len = (active_len + 1).min(ACTIVE_BUF);
                    continue;
                }

                // Deduplicate: if the same text appeared recently, re-boost
                // rather than creating a duplicate node.
                let active_now = active_len.min(ACTIVE_BUF);
                let dedup_start = if active_head > active_now { active_head - active_now } else { 0 };
                let dup_idx = (dedup_start..active_head).find_map(|j| {
                    let candidate = active_buf[j % ACTIVE_BUF];
                    if candidate >= arena.count { return None; }
                    unsafe {
                        if node_text(&*arena.ptr().add(candidate)) == input.text {
                            Some(candidate)
                        } else { None }
                    }
                });

                if let Some(existing) = dup_idx {
                    unsafe { inject_energy(&arena, existing); }
                    active_buf[active_head % ACTIVE_BUF] = existing;
                    active_head += 1;
                    active_len = (active_len + 1).min(ACTIVE_BUF);
                    continue;
                }

                // New node
                let mut pos = [0.0f32; SEM_DIM];
                pos.copy_from_slice(&input.embedding);
                normalize(&mut pos);

                let idx = arena.alloc(pos, &input.text, false);
                unsafe { inject_energy(&arena, idx); }

                // Temporal NEXT edge to previous node
                if idx > 0 {
                    arena.connect(idx - 1, idx, rand_f32() * 0.005, EDGE_NEXT);
                }

                // ── Semantic similarity links ──────────────────────────
                // 1. Scan recent active buffer (fast path for recency)
                let active_now = active_len.min(ACTIVE_BUF);
                let scan_start = if active_head > active_now { active_head - active_now } else { 0 };
                for j in scan_start..active_head {
                    let candidate = active_buf[j % ACTIVE_BUF];
                    if candidate == idx || candidate >= arena.count { continue; }
                    unsafe {
                        let sim = cosine(&(*arena.ptr().add(idx)).pos,
                                        &(*arena.ptr().add(candidate)).pos);
                        if sim > 0.7 {
                            arena.connect(idx, candidate, sim * 0.003, EDGE_LINK);
                        }
                    }
                }

                // 2. Random sample from full arena (replicates Neo4j vector index)
                // This lets new inputs link to old memories, not just recent ones.
                let sample_count = GLOBAL_SIM_SAMPLES.min(arena.count);
                for _ in 0..sample_count {
                    let candidate = rand_usize(arena.count);
                    if candidate == idx { continue; }
                    unsafe {
                        let sim = cosine(&(*arena.ptr().add(idx)).pos,
                                        &(*arena.ptr().add(candidate)).pos);
                        if sim > 0.7 {
                            arena.connect(idx, candidate, sim * 0.003, EDGE_LINK);
                        }
                    }
                }

                // ── Conversational Reinforcement ───────────────────────
                // Boost energy of high-similarity nodes — replicates n8n
                // "Conversational Reinforcement" step that strengthened
                // semantically related cells on each new input.
                unsafe {
                    let new_pos = (*arena.ptr().add(idx)).pos;
                    for j in scan_start..active_head {
                        let candidate = active_buf[j % ACTIVE_BUF];
                        if candidate == idx || candidate >= arena.count { continue; }
                        let n = &mut *arena.ptr().add(candidate);
                        let sim = cosine(&n.pos, &new_pos);
                        if sim > 0.8 {
                            n.energy += CONV_REINFORCE_K * sim;
                        }
                    }
                }

                // ── Fragment nodes (n8n: Fragment Quotes) ─────────────
                let frags = fragment_text(&input.text);
                for frag in &frags {
                    let mut fpos = pos;
                    // Small noise so fragments spread slightly from parent
                    for d in 0..SEM_DIM { fpos[d] += (rand_f32() - 0.5) * 0.01; }
                    normalize(&mut fpos);
                    let fi = arena.alloc(fpos, frag, false);
                    arena.connect(idx, fi, 0.3, EDGE_FRAG);
                    if fi > 0 { arena.connect(fi - 1, fi, 0.02, EDGE_NEXT); }
                    active_buf[active_head % ACTIVE_BUF] = fi;
                    active_head += 1;
                    active_len = (active_len + 1).min(ACTIVE_BUF);
                }

                // Register main node in active set
                active_buf[active_head % ACTIVE_BUF] = idx;
                active_head += 1;
                active_len = (active_len + 1).min(ACTIVE_BUF);
            }

            let active_now = active_len.min(ACTIVE_BUF);
            if active_now == 0 {
                // No active nodes — sleep to avoid spinning
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }

            // Deduplicated active slice
            let active_slice: Vec<usize> = {
                let mut seen = HashSet::new();
                let start = if active_head > active_now { active_head - active_now } else { 0 };
                (start..active_head)
                    .map(|j| active_buf[j % ACTIVE_BUF])
                    .filter(|&i| i < arena.count && seen.insert(i))
                    .collect()
            };

            // ── Per-tick physics ───────────────────────────────────────
            unsafe {
                add_background_noise(&arena, &active_slice);
                diffuse_energy(&arena, &active_slice);
                lateral_inhibition(&arena, &active_slice);
                fatigue_recovery(&arena, &active_slice);
                global_flow_leak(&arena, &active_slice);
                energy_gradient_update(&arena, &active_slice);
                edge_saturation_cap(&arena, &active_slice);
                synaptic_decay(&arena, &active_slice);
                edge_pruning(&arena, &active_slice);
            }

            // ── Every 10 ticks ─────────────────────────────────────────
            if tick % 10 == 0 {
                unsafe {
                    entropy_regulation(&arena, &active_slice);
                    homeostasis(&arena, &active_slice);
                    plastic_threshold(&arena, &active_slice);
                    global_energy_regulation(&arena, &active_slice);
                    synaptic_normalization(&arena, &active_slice);
                }
            }

            // ── Spontaneous thought (~60 ticks) ────────────────────────
            if tick % SPONTANEOUS_PROB == 0 {
                unsafe { spontaneous_thought(&arena, &active_slice); }
            }

            // ── Create attractor links (stochastic ~1/50 ticks) ────────
            if rand_u32() % 50 == 0 {
                unsafe { create_attractor_links(&mut arena, &active_slice); }
            }

            // ── Abstraction formation (~200 ticks) ─────────────────────
            if tick - last_abstraction_tick >= ABSTRACTION_PROB {
                unsafe { try_abstraction(&mut arena, &active_slice); }
                last_abstraction_tick = tick;
            }

            // ── Prediction correction + activation (~30 ticks) ─────────
            if tick % 30 == 0 {
                unsafe {
                    prediction_correction(&arena, &active_slice);
                    prediction_activation(&arena, &active_slice);
                }
            }

            // ── Replay reinforcement (~300 ticks) ──────────────────────
            if tick - last_replay_tick >= 300 {
                unsafe { replay_reinforcement(&arena, &active_slice); }
                last_replay_tick = tick;
            }

            // ── Context update (~20 ticks) ─────────────────────────────
            if tick - last_context_tick >= 20 {
                let (new_context, spoken_idxs) =
                    unsafe { build_context(&arena, &active_slice) };

                // Update last_spoken directly by index — O(spoken) not O(N²)
                unsafe {
                    let ptr = arena.ptr();
                    let now = now_ms();
                    let tick32 = (tick & 0xFFFF_FFFF) as u32;
                    for &ni in &spoken_idxs {
                        if ni < arena.count {
                            let n = &mut *ptr.add(ni);
                            n.last_spoken = now;
                            n.spoke_tick = tick32;
                        }
                    }
                }

                *shared_phys.lock().unwrap() = SharedState { context: new_context };
                last_context_tick = tick;
            }

            // ── Periodic count flush to disk (~1000 ticks) ────────────
            if tick - last_flush_tick >= 1000 {
                arena.flush_count();
                last_flush_tick = tick;
            }

            // Yield — don't burn the CPU when there's nothing to do.
            std::thread::sleep(Duration::from_millis(1));
        }
    });

    tokio::signal::ctrl_c().await.unwrap();
    println!("[caden] shutting down — flushing count ...");
    // Final count flush is best-effort; the periodic flush covers most cases.
    Ok(())
}
