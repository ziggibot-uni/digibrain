/// CadenGraph v4 — Spiking cognitive substrate
///
/// Three mechanisms. Everything else emerges.
///   1. Leaky integrate-and-fire spiking
///   2. Spike-timing-dependent plasticity (STDP)
///   3. Surprise modulation
///
/// No LLM. No transformers. No pretrained embeddings.
/// CPU + RAM only. Text → embedding via character-trigram hashing.
///
///   POST /input  { "text": "..." }
///   GET  /context → active node texts

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

// ── Dimensions ────────────────────────────────────────────────────────
const DIM: usize = 256;
const MAX_EDGES: usize = 32;
const TEXT_LEN: usize = 128;
const NODE_CAP: usize = 5_000_000;
const HEADER_SIZE: usize = 64;
const BRAIN_MAGIC: u64 = 0xCADE_0004_0000_0000;
const FILE_SIZE: usize = HEADER_SIZE + NODE_CAP * size_of::<Node>();

// ── Physics (the only tuning knobs) ───────────────────────────────────
const LEAK: f32 = 0.95;
const THRESHOLD_REST: f32 = 1.0;
const THRESHOLD_UP: f32 = 0.15;
const THRESHOLD_DOWN: f32 = 0.005;
const REFRACTORY_TICKS: u8 = 5;
const STDP_WINDOW: u32 = 20;
const LTP_RATE: f32 = 0.01;
const LTD_RATE: f32 = 0.008;
const NOISE_AMP: f32 = 0.02;
const WEIGHT_CAP: f32 = 1.0;
const SURPRISE_DECAY: f32 = 0.95;
const INPUT_STRENGTH: f32 = 2.0;
const SIM_THRESHOLD: f32 = 0.3;
const EMBED_DRIFT: f32 = 0.001;
const ACTIVE_EPSILON: f32 = 0.05;
const GLOBAL_SAMPLES: usize = 50;
const ACTIVE_CAP: usize = 2048;
const INGEST_PER_TICK: usize = 8;
const LTP_SAMPLE_CAP: usize = 64;

// ── Data structures ───────────────────────────────────────────────────
// One node type. One edge type. That's it.

#[repr(C)]
#[derive(Clone, Copy)]
struct Edge {
    target: u32,
    weight: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Node {
    membrane: f32,
    threshold: f32,
    refractory: u8,
    _p0: [u8; 3],
    last_spike: u32,
    pos: [f32; DIM],
    edges: [Edge; MAX_EDGES],
    edge_count: u8,
    text: [u8; TEXT_LEN],
    text_len: u8,
    _p1: [u8; 2],
}

#[repr(C)]
struct BrainHeader {
    magic: u64,
    node_count: u64,
    tick: u64,
    _reserved: [u64; 5],
}

const _: () = assert!(size_of::<BrainHeader>() == HEADER_SIZE);

// ── Arena ─────────────────────────────────────────────────────────────

struct Arena {
    mmap: MmapMut,
    count: usize,
}

impl Arena {
    fn open(path: &str) -> io::Result<Self> {
        let f = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        f.set_len(FILE_SIZE as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&f)? };
        let hdr = unsafe { &*(mmap.as_ptr() as *const BrainHeader) };
        let count = if hdr.magic == BRAIN_MAGIC {
            (hdr.node_count as usize).min(NODE_CAP)
        } else { 0 };
        let mut a = Self { mmap, count };
        if count == 0 { a.flush(0); } // write magic on fresh file
        println!("[brain] {} nodes restored", count);
        Ok(a)
    }

    fn ptr(&self) -> *mut Node {
        unsafe { (self.mmap.as_ptr() as *mut u8).add(HEADER_SIZE) as *mut Node }
    }

    fn flush(&mut self, tick: u64) {
        let h = unsafe { &mut *(self.mmap.as_mut_ptr() as *mut BrainHeader) };
        h.magic = BRAIN_MAGIC;
        h.node_count = self.count as u64;
        h.tick = tick;
    }

    fn alloc(&mut self, pos: [f32; DIM], text: &str) -> usize {
        if self.count >= NODE_CAP { return self.count - 1; }
        let idx = self.count;
        self.count += 1;
        let n = unsafe { &mut *self.ptr().add(idx) };
        *n = Node {
            membrane: 0.0, threshold: THRESHOLD_REST,
            refractory: 0, _p0: [0; 3], last_spike: 0,
            pos,
            edges: [Edge { target: 0, weight: 0.0 }; MAX_EDGES],
            edge_count: 0,
            text: [0; TEXT_LEN], text_len: 0, _p1: [0; 2],
        };
        let b = text.as_bytes();
        let len = b.len().min(TEXT_LEN);
        n.text[..len].copy_from_slice(&b[..len]);
        n.text_len = len as u8;
        if idx % 256 == 0 { self.flush(0); }
        idx
    }
}

// ── Utilities ─────────────────────────────────────────────────────────

fn rand_u32() -> u32 {
    use std::cell::Cell;
    thread_local! { static S: Cell<u64> = Cell::new(0x12345678abcdef01); }
    S.with(|s| {
        let mut x = s.get();
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        s.set(x); x as u32
    })
}
fn rand_f32() -> f32 { rand_u32() as f32 / u32::MAX as f32 }
fn rand_usize(n: usize) -> usize { (rand_u32() as usize) % n.max(1) }

fn hash64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

/// Character-trigram hashing → DIM-dimensional embedding.
/// Similar text → similar vectors. No transformer needed.
/// STDP will refine these positions over time through experience.
fn text_to_pos(text: &str) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    let low = text.to_lowercase();
    let b = low.as_bytes();
    if b.len() < 3 {
        let h = hash64(b);
        for i in 0..DIM {
            v[i] = ((h.wrapping_mul(i as u64 + 1)) as f32 * 1e-18).sin();
        }
    } else {
        for w in b.windows(3) {
            let h = hash64(w);
            for k in 0..8u64 {
                let dim = h.wrapping_mul(k.wrapping_mul(0x517cc1b727220a95).wrapping_add(1))
                    as usize % DIM;
                let sign = if (h >> (k + 3)) & 1 == 0 { 1.0f32 } else { -1.0 };
                v[dim] += sign;
            }
        }
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 { for x in v.iter_mut() { *x /= norm; } }
    v
}

fn cosine(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..DIM { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    d / ((na.sqrt() * nb.sqrt()) + 1e-9)
}

fn node_text(n: &Node) -> String {
    String::from_utf8_lossy(&n.text[..(n.text_len as usize).min(TEXT_LEN)]).into_owned()
}

// ── Edge operations (raw pointer, no borrow conflicts) ────────────────

unsafe fn add_edge(ptr: *mut Node, count: usize, from: usize, to: usize, w: f32) {
    if from == to || from >= count || to >= count { return; }
    let n = &mut *ptr.add(from);
    for i in 0..n.edge_count as usize {
        if n.edges[i].target == to as u32 {
            n.edges[i].weight = (n.edges[i].weight + w).clamp(-WEIGHT_CAP, WEIGHT_CAP);
            return;
        }
    }
    if (n.edge_count as usize) < MAX_EDGES {
        let i = n.edge_count as usize;
        n.edges[i] = Edge { target: to as u32, weight: w.clamp(-WEIGHT_CAP, WEIGHT_CAP) };
        n.edge_count += 1;
    } else {
        // Overwrite weakest edge if new one is stronger
        let mut weakest = 0;
        for i in 1..MAX_EDGES {
            if n.edges[i].weight.abs() < n.edges[weakest].weight.abs() { weakest = i; }
        }
        if w.abs() > n.edges[weakest].weight.abs() {
            n.edges[weakest] = Edge { target: to as u32, weight: w.clamp(-WEIGHT_CAP, WEIGHT_CAP) };
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// THE TICK — this is the entire physics engine.
//
// ~80 lines. Three mechanisms:
//   1. Leaky integrate-and-fire (membrane, threshold, refractory)
//   2. STDP (LTD on outgoing edges, LTP via co-activation)
//   3. Surprise modulation (scales learning rates)
//
// Everything else — inhibition, fatigue, homeostasis, normalization,
// pruning, abstraction, prediction — emerges from these three.
// ══════════════════════════════════════════════════════════════════════

unsafe fn tick(ptr: *mut Node, count: usize, active: &mut HashSet<usize>,
              tick_n: u32, surprise: &mut f32) {
    // ── Phase 1: Leak, noise, threshold relaxation ────────────────────
    let mut to_remove: Vec<usize> = Vec::new();
    for &i in active.iter() {
        if i >= count { to_remove.push(i); continue; }
        let n = &mut *ptr.add(i);
        if n.refractory > 0 {
            n.refractory -= 1;
            continue;
        }
        n.membrane *= LEAK;
        n.membrane += (rand_f32() - 0.5) * NOISE_AMP;
        n.threshold = (n.threshold - THRESHOLD_DOWN).max(THRESHOLD_REST * 0.5);
        if n.membrane.abs() < ACTIVE_EPSILON { to_remove.push(i); }
    }
    for i in &to_remove { active.remove(i); }

    // ── Phase 2: Fire → propagate → LTD ──────────────────────────────
    let mut fired: Vec<usize> = Vec::new();
    let mut wake: Vec<usize> = Vec::new();

    for &i in active.iter() {
        if i >= count { continue; }
        let n = &*ptr.add(i);
        if n.refractory > 0 || n.membrane < n.threshold { continue; }

        fired.push(i);
        let n = &mut *ptr.add(i);
        n.last_spike = tick_n;
        n.membrane = -0.5;
        n.refractory = REFRACTORY_TICKS;
        n.threshold += THRESHOLD_UP;

        let ec = n.edge_count as usize;
        for ei in 0..ec {
            let ti = n.edges[ei].target as usize;
            let w = n.edges[ei].weight;
            if ti >= count { continue; }

            // Propagate spike
            (*ptr.add(ti)).membrane += w;
            wake.push(ti);

            // LTD: target fired recently before us → anticausal → weaken
            let target_spike = (*ptr.add(ti)).last_spike;
            if target_spike > 0 {
                let dt = tick_n.saturating_sub(target_spike);
                if dt > 0 && dt < STDP_WINDOW {
                    (*ptr.add(i)).edges[ei].weight =
                        ((*ptr.add(i)).edges[ei].weight - LTD_RATE * (1.0 + *surprise))
                        .clamp(-WEIGHT_CAP, WEIGHT_CAP);
                }
            }
        }
    }

    for i in wake { active.insert(i); }

    // ── Phase 3: STDP LTP — co-firing nodes form causal edges ────────
    // Sample active set instead of full scan to stay O(fired × LTP_SAMPLE_CAP)
    let active_vec: Vec<usize> = active.iter().copied().collect();
    for &si in &fired {
        if si >= count { continue; }
        let si_pos = (*ptr.add(si)).pos;

        let (offset, len) = if active_vec.len() > LTP_SAMPLE_CAP {
            (rand_usize(active_vec.len() - LTP_SAMPLE_CAP), LTP_SAMPLE_CAP)
        } else {
            (0, active_vec.len())
        };

        for &other in &active_vec[offset..offset + len] {
            if other == si || other >= count { continue; }
            let on = &*ptr.add(other);
            if on.last_spike == 0 { continue; }
            let dt = tick_n.saturating_sub(on.last_spike);
            if dt == 0 || dt >= STDP_WINDOW { continue; }

            // other fired before si → causal → strengthen other→si
            let sim = cosine(&(*ptr.add(other)).pos, &si_pos);
            if sim > SIM_THRESHOLD {
                let dw = LTP_RATE * sim * (1.0 + *surprise);
                add_edge(ptr, count, other, si, dw);

                // Embedding drift: co-firing pulls positions together
                for d in 0..DIM {
                    let diff = (*ptr.add(other)).pos[d] - (*ptr.add(si)).pos[d];
                    (*ptr.add(si)).pos[d] += diff * EMBED_DRIFT;
                    (*ptr.add(other)).pos[d] -= diff * EMBED_DRIFT * 0.5;
                }
            }
        }
    }

    // ── Phase 4: Surprise decay ──────────────────────────────────────
    *surprise *= SURPRISE_DECAY;
}

// ── Context (what the brain is "thinking about") ──────────────────────

unsafe fn build_context(ptr: *mut Node, count: usize,
                        active: &HashSet<usize>, tick_n: u32) -> Vec<ContextEntry> {
    let mut entries: Vec<ContextEntry> = Vec::new();
    let mut seen = HashSet::new();
    for &i in active {
        if i >= count { continue; }
        let n = &*ptr.add(i);
        if n.text_len == 0 || n.last_spike == 0 { continue; }
        let text = node_text(n);
        let trimmed = text.trim();
        if trimmed.len() <= 3 { continue; }
        let key = trimmed.to_lowercase();
        if !seen.insert(key) { continue; }
        entries.push(ContextEntry {
            text: trimmed.to_string(),
            recency: tick_n.saturating_sub(n.last_spike),
        });
    }
    entries.sort_by_key(|e| e.recency);
    entries.truncate(16);
    entries
}

// ── HTTP types ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct InputPayload { text: String }

#[derive(Serialize, Clone)]
struct ContextEntry { text: String, recency: u32 }

#[derive(Serialize)]
struct ContextOutput { context: Vec<ContextEntry>, active_count: usize, node_count: usize }

struct Shared { context: Vec<ContextEntry>, active_count: usize, node_count: usize }

// ══════════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> io::Result<()> {
    println!("[brain] CadenGraph v4 — spiking cognitive substrate");
    println!("[brain] DIM={DIM} EDGES={MAX_EDGES} CAP={}M FILE={:.1}GB",
        NODE_CAP / 1_000_000, FILE_SIZE as f64 / 1e9);

    let mut arena = Arena::open("brain.bin")?;

    let (tx, rx): (Sender<InputPayload>, Receiver<InputPayload>) =
        crossbeam_channel::bounded(4096);
    let shared = Arc::new(Mutex::new(Shared {
        context: vec![], active_count: 0, node_count: 0,
    }));

    // ── HTTP server ──────────────────────────────────────────────────
    let app = Router::new()
        .route("/input", post({
            let tx = tx.clone();
            move |Json(p): Json<InputPayload>| {
                let tx = tx.clone();
                async move { let _ = tx.try_send(p); Json(serde_json::json!({"status": "ok"})) }
            }
        }))
        .route("/context", get({
            let s = shared.clone();
            move || { let s = s.clone(); async move {
                let g = s.lock().unwrap();
                Json(ContextOutput {
                    context: g.context.clone(),
                    active_count: g.active_count,
                    node_count: g.node_count,
                })
            }}
        }))
        .route("/think", get({
            let s = shared.clone();
            move || { let s = s.clone(); async move {
                let g = s.lock().unwrap();
                let thought: String = g.context.iter()
                    .map(|e| e.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                thought
            }}
        }));

    let addr: SocketAddr = "0.0.0.0:7070".parse().unwrap();
    println!("[brain] http://{addr}");
    tokio::spawn(async move {
        axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app)
            .await.unwrap();
    });

    // ── Physics thread ───────────────────────────────────────────────
    let shared_phys = shared.clone();
    std::thread::spawn(move || {
        let mut active: HashSet<usize> = HashSet::new();
        let mut tick_n: u32 = {
            let h = unsafe { &*(arena.mmap.as_ptr() as *const BrainHeader) };
            if h.magic == BRAIN_MAGIC { h.tick as u32 } else { 0 }
        };
        let mut surprise: f32 = 0.0;
        let mut last_ctx: u32 = 0;
        let mut last_flush: u32 = 0;

        loop {
            tick_n = tick_n.wrapping_add(1);

            // ── Ingest (rate-limited) ─────────────────────────────────
            for _ in 0..INGEST_PER_TICK {
                let input = match rx.try_recv() { Ok(v) => v, Err(_) => break };
                let pos = text_to_pos(&input.text);
                let idx = arena.alloc(pos, &input.text);
                let ptr = arena.ptr();
                let count = arena.count;

                unsafe {
                    (*ptr.add(idx)).membrane = INPUT_STRENGTH;

                    // Temporal link: previous node → this one
                    if idx > 0 {
                        add_edge(ptr, count, idx - 1, idx, 0.3);
                        active.insert(idx - 1);
                    }

                    // Similarity links: sample the full arena
                    let samples = GLOBAL_SAMPLES.min(count);
                    for _ in 0..samples {
                        let c = rand_usize(count);
                        if c == idx { continue; }
                        let sim = cosine(&(*ptr.add(idx)).pos, &(*ptr.add(c)).pos);
                        if sim > SIM_THRESHOLD {
                            add_edge(ptr, count, idx, c, sim * 0.3);
                            add_edge(ptr, count, c, idx, sim * 0.3);
                            active.insert(c);
                        }
                    }
                }

                active.insert(idx);
                surprise = 1.0;
            }

            if active.is_empty() {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }

            // ── Physics ───────────────────────────────────────────────
            unsafe {
                tick(arena.ptr(), arena.count, &mut active, tick_n, &mut surprise);
            }

            // ── Evict if active set too large ─────────────────────────
            if active.len() > ACTIVE_CAP {
                let ptr = arena.ptr();
                let mut by_membrane: Vec<(usize, f32)> = active.iter()
                    .map(|&i| (i, unsafe { (*ptr.add(i)).membrane.abs() }))
                    .collect();
                by_membrane.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                by_membrane.truncate(ACTIVE_CAP);
                active = by_membrane.into_iter().map(|(i, _)| i).collect();
            }

            // ── Context update (~20 ticks) ────────────────────────────
            if tick_n.wrapping_sub(last_ctx) >= 20 {
                let ctx = unsafe {
                    build_context(arena.ptr(), arena.count, &active, tick_n)
                };
                *shared_phys.lock().unwrap() = Shared {
                    context: ctx,
                    active_count: active.len(),
                    node_count: arena.count,
                };
                last_ctx = tick_n;
            }

            // ── Flush (~1000 ticks) ───────────────────────────────────
            if tick_n.wrapping_sub(last_flush) >= 1000 {
                arena.flush(tick_n as u64);
                last_flush = tick_n;
            }

            // Adaptive sleep: fast when busy, slow when idle
            if !rx.is_empty() || active.len() > 100 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    });

    tokio::signal::ctrl_c().await.unwrap();
    println!("[brain] shutdown");
    Ok(())
}
