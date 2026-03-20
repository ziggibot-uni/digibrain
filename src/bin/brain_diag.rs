/// brain_diag — Real-time diagnostics for CadenGraph v4 spiking brain
///
/// Reads brain.bin (read-only mmap) and displays live metrics:
///   - Node/tick counts, active firing stats
///   - Membrane & threshold distributions
///   - Edge connectivity & weight health
///   - Embedding norm distribution
///   - Top active nodes with text
///   - History plots over time

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use memmap2::Mmap;
use std::{
    fs::OpenOptions,
    mem::size_of,
    sync::mpsc,
    time::Duration,
};

// ── Must match main.rs exactly ──────────────────────────────────────────
const DIM: usize = 256;
const MAX_EDGES: usize = 32;
const TEXT_LEN: usize = 128;
const NODE_CAP: usize = 5_000_000;
const HEADER_SIZE: usize = 64;
const BRAIN_MAGIC: u64 = 0xCADE_0004_0000_0000;

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

// ── Snapshot ─────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Snapshot {
    valid: bool,
    node_count: usize,
    tick: u64,

    // Membrane stats
    mean_membrane: f32,
    std_membrane: f32,
    min_membrane: f32,
    max_membrane: f32,

    // Threshold stats
    mean_threshold: f32,
    max_threshold: f32,

    // Firing
    recently_fired: usize,   // nodes with last_spike within 100 ticks
    refractory_count: usize, // nodes currently refractory
    firing_rate: f32,        // recently_fired / node_count

    // Edges
    total_edges: usize,
    mean_edge_count: f32,
    mean_weight: f32,
    std_weight: f32,
    positive_edges: usize,
    negative_edges: usize,
    zero_edges: usize,
    max_weight: f32,
    min_weight: f32,
    nodes_at_max_edges: usize,

    // Embedding
    mean_embed_norm: f32,
    std_embed_norm: f32,
    min_embed_norm: f32,
    max_embed_norm: f32,

    // Top active nodes
    top_active: Vec<(String, f32, u32, u8)>, // (text, membrane, last_spike, edge_count)
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            valid: false, node_count: 0, tick: 0,
            mean_membrane: 0.0, std_membrane: 0.0, min_membrane: 0.0, max_membrane: 0.0,
            mean_threshold: 0.0, max_threshold: 0.0,
            recently_fired: 0, refractory_count: 0, firing_rate: 0.0,
            total_edges: 0, mean_edge_count: 0.0, mean_weight: 0.0, std_weight: 0.0,
            positive_edges: 0, negative_edges: 0, zero_edges: 0,
            max_weight: 0.0, min_weight: 0.0, nodes_at_max_edges: 0,
            mean_embed_norm: 0.0, std_embed_norm: 0.0, min_embed_norm: 0.0, max_embed_norm: 0.0,
            top_active: vec![],
        }
    }
}

fn take_snapshot(path: &str) -> Snapshot {
    let f = match OpenOptions::new().read(true).open(path) {
        Ok(f) => f,
        Err(_) => return Snapshot::default(),
    };
    let meta = f.metadata().unwrap();
    if meta.len() < HEADER_SIZE as u64 {
        return Snapshot::default();
    }

    let mmap = unsafe { Mmap::map(&f).unwrap() };
    let hdr = unsafe { &*(mmap.as_ptr() as *const BrainHeader) };

    if hdr.magic != BRAIN_MAGIC {
        return Snapshot::default();
    }

    let count = (hdr.node_count as usize).min(NODE_CAP);
    let tick = hdr.tick;

    if count == 0 {
        return Snapshot { valid: true, tick, ..Snapshot::default() };
    }

    let expected = HEADER_SIZE + count * size_of::<Node>();
    if mmap.len() < expected {
        return Snapshot::default();
    }

    let nodes_ptr = unsafe { mmap.as_ptr().add(HEADER_SIZE) as *const Node };
    let tick32 = tick as u32;

    // Accumulators
    let mut sum_m = 0.0f64;
    let mut sum_m2 = 0.0f64;
    let mut min_m = f32::MAX;
    let mut max_m = f32::MIN;
    let mut sum_th = 0.0f64;
    let mut max_th: f32 = 0.0;
    let mut recently_fired: usize = 0;
    let mut refractory_count: usize = 0;
    let mut total_edges: usize = 0;
    let mut sum_w = 0.0f64;
    let mut sum_w2 = 0.0f64;
    let mut pos_e: usize = 0;
    let mut neg_e: usize = 0;
    let mut zero_e: usize = 0;
    let mut max_w: f32 = f32::MIN;
    let mut min_w: f32 = f32::MAX;
    let mut full_nodes: usize = 0;
    let mut sum_norm = 0.0f64;
    let mut sum_norm2 = 0.0f64;
    let mut min_norm = f32::MAX;
    let mut max_norm: f32 = 0.0;

    // For top active: collect nodes with recent spikes, sort by recency
    let mut active_candidates: Vec<(usize, u32)> = Vec::new();

    for i in 0..count {
        let n = unsafe { &*nodes_ptr.add(i) };

        // Membrane
        let m = n.membrane;
        sum_m += m as f64;
        sum_m2 += (m as f64) * (m as f64);
        if m < min_m { min_m = m; }
        if m > max_m { max_m = m; }

        // Threshold
        sum_th += n.threshold as f64;
        if n.threshold > max_th { max_th = n.threshold; }

        // Firing
        if n.last_spike > 0 && tick32.saturating_sub(n.last_spike) < 100 {
            recently_fired += 1;
            if n.text_len > 0 {
                active_candidates.push((i, n.last_spike));
            }
        }
        if n.refractory > 0 { refractory_count += 1; }

        // Edges
        let ec = n.edge_count as usize;
        total_edges += ec;
        if ec >= MAX_EDGES { full_nodes += 1; }
        for ei in 0..ec {
            let w = n.edges[ei].weight;
            sum_w += w as f64;
            sum_w2 += (w as f64) * (w as f64);
            if w > 0.001 { pos_e += 1; }
            else if w < -0.001 { neg_e += 1; }
            else { zero_e += 1; }
            if w > max_w { max_w = w; }
            if w < min_w { min_w = w; }
        }

        // Embedding norm
        let norm: f32 = n.pos.iter().map(|x| x * x).sum::<f32>().sqrt();
        sum_norm += norm as f64;
        sum_norm2 += (norm as f64) * (norm as f64);
        if norm < min_norm { min_norm = norm; }
        if norm > max_norm { max_norm = norm; }
    }

    let cn = count as f64;
    let mean_m = (sum_m / cn) as f32;
    let std_m = ((sum_m2 / cn - (sum_m / cn).powi(2)).max(0.0).sqrt()) as f32;
    let mean_th = (sum_th / cn) as f32;

    let (mean_w, std_w) = if total_edges > 0 {
        let en = total_edges as f64;
        let mw = (sum_w / en) as f32;
        let sw = ((sum_w2 / en - (sum_w / en).powi(2)).max(0.0).sqrt()) as f32;
        (mw, sw)
    } else { (0.0, 0.0) };

    let mean_en = (sum_norm / cn) as f32;
    let std_en = ((sum_norm2 / cn - (sum_norm / cn).powi(2)).max(0.0).sqrt()) as f32;

    // Top active: sort by most recent spike, take top 20
    active_candidates.sort_by(|a, b| b.1.cmp(&a.1));
    active_candidates.truncate(20);
    let top_active: Vec<(String, f32, u32, u8)> = active_candidates.iter().map(|&(i, _)| {
        let n = unsafe { &*nodes_ptr.add(i) };
        let text = String::from_utf8_lossy(&n.text[..(n.text_len as usize).min(TEXT_LEN)]).into_owned();
        (text, n.membrane, n.last_spike, n.edge_count)
    }).collect();

    Snapshot {
        valid: true,
        node_count: count,
        tick,
        mean_membrane: mean_m, std_membrane: std_m, min_membrane: min_m, max_membrane: max_m,
        mean_threshold: mean_th, max_threshold: max_th,
        recently_fired, refractory_count,
        firing_rate: recently_fired as f32 / count as f32,
        total_edges, mean_edge_count: total_edges as f32 / count as f32,
        mean_weight: mean_w, std_weight: std_w,
        positive_edges: pos_e, negative_edges: neg_e, zero_edges: zero_e,
        max_weight: if total_edges > 0 { max_w } else { 0.0 },
        min_weight: if total_edges > 0 { min_w } else { 0.0 },
        nodes_at_max_edges: full_nodes,
        mean_embed_norm: mean_en, std_embed_norm: std_en,
        min_embed_norm: min_norm, max_embed_norm: max_norm,
        top_active,
    }
}

// ── History ring buffer ─────────────────────────────────────────────────

const HISTORY_LEN: usize = 600; // 10 minutes at 1 sample/sec

struct History {
    node_count: Vec<f64>,
    firing_rate: Vec<f64>,
    mean_membrane: Vec<f64>,
    mean_weight: Vec<f64>,
    mean_threshold: Vec<f64>,
    total_edges: Vec<f64>,
    recently_fired: Vec<f64>,
    tick: Vec<f64>,
}

impl History {
    fn new() -> Self {
        Self {
            node_count: Vec::with_capacity(HISTORY_LEN),
            firing_rate: Vec::with_capacity(HISTORY_LEN),
            mean_membrane: Vec::with_capacity(HISTORY_LEN),
            mean_weight: Vec::with_capacity(HISTORY_LEN),
            mean_threshold: Vec::with_capacity(HISTORY_LEN),
            total_edges: Vec::with_capacity(HISTORY_LEN),
            recently_fired: Vec::with_capacity(HISTORY_LEN),
            tick: Vec::with_capacity(HISTORY_LEN),
        }
    }

    fn push(&mut self, s: &Snapshot) {
        if self.node_count.len() >= HISTORY_LEN {
            self.node_count.remove(0);
            self.firing_rate.remove(0);
            self.mean_membrane.remove(0);
            self.mean_weight.remove(0);
            self.mean_threshold.remove(0);
            self.total_edges.remove(0);
            self.recently_fired.remove(0);
            self.tick.remove(0);
        }
        self.node_count.push(s.node_count as f64);
        self.firing_rate.push(s.firing_rate as f64);
        self.mean_membrane.push(s.mean_membrane as f64);
        self.mean_weight.push(s.mean_weight as f64);
        self.mean_threshold.push(s.mean_threshold as f64);
        self.total_edges.push(s.total_edges as f64);
        self.recently_fired.push(s.recently_fired as f64);
        self.tick.push(s.tick as f64);
    }

    fn line(&self, data: &[f64]) -> Line<'_> {
        let pts: PlotPoints = data.iter().enumerate()
            .map(|(i, &v)| [i as f64, v])
            .collect();
        Line::new(pts)
    }
}

// ── App ─────────────────────────────────────────────────────────────────

struct DiagApp {
    rx: mpsc::Receiver<Snapshot>,
    snap: Snapshot,
    history: History,
}

impl DiagApp {
    fn new(rx: mpsc::Receiver<Snapshot>) -> Self {
        Self {
            rx,
            snap: Snapshot::default(),
            history: History::new(),
        }
    }
}

impl eframe::App for DiagApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain channel, keep latest
        while let Ok(s) = self.rx.try_recv() {
            if s.valid {
                self.history.push(&s);
            }
            self.snap = s;
        }

        ctx.request_repaint_after(Duration::from_millis(250));

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let s = &self.snap;

                if !s.valid {
                    ui.heading("brain_diag v4 — waiting for brain.bin ...");
                    return;
                }

                ui.heading(format!(
                    "CadenGraph v4 | {} nodes | tick {} | {:.1} KB/node",
                    s.node_count, s.tick, size_of::<Node>() as f64 / 1024.0
                ));
                ui.separator();

                // ── Overview ────────────────────────────────────────────
                ui.columns(3, |cols| {
                    cols[0].label(format!("Nodes: {}", s.node_count));
                    cols[0].label(format!("Tick: {}", s.tick));
                    cols[0].label(format!("Node size: {} B", size_of::<Node>()));

                    cols[1].label(format!("Recently fired: {}", s.recently_fired));
                    cols[1].label(format!("Firing rate: {:.4}", s.firing_rate));
                    cols[1].label(format!("Refractory: {}", s.refractory_count));

                    cols[2].label(format!("Total edges: {}", s.total_edges));
                    cols[2].label(format!("Mean edges/node: {:.2}", s.mean_edge_count));
                    cols[2].label(format!("Full nodes ({}): {}", MAX_EDGES, s.nodes_at_max_edges));
                });
                ui.separator();

                // ── Membrane ────────────────────────────────────────────
                ui.collapsing("Membrane Potential", |ui| {
                    ui.columns(2, |cols| {
                        cols[0].label(format!("Mean: {:.6}", s.mean_membrane));
                        cols[0].label(format!("Std:  {:.6}", s.std_membrane));
                        cols[1].label(format!("Min:  {:.6}", s.min_membrane));
                        cols[1].label(format!("Max:  {:.6}", s.max_membrane));
                    });
                    if s.mean_membrane.abs() > 0.5 {
                        ui.colored_label(egui::Color32::YELLOW, "! High mean membrane — possible runaway");
                    }
                    if s.max_membrane > 5.0 {
                        ui.colored_label(egui::Color32::RED, "! Extreme membrane detected — seizure risk");
                    }
                });

                // ── Threshold ───────────────────────────────────────────
                ui.collapsing("Threshold Adaptation", |ui| {
                    ui.label(format!("Mean threshold: {:.4} (rest = 1.0)", s.mean_threshold));
                    ui.label(format!("Max threshold:  {:.4}", s.max_threshold));
                    if s.mean_threshold > 2.0 {
                        ui.colored_label(egui::Color32::YELLOW, "! High mean threshold — neurons becoming hard to activate");
                    }
                });

                // ── Edges ───────────────────────────────────────────────
                ui.collapsing("Connectivity", |ui| {
                    ui.columns(2, |cols| {
                        cols[0].label(format!("Total edges: {}", s.total_edges));
                        cols[0].label(format!("Mean weight: {:.6}", s.mean_weight));
                        cols[0].label(format!("Std weight:  {:.6}", s.std_weight));
                        cols[1].label(format!("Positive: {}", s.positive_edges));
                        cols[1].label(format!("Negative: {}", s.negative_edges));
                        cols[1].label(format!("~Zero:    {}", s.zero_edges));
                    });
                    ui.label(format!("Weight range: [{:.4}, {:.4}]", s.min_weight, s.max_weight));
                    let ratio = if s.positive_edges + s.negative_edges > 0 {
                        s.positive_edges as f32 / (s.positive_edges + s.negative_edges) as f32
                    } else { 0.5 };
                    ui.label(format!("Excitatory ratio: {:.1}%", ratio * 100.0));
                    if ratio < 0.3 {
                        ui.colored_label(egui::Color32::YELLOW, "! Low excitatory ratio — network may go silent");
                    }
                    if ratio > 0.95 {
                        ui.colored_label(egui::Color32::YELLOW, "! Very high excitatory ratio — seizure risk");
                    }
                });

                // ── Embeddings ──────────────────────────────────────────
                ui.collapsing("Embedding Health", |ui| {
                    ui.columns(2, |cols| {
                        cols[0].label(format!("Mean norm: {:.4}", s.mean_embed_norm));
                        cols[0].label(format!("Std norm:  {:.4}", s.std_embed_norm));
                        cols[1].label(format!("Min norm:  {:.4}", s.min_embed_norm));
                        cols[1].label(format!("Max norm:  {:.4}", s.max_embed_norm));
                    });
                    if s.std_embed_norm > 0.5 {
                        ui.colored_label(egui::Color32::YELLOW, "! High norm variance — embeddings drifting apart");
                    }
                    if s.mean_embed_norm < 0.1 {
                        ui.colored_label(egui::Color32::RED, "! Near-zero norms — embedding collapse");
                    }
                });

                ui.separator();

                // ── Top active nodes ─────────────────────────────────────
                ui.collapsing("Top Active Nodes (by recency)", |ui| {
                    egui::Grid::new("active_grid").striped(true).show(ui, |ui| {
                        ui.label("Text");
                        ui.label("Membrane");
                        ui.label("Last Spike");
                        ui.label("Edges");
                        ui.end_row();
                        for (text, membrane, spike, edges) in &s.top_active {
                            let display = if text.len() > 60 {
                                format!("{}...", &text[..60])
                            } else {
                                text.clone()
                            };
                            ui.label(&display);
                            ui.label(format!("{:.4}", membrane));
                            ui.label(format!("{}", spike));
                            ui.label(format!("{}", edges));
                            ui.end_row();
                        }
                    });
                });

                ui.separator();

                // ── History plots ────────────────────────────────────────
                let h = &self.history;
                let plot_h = 120.0;

                ui.columns(2, |cols| {
                    cols[0].label("Node Count");
                    Plot::new("nodes_plot").height(plot_h).show(&mut cols[0], |plot_ui| {
                        plot_ui.line(h.line(&h.node_count));
                    });

                    cols[1].label("Firing Rate");
                    Plot::new("firing_plot").height(plot_h).show(&mut cols[1], |plot_ui| {
                        plot_ui.line(h.line(&h.firing_rate));
                    });
                });

                ui.columns(2, |cols| {
                    cols[0].label("Recently Fired");
                    Plot::new("fired_plot").height(plot_h).show(&mut cols[0], |plot_ui| {
                        plot_ui.line(h.line(&h.recently_fired));
                    });

                    cols[1].label("Mean Membrane");
                    Plot::new("membrane_plot").height(plot_h).show(&mut cols[1], |plot_ui| {
                        plot_ui.line(h.line(&h.mean_membrane));
                    });
                });

                ui.columns(2, |cols| {
                    cols[0].label("Mean Weight");
                    Plot::new("weight_plot").height(plot_h).show(&mut cols[0], |plot_ui| {
                        plot_ui.line(h.line(&h.mean_weight));
                    });

                    cols[1].label("Mean Threshold");
                    Plot::new("threshold_plot").height(plot_h).show(&mut cols[1], |plot_ui| {
                        plot_ui.line(h.line(&h.mean_threshold));
                    });
                });

                ui.columns(2, |cols| {
                    cols[0].label("Total Edges");
                    Plot::new("edges_plot").height(plot_h).show(&mut cols[0], |plot_ui| {
                        plot_ui.line(h.line(&h.total_edges));
                    });

                    cols[1].label("Tick");
                    Plot::new("tick_plot").height(plot_h).show(&mut cols[1], |plot_ui| {
                        plot_ui.line(h.line(&h.tick));
                    });
                });
            });
        });
    }
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let (tx, rx) = mpsc::channel::<Snapshot>();

    // Worker thread: snapshot brain.bin every second
    std::thread::spawn(move || {
        loop {
            let snap = take_snapshot("brain.bin");
            if tx.send(snap).is_err() { break; }
            std::thread::sleep(Duration::from_secs(1));
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 800.0])
            .with_title("CadenGraph v4 — Brain Diagnostics"),
        ..Default::default()
    };

    eframe::run_native(
        "brain_diag",
        options,
        Box::new(|_cc| Ok(Box::new(DiagApp::new(rx)))),
    ).unwrap();
}
