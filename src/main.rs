// main.rs
// Description: Binary entry point with interactive menu loop.
//              Builds model from tokenizer, supports checkpoint save and load, and provides
//              parallel training and serving via separate model instances.
//
//              Extensions:
//              - Background training thread with live metrics
//              - Parallel ask during training via separate serving model instance
//              - Snapshot based parameter hot updates from training to serving
//              - Continuous learning support for partial path availability and incremental updates
//              - Online data ingestion: add training data files while training is running
//
// History:
// - 2026-02-01: Add menu loop and checkpoint save and load.
// - 2026-02-07: Add MBT parallel block group layer to support multi branch topology.
// - 2026-02-08: Add predict_with_stats and post predict metrics.
// - 2026-02-13: Add background training, live metrics, cooperative stop.
// - 2026-02-13: Add serving model with snapshot updates for true parallel ask during training.
// - 2026-02-14: Add online ingestion channel and robust menu wiring for command b.
// - 2026-02-15: Add help function for menu and metrics documentation (ASCII).
// Author: Marcus Schlieper (ExpChat.ai)

#![allow(warnings)]

mod layer;
mod math;
mod tokenizer;
mod train;
mod utils;

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

use crate::layer::{
    ContinuousLearningConfig, Embeddings, Layer, Llm, OutputProjection, ParallelBlockGroup,
    PredictStats, TrainingDataEventAscii, TrainingProgressEventAscii, TransformerBlock,
    TransformerSequence, phase_strategy_config_ascii, training_phase_ascii,
};
use crate::tokenizer::{BpeTokenizer, BpeTokenizerConfig};
use crate::train::{Dataset, DatasetType};

pub const MAX_SEQ_LEN: usize = 80;
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;

fn read_line_ascii_trimmed() -> Result<String, String> {
    let mut s_input = String::new();
    std::io::stdin()
        .read_line(&mut s_input)
        .map_err(|_| "input_read_error".to_string())?;
    Ok(s_input.trim().to_string())
}

// Build a fresh model whose dimensions match the tokenizer vocab.
fn build_llm_from_tokenizer(bpe: crate::tokenizer::BpeTokenizer) -> Llm {
    let vocab = bpe.vocab.clone();

    let embeddings = Embeddings::new(vocab.clone());
    let block1 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);

    // MTB stage: parallel branches inside one logical layer position, now as sequences.
    let block2_1 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_2 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_3 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_4 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_5 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_6 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_7 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_8 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);

    let seq_2_1 =
        TransformerSequence::new(vec![block2_1, block2_2]).expect("transformer_sequence_new_failed");
    let seq_2_2 =
        TransformerSequence::new(vec![block2_3, block2_4]).expect("transformer_sequence_new_failed");
    let seq_2_3 =
        TransformerSequence::new(vec![block2_5, block2_6]).expect("transformer_sequence_new_failed");
    let seq_2_4 =
        TransformerSequence::new(vec![block2_7, block2_8]).expect("transformer_sequence_new_failed");

    let parallel_block2 = ParallelBlockGroup::new(vec![
        Box::new(seq_2_1) as Box<dyn Layer>,
        Box::new(seq_2_2) as Box<dyn Layer>,
        Box::new(seq_2_3) as Box<dyn Layer>,
        Box::new(seq_2_4) as Box<dyn Layer>,
    ])
    .expect("parallel_block_group_new_failed");


    let block3_1 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block3_2 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block3_3 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block3_4 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block3_5 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block3_6 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block3_7 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block3_8 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);

    let seq_3_1 =
        TransformerSequence::new(vec![block3_1, block3_2]).expect("transformer_sequence_new_failed");
    
    let seq_3_2 =
        TransformerSequence::new(vec![block3_5, block3_6]).expect("transformer_sequence_new_failed");

    let parallel_block3 = ParallelBlockGroup::new(vec![
        Box::new(seq_3_1) as Box<dyn Layer>,
        Box::new(seq_3_2) as Box<dyn Layer>,
        
    ])
    .expect("parallel_block_group_new_failed");

    let block4 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let out = OutputProjection::new(crate::EMBEDDING_DIM, vocab.words.len());

    let mut llm = Llm::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(block1),
            Box::new(parallel_block2),
            Box::new(parallel_block3),
            Box::new(block4),
            Box::new(out),
        ],
    );

    llm.set_bpe_tokenizer(bpe);
    llm.set_residual_dropout_p(0.1);
    llm.set_training(true);

    let _ = llm.set_sampling_config(0.9, 40, 0.95, 987654321);
    llm
}

fn topology_to_ascii_lines(llm: &mut Llm) -> Vec<String> {
    let mut v_out: Vec<String> = Vec::new();

    v_out.push("=== Model Topology (ASCII) ===".to_string());
    v_out.push(format!(
        "max_seq_len={}, embedding_dim={}, hidden_dim={}",
        crate::MAX_SEQ_LEN,
        crate::EMBEDDING_DIM,
        crate::HIDDEN_DIM
    ));
    v_out.push(format!("total_parameters={}", llm.total_parameters()));
    v_out.push("".to_string());

    for (i_idx, layer) in llm.network.iter_mut().enumerate() {
        let s_t = layer.layer_type().to_string();

        if s_t == "ParallelBlockGroup" {
            let opt_pg = layer.as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
            if let Some(pg) = opt_pg {
                v_out.push(format!(
                    "[{}] ParallelBlockGroup branches={}",
                    i_idx,
                    pg.num_branches()
                ));
                let v_branch_types = pg.branch_layer_types_ascii();
                for (i_b, s_bt) in v_branch_types.iter().enumerate() {
                    v_out.push(format!("  - branch[{}] {}", i_b, s_bt));
                }
                continue;
            }

            v_out.push(format!("[{}] ParallelBlockGroup (downcast_failed)", i_idx));
            continue;
        }

        v_out.push(format!(
            "[{}] {} parameters={}",
            i_idx,
            s_t,
            layer.parameters()
        ));
    }

    v_out
}

fn print_metrics_ascii(llm: &mut Llm) {
    println!();
    println!("=== Metrics (MTB diagnostics) ===");
    llm.run_post_load_mtb_diagnostics_ascii();
}

#[derive(Clone, Debug)]
struct predict_metrics_ascii {
    d_duration_ms: f64,
    i_generated_tokens: usize,
    d_tokens_per_sec: f64,
    i_input_tokens: usize,
    i_total_tokens: usize,
    i_output_chars: usize,
    d_avg_chars_per_token_out: f64,
    d_output_chars_per_sec: f64,
    d_effective_context_utilization: f64,
    d_avg_selected_token_prob: f64,
    d_perplexity_selected: f64,
    d_avg_next_token_entropy_nat: f64,
    d_avg_top1_top2_margin: f64,
    i_pred_stats_steps: usize,
}

#[derive(Clone, Debug)]
struct training_metrics_snapshot_ascii {
    b_running: bool,
    b_cancel_requested: bool,
    s_phase: String,
    i_epoch_current: usize,
    i_epochs_total: usize,
    d_last_epoch_loss: f32,
    d_last_step_loss: f32,
    i_rows_used_last_epoch: usize,
    i_total_steps: usize,
    s_last_error: String,

    i_skips_empty_act: usize,
    i_skips_empty_logits: usize,
    i_skips_pg_downcast_failed: usize,
    i_skips_pg_no_branches: usize,

    // ---- New metrics fields (mirrors TrainingProgressEventAscii) ----

    // 1) Ingestion
    d_ingest_rows_per_sec_window: f32,
    d_ingest_events_per_sec_window: f32,
    i_ingest_rows_added_total: usize,
    i_ingest_events_processed_total: usize,
    i_ingest_parse_errors_total: usize,
    i_ingest_rows_rejected_total: usize,
    i_ingest_pending_events_observed_peak: usize,

    // 2) Coverage
    d_coverage_ratio_used_over_available: f32,
    d_new_data_ratio_in_available: f32,
    i_new_rows_added_during_epoch: usize,
    i_epoch_token_rows_start: usize,
    i_epoch_token_rows_end: usize,

    // 3) Mask stats
    d_active_branches_mean: f32,
    d_active_branches_std: f32,
    i_active_branches_min: usize,
    i_active_branches_max: usize,
    d_mask_sparsity_mean: f32,
    d_mask_sparsity_std: f32,
    d_steps_at_min_active_share: f32,

    // 4) Scaling proxy
    d_grad_norm_ratio_scaled_over_unscaled_mean: f32,
    d_grad_norm_ratio_scaled_over_unscaled_std: f32,
    d_grad_norm_scaled_mean: f32,
    d_grad_norm_unscaled_mean: f32,

    // 5) Replay
    d_replay_share: f32,
    d_replay_p_last: f32,
    d_replay_delta_loss_mean: f32,
    d_replay_delta_loss_std: f32,

    // 6) Retention
    d_loss_control_old: f32,
    d_loss_control_new: f32,
    d_retention_delta_old: f32,
    d_retention_delta_new: f32,

    // 7) Fairness
    d_branch_select_gini: f32,
    d_branch_select_top1_share: f32,

    // 8) Snapshot (training side only in this patch)
    i_snapshots_sent_total: usize,

    // 9) Expansion telemetry
    i_expansion_events_total: usize,
    i_branches_before_last_expand: usize,
    i_branches_after_last_expand: usize,
    d_eta_injection_last: f32,
    d_sum_w_new_last: f32,

    // 10) Drift
    d_expand_drift_logits_l2_mean: f32,
    d_expand_drift_logits_l2_std: f32,
    d_expand_drift_logits_cos_dist_mean: f32,
    d_expand_drift_logits_cos_dist_std: f32,

    // EMA
    b_ema_active: bool,
    i_ema_last_selected_branch: isize,
}

impl training_metrics_snapshot_ascii {
    fn new_idle() -> Self {
        Self {
            b_running: false,
            b_cancel_requested: false,
            s_phase: "idle".to_string(),
            i_epoch_current: 0,
            i_epochs_total: 0,
            d_last_epoch_loss: 0.0,
            d_last_step_loss: 0.0,
            i_rows_used_last_epoch: 0,
            i_total_steps: 0,
            s_last_error: "".to_string(),

            i_skips_empty_act: 0,
            i_skips_empty_logits: 0,
            i_skips_pg_downcast_failed: 0,
            i_skips_pg_no_branches: 0,

            d_ingest_rows_per_sec_window: 0.0,
            d_ingest_events_per_sec_window: 0.0,
            i_ingest_rows_added_total: 0,
            i_ingest_events_processed_total: 0,
            i_ingest_parse_errors_total: 0,
            i_ingest_rows_rejected_total: 0,
            i_ingest_pending_events_observed_peak: 0,

            d_coverage_ratio_used_over_available: 0.0,
            d_new_data_ratio_in_available: 0.0,
            i_new_rows_added_during_epoch: 0,
            i_epoch_token_rows_start: 0,
            i_epoch_token_rows_end: 0,

            d_active_branches_mean: 0.0,
            d_active_branches_std: 0.0,
            i_active_branches_min: 0,
            i_active_branches_max: 0,
            d_mask_sparsity_mean: 0.0,
            d_mask_sparsity_std: 0.0,
            d_steps_at_min_active_share: 0.0,

            d_grad_norm_ratio_scaled_over_unscaled_mean: 0.0,
            d_grad_norm_ratio_scaled_over_unscaled_std: 0.0,
            d_grad_norm_scaled_mean: 0.0,
            d_grad_norm_unscaled_mean: 0.0,

            d_replay_share: 0.0,
            d_replay_p_last: 0.0,
            d_replay_delta_loss_mean: 0.0,
            d_replay_delta_loss_std: 0.0,

            d_loss_control_old: 0.0,
            d_loss_control_new: 0.0,
            d_retention_delta_old: 0.0,
            d_retention_delta_new: 0.0,

            d_branch_select_gini: 0.0,
            d_branch_select_top1_share: 0.0,

            i_snapshots_sent_total: 0,

            i_expansion_events_total: 0,
            i_branches_before_last_expand: 0,
            i_branches_after_last_expand: 0,
            d_eta_injection_last: 0.0,
            d_sum_w_new_last: 0.0,

            d_expand_drift_logits_l2_mean: 0.0,
            d_expand_drift_logits_l2_std: 0.0,
            d_expand_drift_logits_cos_dist_mean: 0.0,
            d_expand_drift_logits_cos_dist_std: 0.0,

            b_ema_active: false,
            i_ema_last_selected_branch: -1,
        }
    }
}

fn clamp_f64(d_x: f64, d_min: f64, d_max: f64) -> f64 {
    if !d_x.is_finite() {
        return d_min;
    }
    if d_x < d_min {
        d_min
    } else if d_x > d_max {
        d_max
    } else {
        d_x
    }
}

fn compute_predict_metrics_ascii(
    llm: &Llm,
    s_prompt: &str,
    s_output: &str,
    d_duration_ms: f64,
    opt_stats: Option<&PredictStats>,
) -> predict_metrics_ascii {
    let i_input_tokens = llm.tokenize(s_prompt).map(|v| v.len()).unwrap_or(0);
    let i_output_tokens = llm.tokenize(s_output).map(|v| v.len()).unwrap_or(0);
    let i_total_tokens = i_input_tokens.saturating_add(i_output_tokens);

    let d_sec = (d_duration_ms / 1000.0).max(1e-9);
    let d_tokens_per_sec = (i_output_tokens as f64) / d_sec;

    let i_output_chars = s_output.len();
    let d_avg_chars_per_token_out = if i_output_tokens == 0 {
        0.0
    } else {
        (i_output_chars as f64) / (i_output_tokens as f64)
    };
    let d_output_chars_per_sec = (i_output_chars as f64) / d_sec;

    let d_effective_context_utilization =
        (i_total_tokens as f64) / (crate::MAX_SEQ_LEN as f64).max(1.0);

    let (d_avg_p, d_ppl, d_h, d_margin, i_steps) = match opt_stats {
        Some(st) => (
            st.d_avg_selected_token_prob as f64,
            st.d_perplexity_selected as f64,
            st.d_avg_next_token_entropy_nat as f64,
            st.d_avg_top1_top2_margin as f64,
            st.i_steps,
        ),
        None => (0.0, 0.0, 0.0, 0.0, 0),
    };

    predict_metrics_ascii {
        d_duration_ms: clamp_f64(d_duration_ms, 0.0, 1.0e12),
        i_generated_tokens: i_output_tokens,
        d_tokens_per_sec: clamp_f64(d_tokens_per_sec, 0.0, 1.0e12),

        i_input_tokens,
        i_total_tokens,
        i_output_chars,
        d_avg_chars_per_token_out: clamp_f64(d_avg_chars_per_token_out, 0.0, 1.0e9),
        d_output_chars_per_sec: clamp_f64(d_output_chars_per_sec, 0.0, 1.0e12),
        d_effective_context_utilization: clamp_f64(d_effective_context_utilization, 0.0, 1.0),

        d_avg_selected_token_prob: clamp_f64(d_avg_p, 0.0, 1.0),
        d_perplexity_selected: clamp_f64(d_ppl, 0.0, 1.0e12),
        d_avg_next_token_entropy_nat: clamp_f64(d_h, 0.0, 1.0e12),
        d_avg_top1_top2_margin: clamp_f64(d_margin, 0.0, 1.0),
        i_pred_stats_steps: i_steps,
    }
}

fn print_predict_metrics_ascii(m: &predict_metrics_ascii) {
    println!();
    println!("=== Predict Metrics ===");
    println!("duration_ms: {:.3}", m.d_duration_ms);
    println!("generated_tokens: {}", m.i_generated_tokens);
    println!("tokens_per_sec: {:.3}", m.d_tokens_per_sec);

    println!("input_tokens: {}", m.i_input_tokens);
    println!("total_tokens: {}", m.i_total_tokens);
    println!("output_chars: {}", m.i_output_chars);
    println!("avg_chars_per_token_out: {:.3}", m.d_avg_chars_per_token_out);
    println!("output_chars_per_sec: {:.3}", m.d_output_chars_per_sec);
    println!(
        "effective_context_utilization: {:.6}",
        m.d_effective_context_utilization
    );

    println!("avg_selected_token_prob: {:.6}", m.d_avg_selected_token_prob);
    println!("perplexity_selected: {:.6}", m.d_perplexity_selected);
    println!(
        "avg_next_token_entropy_nat: {:.6}",
        m.d_avg_next_token_entropy_nat
    );
    println!("avg_top1_top2_margin: {:.6}", m.d_avg_top1_top2_margin);
    println!("pred_stats_steps: {}", m.i_pred_stats_steps);
}

fn drain_training_progress_non_blocking(
    opt_rx: &mut Option<std::sync::mpsc::Receiver<crate::layer::TrainingProgressEventAscii>>,
    metrics_shared: &std::sync::Arc<std::sync::Mutex<training_metrics_snapshot_ascii>>,
    b_cancel_train: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::mpsc;

    let rx = match opt_rx.as_mut() {
        Some(r) => r,
        None => return,
    };

    loop {
        let ev = match rx.try_recv() {
            Ok(v) => v,
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                *opt_rx = None;
                break;
            }
        };

        let mut m = match metrics_shared.lock() {
            Ok(g) => g,
            Err(_) => return,
        };

        m.b_running = true;
        m.b_cancel_requested = b_cancel_train.load(std::sync::atomic::Ordering::SeqCst);
        m.s_phase = ev.s_phase;
        m.i_epoch_current = ev.i_epoch_current;
        m.i_epochs_total = ev.i_epochs_total;
        m.d_last_epoch_loss = ev.d_last_epoch_loss;
        m.d_last_step_loss = ev.d_last_step_loss;
        m.i_rows_used_last_epoch = ev.i_rows_used_last_epoch;
        m.i_total_steps = ev.i_total_steps;

        m.i_skips_empty_act = ev.i_skips_empty_act;
        m.i_skips_empty_logits = ev.i_skips_empty_logits;
        m.i_skips_pg_downcast_failed = ev.i_skips_pg_downcast_failed;
        m.i_skips_pg_no_branches = ev.i_skips_pg_no_branches;

        // New metrics copy.
        m.d_ingest_rows_per_sec_window = ev.d_ingest_rows_per_sec_window;
        m.d_ingest_events_per_sec_window = ev.d_ingest_events_per_sec_window;
        m.i_ingest_rows_added_total = ev.i_ingest_rows_added_total;
        m.i_ingest_events_processed_total = ev.i_ingest_events_processed_total;
        m.i_ingest_parse_errors_total = ev.i_ingest_parse_errors_total;
        m.i_ingest_rows_rejected_total = ev.i_ingest_rows_rejected_total;
        m.i_ingest_pending_events_observed_peak = ev.i_ingest_pending_events_observed_peak;

        m.d_coverage_ratio_used_over_available = ev.d_coverage_ratio_used_over_available;
        m.d_new_data_ratio_in_available = ev.d_new_data_ratio_in_available;
        m.i_new_rows_added_during_epoch = ev.i_new_rows_added_during_epoch;
        m.i_epoch_token_rows_start = ev.i_epoch_token_rows_start;
        m.i_epoch_token_rows_end = ev.i_epoch_token_rows_end;

        m.d_active_branches_mean = ev.d_active_branches_mean;
        m.d_active_branches_std = ev.d_active_branches_std;
        m.i_active_branches_min = ev.i_active_branches_min;
        m.i_active_branches_max = ev.i_active_branches_max;
        m.d_mask_sparsity_mean = ev.d_mask_sparsity_mean;
        m.d_mask_sparsity_std = ev.d_mask_sparsity_std;
        m.d_steps_at_min_active_share = ev.d_steps_at_min_active_share;

        m.d_grad_norm_ratio_scaled_over_unscaled_mean = ev.d_grad_norm_ratio_scaled_over_unscaled_mean;
        m.d_grad_norm_ratio_scaled_over_unscaled_std = ev.d_grad_norm_ratio_scaled_over_unscaled_std;
        m.d_grad_norm_scaled_mean = ev.d_grad_norm_scaled_mean;
        m.d_grad_norm_unscaled_mean = ev.d_grad_norm_unscaled_mean;

        m.d_replay_share = ev.d_replay_share;
        m.d_replay_p_last = ev.d_replay_p_last;
        m.d_replay_delta_loss_mean = ev.d_replay_delta_loss_mean;
        m.d_replay_delta_loss_std = ev.d_replay_delta_loss_std;

        m.d_loss_control_old = ev.d_loss_control_old;
        m.d_loss_control_new = ev.d_loss_control_new;
        m.d_retention_delta_old = ev.d_retention_delta_old;
        m.d_retention_delta_new = ev.d_retention_delta_new;

        m.d_branch_select_gini = ev.d_branch_select_gini;
        m.d_branch_select_top1_share = ev.d_branch_select_top1_share;

        m.i_snapshots_sent_total = ev.i_snapshots_sent_total;

        m.i_expansion_events_total = ev.i_expansion_events_total;
        m.i_branches_before_last_expand = ev.i_branches_before_last_expand;
        m.i_branches_after_last_expand = ev.i_branches_after_last_expand;
        m.d_eta_injection_last = ev.d_eta_injection_last;
        m.d_sum_w_new_last = ev.d_sum_w_new_last;

        m.d_expand_drift_logits_l2_mean = ev.d_expand_drift_logits_l2_mean;
        m.d_expand_drift_logits_l2_std = ev.d_expand_drift_logits_l2_std;
        m.d_expand_drift_logits_cos_dist_mean = ev.d_expand_drift_logits_cos_dist_mean;
        m.d_expand_drift_logits_cos_dist_std = ev.d_expand_drift_logits_cos_dist_std;

        m.b_ema_active = ev.b_ema_active;
        m.i_ema_last_selected_branch = ev.i_ema_last_selected_branch;
    }
}
fn print_training_metrics_snapshot_ascii(m: &training_metrics_snapshot_ascii) {
    // Description: Print training metrics snapshot with inline evaluation against heuristic
    //              thresholds, including short justifications appended per metric line.
    //              Color rule: WARN in yellow, OK in green (ANSI). BAD stays red for clarity.
    //
    // History:
    // - 2026-02-13: Add background training metrics printer.
    // - 2026-02-15: Extend printer with continuous learning metrics fields.
    // - 2026-02-15: Add threshold based evaluation text appended to each metric output line.
    // - 2026-02-15: Add ANSI color output: OK green, WARN yellow (and BAD red).
    //
    // Author: Marcus Schlieper (ExpChat.ai)

    // ANSI color helpers (ASCII only).
    const S_ANSI_RESET: &str = "\x1b[0m";
    const S_ANSI_GREEN: &str = "\x1b[32m";
    const S_ANSI_YELLOW: &str = "\x1b[33m";
    const S_ANSI_RED: &str = "\x1b[31m";

    fn s_colorize_eval_ascii(s_eval: &str) -> String {
        // Colorize by prefix: "EVAL: ok", "EVAL: warn", "EVAL: bad".
        let s_lc = s_eval.to_ascii_lowercase();
        if s_lc.starts_with("eval: ok") {
            format!("{}{}{}", S_ANSI_GREEN, s_eval, S_ANSI_RESET)
        } else if s_lc.starts_with("eval: warn") {
            format!("{}{}{}", S_ANSI_YELLOW, s_eval, S_ANSI_RESET)
        } else if s_lc.starts_with("eval: bad") {
            format!("{}{}{}", S_ANSI_RED, s_eval, S_ANSI_RESET)
        } else {
            s_eval.to_string()
        }
    }

    fn fmt_bool_ascii(b_v: bool) -> &'static str {
        if b_v { "true" } else { "false" }
    }

    fn eval_loss_ascii(d_loss: f32) -> &'static str {
        if !d_loss.is_finite() {
            "EVAL: bad (non_finite)"
        } else if d_loss > 8.0 {
            "EVAL: bad (very_high_loss, likely config_or_data_issue)"
        } else if d_loss > 4.0 {
            "EVAL: warn (high_loss, early_training_or_underfit)"
        } else if d_loss > 2.0 {
            "EVAL: ok (moderate_loss, improving_expected)"
        } else {
            "EVAL: ok (low_loss, watch_overfit_signals)"
        }
    }

    fn eval_rows_used_ascii(i_used: usize, i_avail: usize) -> String {
        if i_avail == 0 {
            return "EVAL: warn (no_available_rows)".to_string();
        }
        let d_ratio = (i_used as f32) / (i_avail as f32).max(1.0);
        if !d_ratio.is_finite() {
            "EVAL: bad (ratio_non_finite)".to_string()
        } else if d_ratio < 0.50 {
            format!(
                "EVAL: bad (used_over_avail={:.3}, very_low_coverage_or_many_skips)",
                d_ratio
            )
        } else if d_ratio < 0.90 {
            format!(
                "EVAL: warn (used_over_avail={:.3}, investigate_skips_and_data_quality)",
                d_ratio
            )
        } else {
            format!("EVAL: ok (used_over_avail={:.3}, high_coverage)", d_ratio)
        }
    }

    fn eval_skip_counter_zero_ascii(i_v: usize, s_name: &str) -> String {
        if i_v == 0 {
            format!("EVAL: ok ({}=0)", s_name)
        } else {
            format!("EVAL: bad ({}>0, correctness_warning)", s_name)
        }
    }

    fn eval_nonneg_rate_ascii(d_v: f32, s_name: &str) -> String {
        if !d_v.is_finite() {
            format!("EVAL: bad ({}_non_finite)", s_name)
        } else if d_v < 0.0 {
            format!("EVAL: bad ({}_negative)", s_name)
        } else if d_v == 0.0 {
            format!("EVAL: warn ({}_zero, may_be_idle_or_stalled)", s_name)
        } else {
            format!("EVAL: ok ({}_positive)", s_name)
        }
    }

    fn eval_parse_errors_ascii(i_err: usize) -> String {
        if i_err == 0 {
            "EVAL: ok (parse_errors=0)".to_string()
        } else if i_err <= 10 {
            "EVAL: warn (parse_errors_low, occasional_bad_rows_possible)".to_string()
        } else {
            "EVAL: bad (parse_errors_high, systematic_format_or_tokenizer_issue)".to_string()
        }
    }

    fn eval_reject_ratio_ascii(i_rej: usize, i_add: usize) -> String {
        let i_den = i_add.saturating_add(i_rej);
        if i_den == 0 {
            return "EVAL: warn (no_ingested_rows_observed)".to_string();
        }
        let d_r = (i_rej as f32) / (i_den as f32).max(1.0);
        if !d_r.is_finite() {
            "EVAL: bad (reject_ratio_non_finite)".to_string()
        } else if d_r < 0.05 {
            format!("EVAL: ok (reject_ratio={:.3}, healthy)", d_r)
        } else if d_r <= 0.20 {
            format!(
                "EVAL: warn (reject_ratio={:.3}, data_cleanliness_or_budget_issue)",
                d_r
            )
        } else {
            format!("EVAL: bad (reject_ratio={:.3}, pipeline_problem_likely)", d_r)
        }
    }

    fn eval_pending_peak_ascii(i_peak: usize) -> String {
        if i_peak <= 2 {
            format!("EVAL: ok (pending_peak={}, drains_keep_up)", i_peak)
        } else if i_peak <= 10 {
            format!("EVAL: warn (pending_peak={}, mild_backlog)", i_peak)
        } else {
            format!("EVAL: warn (pending_peak={}, sustained_backlog_likely)", i_peak)
        }
    }

    fn eval_ratio_0_1_ascii(d_v: f32, s_name: &str) -> String {
        if !d_v.is_finite() {
            format!("EVAL: bad ({}_non_finite)", s_name)
        } else if d_v < 0.0 || d_v > 1.0 {
            format!("EVAL: warn ({}_out_of_range={:.6})", s_name, d_v)
        } else {
            format!("EVAL: ok ({}_in_range={:.6})", s_name, d_v)
        }
    }

    fn eval_coverage_ratio_ascii(d_cov: f32) -> &'static str {
        if !d_cov.is_finite() {
            "EVAL: bad (coverage_non_finite)"
        } else if d_cov < 0.50 {
            "EVAL: bad (very_low_coverage, skips_or_stop)"
        } else if d_cov < 0.90 {
            "EVAL: warn (partial_coverage, acceptable_if_intended)"
        } else {
            "EVAL: ok (high_coverage)"
        }
    }

    fn eval_new_data_ratio_ascii(d_r: f32) -> &'static str {
        if !d_r.is_finite() {
            "EVAL: bad (new_data_ratio_non_finite)"
        } else if d_r < 0.01 {
            "EVAL: ok (near_static_pool_or_low_ingest)"
        } else if d_r <= 0.10 {
            "EVAL: ok (incremental_online_learning)"
        } else {
            "EVAL: warn (strong_nonstationarity, watch_retention)"
        }
    }

    fn eval_mask_sparsity_ascii(d_s: f32) -> &'static str {
        if !d_s.is_finite() {
            "EVAL: bad (mask_sparsity_non_finite)"
        } else if d_s > 0.70 {
            "EVAL: warn (highly_sparse, higher_variance_expected)"
        } else if d_s >= 0.30 {
            "EVAL: ok (moderate_sparsity)"
        } else {
            "EVAL: ok (dense_participation)"
        }
    }

    fn eval_steps_at_min_active_share_ascii(d_share: f32) -> &'static str {
        if !d_share.is_finite() {
            "EVAL: bad (min_active_share_non_finite)"
        } else if d_share > 0.60 {
            "EVAL: warn (frequent_clamping, p_i_too_low_likely)"
        } else if d_share >= 0.20 {
            "EVAL: ok (moderate_clamping)"
        } else {
            "EVAL: ok (rare_clamping)"
        }
    }

    fn eval_grad_norm_ratio_ascii(d_ratio: f32) -> &'static str {
        if !d_ratio.is_finite() {
            "EVAL: bad (grad_ratio_non_finite)"
        } else if d_ratio < 1.0 {
            "EVAL: ok (ratio_lt_1, proxy_or_clipping_effect)"
        } else if d_ratio <= 2.0 {
            "EVAL: ok (moderate_amplification)"
        } else {
            "EVAL: warn (strong_amplification, may_need_lower_lr_or_tighter_clip)"
        }
    }

    fn eval_replay_share_ascii(d_share: f32) -> &'static str {
        if !d_share.is_finite() {
            "EVAL: bad (replay_share_non_finite)"
        } else if d_share <= 0.05 {
            "EVAL: warn (minimal_replay, forgetting_risk_if_nonstationary)"
        } else if d_share <= 0.25 {
            "EVAL: ok (moderate_replay)"
        } else {
            "EVAL: warn (heavy_replay, may_slow_adaptation)"
        }
    }

    fn eval_replay_delta_loss_ascii(d_mean: f32) -> &'static str {
        if !d_mean.is_finite() {
            "EVAL: bad (replay_delta_loss_non_finite)"
        } else if d_mean > 0.50 {
            "EVAL: warn (strong_mismatch, model_drifting_from_old_data)"
        } else if d_mean < -0.50 {
            "EVAL: warn (replay_too_easy_or_overfit_old)"
        } else {
            "EVAL: ok (delta_loss_moderate)"
        }
    }

    fn eval_retention_delta_ascii(d_delta: f32, s_name: &str) -> String {
        if !d_delta.is_finite() {
            return format!("EVAL: bad ({}_non_finite)", s_name);
        }
        let d_abs = d_delta.abs();
        if d_abs <= 0.10 {
            format!("EVAL: ok ({}={:.6}, stable)", s_name, d_delta)
        } else if d_abs <= 0.30 {
            format!("EVAL: warn ({}={:.6}, mild_shift)", s_name, d_delta)
        } else {
            format!("EVAL: warn ({}={:.6}, substantial_shift)", s_name, d_delta)
        }
    }

    fn eval_gini_ascii(d_g: f32, s_name: &str) -> String {
        if !d_g.is_finite() {
            return format!("EVAL: bad ({}_non_finite)", s_name);
        }
        if d_g < 0.10 {
            format!("EVAL: ok ({}={:.6}, near_uniform)", s_name, d_g)
        } else if d_g <= 0.30 {
            format!("EVAL: warn ({}={:.6}, moderate_concentration)", s_name, d_g)
        } else {
            format!("EVAL: warn ({}={:.6}, strong_dominance)", s_name, d_g)
        }
    }

    fn eval_top1_share_ascii(d_s: f32, s_name: &str) -> String {
        if !d_s.is_finite() {
            return format!("EVAL: bad ({}_non_finite)", s_name);
        }
        if d_s < 0.40 {
            format!("EVAL: ok ({}={:.6}, healthy_diversity)", s_name, d_s)
        } else if d_s <= 0.60 {
            format!("EVAL: warn ({}={:.6}, moderate_dominance)", s_name, d_s)
        } else {
            format!("EVAL: warn ({}={:.6}, severe_dominance)", s_name, d_s)
        }
    }

    fn eval_eta_ascii(d_eta: f32) -> &'static str {
        if !d_eta.is_finite() {
            "EVAL: bad (eta_non_finite)"
        } else if d_eta <= 0.05 {
            "EVAL: ok (very_conservative, new_branches_learn_slow)"
        } else if d_eta <= 0.20 {
            "EVAL: ok (balanced_injection)"
        } else {
            "EVAL: warn (aggressive_injection, drift_risk)"
        }
    }

    fn eval_drift_l2_ascii(d_l2: f32) -> &'static str {
        if !d_l2.is_finite() {
            "EVAL: bad (drift_l2_non_finite)"
        } else if d_l2 < 0.10 {
            "EVAL: ok (very_stable)"
        } else if d_l2 <= 0.50 {
            "EVAL: ok (moderate_drift)"
        } else {
            "EVAL: warn (large_change, reduce_eta_or_frequency)"
        }
    }

    fn eval_drift_cos_ascii(d_cd: f32) -> &'static str {
        if !d_cd.is_finite() {
            "EVAL: bad (drift_cos_non_finite)"
        } else if d_cd < 0.05 {
            "EVAL: ok (very_stable_direction)"
        } else if d_cd <= 0.20 {
            "EVAL: ok (moderate_direction_change)"
        } else {
            "EVAL: warn (large_direction_change)"
        }
    }

    println!();
    println!("=== Training Metrics ===");

    // Core flags.
    println!(
        "running: {}  {}",
        fmt_bool_ascii(m.b_running),
        s_colorize_eval_ascii(if m.b_running { "EVAL: ok (active)" } else { "EVAL: ok (inactive)" })
    );
    println!(
        "cancel_requested: {}  {}",
        fmt_bool_ascii(m.b_cancel_requested),
        s_colorize_eval_ascii(if m.b_cancel_requested {
            "EVAL: warn (cooperative_stop_pending_or_done)"
        } else {
            "EVAL: ok (no_cancel)"
        })
    );
    println!(
        "phase: {}  {}",
        m.s_phase,
        s_colorize_eval_ascii(if m.s_phase == "error" {
            "EVAL: bad (phase_error)"
        } else if m.s_phase == "canceled" || m.s_phase == "cancel_requested" {
            "EVAL: warn (phase_cancel)"
        } else if m.s_phase == "idle" {
            "EVAL: ok (idle)"
        } else {
            "EVAL: ok (running_phase)"
        })
    );

    // Progress.
    println!(
        "epoch: {} / {}  {}",
        m.i_epoch_current,
        m.i_epochs_total,
        s_colorize_eval_ascii(if m.i_epochs_total == 0 {
            "EVAL: warn (epochs_total_zero)"
        } else if m.i_epoch_current > m.i_epochs_total {
            "EVAL: warn (epoch_oob)"
        } else {
            "EVAL: ok (epoch_in_range)"
        })
    );
    println!(
        "last_epoch_loss: {:.6}  {}",
        m.d_last_epoch_loss,
        s_colorize_eval_ascii(eval_loss_ascii(m.d_last_epoch_loss))
    );
    println!(
        "last_step_loss: {:.6}  {}",
        m.d_last_step_loss,
        s_colorize_eval_ascii(eval_loss_ascii(m.d_last_step_loss))
    );

    // This uses epoch_token_rows_start as best available-rows proxy.
    let i_avail_rows_proxy = m.i_epoch_token_rows_start;
    println!(
        "rows_used_last_epoch: {}  {}",
        m.i_rows_used_last_epoch,
        s_colorize_eval_ascii(&eval_rows_used_ascii(m.i_rows_used_last_epoch, i_avail_rows_proxy))
    );
    println!(
        "total_steps: {}  {}",
        m.i_total_steps,
        s_colorize_eval_ascii(if m.i_total_steps == 0 {
            "EVAL: warn (no_steps_yet)"
        } else {
            "EVAL: ok (steps_positive)"
        })
    );

    // Skip counters.
    println!(
        "skips_empty_act: {}  {}",
        m.i_skips_empty_act,
        s_colorize_eval_ascii(&eval_skip_counter_zero_ascii(m.i_skips_empty_act, "skips_empty_act"))
    );
    println!(
        "skips_empty_logits: {}  {}",
        m.i_skips_empty_logits,
        s_colorize_eval_ascii(&eval_skip_counter_zero_ascii(
            m.i_skips_empty_logits,
            "skips_empty_logits"
        ))
    );
    println!(
        "skips_pg_downcast_failed: {}  {}",
        m.i_skips_pg_downcast_failed,
        s_colorize_eval_ascii(&eval_skip_counter_zero_ascii(
            m.i_skips_pg_downcast_failed,
            "skips_pg_downcast_failed"
        ))
    );
    println!(
        "skips_pg_no_branches: {}  {}",
        m.i_skips_pg_no_branches,
        s_colorize_eval_ascii(&eval_skip_counter_zero_ascii(
            m.i_skips_pg_no_branches,
            "skips_pg_no_branches"
        ))
    );

    println!();
    println!("--- Continuous Learning Metrics ---");

    // (1) Ingestion.
    println!(
        "ingest_rows_per_sec_window: {:.3}  {}",
        m.d_ingest_rows_per_sec_window,
        s_colorize_eval_ascii(&eval_nonneg_rate_ascii(
            m.d_ingest_rows_per_sec_window,
            "ingest_rows_per_sec_window"
        ))
    );
    println!(
        "ingest_events_per_sec_window: {:.3}  {}",
        m.d_ingest_events_per_sec_window,
        s_colorize_eval_ascii(&eval_nonneg_rate_ascii(
            m.d_ingest_events_per_sec_window,
            "ingest_events_per_sec_window"
        ))
    );
    println!(
        "ingest_rows_added_total: {}  {}",
        m.i_ingest_rows_added_total,
        s_colorize_eval_ascii(if m.i_ingest_rows_added_total == 0 {
            "EVAL: ok (no_online_rows_added_yet)"
        } else {
            "EVAL: ok (monotone_counter_positive)"
        })
    );
    println!(
        "ingest_events_processed_total: {}  {}",
        m.i_ingest_events_processed_total,
        s_colorize_eval_ascii(if m.i_ingest_events_processed_total == 0 {
            "EVAL: warn (no_events_processed_yet)"
        } else {
            "EVAL: ok (events_processed_positive)"
        })
    );
    println!(
        "ingest_parse_errors_total: {}  {}",
        m.i_ingest_parse_errors_total,
        s_colorize_eval_ascii(&eval_parse_errors_ascii(m.i_ingest_parse_errors_total))
    );
    println!(
        "ingest_rows_rejected_total: {}  {}",
        m.i_ingest_rows_rejected_total,
        s_colorize_eval_ascii(&eval_reject_ratio_ascii(
            m.i_ingest_rows_rejected_total,
            m.i_ingest_rows_added_total
        ))
    );
    println!(
        "ingest_pending_events_observed_peak: {}  {}",
        m.i_ingest_pending_events_observed_peak,
        s_colorize_eval_ascii(&eval_pending_peak_ascii(m.i_ingest_pending_events_observed_peak))
    );

    // (2) Coverage.
    println!(
        "coverage_ratio_used_over_available: {:.6}  {}",
        m.d_coverage_ratio_used_over_available,
        s_colorize_eval_ascii(eval_coverage_ratio_ascii(m.d_coverage_ratio_used_over_available))
    );
    println!(
        "new_data_ratio_in_available: {:.6}  {}",
        m.d_new_data_ratio_in_available,
        s_colorize_eval_ascii(eval_new_data_ratio_ascii(m.d_new_data_ratio_in_available))
    );
    println!(
        "new_rows_added_during_epoch: {}  {}",
        m.i_new_rows_added_during_epoch,
        s_colorize_eval_ascii(if m.i_new_rows_added_during_epoch == 0 {
            "EVAL: ok (no_new_rows_this_epoch)"
        } else {
            "EVAL: ok (online_data_present)"
        })
    );
    println!(
        "epoch_token_rows_start: {}  {}",
        m.i_epoch_token_rows_start,
        s_colorize_eval_ascii(if m.i_epoch_token_rows_start == 0 {
            "EVAL: warn (no_rows_at_epoch_start)"
        } else {
            "EVAL: ok (rows_available)"
        })
    );
    println!(
        "epoch_token_rows_end: {}  {}",
        m.i_epoch_token_rows_end,
        s_colorize_eval_ascii(if m.i_epoch_token_rows_end < m.i_epoch_token_rows_start {
            "EVAL: warn (end_lt_start_unexpected)"
        } else if m.i_epoch_token_rows_end == 0 {
            "EVAL: warn (no_rows_at_epoch_end)"
        } else {
            "EVAL: ok (nondecreasing_pool)"
        })
    );

    // (3) Mask stats.
    println!(
        "active_branches_mean: {:.6}  {}",
        m.d_active_branches_mean,
        s_colorize_eval_ascii(if !m.d_active_branches_mean.is_finite() {
            "EVAL: bad (active_mean_non_finite)"
        } else if m.d_active_branches_mean <= 0.0 {
            "EVAL: warn (active_mean_zero_or_negative)"
        } else {
            "EVAL: ok (active_mean_positive)"
        })
    );
    println!(
        "active_branches_std: {:.6}  {}",
        m.d_active_branches_std,
        s_colorize_eval_ascii(if !m.d_active_branches_std.is_finite() {
            "EVAL: bad (active_std_non_finite)"
        } else if m.d_active_branches_std == 0.0 {
            "EVAL: ok (std_zero, stable_mask_or_low_samples)"
        } else {
            "EVAL: ok (std_positive)"
        })
    );
    println!(
        "active_branches_min: {}  {}",
        m.i_active_branches_min,
        s_colorize_eval_ascii(if m.i_active_branches_min == 0 {
            "EVAL: warn (min_active_zero, mask_or_pg_issue)"
        } else {
            "EVAL: ok (min_active_positive)"
        })
    );
    println!(
        "active_branches_max: {}  {}",
        m.i_active_branches_max,
        s_colorize_eval_ascii(if m.i_active_branches_max < m.i_active_branches_min {
            "EVAL: warn (max_lt_min_unexpected)"
        } else if m.i_active_branches_max == 0 {
            "EVAL: warn (max_active_zero)"
        } else {
            "EVAL: ok (max_active_consistent)"
        })
    );
    println!(
        "mask_sparsity_mean: {:.6}  {}",
        m.d_mask_sparsity_mean,
        s_colorize_eval_ascii(eval_mask_sparsity_ascii(m.d_mask_sparsity_mean))
    );
    println!(
        "mask_sparsity_std: {:.6}  {}",
        m.d_mask_sparsity_std,
        s_colorize_eval_ascii(if !m.d_mask_sparsity_std.is_finite() {
            "EVAL: bad (sparsity_std_non_finite)"
        } else {
            "EVAL: ok (sparsity_std_finite)"
        })
    );
    println!(
        "steps_at_min_active_share: {:.6}  {}",
        m.d_steps_at_min_active_share,
        s_colorize_eval_ascii(eval_steps_at_min_active_share_ascii(
            m.d_steps_at_min_active_share
        ))
    );

    // (4) Scaling proxy.
    println!(
        "grad_norm_ratio_scaled_over_unscaled_mean: {:.6}  {}",
        m.d_grad_norm_ratio_scaled_over_unscaled_mean,
        s_colorize_eval_ascii(eval_grad_norm_ratio_ascii(
            m.d_grad_norm_ratio_scaled_over_unscaled_mean
        ))
    );
    println!(
        "grad_norm_ratio_scaled_over_unscaled_std: {:.6}  {}",
        m.d_grad_norm_ratio_scaled_over_unscaled_std,
        s_colorize_eval_ascii(if !m.d_grad_norm_ratio_scaled_over_unscaled_std.is_finite() {
            "EVAL: bad (grad_ratio_std_non_finite)"
        } else {
            "EVAL: ok (grad_ratio_std_finite)"
        })
    );
    println!(
        "grad_norm_scaled_mean: {:.6}  {}",
        m.d_grad_norm_scaled_mean,
        s_colorize_eval_ascii(if !m.d_grad_norm_scaled_mean.is_finite() {
            "EVAL: bad (grad_norm_scaled_non_finite)"
        } else if m.d_grad_norm_scaled_mean == 0.0 {
            "EVAL: warn (grad_norm_scaled_zero, may_be_no_updates_or_proxy_low)"
        } else {
            "EVAL: ok (grad_norm_scaled_finite)"
        })
    );
    println!(
        "grad_norm_unscaled_mean: {:.6}  {}",
        m.d_grad_norm_unscaled_mean,
        s_colorize_eval_ascii(if !m.d_grad_norm_unscaled_mean.is_finite() {
            "EVAL: bad (grad_norm_unscaled_non_finite)"
        } else if m.d_grad_norm_unscaled_mean == 0.0 {
            "EVAL: warn (grad_norm_unscaled_zero, may_be_no_updates_or_proxy_low)"
        } else {
            "EVAL: ok (grad_norm_unscaled_finite)"
        })
    );

    // (5) Replay.
    println!(
        "replay_share: {:.6}  {}",
        m.d_replay_share,
        s_colorize_eval_ascii(eval_replay_share_ascii(m.d_replay_share))
    );
    println!(
        "replay_p_last: {:.6}  {}",
        m.d_replay_p_last,
        s_colorize_eval_ascii(&eval_ratio_0_1_ascii(m.d_replay_p_last, "replay_p_last"))
    );
    println!(
        "replay_delta_loss_mean: {:.6}  {}",
        m.d_replay_delta_loss_mean,
        s_colorize_eval_ascii(eval_replay_delta_loss_ascii(m.d_replay_delta_loss_mean))
    );
    println!(
        "replay_delta_loss_std: {:.6}  {}",
        m.d_replay_delta_loss_std,
        s_colorize_eval_ascii(if !m.d_replay_delta_loss_std.is_finite() {
            "EVAL: bad (replay_delta_loss_std_non_finite)"
        } else {
            "EVAL: ok (replay_delta_loss_std_finite)"
        })
    );

    // (6) Retention.
    println!(
        "loss_control_old: {:.6}  {}",
        m.d_loss_control_old,
        s_colorize_eval_ascii(eval_loss_ascii(m.d_loss_control_old))
    );
    println!(
        "loss_control_new: {:.6}  {}",
        m.d_loss_control_new,
        s_colorize_eval_ascii(eval_loss_ascii(m.d_loss_control_new))
    );
    println!(
        "retention_delta_old: {:.6}  {}",
        m.d_retention_delta_old,
        s_colorize_eval_ascii(&eval_retention_delta_ascii(
            m.d_retention_delta_old,
            "retention_delta_old"
        ))
    );
    println!(
        "retention_delta_new: {:.6}  {}",
        m.d_retention_delta_new,
        s_colorize_eval_ascii(&eval_retention_delta_ascii(
            m.d_retention_delta_new,
            "retention_delta_new"
        ))
    );

    // (7) Fairness.
    println!(
        "branch_select_gini: {:.6}  {}",
        m.d_branch_select_gini,
        s_colorize_eval_ascii(&eval_gini_ascii(m.d_branch_select_gini, "branch_select_gini"))
    );
    println!(
        "branch_select_top1_share: {:.6}  {}",
        m.d_branch_select_top1_share,
        s_colorize_eval_ascii(&eval_top1_share_ascii(
            m.d_branch_select_top1_share,
            "branch_select_top1_share"
        ))
    );

    // (8) Snapshot counters (training side only in this build).
    println!(
        "snapshots_sent_total: {}  {}",
        m.i_snapshots_sent_total,
        s_colorize_eval_ascii(if m.i_snapshots_sent_total == 0 {
            "EVAL: warn (no_snapshots_sent_yet)"
        } else {
            "EVAL: ok (snapshots_sent_positive)"
        })
    );

    // (9) Expansion telemetry.
    println!(
        "expansion_events_total: {}  {}",
        m.i_expansion_events_total,
        s_colorize_eval_ascii(if m.i_expansion_events_total == 0 {
            "EVAL: ok (no_expansion)"
        } else {
            "EVAL: warn (expansion_occurred, watch_compute_and_drift)"
        })
    );
    println!(
        "branches_before_last_expand: {}  {}",
        m.i_branches_before_last_expand,
        s_colorize_eval_ascii(if m.i_expansion_events_total == 0 {
            "EVAL: ok (n_a_no_expansion)"
        } else {
            "EVAL: ok (reported)"
        })
    );
    println!(
        "branches_after_last_expand: {}  {}",
        m.i_branches_after_last_expand,
        s_colorize_eval_ascii(if m.i_expansion_events_total == 0 {
            "EVAL: ok (n_a_no_expansion)"
        } else if m.i_branches_after_last_expand <= m.i_branches_before_last_expand {
            "EVAL: warn (after_not_gt_before)"
        } else {
            "EVAL: ok (expanded)"
        })
    );
    println!(
        "eta_injection_last: {:.6}  {}",
        m.d_eta_injection_last,
        s_colorize_eval_ascii(eval_eta_ascii(m.d_eta_injection_last))
    );
    println!(
        "sum_w_new_last: {:.6}  {}",
        m.d_sum_w_new_last,
        s_colorize_eval_ascii(if !m.d_sum_w_new_last.is_finite() {
            "EVAL: bad (sum_w_new_non_finite)"
        } else if m.d_sum_w_new_last < 0.0 {
            "EVAL: warn (sum_w_new_negative)"
        } else if m.d_sum_w_new_last == 0.0 && m.i_expansion_events_total > 0 {
            "EVAL: warn (sum_w_new_zero_despite_expansion)"
        } else {
            "EVAL: ok (sum_w_new_finite)"
        })
    );

    // (10) Drift.
    println!(
        "expand_drift_logits_l2_mean: {:.6}  {}",
        m.d_expand_drift_logits_l2_mean,
        s_colorize_eval_ascii(eval_drift_l2_ascii(m.d_expand_drift_logits_l2_mean))
    );
    println!(
        "expand_drift_logits_l2_std: {:.6}  {}",
        m.d_expand_drift_logits_l2_std,
        s_colorize_eval_ascii(if !m.d_expand_drift_logits_l2_std.is_finite() {
            "EVAL: bad (drift_l2_std_non_finite)"
        } else {
            "EVAL: ok (drift_l2_std_finite)"
        })
    );
    println!(
        "expand_drift_logits_cos_dist_mean: {:.6}  {}",
        m.d_expand_drift_logits_cos_dist_mean,
        s_colorize_eval_ascii(eval_drift_cos_ascii(m.d_expand_drift_logits_cos_dist_mean))
    );
    println!(
        "expand_drift_logits_cos_dist_std: {:.6}  {}",
        m.d_expand_drift_logits_cos_dist_std,
        s_colorize_eval_ascii(if !m.d_expand_drift_logits_cos_dist_std.is_finite() {
            "EVAL: bad (drift_cos_std_non_finite)"
        } else {
            "EVAL: ok (drift_cos_std_finite)"
        })
    );

    // EMA.
    println!(
        "ema_active: {}  {}",
        fmt_bool_ascii(m.b_ema_active),
        s_colorize_eval_ascii(if m.b_ema_active {
            "EVAL: ok (ema_routing_active)"
        } else {
            "EVAL: ok (ema_routing_inactive_or_warmup)"
        })
    );
    println!(
        "ema_last_selected_branch: {}  {}",
        m.i_ema_last_selected_branch,
        s_colorize_eval_ascii(if m.i_ema_last_selected_branch < 0 {
            "EVAL: ok (n_a_or_not_selected_yet)"
        } else {
            "EVAL: ok (selected_branch_reported)"
        })
    );

    if !m.s_last_error.is_empty() {
        println!(
            "last_error: {}  {}",
            m.s_last_error,
            s_colorize_eval_ascii("EVAL: bad (last_error_present)")
        );
    }
}


// Help function: explains all menu items and all metrics in ASCII only.
fn print_help_ascii() {
    // Description: Print interactive help text for menu and metrics (ASCII only).
    //              This version extends metric descriptions with threshold guidance and
    //              interpretation hints for expert users.
    //
    // History:
    // - 2026-02-15: Add help function for menu and metrics documentation (ASCII).
    // - 2026-02-15: Extend metrics help with threshold ranges and interpretations.
    //
    // Author: Marcus Schlieper (ExpChat.ai)

    println!();
    println!("=== Help (ASCII) ===");
    println!();

    println!("Menu commands:");
    println!("  t  Train (background, continuous learning)");
    println!("     - Starts background training on llm_train.");
    println!("     - Serving (ask) continues on llm_serve and receives snapshot updates.");
    println!("     - Training uses continuous learning mask logic (partial branch availability),");
    println!("       optional EMA based branch selection, replay, and optional autonomous expansion.");
    println!();

    println!("  b  Training metrics");
    println!("     - Prints last training progress snapshot received from training thread.");
    println!("     - Includes base loss metrics, diagnostic skip counters, and advanced validation metrics.");
    println!();

    println!("  s  Stop training");
    println!("     - Requests cooperative cancellation and joins training thread.");
    println!("     - Also signals shutdown to online ingestion receiver when present.");
    println!();

    println!("  n  Add new training data file (online ingestion)");
    println!("     - Requires running training thread.");
    println!("     - Expects a JSON file containing an array of strings (training examples).");
    println!("     - Data are tokenized and appended to the active training pool (append only).");
    println!();

    println!("  l  Load checkpoint (serve model)");
    println!("     - Loads checkpoint file and rebuilds topology for llm_serve only.");
    println!("     - Training model instance is not replaced by this action.");
    println!();

    println!("  w  Save checkpoint (serve model)");
    println!("     - Saves tokenizer, topology spec, and all parameters from llm_serve.");
    println!();

    println!("  a  Ask (serve model, parallel to training)");
    println!("     - Interactive inference loop on llm_serve.");
    println!("     - Reports prediction metrics (throughput, entropy, margin, perplexity proxy).");
    println!();

    println!("  o  Toggle outage simulation (serve model, test only)");
    println!("     - Enables fault injection in ParallelBlockGroup during predict.");
    println!("     - Drops a randomly chosen branch per predict call for robustness testing.");
    println!();

    println!("  y  Topology (ASCII, serve model)");
    println!("     - Prints layer list and for ParallelBlockGroup also branch layer types.");
    println!();

    println!("  x  Metrics (MTB diagnostics, serve model)");
    println!("     - Runs post load diagnostics on ParallelBlockGroup to quantify path usage and diversity.");
    println!();

    println!("  h  Help");
    println!("     - Prints this help text.");
    println!();

    println!("  e  Exit");
    println!("     - Requests ingestion shutdown, cancels training if running, joins thread, exits program.");
    println!();

    println!("Training metrics (printed by command b):");
    println!("  Base progress fields:");
    println!("    phase: current phase name (e.g. pretraining, instruction_tuning, realtime)");
    println!("    epoch: current epoch and total epochs");
    println!("    last_epoch_loss: mean loss of last completed epoch (or running mean during epoch in step events)");
    println!("      - Interpretation: lower is better, but only comparable within same dataset and vocab.");
    println!("      - Practical thresholds (heuristic):");
    println!("          > 8.0  very high / likely broken tokenization, too high lr, or severe mismatch");
    println!("          4.0-8.0 high / early training or underfit");
    println!("          2.0-4.0 moderate / improving but not yet stable");
    println!("          < 2.0  low / potentially good fit, risk of overfit depends on validation signals");
    println!("    last_step_loss: last observed step loss (cross entropy)");
    println!("      - Interpretation: high variance is normal; use trend, not single value.");
    println!("      - Thresholds (heuristic): spikes > (2x running mean) suggest instability or bad rows.");
    println!("    rows_used_last_epoch: number of training rows that produced a valid update");
    println!("      - Interpretation: should be close to available rows; large gaps indicate skips.");
    println!("      - Thresholds: if used/available < 0.90, investigate skip counters and data quality.");
    println!("    total_steps: number of successful update steps across epochs");
    println!();

    println!("  Diagnostic skip counters:");
    println!("    skips_empty_act: forward produced empty activations, row skipped");
    println!("      - Interpretation: indicates shape or topology errors, or invalid token rows.");
    println!("      - Thresholds: should be 0. Any nonzero value is a correctness warning.");
    println!("    skips_empty_logits: forward produced empty logits, row skipped");
    println!("      - Interpretation: severe forward path failure (e.g. OutputProjection not reached).");
    println!("      - Thresholds: should be 0. Any nonzero value is a correctness warning.");
    println!("    skips_pg_downcast_failed: ParallelBlockGroup downcast failed, row skipped");
    println!("      - Interpretation: type wiring error; training cannot access PG specialized functions.");
    println!("      - Thresholds: should be 0. Any nonzero value is a correctness warning.");
    println!("    skips_pg_no_branches: ParallelBlockGroup had zero branches, row skipped");
    println!("      - Interpretation: invalid model state; expansion or build logic failure.");
    println!("      - Thresholds: should be 0. Any nonzero value is a correctness warning.");
    println!();

    println!("  Advanced validation metrics (continuous learning and expandable width):");
    println!("  (1) Ingestion throughput and queue proxies:");
    println!("    ingest_rows_per_sec_window: windowed rate of accepted rows added to token pool");
    println!("      - Expected range: >= 0.0.");
    println!("      - Thresholds:");
    println!("          0.0 while ingestion events exist: ingestion stalled or drain not called frequently enough");
    println!("          very high spikes: normal for bursty drains; interpret with events_per_sec_window");
    println!("    ingest_events_per_sec_window: windowed rate of processed ingestion events");
    println!("      - Expected range: >= 0.0.");
    println!("      - Thresholds:");
    println!("          near 0.0 with repeated user add actions: receiver not alive or channel blocked");
    println!("    ingest_rows_added_total: total accepted rows added since start of phase");
    println!("      - Interpretation: monotone counter; used to confirm online learning actually extends pool.");
    println!("    ingest_events_processed_total: total processed ingestion events since start of phase");
    println!("      - Interpretation: monotone counter; should increase after each add request.");
    println!("    ingest_parse_errors_total: total parse or tokenization errors during ingestion");
    println!("      - Thresholds:");
    println!("          0: ideal");
    println!("          1-10: likely occasional malformed rows or encoding issues");
    println!("          > 10: systematic format mismatch (not JSON array of strings), invalid UTF-8, or tokenizer failures");
    println!("    ingest_rows_rejected_total: total rows rejected (empty, too short, over budget)");
    println!("      - Thresholds (heuristic):");
    println!("          rejected/added < 0.05: healthy");
    println!("          0.05-0.20: data cleanliness issue or aggressive min length/budget");
    println!("          > 0.20: data pipeline problem; many rows too short or empty");
    println!("    ingest_pending_events_observed_peak: coarse proxy for pending events observed during drains");
    println!("      - Interpretation: not an exact queue length, but a backpressure indicator.");
    println!("      - Thresholds (heuristic):");
    println!("          <= 2: healthy / drains keep up");
    println!("          3-10: mild backlog; consider more frequent drains or smaller events");
    println!("          > 10: sustained backlog likely; ingestion may lag far behind user inputs");
    println!();

    println!("  (2) Coverage ratio (effective data coverage per epoch):");
    println!("    epoch_token_rows_start: token rows available at epoch start (snapshot length)");
    println!("    epoch_token_rows_end: token rows available at epoch end (after ingestion)");
    println!("    new_rows_added_during_epoch: epoch_token_rows_end - epoch_token_rows_start");
    println!("    coverage_ratio_used_over_available: used_rows / epoch_token_rows_start (approx)");
    println!("      - Expected range: [0.0, 1.0] in typical settings.");
    println!("      - Thresholds (heuristic):");
    println!("          < 0.50: very low coverage; training likely dominated by skips or early stop");
    println!("          0.50-0.90: partial coverage; acceptable if replay or sampling is intended");
    println!("          >= 0.90: high coverage; typical for sequential full-pass epochs");
    println!("      - Interpretation: low values reduce effective learning rate per epoch and bias updates.");
    println!("    new_data_ratio_in_available: new_rows_added_during_epoch / epoch_token_rows_end");
    println!("      - Expected range: [0.0, 1.0].");
    println!("      - Thresholds (heuristic):");
    println!("          < 0.01: near static pool; online ingestion not materially affecting epoch");
    println!("          0.01-0.10: incremental online learning");
    println!("          > 0.10: strongly nonstationary data; consider replay and retention monitoring");
    println!();

    println!("  (3) Availability mask statistics (participation and sparsity):");
    println!("    active_branches_mean/std/min/max: statistics of active branches per step");
    println!("      - Expected constraints: min >= min_active, max <= total_branches.");
    println!("      - Thresholds (heuristic):");
    println!("          active_mean close to min_active: very sparse training; increases variance, may slow convergence");
    println!("          active_mean close to total_branches: near full participation; reduces variance, higher compute");
    println!("    mask_sparsity_mean/std: fraction of inactive branches per step");
    println!("      - Expected range: [0.0, 1.0].");
    println!("      - Thresholds (heuristic):");
    println!("          > 0.70: highly sparse; expect stronger need for inverse scaling and replay");
    println!("          0.30-0.70: moderate sparsity; typical for partial availability regimes");
    println!("          < 0.30: dense; similar to standard multi-branch averaging");
    println!("    steps_at_min_active_share: share of steps where active branches equal min_active");
    println!("      - Expected range: [0.0, 1.0].");
    println!("      - Thresholds (heuristic):");
    println!("          > 0.60: mask sampler frequently clamps to minimum; participation probabilities likely too low");
    println!("          0.20-0.60: moderate clamping; acceptable");
    println!("          < 0.20: sampler rarely clamped; probabilities likely sufficient");
    println!();

    println!("  (4) Inverse participation scaling impact (unbiasedness proxy):");
    println!("    grad_norm_ratio_scaled_over_unscaled_mean/std: proxy ratio for gradient magnitude");
    println!("      - Expected range: > 0.0 (finite).");
    println!("      - Thresholds (heuristic):");
    println!("          ~1.0: scaling has limited effect (either dense masks or similar p_i)");
    println!("          1.0-2.0: moderate amplification; typical when many branches are inactive");
    println!("          > 2.0: strong amplification; may require tighter gradient clipping or lower lr");
    println!("      - Interpretation: high ratio increases update variance, but targets unbiasedness in expectation.");
    println!("    grad_norm_scaled_mean/unscaled_mean: proxy means of compared gradient norms");
    println!("      - Thresholds: monotonic growth without loss improvement suggests instability or too high lr.");
    println!("    Note: this implementation uses a lightweight proxy and does not clone full weights.");
    println!();

    println!("  (5) Replay usage and replay effect strength:");
    println!("    replay_share: replay_steps / (fresh_steps + replay_steps)");
    println!("      - Expected range: [0.0, 1.0].");
    println!("      - Thresholds (heuristic):");
    println!("          0.00-0.05: minimal replay; risk of forgetting under high nonstationarity");
    println!("          0.05-0.25: moderate replay; common tradeoff");
    println!("          > 0.25: heavy replay; may slow adaptation to new data");
    println!("    replay_p_last: last replay probability used by phase strategy");
    println!("      - Expected range: [0.0, 1.0]. If it saturates at max early, ramp_steps may be too small.");
    println!("    replay_delta_loss_mean/std: mean/std of (loss_replay - loss_fresh) pairs");
    println!("      - Interpretation:");
    println!("          positive mean: replay examples are harder now than fresh ones; possible drift away from old data");
    println!("          near zero: replay and fresh have similar difficulty under current model");
    println!("          negative mean: replay is easier; replay may be redundant or indicates overfitting to old data");
    println!("      - Thresholds (heuristic):");
    println!("          > +0.50: strong mismatch; increase replay or adjust lr");
    println!("          < -0.50: replay likely too easy; reduce share or refresh buffer diversity");
    println!();

    println!("  (6) Forgetting indicator (retention score) on fixed control sets:");
    println!("    loss_control_old/new: forward only loss on fixed control slices");
    println!("      - Interpretation: absolute values depend on dataset; monitor trends over time.");
    println!("    retention_delta_old/new: loss_now - loss_baseline (positive indicates forgetting)");
    println!("      - Expected range: any finite value; stable systems keep it near 0.");
    println!("      - Thresholds (heuristic):");
    println!("          0.00-0.10: stable retention");
    println!("          0.10-0.30: mild forgetting; consider increasing replay_share or lowering lr");
    println!("          > 0.30: substantial forgetting; likely requires stronger regularization or replay");
    println!("      - Interpretation caveat: control sets are deterministic slices of initial token pool in this implementation.");
    println!();

    println!("  (7) Branch selection fairness and dominance:");
    println!("    branch_select_gini: Gini coefficient of EMA selection frequencies");
    println!("      - Expected range: [0.0, 1.0] (0 uniform, 1 extreme dominance).");
    println!("      - Thresholds (heuristic):");
    println!("          < 0.10: near uniform usage");
    println!("          0.10-0.30: moderate concentration; monitor starvation risk");
    println!("          > 0.30: strong dominance; path starvation likely, consider adjusting EMA strategy or expansion");
    println!("    branch_select_top1_share: max selection share across branches");
    println!("      - Expected range: [0.0, 1.0].");
    println!("      - Thresholds (heuristic):");
    println!("          < 0.40: healthy diversity");
    println!("          0.40-0.60: moderate dominance");
    println!("          > 0.60: severe dominance; unused paths waste capacity and may drift");
    println!("    Interpretation: higher values indicate dominance and possible path starvation.");
    println!();

    println!("  (8) Snapshot telemetry (train to serve):");
    println!("    snapshots_sent_total: count of parameter snapshots sent from training");
    println!("      - Interpretation: should increase regularly when training is running.");
    println!("      - Thresholds (heuristic):");
    println!("          no increase for > 2 * snapshot_every_steps: snapshot send path broken or training not stepping");
    println!("    Note: latency and staleness require snapshot metadata in payload;");
    println!("          receiver side must carry send_time_ms and train_step for full measurement.");
    println!();

    println!("  (9) Expansion events and injection telemetry:");
    println!("    expansion_events_total: number of width expansion operations performed");
    println!("      - Interpretation: should remain low; frequent expansion indicates unstable routing or insufficient capacity.");
    println!("      - Thresholds (heuristic):");
    println!("          > 1 per 5k steps: aggressive growth; may cause compute blowup and instability");
    println!("    branches_before_last_expand / branches_after_last_expand: last expansion size");
    println!("      - Interpretation: jump size should be small (commonly +1) for stability.");
    println!("    eta_injection_last: conservative injection parameter used by phase strategy");
    println!("      - Expected range: [0.0, 0.5] by validation.");
    println!("      - Thresholds (heuristic):");
    println!("          0.00-0.05: very conservative; new branches learn slowly");
    println!("          0.05-0.20: balanced; typical for stable injection");
    println!("          > 0.20: aggressive; may cause noticeable output drift and instability");
    println!("    sum_w_new_last: approximate total weight mass assigned to new branches (best effort)");
    println!("      - Expected range: approx eta_injection_last in this implementation.");
    println!();

    println!("  (10) Functional continuity on expansion (output drift proxy):");
    println!("    expand_drift_logits_l2_mean/std: L2 distance of last step logits before vs after expansion");
    println!("      - Expected range: >= 0.0.");
    println!("      - Thresholds (heuristic, model dependent):");
    println!("          < 0.10: very stable expansion");
    println!("          0.10-0.50: moderate drift; typically acceptable");
    println!("          > 0.50: large functional change; reduce eta_injection or expansion frequency");
    println!("    expand_drift_logits_cos_dist_mean/std: cosine distance of last step logits before vs after");
    println!("      - Expected range: [0.0, 2.0], typical [0.0, 1.0].");
    println!("      - Thresholds (heuristic):");
    println!("          < 0.05: very stable directionally");
    println!("          0.05-0.20: moderate change");
    println!("          > 0.20: large directional change; indicates behavior shift under expansion");
    println!("    Interpretation: smaller values indicate more stable behavior under expansion.");
    println!();

    println!("  EMA state:");
    println!("    ema_active: whether EMA based branch selection is active at current step");
    println!("      - Interpretation: false during warmup means routing is not yet selective.");
    println!("    ema_last_selected_branch: last selected branch index (or -1 if not applicable)");
    println!("      - Interpretation: stable constant values imply dominance; correlate with fairness metrics.");
    println!();

    println!("Prediction metrics (printed after each ask in interactive mode):");
    println!("  duration_ms: wall clock time for predict");
    println!("    - Thresholds (heuristic): sudden jumps indicate snapshot import contention or CPU saturation.");
    println!("  generated_tokens: number of output tokens generated (tokenizer based)");
    println!("    - Interpretation: depends on EOS behavior and sampling; not a quality metric alone.");
    println!("  tokens_per_sec: throughput based on generated_tokens and duration");
    println!("    - Thresholds: low throughput indicates heavy topology, debug builds, or contention.");
    println!("  effective_context_utilization: (input_tokens + output_tokens) / MAX_SEQ_LEN");
    println!("    - Expected range: [0.0, 1.0].");
    println!("    - Thresholds (heuristic):");
    println!("        > 0.85: near context limit; truncation risk and degraded coherence possible");
    println!("        0.40-0.85: typical working range");
    println!("        < 0.40: short prompts/outputs; usually fine");
    println!("  avg_selected_token_prob: mean probability of selected token across generation steps");
    println!("    - Expected range: [0.0, 1.0].");
    println!("    - Thresholds (heuristic):");
    println!("        < 0.05: very uncertain generation; likely high entropy, noisy output");
    println!("        0.05-0.20: moderate uncertainty; common in creative sampling");
    println!("        > 0.20: confident selection; risk of repetitive output increases if too high");
    println!("  perplexity_selected: exp(mean(-ln(p_selected))) proxy from selected token probabilities");
    println!("    - Expected range: >= 1.0 (when probabilities are valid).");
    println!("    - Thresholds (heuristic):");
    println!("        > 50: highly uncertain generation");
    println!("        10-50: moderate uncertainty");
    println!("        < 10: relatively confident generation");
    println!("  avg_next_token_entropy_nat: mean entropy of next token distribution per step (nats)");
    println!("    - Expected range: >= 0.0.");
    println!("    - Thresholds (heuristic, vocab dependent):");
    println!("        high entropy: more randomness and diversity, lower determinism");
    println!("        low entropy: more determinism, possible repetition if combined with high top1 dominance");
    println!("  avg_top1_top2_margin: mean difference between top1 and top2 probabilities per step");
    println!("    - Expected range: [0.0, 1.0].");
    println!("    - Thresholds (heuristic):");
    println!("        < 0.02: ambiguous next token choice; output sensitive to sampling noise");
    println!("        0.02-0.10: moderate decisiveness");
    println!("        > 0.10: highly decisive; may correlate with repetitive patterns in greedy-like regimes");
    println!();
}


fn drain_snapshot_updates_non_blocking(
    opt_rx: &mut Option<std::sync::mpsc::Receiver<Vec<f32>>>,
    llm_serve: &std::sync::Arc<std::sync::Mutex<crate::layer::Llm>>,
) {
    use std::sync::mpsc;

    let rx = match opt_rx.as_mut() {
        Some(r) => r,
        None => return,
    };

    let mut opt_last: Option<Vec<f32>> = None;

    loop {
        match rx.try_recv() {
            Ok(v) => opt_last = Some(v),
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                *opt_rx = None;
                break;
            }
        }
    }

    if let Some(v_params) = opt_last {
        let mut llm = match llm_serve.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let _ = llm.import_parameters_snapshot(&v_params);

        // Minimal: no further metrics possible without snapshot metadata.
        // Full implementation requires SnapshotPacketAscii as described above.
    }
}

fn main() {
    let mut s_checkpoint_path: String = "../../checkpoints/llm_checkpoint.json".to_string();

    let dataset = Dataset::new(
        "../../data/data_to_pretrain.json",
        "../../data/data_to_train.json",
        DatasetType::JSON,
    );

    // Initial tokenizer training.
    let mut v_corpus: Vec<String> = Vec::new();
    v_corpus.extend(dataset.pretraining_data.clone());
    v_corpus.extend(dataset.chat_training_data.clone());

    let mut config = BpeTokenizerConfig::default();
    config.i_vocab_target = 2000;
    config.i_min_pair_count = 2;

    let bpe = match BpeTokenizer::train_from_corpus_with_config(&v_corpus, config) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("Tokenizer training failed: {}", e);
            return;
        }
    };

    // Two models: one for training (exclusive), one for serving (parallel asks).
    let llm_train: Arc<Mutex<Llm>> = Arc::new(Mutex::new(build_llm_from_tokenizer(bpe.clone())));
    let llm_serve: Arc<Mutex<Llm>> = Arc::new(Mutex::new(build_llm_from_tokenizer(bpe)));

    // Background training control and metrics.
    let b_cancel_train: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let metrics_shared: Arc<Mutex<training_metrics_snapshot_ascii>> =
        Arc::new(Mutex::new(training_metrics_snapshot_ascii::new_idle()));

    // Receivers in main thread.
    let mut opt_progress_rx: Option<mpsc::Receiver<TrainingProgressEventAscii>> = None;
    let mut opt_snapshot_rx: Option<mpsc::Receiver<Vec<f32>>> = None;

    // Online ingestion sender (main to training thread).
    let mut opt_data_tx: Option<mpsc::Sender<TrainingDataEventAscii>> = None;

    // Training thread handle.
    let mut opt_train_handle: Option<thread::JoinHandle<()>> = None;

    {
        let llm = llm_serve.lock().expect("llm_mutex_poisoned");
        println!("\n=== MODEL INFORMATION ===");
        println!("Network architecture: {}", llm.network_description());
        println!(
            "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
            MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
        );
        println!("Total parameters: {}", llm.total_parameters());
    }

    loop {
        // Keep live state fresh.
        drain_training_progress_non_blocking(&mut opt_progress_rx, &metrics_shared, &b_cancel_train);
        drain_snapshot_updates_non_blocking(&mut opt_snapshot_rx, &llm_serve);

        println!("\n--- Menu Mode ---");
        println!("Commands:");
        println!("  t Train (background, continuous learning)");
        println!("  b Training metrics");
        println!("  s Stop training");
        println!("  n Add new training data file (online ingestion)");
        println!("  l Load checkpoint (serve model)");
        println!("  w Save checkpoint (serve model)");
        println!("  a Ask (serve model, parallel to training, done to exit)");
        println!("  o Toggle outage simulation (serve model, test only)");
        println!("  y Topology (ASCII, serve model)");
        println!("  x Metrics (MTB diagnostics, serve model)");
        println!("  h Help");
        println!("  e Exit");

        print!("\nEnter command: ");
        let _ = std::io::stdout().flush();

        let s_cmd = match read_line_ascii_trimmed() {
            Ok(s) => s,
            Err(e) => {
                println!("Input error: {}", e);
                continue;
            }
        };

        // Drain again after input.
        drain_training_progress_non_blocking(&mut opt_progress_rx, &metrics_shared, &b_cancel_train);
        drain_snapshot_updates_non_blocking(&mut opt_snapshot_rx, &llm_serve);

        let s_cmd_lc = s_cmd.to_lowercase();


        if s_cmd_lc == "e" {
            if let Some(tx) = opt_data_tx.as_ref() {
                let _ = tx.send(TrainingDataEventAscii::shutdown);
            }

            if let Some(h) = opt_train_handle.take() {
                b_cancel_train.store(true, Ordering::SeqCst);
                let _ = h.join();
            }

            println!("Exit.");
            break;
        }
        if s_cmd_lc == "h" {
            print_help_ascii();
            continue;
        }

        if s_cmd_lc == "t" {
            let b_already_running = {
                let m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                m.b_running
            };
            if b_already_running {
                println!("Training already running.");
                continue;
            }

            b_cancel_train.store(false, Ordering::SeqCst);
            {
                let mut m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                *m = training_metrics_snapshot_ascii::new_idle();
                m.b_running = true;
                m.s_phase = "starting".to_string();
            }

            // Progress channel.
            let (tx_progress, rx_progress) = mpsc::channel::<TrainingProgressEventAscii>();
            opt_progress_rx = Some(rx_progress);

            // Snapshot channel for serving updates.
            let (tx_snapshot, rx_snapshot) = mpsc::channel::<Vec<f32>>();
            opt_snapshot_rx = Some(rx_snapshot);

            // Online data ingestion channel.
            let (tx_data, rx_data) = mpsc::channel::<TrainingDataEventAscii>();
            opt_data_tx = Some(tx_data.clone());

            let llm_for_train = Arc::clone(&llm_train);
            let metrics_for_train = Arc::clone(&metrics_shared);
            let cancel_for_train = Arc::clone(&b_cancel_train);

            let v_pretraining_examples: Vec<String> = dataset.pretraining_data.clone();
            let v_chat_training_examples: Vec<String> = dataset.chat_training_data.clone();

            opt_train_handle = Some(thread::spawn(move || {
                let r_run = (|| -> Result<(), String> {
                    let i_snapshot_every_steps: usize = 200;

                    let i_epochs_total_pretrain = 30;
                    let i_epochs_total_train = 5000;

                    let cl_cfg_pre = ContinuousLearningConfig {
                        v_branch_participation_p: vec![0.75, 0.75, 0.75, 0.75],
                        i_min_active_branches: 2,
                        b_scale_by_inverse_participation: true,
                        u64_mask_seed: 20260213,
                    };

                    let cl_cfg_tune = ContinuousLearningConfig {
                        v_branch_participation_p: vec![0.60, 0.70, 0.80, 0.65],
                        i_min_active_branches: 2,
                        b_scale_by_inverse_participation: true,
                        u64_mask_seed: 20260214,
                    };

                    let cfg_phase_pre = phase_strategy_config_ascii {
                        e_phase: training_phase_ascii::realtime,
                        b_enable_ema_branch_selection: true,
                        i_ema_warmup_steps: 500,

                        b_enable_replay: true,
                        d_replay_p_start: 0.0,
                        d_replay_p_max: 0.25,
                        i_replay_ramp_steps: 2000,

                        b_enable_autonomous_expansion: true,
                        i_expand_check_every_steps: 500,
                        d_eta_injection: 0.05,

                        i_max_total_branches: 16,
                    };

                    let cfg_phase_tune = phase_strategy_config_ascii {
                        e_phase: training_phase_ascii::realtime,
                        b_enable_ema_branch_selection: true,
                        i_ema_warmup_steps: 500,

                        b_enable_replay: true,
                        d_replay_p_start: 0.0,
                        d_replay_p_max: 0.25,
                        i_replay_ramp_steps: 2000,

                        b_enable_autonomous_expansion: true,
                        i_expand_check_every_steps: 500,
                        d_eta_injection: 0.05,

                        i_max_total_branches: 16,
                    };

                    // Update phase in shared metrics.
                    {
                        let mut m = metrics_for_train
                            .lock()
                            .map_err(|_| "metrics_lock_failed".to_string())?;
                        m.s_phase = "pretraining".to_string();
                        m.i_epoch_current = 0;
                        m.i_epochs_total = i_epochs_total_pretrain;
                        m.s_last_error = "".to_string();
                    }

                    // Training orchestration: if available, keep receiver alive across phases.
                    // NOTE: This method must exist in layer.rs for the robust solution.
                    {
                        let mut llm = llm_for_train.lock().map_err(|_| "llm_lock_failed".to_string())?;

                        llm.train_two_phase_with_progress_online_ascii(
                            v_pretraining_examples.iter().map(|s| s.as_str()).collect(),
                            i_epochs_total_pretrain,
                            0.0005,
                            "pretraining",
                            Some(cl_cfg_pre),
                            cfg_phase_pre,
                            v_chat_training_examples.iter().map(|s| s.as_str()).collect(),
                            i_epochs_total_train,
                            0.0001,
                            "instruction_tuning",
                            Some(cl_cfg_tune),
                            cfg_phase_tune,
                            Arc::clone(&cancel_for_train),
                            tx_progress.clone(),
                            i_snapshot_every_steps,
                            Some(tx_snapshot.clone()),
                            rx_data,
                        )?;
                    }

                    Ok(())
                })();

                drop(tx_progress);
                drop(tx_snapshot);

                let mut m = match metrics_for_train.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                m.b_cancel_requested = cancel_for_train.load(Ordering::SeqCst);
                m.b_running = false;

                match r_run {
                    Ok(()) => {
                        if m.b_cancel_requested {
                            m.s_phase = "canceled".to_string();
                        } else {
                            m.s_phase = "done".to_string();
                        }
                    }
                    Err(e) => {
                        m.s_phase = "error".to_string();
                        m.s_last_error = e;
                    }
                }
            }));

            println!("Training started in background. Serving continues on llm_serve.");
            continue;
        }

        if s_cmd_lc == "b" {
            drain_training_progress_non_blocking(&mut opt_progress_rx, &metrics_shared, &b_cancel_train);
            let m = metrics_shared
                .lock()
                .expect("metrics_mutex_poisoned")
                .clone();
            print_training_metrics_snapshot_ascii(&m);
            continue;
        }

        if s_cmd_lc == "s" {
            let b_running = {
                let m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                m.b_running
            };
            if !b_running {
                println!("Training not running.");
                if let Some(h) = opt_train_handle.take() {
                    let _ = h.join();
                }
                opt_data_tx = None;
                continue;
            }

            b_cancel_train.store(true, Ordering::SeqCst);
            {
                let mut m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                m.b_cancel_requested = true;
                m.s_phase = "cancel_requested".to_string();
            }

            if let Some(tx) = opt_data_tx.as_ref() {
                let _ = tx.send(TrainingDataEventAscii::shutdown);
            }

            if let Some(h) = opt_train_handle.take() {
                let _ = h.join();
            }

            opt_data_tx = None;
            println!("Training stop requested and thread joined.");
            continue;
        }

        if s_cmd_lc == "n" {
            let b_running = {
                let m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                m.b_running
            };
            if !b_running {
                println!("Training not running. Online ingestion requires a running training thread.");
                continue;
            }

            let tx = match opt_data_tx.as_ref() {
                Some(v) => v,
                None => {
                    println!("Online ingestion channel not available.");
                    continue;
                }
            };

            print!("Enter path to JSON training file (array of strings): ");
            let _ = std::io::stdout().flush();

            let s_path = match read_line_ascii_trimmed() {
                Ok(s) => s,
                Err(e) => {
                    println!("Input error: {}", e);
                    continue;
                }
            };

            if s_path.trim().is_empty() {
                println!("Empty path.");
                continue;
            }

            match tx.send(TrainingDataEventAscii::add_training_file_json_array { s_path: s_path.clone() }) {
                Ok(()) => println!("Queued training data file for ingestion: {}", s_path),
                Err(_) => println!("Failed to send ingestion event (receiver not alive)."),
            }

            continue;
        }

        if s_cmd_lc == "w" {
            print!("Enter checkpoint path or press Enter for default: ");
            let _ = std::io::stdout().flush();

            let s_path = match read_line_ascii_trimmed() {
                Ok(s) => s,
                Err(e) => {
                    println!("Input error: {}", e);
                    continue;
                }
            };

            if !s_path.is_empty() {
                s_checkpoint_path = s_path;
            }

            let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
            match llm.save_checkpoint_llm_checkpoint_v2(&s_checkpoint_path) {
                Ok(()) => println!("Saved checkpoint: {}", s_checkpoint_path),
                Err(e) => println!("Save failed: {}", e),
            }

            continue;
        }

        if s_cmd_lc == "l" {
            print!("Enter checkpoint path or press Enter for default: ");
            let _ = std::io::stdout().flush();

            let s_path = match read_line_ascii_trimmed() {
                Ok(s) => s,
                Err(e) => {
                    println!("Input error: {}", e);
                    continue;
                }
            };

            if !s_path.is_empty() {
                s_checkpoint_path = s_path;
            }

            match Llm::load_checkpoint_llm_checkpoint_v2_rebuild(&s_checkpoint_path) {
                Ok(llm_loaded) => {
                    let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
                    *llm = llm_loaded;
                    println!("Loaded checkpoint into serve model: {}", s_checkpoint_path);
                }
                Err(e) => println!("Load failed: {}", e),
            }

            continue;
        }

        if s_cmd_lc == "o" {
            let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
            let b_new = !llm.is_outage_simulation_enabled();
            llm.set_outage_simulation_enabled(b_new);
            println!(
                "Outage simulation: {}",
                if b_new { "enabled" } else { "disabled" }
            );
            continue;
        }

        if s_cmd_lc == "y" {
            let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
            let v_lines = topology_to_ascii_lines(&mut llm);
            println!();
            for s_line in v_lines {
                println!("{}", s_line);
            }
            continue;
        }

        if s_cmd_lc == "x" {
            let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
            print_metrics_ascii(&mut llm);
            continue;
        }

        if s_cmd_lc == "a" {
            println!("Interactive mode. Type 'done' to exit.");
            loop {
                // While in interactive mode, keep snapshots applied.
                drain_snapshot_updates_non_blocking(&mut opt_snapshot_rx, &llm_serve);

                print!("Enter prompt: ");
                let _ = std::io::stdout().flush();

                let s_user = match read_line_ascii_trimmed() {
                    Ok(s) => s,
                    Err(e) => {
                        println!("Input error: {}", e);
                        continue;
                    }
                };

                if s_user.is_empty() {
                    println!("Empty prompt.");
                    continue;
                }
                if s_user.eq_ignore_ascii_case("done") {
                    break;
                }

                let s_formatted = format!("User: {}", s_user);

                let t0 = Instant::now();
                let r_predict = {
                    let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
                    llm.predict_with_stats(&s_formatted)
                };
                let d_ms = t0.elapsed().as_secs_f64() * 1000.0;

                match r_predict {
                    Ok((s_out, st)) => {
                        println!("Model output: {}", s_out);
                        let llm = llm_serve.lock().expect("llm_mutex_poisoned");
                        let m =
                            compute_predict_metrics_ascii(&llm, &s_formatted, &s_out, d_ms, Some(&st));
                        print_predict_metrics_ascii(&m);
                    }
                    Err(e) => {
                        println!("Model output error: {}", e);
                        let llm = llm_serve.lock().expect("llm_mutex_poisoned");
                        let m = compute_predict_metrics_ascii(&llm, &s_formatted, "", d_ms, None);
                        print_predict_metrics_ascii(&m);
                    }
                }
            }
            continue;
        }

        println!("Unknown command.");
    }
}
