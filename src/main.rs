// main.rs
// Description: Binary entry point with menu loop. Builds model from tokenizer,
//              supports checkpoint save and load.
//
//              Extension:
//              - Add command z to run an A B benchmark over 10 fixed prompts:
//                (A) outage_simulation=false, predict_with_stats, aggregate mean metrics
//                (B) outage_simulation=true,  predict_with_stats, aggregate mean metrics
//
// History:
// - 2026-02-01: Add menu loop and checkpoint save and load.
// - 2026-02-07: Add MTB parallel block group layer to support multi branch topology.
// - 2026-02-08: Add predict_with_stats and post predict metrics.
// - 2026-02-08: Add command z to compute mean metrics with outage simulation off and on.
// Author: Marcus Schlieper

mod layer;
mod math;
mod tokenizer;
mod train;
mod utils;

use std::io::Write;
use std::time::Instant;

use crate::layer::{
    Embeddings, Layer, Llm, OutputProjection, ParallelBlockGroup, PredictStats, TransformerBlock,
    TransformerSequence,
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

    let seq_2_1 = TransformerSequence::new(vec![block2_1, block2_2])
        .expect("transformer_sequence_new_failed");
    let seq_2_2 = TransformerSequence::new(vec![block2_3, block2_4])
        .expect("transformer_sequence_new_failed");
    let seq_2_3 = TransformerSequence::new(vec![block2_5, block2_6])
        .expect("transformer_sequence_new_failed");
    let seq_2_4 = TransformerSequence::new(vec![block2_7, block2_8])
        .expect("transformer_sequence_new_failed");

    let parallel_block2 = ParallelBlockGroup::new(vec![
        Box::new(seq_2_1) as Box<dyn Layer>,
        Box::new(seq_2_2) as Box<dyn Layer>,
        Box::new(seq_2_3) as Box<dyn Layer>,
        Box::new(seq_2_4) as Box<dyn Layer>,
    ])
    .expect("parallel_block_group_new_failed");

    let block3 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let out = OutputProjection::new(crate::EMBEDDING_DIM, vocab.words.len());

    let mut llm = Llm::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(block1),
            Box::new(parallel_block2),
            Box::new(block3),
            Box::new(out),
        ],
    );

    llm.set_bpe_tokenizer(bpe);
    llm.set_residual_dropout_p(0.1);
    llm.set_training(true);

    let _ = llm.set_sampling_config(0.9, 40, 0.95, 987654321);
    llm
}

// ASCII topology rendering for the current in-memory network.
fn topology_to_ascii_lines(llm: &mut Llm) -> Vec<String> {
    // History:
    // - 2026-02-08: Add ASCII topology report for menu command y.
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

                // NOTE: Branch list is accessed via public helper (encapsulation).
                let v_branch_types = pg.branch_layer_types_ascii();
                for (i_b, s_bt) in v_branch_types.iter().enumerate() {
                    v_out.push(format!("  - branch[{}] {}", i_b, s_bt));
                }
                continue;
            }

            v_out.push(format!(
                "[{}] ParallelBlockGroup (downcast_failed)",
                i_idx
            ));
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

// Print MTB metrics and an ASCII report (manual trigger).
fn print_metrics_ascii(llm: &mut Llm) {
    // History:
    // - 2026-02-08: Add menu command x to print MTB diagnostics.
    println!();
    println!("=== Metrics (MTB diagnostics) ===");
    llm.run_post_load_mtb_diagnostics_ascii();
}

// Post predict runtime metrics (single run).
#[derive(Clone, Debug)]
struct predict_metrics_ascii {
    // Runtime / throughput.
    d_duration_ms: f64,
    i_generated_tokens: usize,
    d_tokens_per_sec: f64,

    // Token / text.
    i_input_tokens: usize,
    i_total_tokens: usize,
    i_output_chars: usize,
    d_avg_chars_per_token_out: f64,
    d_output_chars_per_sec: f64,
    d_effective_context_utilization: f64,

    // Prediction quality proxies from PredictStats.
    d_avg_selected_token_prob: f64,
    d_perplexity_selected: f64,
    d_avg_next_token_entropy_nat: f64,
    d_avg_top1_top2_margin: f64,
    i_pred_stats_steps: usize,
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

    // Effective context utilization indicates proximity to MAX_SEQ_LEN and truncation risk.
    let d_effective_context_utilization =
        (i_total_tokens as f64) / (crate::MAX_SEQ_LEN as f64).max(1.0);

    // Prediction stats (safe defaults if not available).
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

// Mean aggregation for command z (10 prompts).
#[derive(Clone, Debug)]
struct predict_metrics_mean_agg_ascii {
    i_runs_total: usize,
    i_runs_ok: usize,

    d_duration_ms_mean: f64,
    d_generated_tokens_mean: f64,
    d_tokens_per_sec_mean: f64,

    d_avg_selected_token_prob_mean: f64,
    d_perplexity_selected_mean: f64,
    d_avg_next_token_entropy_nat_mean: f64,
    d_avg_top1_top2_margin_mean: f64,

    d_effective_context_utilization_mean: f64,
}

fn mean_safe_f64(v_x: &[f64]) -> f64 {
    if v_x.is_empty() {
        return 0.0;
    }
    let mut d_sum: f64 = 0.0;
    let mut d_cnt: f64 = 0.0;
    for &d in v_x.iter() {
        if d.is_finite() {
            d_sum += d;
            d_cnt += 1.0;
        }
    }
    if d_cnt <= 0.0 {
        0.0
    } else {
        d_sum / d_cnt
    }
}

fn run_predict_mean_benchmark_ascii(
    llm: &mut Llm,
    v_prompts: &[String],
    i_limit: usize,
) -> predict_metrics_mean_agg_ascii {
    // History:
    // - 2026-02-08: Add command z mean benchmark runner (A/B outage simulation).
    let i_take = i_limit.min(v_prompts.len());
    let mut i_runs_total: usize = 0;
    let mut i_runs_ok: usize = 0;

    let mut v_duration_ms: Vec<f64> = Vec::new();
    let mut v_generated_tokens: Vec<f64> = Vec::new();
    let mut v_tokens_per_sec: Vec<f64> = Vec::new();

    let mut v_avg_p: Vec<f64> = Vec::new();
    let mut v_ppl: Vec<f64> = Vec::new();
    let mut v_entropy: Vec<f64> = Vec::new();
    let mut v_margin: Vec<f64> = Vec::new();
    let mut v_ctx_util: Vec<f64> = Vec::new();

    for s_prompt in v_prompts.iter().take(i_take) {
        i_runs_total = i_runs_total.saturating_add(1);

        let t0 = Instant::now();
        let r = llm.predict_with_stats(s_prompt);
        let d_ms = t0.elapsed().as_secs_f64() * 1000.0;

        match r {
            Ok((s_out, st)) => {
                i_runs_ok = i_runs_ok.saturating_add(1);
                let m = compute_predict_metrics_ascii(llm, s_prompt, &s_out, d_ms, Some(&st));

                v_duration_ms.push(m.d_duration_ms);
                v_generated_tokens.push(m.i_generated_tokens as f64);
                v_tokens_per_sec.push(m.d_tokens_per_sec);

                v_avg_p.push(m.d_avg_selected_token_prob);
                v_ppl.push(m.d_perplexity_selected);
                v_entropy.push(m.d_avg_next_token_entropy_nat);
                v_margin.push(m.d_avg_top1_top2_margin);
                v_ctx_util.push(m.d_effective_context_utilization);
            }
            Err(_) => {
                // On error, skip from mean to avoid skewing with zeros.
                // This is intentionally strict for expert diagnostics.
                continue;
            }
        }
    }

    predict_metrics_mean_agg_ascii {
        i_runs_total,
        i_runs_ok,

        d_duration_ms_mean: mean_safe_f64(&v_duration_ms),
        d_generated_tokens_mean: mean_safe_f64(&v_generated_tokens),
        d_tokens_per_sec_mean: mean_safe_f64(&v_tokens_per_sec),

        d_avg_selected_token_prob_mean: mean_safe_f64(&v_avg_p),
        d_perplexity_selected_mean: mean_safe_f64(&v_ppl),
        d_avg_next_token_entropy_nat_mean: mean_safe_f64(&v_entropy),
        d_avg_top1_top2_margin_mean: mean_safe_f64(&v_margin),

        d_effective_context_utilization_mean: mean_safe_f64(&v_ctx_util),
    }
}

fn print_mean_benchmark_report_ascii(s_title: &str, agg: &predict_metrics_mean_agg_ascii) {
    println!();
    println!("=== Mean Benchmark Report ===");
    println!("title: {}", s_title);
    println!("runs_total: {}", agg.i_runs_total);
    println!("runs_ok: {}", agg.i_runs_ok);

    println!("duration_ms_mean: {:.6}", agg.d_duration_ms_mean);
    println!("generated_tokens_mean: {:.6}", agg.d_generated_tokens_mean);
    println!("tokens_per_sec_mean: {:.6}", agg.d_tokens_per_sec_mean);

    println!(
        "avg_selected_token_prob_mean: {:.6}",
        agg.d_avg_selected_token_prob_mean
    );
    println!(
        "perplexity_selected_mean: {:.6}",
        agg.d_perplexity_selected_mean
    );
    println!(
        "avg_next_token_entropy_nat_mean: {:.6}",
        agg.d_avg_next_token_entropy_nat_mean
    );
    println!(
        "avg_top1_top2_margin_mean: {:.6}",
        agg.d_avg_top1_top2_margin_mean
    );
    println!(
        "effective_context_utilization_mean: {:.6}",
        agg.d_effective_context_utilization_mean
    );
}

fn main() {
    let mut s_checkpoint_path: String = "../../checkpoints/llm_checkpoint.json".to_string();

    let dataset = Dataset::new(
        "../../data/data_to_pretrain.json",
        "../../data/data_to_train.json",
        DatasetType::JSON,
    );

    // Keep initial tokenizer training to allow immediate usage.
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

    let mut llm = build_llm_from_tokenizer(bpe);

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!(
        "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("Total parameters: {}", llm.total_parameters());

    loop {
        println!("\n--- Menu Mode ---");
        println!("Commands:");
        println!("  t Train");
        println!("  l Load checkpoint");
        println!("  s Save checkpoint");
        println!("  a Ask");
        println!("  o Toggle outage simulation (test only)");
        println!("  y Topology (ASCII)");
        println!("  x Metrics (MTB diagnostics)");
        println!("  z Mean benchmark (10 prompts, outage off/on)");
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

        // Normalize once. From here on, only compare s_cmd_lc.
        let s_cmd_lc = s_cmd.to_lowercase();

        if s_cmd_lc == "e" {
            println!("Exit.");
            break;
        }

        if s_cmd_lc == "t" {
            let v_pretraining_examples: Vec<&str> = dataset
                .pretraining_data
                .iter()
                .map(|s| s.as_str())
                .collect();

            let v_chat_training_examples: Vec<&str> = dataset
                .chat_training_data
                .iter()
                .map(|s| s.as_str())
                .collect();

            println!("\n=== PRE-TRAINING MODEL ===");
            println!(
                "Pre-training on {} examples for {} epochs with learning rate {}",
                dataset.pretraining_data.len(),
                100,
                0.0005
            );

            if let Err(e) = llm.train(v_pretraining_examples, 30, 0.0005) {
                eprintln!("Training failed: {}", e);
                continue;
            }

            println!("\n=== INSTRUCTION TUNING ===");
            println!(
                "Instruction tuning on {} examples for {} epochs with learning rate {}",
                dataset.chat_training_data.len(),
                200,
                0.0001
            );

            if let Err(e) = llm.train(v_chat_training_examples, 50, 0.0001) {
                eprintln!("Training failed: {}", e);
                continue;
            }

            continue;
        }

        if s_cmd_lc == "s" {
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

            // IMPORTANT: Rebuild model to match checkpoint tokenizer and vocab size.
            match Llm::load_checkpoint_llm_checkpoint_v2_rebuild(&s_checkpoint_path) {
                Ok(llm_loaded) => {
                    llm = llm_loaded;
                    println!("Loaded checkpoint: {}", s_checkpoint_path);
                }
                Err(e) => println!("Load failed: {}", e),
            }

            continue;
        }

        if s_cmd_lc == "o" {
            let b_new = !llm.is_outage_simulation_enabled();
            llm.set_outage_simulation_enabled(b_new);

            if b_new {
                println!("Outage simulation: enabled");
            } else {
                println!("Outage simulation: disabled");
            }
            continue;
        }

        if s_cmd_lc == "y" {
            let v_lines = topology_to_ascii_lines(&mut llm);
            println!();
            for s_line in v_lines {
                println!("{}", s_line);
            }
            continue;
        }

        if s_cmd_lc == "x" {
            print_metrics_ascii(&mut llm);
            continue;
        }

        if s_cmd_lc == "z" {
            // Fixed prompt set for a stable A/B comparison (ASCII only prompts by convention).
            // If desired, this can be extended to include UTF-8 prompts as well.
            let v_prompts: Vec<String> = vec![
                "User: Wo liegt der Mount Everest?".to_string(),
                "User: was ist der  Mount everest".to_string(),
                "User: What is gradient clipping?".to_string(),
                "User: In welchem Gebirge befindet sich der Mount Everest?".to_string(),
                "User: Was ist der Amazonas-Regenwald?".to_string(),
                "User: Wie lange braucht der Mond fuer eine Umrundung der Erde?".to_string(),
                "User: Was ist Schall?".to_string(),
                "User: Wie ist Jupiter?".to_string(),
                "User: Was bedeckt den groessten Teil der Erde?".to_string(),
                "User: Wie wird Schokolade hergestellt?".to_string(),
            ];

            println!();
            println!("=== Mean benchmark start ===");
            println!("prompts: {}", v_prompts.len());

            // Phase A: outage simulation disabled.
            llm.set_outage_simulation_enabled(false);
            let agg_off = run_predict_mean_benchmark_ascii(&mut llm, &v_prompts, 10);
            print_mean_benchmark_report_ascii("outage_simulation=false", &agg_off);

            // Phase B: outage simulation enabled.
            llm.set_outage_simulation_enabled(true);
            let agg_on = run_predict_mean_benchmark_ascii(&mut llm, &v_prompts, 10);
            print_mean_benchmark_report_ascii("outage_simulation=true", &agg_on);

            // Restore default: keep previous behavior conservative (disabled).
            llm.set_outage_simulation_enabled(false);

            println!();
            println!("=== Mean benchmark end ===");
            continue;
        }

        if s_cmd_lc == "a" {
            println!("Interactive mode. Type 'done' to exit.");
            loop {
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

                // Metrics timing starts immediately before predict.
                let t0 = Instant::now();
                let r_predict = llm.predict_with_stats(&s_formatted);
                let d_ms = t0.elapsed().as_secs_f64() * 1000.0;

                match r_predict {
                    Ok((s_out, st)) => {
                        println!("Model output: {}", s_out);
                        let m = compute_predict_metrics_ascii(
                            &llm,
                            &s_formatted,
                            &s_out,
                            d_ms,
                            Some(&st),
                        );
                        print_predict_metrics_ascii(&m);
                    }
                    Err(e) => {
                        println!("Model output error: {}", e);
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
