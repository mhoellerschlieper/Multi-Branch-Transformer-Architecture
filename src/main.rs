// main.rs
// Description: Binary entry point with menu loop. Builds model from tokenizer,
//              supports checkpoint save and load.
//
//              Extension:
//              - After each Ask/predict, print runtime metrics:
//                duration, token_count, tokens_per_sec, and 5 additional metrics.
//
// History:
// - 2026-02-01: Add menu loop and checkpoint save and load.
// - 2026-02-01: Fix checkpoint load by rebuilding model from checkpoint tokenizer vocab.
// - 2026-02-07: Add MTB parallel block group layer to support multi branch topology.
// - 2026-02-07: Add TransformerSequence and generalize ParallelBlockGroup to accept Layer branches.
// - 2026-02-08: Add menu commands T (topology) and M (metrics) to main loop.
// - 2026-02-08: Add post predict metrics: duration, token count, tokens/sec, plus 5 extra metrics.
// Author: Marcus Schlieper

mod layer;
mod math;
mod tokenizer;
mod train;
mod utils;

use std::io::Write;
use std::time::Instant;

use crate::layer::{
    Embeddings, Layer, Llm, OutputProjection, ParallelBlockGroup, TransformerBlock,
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
    // - 2026-02-08: Add ASCII topology report for menu command T.
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
            let opt_pg = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>());

            if let Some(pg) = opt_pg {
                v_out.push(format!(
                    "[{}] ParallelBlockGroup branches={}",
                    i_idx,
                    pg.num_branches()
                ));

                // NOTE: v_branches is private. This requires a public helper in layer.rs:
                // ParallelBlockGroup::branch_layer_types_ascii()
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

// Print MTB metrics and an ASCII report (manual trigger).
fn print_metrics_ascii(llm: &mut Llm) {
    // History:
    // - 2026-02-08: Add menu command M to print MTB diagnostics.
    println!("");
    println!("=== Metrics (MTB diagnostics) ===");
    llm.run_post_load_mtb_diagnostics_ascii();
}

// Post predict runtime metrics.
#[derive(Clone, Debug)]
struct predict_metrics_ascii {
    // Primary metrics requested.
    d_duration_ms: f64,
    i_generated_tokens: usize,
    d_tokens_per_sec: f64,

    // Five additional metrics (new).
    i_input_tokens: usize,
    i_total_tokens: usize,
    d_avg_chars_per_token_out: f64,
    d_output_chars_per_sec: f64,
    d_effective_context_utilization: f64,
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
) -> predict_metrics_ascii {
    // NOTE:
    // Token counts are computed via the tokenizer path if available.
    // This is safe but can be computationally nontrivial; however, it is acceptable
    // for interactive CLI diagnostics.

    let i_input_tokens = llm.tokenize(s_prompt).map(|v| v.len()).unwrap_or(0);

    // Output token count: encode output as standalone text.
    // This approximates generated token count; if the model emits special tokens,
    // decode filters them, so the re-encoding count can be slightly different.
    let i_output_tokens = llm.tokenize(s_output).map(|v| v.len()).unwrap_or(0);

    let i_total_tokens = i_input_tokens.saturating_add(i_output_tokens);

    let d_sec = (d_duration_ms / 1000.0).max(1e-9);
    let d_tokens_per_sec = (i_output_tokens as f64) / d_sec;

    let i_out_chars = s_output.len();
    let d_avg_chars_per_token_out = if i_output_tokens == 0 {
        0.0
    } else {
        (i_out_chars as f64) / (i_output_tokens as f64)
    };
    let d_output_chars_per_sec = (i_out_chars as f64) / d_sec;

    // Effective context utilization: total tokens relative to MAX_SEQ_LEN.
    // This provides a quick indication of truncation risk and headroom.
    let d_effective_context_utilization =
        (i_total_tokens as f64) / (crate::MAX_SEQ_LEN as f64).max(1.0);

    predict_metrics_ascii {
        d_duration_ms: clamp_f64(d_duration_ms, 0.0, 1.0e12),
        i_generated_tokens: i_output_tokens,
        d_tokens_per_sec: clamp_f64(d_tokens_per_sec, 0.0, 1.0e12),
        i_input_tokens,
        i_total_tokens,
        d_avg_chars_per_token_out: clamp_f64(d_avg_chars_per_token_out, 0.0, 1.0e9),
        d_output_chars_per_sec: clamp_f64(d_output_chars_per_sec, 0.0, 1.0e12),
        d_effective_context_utilization: clamp_f64(d_effective_context_utilization, 0.0, 1.0),
    }
}

fn print_predict_metrics_ascii(m: &predict_metrics_ascii) {
    // Output format is ASCII only.
    println!("");
    println!("=== Predict Metrics ===");
    println!("duration_ms: {:.3}", m.d_duration_ms);
    println!("generated_tokens: {}", m.i_generated_tokens);
    println!("tokens_per_sec: {:.3}", m.d_tokens_per_sec);
    println!("input_tokens: {}", m.i_input_tokens);
    println!("total_tokens: {}", m.i_total_tokens);
    println!("avg_chars_per_token_out: {:.3}", m.d_avg_chars_per_token_out);
    println!("output_chars_per_sec: {:.3}", m.d_output_chars_per_sec);
    println!(
        "effective_context_utilization: {:.6}",
        m.d_effective_context_utilization
    );
}

fn main() {
    let mut s_checkpoint_path: String = "../../checkpoints/llm_checkpoint.json".to_string();

    let dataset = Dataset::new(
        "../../data/data_to_pretrain.json",
        "../../data/data_to_train.json",
        DatasetType::JSON,
    );

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

        if s_cmd == "y" {
            let v_lines = topology_to_ascii_lines(&mut llm);
            println!("");
            for s_line in v_lines {
                println!("{}", s_line);
            }
            continue;
        }

        if s_cmd == "x" {
            print_metrics_ascii(&mut llm);
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
                let r_predict = llm.predict(&s_formatted);
                let d_ms = t0.elapsed().as_secs_f64() * 1000.0;

                match r_predict {
                    Ok(s_out) => {
                        println!("Model output: {}", s_out);

                        // After each ask/predict, print the requested metrics.
                        let m = compute_predict_metrics_ascii(&llm, &s_formatted, &s_out, d_ms);
                        print_predict_metrics_ascii(&m);
                    }
                    Err(e) => {
                        println!("Model output error: {}", e);

                        // Still print minimal metrics even on error.
                        let m = compute_predict_metrics_ascii(&llm, &s_formatted, "", d_ms);
                        print_predict_metrics_ascii(&m);
                    }
                }
            }

            continue;
        }

        println!("Unknown command.");
    }
}
