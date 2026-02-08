// tokenizer.rs
// Description: Byte Pair Encoding (BPE) tokenizer including training, encoding, decoding,
//              plus serialization support for checkpoint save and load.
//
//              UTF-8 extension:
//              - Replace ASCII-only pre-tokenization with UTF-8 aware pre-tokenizer.
//              - Remove non-ASCII replacement by UNK.
//              - Keep determinism via explicit normalization and stable sorting.
//
// History:
// - 2026-02-01: Consolidate tokenizer and vocab related logic into tokenizer.rs and layer.rs.
// - 2026-02-01: Add checkpoint serialization for tokenizer (vocab and merges).
// - 2026-02-01: Implement stable special tokens, improved pre-tokenization, and deterministic training.
// - 2026-02-08: Implement UTF-8 aware pre-tokenization and decoding normalization.
// Author: Marcus Schlieper

use std::collections::{HashMap, HashSet};

use crate::layer::Vocab;
use crate::utils;
use serde::{Deserialize, Serialize};

// Special tokens. Keep stable and ASCII.
// NOTE: Must be unique and non-empty to avoid collisions in vocab encoding and decoding.
pub const S_PAD: &str = "<pad>";
pub const S_UNK: &str = "<unk>";
pub const S_BOS: &str = "<bos>";
pub const S_EOS: &str = "<eos>";

// Word boundary marker.
// NOTE: Keep stable and ASCII.
pub const S_WB: &str = "</w>";

// Tokenizer configuration stored for reproducibility.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeTokenizerConfig {
    pub i_vocab_target: usize,
    pub i_min_pair_count: usize,
    pub u64_seed: u64,
    pub s_pre_tokenizer: String,
    // UTF-8 normalization mode, stored to ensure deterministic behavior across runs.
    // Supported: "none", "nfc"
    pub s_unicode_normalization: String,
}

impl Default for BpeTokenizerConfig {
    fn default() -> Self {
        Self {
            i_vocab_target: 2000,
            i_min_pair_count: 2,
            u64_seed: 123456789,
            // Switch default to UTF-8 aware tokenizer while keeping name explicit.
            s_pre_tokenizer: "utf8_ws_punct_v1".to_string(),
            s_unicode_normalization: "nfc".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeMerge {
    pub s_left: String,
    pub s_right: String,
    pub s_merged: String,
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    pub vocab: Vocab,
    pub merges: Vec<BpeMerge>,
    merge_ranks: HashMap<String, usize>,
    pub config: BpeTokenizerConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeTokenizerCheckpoint {
    pub v_vocab_words: Vec<String>,
    pub v_merges: Vec<BpeMerge>,
    pub config: BpeTokenizerConfig,
}

impl BpeTokenizer {
    pub fn new(vocab: Vocab, merges: Vec<BpeMerge>, config: BpeTokenizerConfig) -> Result<Self, String> {
        let mut m_ranks: HashMap<String, usize> = HashMap::new();
        for (i_rank, m) in merges.iter().enumerate() {
            let s_key = format!("{}\t{}", m.s_left, m.s_right);
            m_ranks.insert(s_key, i_rank);
        }

        // Validate special tokens exist and are unique.
        for s_tok in [S_PAD, S_UNK, S_BOS, S_EOS] {
            if vocab.encode(s_tok).is_none() {
                return Err(format!("missing_special_token_in_vocab: {}", s_tok));
            }
        }
        if S_PAD == S_UNK
            || S_PAD == S_BOS
            || S_PAD == S_EOS
            || S_UNK == S_BOS
            || S_UNK == S_EOS
            || S_BOS == S_EOS
        {
            return Err("special_tokens_not_unique".to_string());
        }
        if S_PAD.is_empty() || S_UNK.is_empty() || S_BOS.is_empty() || S_EOS.is_empty() {
            return Err("special_tokens_must_not_be_empty".to_string());
        }

        Ok(Self {
            vocab,
            merges,
            merge_ranks: m_ranks,
            config,
        })
    }

    pub fn to_checkpoint(&self) -> BpeTokenizerCheckpoint {
        BpeTokenizerCheckpoint {
            v_vocab_words: self.vocab.words.clone(),
            v_merges: self.merges.clone(),
            config: self.config.clone(),
        }
    }

    pub fn from_checkpoint(cp: &BpeTokenizerCheckpoint) -> Result<Self, String> {
        if cp.v_vocab_words.len() < 4 {
            return Err("checkpoint_vocab_too_small".to_string());
        }
        let v_refs: Vec<&str> = cp.v_vocab_words.iter().map(|s| s.as_str()).collect();
        let vocab = Vocab::new(v_refs);
        Self::new(vocab, cp.v_merges.clone(), cp.config.clone())
    }

    // Apply Unicode normalization to ensure deterministic training/encoding.
    // NOTE: This uses a minimal internal implementation:
    // - "none": return as-is
    // - "nfc": use utils::normalize_text_utf8_nfc
    fn normalize_unicode_deterministic(&self, s_text: &str) -> String {
        if self.config.s_unicode_normalization == "nfc" {
            utils::normalize_text_utf8_nfc(s_text)
        } else {
            s_text.to_string()
        }
    }

    // UTF-8 aware pre-tokenization:
    // - Split on Unicode whitespace (Rust split_whitespace already supports this)
    // - Separate punctuation using Unicode general category approximation:
    //   - treat any char that is not alphanumeric and not underscore as delimiter token,
    //     but keep it as its own token (except whitespace)
    // - Preserve all UTF-8 characters (no replacement by UNK)
    fn pre_tokenize_utf8_ws_punct(s_text: &str) -> Vec<String> {
        let mut v_out: Vec<String> = Vec::new();

        for s_piece in s_text.split_whitespace() {
            if s_piece.is_empty() {
                continue;
            }

            let mut s_buf: String = String::new();
            for ch in s_piece.chars() {
                // Keep combining marks, letters, digits.
                let b_is_word = ch.is_alphanumeric() || ch == '_';

                if b_is_word {
                    s_buf.push(ch);
                    continue;
                }

                // ch is punctuation or symbol (but not whitespace because split_whitespace removed it)
                if !s_buf.is_empty() {
                    v_out.push(s_buf.clone());
                    s_buf.clear();
                }
                v_out.push(ch.to_string());
            }

            if !s_buf.is_empty() {
                v_out.push(s_buf);
            }
        }

        v_out
    }

    fn pre_tokenize(s_text: &str, config: &BpeTokenizerConfig) -> Vec<String> {
        // Keep dispatch explicit.
        if config.s_pre_tokenizer == "utf8_ws_punct_v1" {
            Self::pre_tokenize_utf8_ws_punct(s_text)
        } else {
            // Safe fallback: UTF-8 path to prevent data loss.
            Self::pre_tokenize_utf8_ws_punct(s_text)
        }
    }

    pub fn encode_text(&self, s_text: &str, b_add_bos_eos: bool) -> Vec<usize> {
        let mut v_ids: Vec<usize> = Vec::new();

        let s_norm = self.normalize_unicode_deterministic(s_text);

        if b_add_bos_eos {
            if let Some(i_bos) = self.vocab.encode(S_BOS) {
                v_ids.push(i_bos);
            }
        }

        for s_word in Self::pre_tokenize(&s_norm, &self.config).into_iter() {
            let mut v_word = self.encode_word(&s_word);
            v_ids.append(&mut v_word);
        }

        if b_add_bos_eos {
            if let Some(i_eos) = self.vocab.encode(S_EOS) {
                v_ids.push(i_eos);
            }
        }

        v_ids
    }

    pub fn decode_ids(&self, v_ids: &[usize]) -> String {
        let mut v_words: Vec<String> = Vec::new();
        let mut s_current: String = String::new();

        for &i_id in v_ids.iter() {
            let s_tok = match self.vocab.decode(i_id) {
                Some(s) => s.to_string(),
                None => S_UNK.to_string(),
            };

            // Skip special tokens.
            if s_tok == S_PAD || s_tok == S_UNK || s_tok == S_BOS || s_tok == S_EOS {
                continue;
            }

            // Keep previous tag filter, but make it UTF-8 safe (no ASCII conversion).
            if utils::is_tag_like_utf8(&s_tok) {
                continue;
            }

            if s_tok == S_WB {
                if !s_current.is_empty() {
                    v_words.push(s_current.clone());
                    s_current.clear();
                }
                continue;
            }

            if s_tok.ends_with(S_WB) {
                let s_piece = s_tok.trim_end_matches(S_WB);
                if !s_piece.is_empty() {
                    s_current.push_str(s_piece);
                }
                if !s_current.is_empty() {
                    v_words.push(s_current.clone());
                    s_current.clear();
                }
                continue;
            }

            // For punctuation tokens (single char and not alphanumeric), close current and emit.
            if s_tok.chars().count() == 1 {
                let ch = s_tok.chars().next().unwrap();
                if !(ch.is_alphanumeric() || ch == '_') {
                    if !s_current.is_empty() {
                        v_words.push(s_current.clone());
                        s_current.clear();
                    }
                    v_words.push(s_tok);
                    continue;
                }
            }

            s_current.push_str(&s_tok);
        }

        if !s_current.is_empty() {
            v_words.push(s_current);
        }

        // UTF-8 safe normalization (whitespace trimming only, no lossy conversion).
        utils::normalize_text_utf8_spacing(&v_words.join(" "))
    }

    pub fn encode_word(&self, s_word: &str) -> Vec<usize> {
        let i_unk = self.vocab.encode(S_UNK).unwrap_or(0);

        if s_word.is_empty() {
            return vec![i_unk];
        }

        // Keep behavior stable for single-character punctuation tokens: treat them as atomic symbols.
        if s_word.chars().count() == 1 {
            let ch = s_word.chars().next().unwrap();
            if !(ch.is_alphanumeric() || ch == '_') {
                return vec![self.vocab.encode(s_word).unwrap_or(i_unk)];
            }
        }

        // BPE symbol stream is Unicode scalar values (Rust char) plus word boundary.
        let mut v_syms: Vec<String> = s_word.chars().map(|c| c.to_string()).collect();
        v_syms.push(S_WB.to_string());

        loop {
            let v_pairs = Self::adjacent_pairs(&v_syms);
            if v_pairs.is_empty() {
                break;
            }

            let mut opt_best: Option<(usize, String, String)> = None;
            for (s_left, s_right) in v_pairs.into_iter() {
                let s_key = format!("{}\t{}", s_left, s_right);
                if let Some(&i_rank) = self.merge_ranks.get(&s_key) {
                    match &opt_best {
                        None => opt_best = Some((i_rank, s_left, s_right)),
                        Some((i_best, _, _)) => {
                            if i_rank < *i_best {
                                opt_best = Some((i_rank, s_left, s_right));
                            }
                        }
                    }
                }
            }

            let (_, s_l, s_r) = match opt_best {
                None => break,
                Some(x) => x,
            };

            v_syms = Self::apply_merge(&v_syms, &s_l, &s_r);
        }

        v_syms
            .iter()
            .map(|s| self.vocab.encode(s).unwrap_or(i_unk))
            .collect()
    }

    fn adjacent_pairs(v_syms: &[String]) -> Vec<(String, String)> {
        if v_syms.len() < 2 {
            return Vec::new();
        }
        let mut v_pairs: Vec<(String, String)> = Vec::new();
        for i in 0..(v_syms.len() - 1) {
            v_pairs.push((v_syms[i].clone(), v_syms[i + 1].clone()));
        }
        v_pairs
    }

    fn apply_merge(v_syms: &[String], s_left: &str, s_right: &str) -> Vec<String> {
        let mut v_out: Vec<String> = Vec::new();
        let mut i_pos: usize = 0;

        while i_pos < v_syms.len() {
            if i_pos + 1 < v_syms.len() && v_syms[i_pos] == s_left && v_syms[i_pos + 1] == s_right {
                v_out.push(format!("{}{}", s_left, s_right));
                i_pos += 2;
            } else {
                v_out.push(v_syms[i_pos].clone());
                i_pos += 1;
            }
        }
        v_out
    }

    // Deterministic training API using a config struct.
    pub fn train_from_corpus_with_config(v_texts: &[String], config: BpeTokenizerConfig) -> Result<Self, String> {
        if config.i_vocab_target < 10 {
            return Err("vocab_target_too_small".to_string());
        }
        if v_texts.is_empty() {
            return Err("empty_corpus".to_string());
        }

        // Deterministic word freq based on stable iteration later.
        let mut m_word_freq: HashMap<String, usize> = HashMap::new();

        // Apply same normalization used by encode_text to ensure training/usage symmetry.
        for s in v_texts.iter() {
            let s_norm = if config.s_unicode_normalization == "nfc" {
                utils::normalize_text_utf8_nfc(s)
            } else {
                s.to_string()
            };

            for w in Self::pre_tokenize(&s_norm, &config).into_iter() {
                *m_word_freq.entry(w).or_insert(0) += 1;
            }
        }

        // Deterministic symbol collection.
        let mut set_symbols: HashSet<String> = HashSet::new();
        set_symbols.insert(S_WB.to_string());

        // Collect chars from words (full Unicode).
        for (s_word, _) in m_word_freq.iter() {
            for c in s_word.chars() {
                set_symbols.insert(c.to_string());
            }
        }

        // Special tokens are included first and remain stable.
        let mut v_vocab: Vec<String> = vec![S_PAD.to_string(), S_UNK.to_string(), S_BOS.to_string(), S_EOS.to_string()];

        let mut v_initial: Vec<String> = set_symbols.into_iter().collect();
        v_initial.sort();
        v_vocab.extend(v_initial);

        let mut v_merges: Vec<BpeMerge> = Vec::new();

        // Deterministic word list.
        let mut v_words: Vec<String> = m_word_freq.keys().cloned().collect();
        v_words.sort();

        let mut m_word_syms: HashMap<String, Vec<String>> = HashMap::new();
        for s_word in v_words.iter() {
            let mut v_syms: Vec<String> = s_word.chars().map(|c| c.to_string()).collect();
            v_syms.push(S_WB.to_string());
            m_word_syms.insert(s_word.clone(), v_syms);
        }

        // Deterministic tie-breaking for merges:
        // - primary: highest count
        // - secondary: lexical order of pair (left, right)
        while v_vocab.len() < config.i_vocab_target {
            let m_pair_counts = Self::count_pairs(&m_word_syms, &m_word_freq);

            let mut opt_best: Option<((String, String), usize)> = None;
            for (k, &i_count) in m_pair_counts.iter() {
                if i_count < config.i_min_pair_count {
                    continue;
                }
                match &opt_best {
                    None => opt_best = Some((k.clone(), i_count)),
                    Some((k_best, i_best)) => {
                        if i_count > *i_best {
                            opt_best = Some((k.clone(), i_count));
                        } else if i_count == *i_best {
                            if k.0 < k_best.0 || (k.0 == k_best.0 && k.1 < k_best.1) {
                                opt_best = Some((k.clone(), i_count));
                            }
                        }
                    }
                }
            }

            let ((s_left, s_right), _) = match opt_best {
                None => break,
                Some(x) => x,
            };

            let s_merged = format!("{}{}", s_left, s_right);
            if !v_vocab.contains(&s_merged) {
                v_vocab.push(s_merged.clone());
            }

            v_merges.push(BpeMerge {
                s_left: s_left.clone(),
                s_right: s_right.clone(),
                s_merged: s_merged.clone(),
            });

            // Apply merge deterministically in a stable word order.
            let mut v_keys: Vec<String> = m_word_syms.keys().cloned().collect();
            v_keys.sort();

            for s_word in v_keys.into_iter() {
                if let Some(v_syms) = m_word_syms.get(&s_word).cloned() {
                    let v_new = Self::apply_merge(&v_syms, &s_left, &s_right);
                    m_word_syms.insert(s_word, v_new);
                }
            }
        }

        let v_refs: Vec<&str> = v_vocab.iter().map(|s| s.as_str()).collect();
        let vocab = Vocab::new(v_refs);

        Self::new(vocab, v_merges, config)
    }

    pub fn train_from_corpus(v_texts: &[String], i_vocab_target: usize, i_min_pair_count: usize) -> Result<Self, String> {
        let mut config = BpeTokenizerConfig::default();
        config.i_vocab_target = i_vocab_target;
        config.i_min_pair_count = i_min_pair_count;
        Self::train_from_corpus_with_config(v_texts, config)
    }

    fn count_pairs(
        m_word_syms: &HashMap<String, Vec<String>>,
        m_word_freq: &HashMap<String, usize>,
    ) -> HashMap<(String, String), usize> {
        let mut m_counts: HashMap<(String, String), usize> = HashMap::new();

        for (s_word, v_syms) in m_word_syms.iter() {
            let i_freq = *m_word_freq.get(s_word).unwrap_or(&0);
            if i_freq == 0 || v_syms.len() < 2 {
                continue;
            }

            for i in 0..(v_syms.len() - 1) {
                let k = (v_syms[i].clone(), v_syms[i + 1].clone());
                *m_counts.entry(k).or_insert(0) += i_freq;
            }
        }

        m_counts
    }
}
