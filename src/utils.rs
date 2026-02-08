// utils.rs
// Description: Small UTF-8 safe utilities for decoding fallbacks and text normalization.
//              Adds safe JSON checkpoint serialization and atomic file write helpers.
//
// History:
// - 2026-02-01: Consolidate utility helpers into utils.rs.
// - 2026-02-01: Add JSON checkpoint helpers and atomic write for save and load.
// - 2026-02-08: Add UTF-8 safe normalization helpers (spacing and NFC normalization).
// Author: Marcus Schlieper

use crate::layer::Vocab;

// UTF-8 safe check for tokens like "<...>".
// NOTE: This is intentionally conservative and does not attempt HTML parsing.
pub fn is_tag_like_utf8(s_tok: &str) -> bool {
    if s_tok.len() < 3 {
        return false;
    }
    let b = s_tok.as_bytes();
    b[0] == b'<' && b[b.len() - 1] == b'>'
}

// UTF-8 safe spacing normalization:
// - collapse multiple ASCII spaces into one space
// - trim leading and trailing whitespace
// NOTE: This does not delete or replace Unicode characters.
pub fn normalize_text_utf8_spacing(s_text: &str) -> String {
    let mut s_out = s_text.to_string();

    while s_out.contains("  ") {
        s_out = s_out.replace("  ", " ");
    }

    s_out.trim().to_string()
}

// UTF-8 deterministic normalization: NFC.
// NOTE: Requires the unicode-normalization crate in Cargo.toml.
// Security: normalization is pure data transformation.
pub fn normalize_text_utf8_nfc(s_text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    s_text.nfc().collect::<String>()
}

pub fn decode_via_vocab_ascii(vocab: &Vocab, v_ids: &[usize]) -> String {
    // NOTE: Keep legacy fallback, but do not destroy UTF-8 in tokens if present.
    let s_unk = "<unk>";
    let i_unk = vocab.encode(s_unk).unwrap_or(0);

    let mut v_out: Vec<String> = Vec::with_capacity(v_ids.len());
    for &i_id in v_ids.iter() {
        if let Some(s_tok) = vocab.decode(i_id) {
            v_out.push(s_tok.to_string());
        } else if let Some(s_tok) = vocab.decode(i_unk) {
            v_out.push(s_tok.to_string());
        }
    }

    normalize_text_utf8_spacing(&v_out.join(" "))
}

// ---- Checkpoint helpers (JSON) ----

pub fn checkpoint_to_json_ascii(cp: &crate::layer::llm_checkpoint_v2) -> Result<String, String> {
    serde_json::to_string(cp).map_err(|_| "checkpoint_serialize_error".to_string())
}

pub fn checkpoint_from_json_ascii(s_json: &str) -> Result<crate::layer::llm_checkpoint_v2, String> {
    if s_json.trim().is_empty() {
        return Err("checkpoint_json_empty".to_string());
    }
    serde_json::from_str(s_json).map_err(|_| "checkpoint_deserialize_error".to_string())
}

pub fn write_file_atomic_ascii(s_path: &str, s_content: &str) -> Result<(), String> {
    if s_path.trim().is_empty() {
        return Err("file_path_empty".to_string());
    }

    let p_path = std::path::Path::new(s_path);

    if let Some(p_parent) = p_path.parent() {
        if !p_parent.as_os_str().is_empty() {
            std::fs::create_dir_all(p_parent).map_err(|_| "checkpoint_mkdir_error".to_string())?;
        }
    }

    let s_tmp = format!("{}.tmp", s_path);

    std::fs::write(&s_tmp, s_content).map_err(|_| "checkpoint_write_error".to_string())?;

    std::fs::rename(&s_tmp, s_path).map_err(|_| {
        let _ = std::fs::remove_file(&s_tmp);
        "checkpoint_rename_error".to_string()
    })?;

    Ok(())
}