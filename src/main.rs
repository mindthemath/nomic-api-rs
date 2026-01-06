use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use ndarray::ShapeError;
use ort::{
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session, SessionInputValue, SessionInputs,
    },
    value::Value,
    Error as OrtError,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
use tower_http::trace::TraceLayer;
use tracing::info;

// ============================================================================
// Batch Mode Configuration
// ============================================================================

/// Controls how multiple texts are processed during inference.
///
/// Set via the `BATCH_MODE` environment variable:
/// - `NO_BATCH` (default): Sequential processing, one text at a time
/// - `SAFE_BATCH`: Same as NO_BATCH (guaranteed identical results)
/// - `PAD_BATCH`: Full batching with padding (fastest, but ~0.5 embedding differences)
///
/// IMPORTANT: This Nomic model has cross-sample computation - when multiple texts
/// are batched together, each text's embedding is affected by OTHER texts in the batch.
/// This happens even without padding (same token counts). Verified:
/// - Same text batched with itself: identical (diff ≈ 0)
/// - Same text batched with any different text: significant diff (~0.5)
///
/// Therefore SAFE_BATCH = NO_BATCH is the ONLY correct implementation for exact results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchMode {
    /// Process each text individually in a loop.
    /// Slowest but guaranteed correct results.
    NoBatch,

    /// Identical to NoBatch - processes each text sequentially.
    /// Exists for API compatibility and as a "safe" batching option.
    /// Future optimization may enable grouping-based batching if we can
    /// make it produce identical results.
    SafeBatch,

    /// Batch all texts together with padding to uniform length.
    /// Fastest for large batches, but produces slightly different embeddings
    /// due to batched inference mechanics.
    /// Differences are typically small (~0.01-0.2 in values).
    PadBatch,
}

impl BatchMode {
    fn from_env() -> Self {
        match std::env::var("BATCH_MODE")
            .unwrap_or_default()
            .to_uppercase()
            .as_str()
        {
            "SAFE_BATCH" => BatchMode::SafeBatch,
            "PAD_BATCH" => BatchMode::PadBatch,
            _ => BatchMode::NoBatch,
        }
    }
}

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug)]
struct Error(StatusCode, String);

impl From<OrtError> for Error {
    fn from(e: OrtError) -> Self {
        Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

impl From<ShapeError> for Error {
    fn from(e: ShapeError) -> Self {
        Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        (self.0, self.1).into_response()
    }
}

// ============================================================================
// Application State
// ============================================================================

#[derive(Clone)]
struct AppState {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    batch_mode: BatchMode,
}

impl AppState {
    async fn new(model: PathBuf, tok: PathBuf, batch_mode: BatchMode) -> anyhow::Result<Self> {
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model)?;
        let mut tokenizer = Tokenizer::from_file(tok).map_err(|e| anyhow::anyhow!(e))?;

        // Enable padding for batch processing (used by PAD_BATCH mode)
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            batch_mode,
        })
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbedRequest {
    Single { inputs: String },
    Multiple { inputs: Vec<String> },
}

#[derive(Serialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
    tokens: Vec<usize>,
    time_ms: f64,
    batch_mode: String,
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::var("MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("model_quantized.onnx"));
    let tok_path = std::env::var("TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("tokenizer.json"));
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let batch_mode = BatchMode::from_env();

    let state = AppState::new(model_path, tok_path, batch_mode).await?;
    info!(
        "🚀 Nomic server ready on http://0.0.0.0:{} (batch_mode={:?})",
        port, batch_mode
    );

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/embed", post(embed_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

// ============================================================================
// HTTP Handlers
// ============================================================================

async fn health_handler() -> &'static str {
    "OK"
}

async fn embed_handler(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, Error> {
    let start = Instant::now();

    let texts = match req {
        EmbedRequest::Single { inputs } => vec![inputs],
        EmbedRequest::Multiple { inputs } => inputs,
    };

    let (embeddings, tokens) = match state.batch_mode {
        BatchMode::NoBatch | BatchMode::SafeBatch => embed_sequential(&state, &texts)?,
        BatchMode::PadBatch => embed_padded_batch(&state, &texts)?,
    };

    Ok(Json(EmbedResponse {
        embeddings,
        tokens,
        time_ms: start.elapsed().as_secs_f64() * 1000.0,
        batch_mode: format!("{:?}", state.batch_mode),
    }))
}

// ============================================================================
// Embedding Functions
// ============================================================================

/// Mean pooling for a single sequence
/// Takes embeddings for one sequence and its attention mask, returns pooled 768-dim vector
fn mean_pool_sequence(embeddings: &[f32], attention_mask: &[i64], seq_len: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; 768];
    let mut mask_sum = 0.0f32;

    for (i, &mask_val) in attention_mask.iter().enumerate().take(seq_len) {
        if mask_val > 0 {
            mask_sum += 1.0;
            let start_idx = i * 768;
            for j in 0..768 {
                pooled[j] += embeddings[start_idx + j];
            }
        }
    }

    if mask_sum > 1e-9 {
        for val in &mut pooled {
            *val /= mask_sum;
        }
    }

    pooled
}

/// Embed a single text (no batching)
fn embed_single(state: &AppState, text: &str) -> Result<(Vec<f32>, usize), Error> {
    let encoding = state
        .tokenizer
        .encode(text, true)
        .map_err(|e| Error(StatusCode::BAD_REQUEST, e.to_string()))?;

    let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
    let tokens = ids.len();
    let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&i| i as i64).collect();
    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&i| i as i64)
        .collect();

    let input_shape = vec![1i64, tokens as i64];
    let input_ids_value: Value = Value::from_array((input_shape.clone(), ids))?.into();
    let token_type_ids_value: Value =
        Value::from_array((input_shape.clone(), token_type_ids))?.into();
    let attention_mask_value: Value =
        Value::from_array((input_shape, attention_mask.clone()))?.into();

    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        "input_ids".to_string(),
        SessionInputValue::from(input_ids_value),
    );
    inputs_map.insert(
        "token_type_ids".to_string(),
        SessionInputValue::from(token_type_ids_value),
    );
    inputs_map.insert(
        "attention_mask".to_string(),
        SessionInputValue::from(attention_mask_value),
    );

    let mut session_guard = state.session.lock().unwrap();
    let outputs = session_guard.run(SessionInputs::from(inputs_map))?;
    let (output_shape, embedding) = outputs[0].try_extract_tensor::<f32>()?.to_owned();

    let embedding_vec = embedding.to_vec();
    let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();

    let embedding = match shape_dims.as_slice() {
        [768] => embedding_vec,
        [1, 768] => embedding_vec,
        [1, num_tokens, 768] | [num_tokens, 768] => {
            mean_pool_sequence(&embedding_vec, &attention_mask, *num_tokens)
        }
        _ => {
            return Err(Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Unexpected embedding shape: {:?}", shape_dims),
            ));
        }
    };

    Ok((embedding, tokens))
}

/// NO_BATCH: Sequential processing using embed_single (for-loop approach)
fn embed_sequential(
    state: &AppState,
    texts: &[String],
) -> Result<(Vec<Vec<f32>>, Vec<usize>), Error> {
    let mut embeddings = Vec::with_capacity(texts.len());
    let mut tokens = Vec::with_capacity(texts.len());

    for text in texts {
        let (emb, tok) = embed_single(state, text)?;
        embeddings.push(emb);
        tokens.push(tok);
    }

    Ok((embeddings, tokens))
}

/// UNUSED: Group texts by token count, batch within groups.
///
/// This was intended to provide batching without padding for texts of equal length.
/// However, testing proved the Nomic model has cross-sample computation:
/// batching ANY different texts together changes embeddings by ~0.5, regardless
/// of padding. This is a model property (likely batch normalization or similar).
///
/// Kept for reference - demonstrates the approach that WOULD work for models
/// without cross-sample effects.
#[allow(dead_code)]
fn embed_grouped_batch(
    state: &AppState,
    texts: &[String],
) -> Result<(Vec<Vec<f32>>, Vec<usize>), Error> {
    if texts.is_empty() {
        return Ok((vec![], vec![]));
    }

    // First, tokenize all texts to get their lengths
    let encodings: Vec<_> = texts
        .iter()
        .map(|t| {
            state
                .tokenizer
                .encode(t.as_str(), true)
                .map_err(|e| Error(StatusCode::BAD_REQUEST, e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Group indices by token count
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, encoding) in encodings.iter().enumerate() {
        let token_count = encoding.get_ids().len();
        groups.entry(token_count).or_default().push(idx);
    }

    // Process each group - texts in same group have identical length
    let mut all_embeddings = vec![Vec::new(); texts.len()];
    let mut all_tokens = vec![0usize; texts.len()];

    for (token_count, indices) in groups {
        if indices.len() == 1 {
            // Single item, use direct embedding
            let idx = indices[0];
            let (emb, tok) = embed_single(state, &texts[idx])?;
            all_embeddings[idx] = emb;
            all_tokens[idx] = tok;
        } else {
            // Multiple items with same token count - batch them
            let batch_texts: Vec<&str> = indices.iter().map(|&i| texts[i].as_str()).collect();
            let (batch_embeddings, batch_tokens) =
                embed_uniform_batch(state, &batch_texts, token_count)?;

            for (i, idx) in indices.iter().enumerate() {
                all_embeddings[*idx] = batch_embeddings[i].clone();
                all_tokens[*idx] = batch_tokens[i];
            }
        }
    }

    Ok((all_embeddings, all_tokens))
}

/// Batch embed texts that all have the same token count (no padding needed)
/// See embed_grouped_batch for why this is currently unused.
#[allow(dead_code)]
fn embed_uniform_batch(
    state: &AppState,
    texts: &[&str],
    token_count: usize,
) -> Result<(Vec<Vec<f32>>, Vec<usize>), Error> {
    let batch_size = texts.len();

    // Tokenize all texts (they should all produce same length)
    let encodings: Vec<_> = texts
        .iter()
        .map(|t| {
            state
                .tokenizer
                .encode(*t, true)
                .map_err(|e| Error(StatusCode::BAD_REQUEST, e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Build batched tensors
    let mut all_input_ids: Vec<i64> = Vec::with_capacity(batch_size * token_count);
    let mut all_token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * token_count);
    let mut all_attention_mask: Vec<i64> = Vec::with_capacity(batch_size * token_count);

    for encoding in &encodings {
        all_input_ids.extend(encoding.get_ids().iter().map(|&i| i as i64));
        all_token_type_ids.extend(encoding.get_type_ids().iter().map(|&i| i as i64));
        all_attention_mask.extend(encoding.get_attention_mask().iter().map(|&i| i as i64));
    }

    let input_shape = vec![batch_size as i64, token_count as i64];
    let input_ids_value: Value = Value::from_array((input_shape.clone(), all_input_ids))?.into();
    let token_type_ids_value: Value =
        Value::from_array((input_shape.clone(), all_token_type_ids))?.into();
    let attention_mask_value: Value =
        Value::from_array((input_shape, all_attention_mask.clone()))?.into();

    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        "input_ids".to_string(),
        SessionInputValue::from(input_ids_value),
    );
    inputs_map.insert(
        "token_type_ids".to_string(),
        SessionInputValue::from(token_type_ids_value),
    );
    inputs_map.insert(
        "attention_mask".to_string(),
        SessionInputValue::from(attention_mask_value),
    );

    let mut session_guard = state.session.lock().unwrap();
    let outputs = session_guard.run(SessionInputs::from(inputs_map))?;
    let (output_shape, embedding) = outputs[0].try_extract_tensor::<f32>()?.to_owned();

    let embedding_vec = embedding.to_vec();
    let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();

    let embeddings = match shape_dims.as_slice() {
        [b, seq_len, 768] if *b == batch_size => {
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let seq_start = i * seq_len * 768;
                let seq_end = seq_start + seq_len * 768;
                let seq_embeddings = &embedding_vec[seq_start..seq_end];

                let mask_start = i * token_count;
                let mask_end = mask_start + token_count;
                let seq_mask = &all_attention_mask[mask_start..mask_end];

                let pooled = mean_pool_sequence(seq_embeddings, seq_mask, *seq_len);
                results.push(pooled);
            }
            results
        }
        _ => {
            return Err(Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Unexpected uniform batch shape: {:?}, expected [{}, {}, 768]",
                    shape_dims, batch_size, token_count
                ),
            ));
        }
    };

    let tokens = vec![token_count; batch_size];
    Ok((embeddings, tokens))
}

/// PAD_BATCH: Full batching with padding to uniform length
/// Fastest but produces slightly different results due to padding
fn embed_padded_batch(
    state: &AppState,
    texts: &[String],
) -> Result<(Vec<Vec<f32>>, Vec<usize>), Error> {
    if texts.is_empty() {
        return Ok((vec![], vec![]));
    }

    // For batch size 1, use single embedding (no padding needed)
    if texts.len() == 1 {
        let (emb, tok) = embed_single(state, &texts[0])?;
        return Ok((vec![emb], vec![tok]));
    }

    // Encode all texts with padding to batch longest
    let encodings = state
        .tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| Error(StatusCode::BAD_REQUEST, e.to_string()))?;

    let batch_size = encodings.len();
    let max_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);

    // Track original token counts (non-padding tokens)
    let token_counts: Vec<usize> = encodings
        .iter()
        .map(|e| e.get_attention_mask().iter().filter(|&&m| m == 1).count())
        .collect();

    // Build batched tensors
    let mut all_input_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
    let mut all_token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
    let mut all_attention_mask: Vec<i64> = Vec::with_capacity(batch_size * max_len);

    for encoding in &encodings {
        all_input_ids.extend(encoding.get_ids().iter().map(|&i| i as i64));
        all_token_type_ids.extend(encoding.get_type_ids().iter().map(|&i| i as i64));
        all_attention_mask.extend(encoding.get_attention_mask().iter().map(|&i| i as i64));
    }

    let input_shape = vec![batch_size as i64, max_len as i64];
    let input_ids_value: Value = Value::from_array((input_shape.clone(), all_input_ids))?.into();
    let token_type_ids_value: Value =
        Value::from_array((input_shape.clone(), all_token_type_ids))?.into();
    let attention_mask_value: Value =
        Value::from_array((input_shape, all_attention_mask.clone()))?.into();

    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        "input_ids".to_string(),
        SessionInputValue::from(input_ids_value),
    );
    inputs_map.insert(
        "token_type_ids".to_string(),
        SessionInputValue::from(token_type_ids_value),
    );
    inputs_map.insert(
        "attention_mask".to_string(),
        SessionInputValue::from(attention_mask_value),
    );

    let mut session_guard = state.session.lock().unwrap();
    let outputs = session_guard.run(SessionInputs::from(inputs_map))?;
    let (output_shape, embedding) = outputs[0].try_extract_tensor::<f32>()?.to_owned();

    let embedding_vec = embedding.to_vec();
    let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();

    let embeddings = match shape_dims.as_slice() {
        [b, seq_len, 768] if *b == batch_size => {
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let seq_start = i * seq_len * 768;
                let seq_end = seq_start + seq_len * 768;
                let seq_embeddings = &embedding_vec[seq_start..seq_end];

                let mask_start = i * max_len;
                let mask_end = mask_start + max_len;
                let seq_mask = &all_attention_mask[mask_start..mask_end];

                let pooled = mean_pool_sequence(seq_embeddings, seq_mask, *seq_len);
                results.push(pooled);
            }
            results
        }
        _ => {
            return Err(Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Unexpected padded batch shape: {:?}, expected [{}, {}, 768]",
                    shape_dims, batch_size, max_len
                ),
            ));
        }
    };

    Ok((embeddings, token_counts))
}
