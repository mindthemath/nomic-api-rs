//! # nomic-serve
//!
//! A fast embedding server for nomic-embed-text-v1.5 using ONNX Runtime.
//!
//! ## Why Sequential Processing Only
//!
//! This server processes each text individually rather than batching. This is **required**
//! for the nomic-embed-text-v1.5 ONNX model because it exhibits cross-sample interference:
//! when multiple texts are batched together, each text's embedding is affected by the
//! other texts in the batch. See README.md for detailed explanation and evidence.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use ndarray::ShapeError;
use ort::{
    execution_providers::CUDAExecutionProvider,
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
use tokenizers::Tokenizer;
use tower_http::trace::TraceLayer;
use tracing::info;

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
}

impl AppState {
    async fn new(model: PathBuf, tok: PathBuf, use_gpu: bool) -> anyhow::Result<Self> {
        let mut builder =
            SessionBuilder::new()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

        // Enable GPU execution provider if requested
        if use_gpu {
            builder =
                builder.with_execution_providers([CUDAExecutionProvider::default().build()])?;
        }

        let session = builder.commit_from_file(model)?;
        let tokenizer = Tokenizer::from_file(tok).map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
        })
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Deserialize)]
struct EmbedRequest {
    inputs: String,
}

#[derive(Deserialize)]
struct BatchRequest {
    inputs: Vec<String>,
}

#[derive(Serialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
    tokens: usize,
    time_ms: f64,
}

#[derive(Serialize)]
struct BatchResponse {
    embeddings: Vec<Vec<f32>>,
    tokens: Vec<usize>,
    time_ms: f64,
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() {
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
    let use_gpu = std::env::var("USE_GPU")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    let state = match AppState::new(model_path.clone(), tok_path.clone(), use_gpu).await {
        Ok(state) => state,
        Err(e) => {
            eprintln!("Failed to initialize server: {}", e);
            eprintln!("Model path: {:?}", model_path);
            eprintln!("Tokenizer path: {:?}", tok_path);
            std::process::exit(1);
        }
    };
    let device = if use_gpu { "GPU" } else { "CPU" };
    info!(
        "🚀 Nomic embedding server ready on http://0.0.0.0:{} ({})",
        port, device
    );

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/embed", post(embed_handler))
        .route("/batch", post(batch_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = match format!("0.0.0.0:{}", port).parse() {
        Ok(addr) => addr,
        Err(e) => {
            eprintln!("Failed to parse address: {}", e);
            std::process::exit(1);
        }
    };

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("Failed to bind to address {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
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

    // Process single text
    let (embedding, tokens) = embed_single(&state, &req.inputs)?;

    Ok(Json(EmbedResponse {
        embedding,
        tokens,
        time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

async fn batch_handler(
    State(state): State<AppState>,
    Json(req): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, Error> {
    let start = Instant::now();

    // Process each text individually - batching causes cross-sample interference
    // with this model (see README.md for explanation)
    let mut embeddings = Vec::with_capacity(req.inputs.len());
    let mut tokens = Vec::with_capacity(req.inputs.len());

    for text in &req.inputs {
        let (emb, tok) = embed_single(&state, text)?;
        embeddings.push(emb);
        tokens.push(tok);
    }

    Ok(Json(BatchResponse {
        embeddings,
        tokens,
        time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

// ============================================================================
// Embedding
// ============================================================================

/// Mean pooling: average token embeddings weighted by attention mask
fn mean_pool(embeddings: &[f32], attention_mask: &[i64], seq_len: usize) -> Vec<f32> {
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

/// Embed a single text, returning the 768-dim mean-pooled embedding
fn embed_single(state: &AppState, text: &str) -> Result<(Vec<f32>, usize), Error> {
    // Tokenize
    let encoding = state
        .tokenizer
        .encode(text, true)
        .map_err(|e| Error(StatusCode::BAD_REQUEST, e.to_string()))?;

    let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
    let token_count = ids.len();
    let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&i| i as i64).collect();
    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&i| i as i64)
        .collect();

    // Build input tensors with shape [1, seq_len]
    let input_shape = vec![1i64, token_count as i64];
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

    // Run inference
    let mut session_guard = state.session.lock().unwrap();
    let outputs = session_guard.run(SessionInputs::from(inputs_map))?;
    let (output_shape, raw_embedding) = outputs[0].try_extract_tensor::<f32>()?.to_owned();

    let embedding_vec = raw_embedding.to_vec();
    let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();

    // Apply mean pooling based on output shape
    let embedding = match shape_dims.as_slice() {
        [768] => embedding_vec,
        [1, 768] => embedding_vec,
        [1, num_tokens, 768] | [num_tokens, 768] => {
            mean_pool(&embedding_vec, &attention_mask, *num_tokens)
        }
        _ => {
            return Err(Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Unexpected embedding shape: {:?}", shape_dims),
            ));
        }
    };

    Ok((embedding, token_count))
}
