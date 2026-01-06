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
use tokenizers::Tokenizer;
use tower_http::trace::TraceLayer;
use tracing::info;

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

#[derive(Clone)]
struct AppState {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
}

#[derive(Deserialize)]
struct EmbedRequest {
    inputs: String,
}

#[derive(Serialize)]
struct EmbedResponse {
    embeddings: Vec<f32>,
    tokens: usize,
    time_ms: f64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::var("MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("model_quantized.onnx"));
    let tok_path = std::env::var("TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("tokenizer.json"));

    let state = AppState::new(model_path, tok_path).await?;
    info!("🚀 Nomic server ready on http://0.0.0.0:8080");

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/embed", post(embed_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:8080".parse()?;
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

impl AppState {
    async fn new(model: PathBuf, tok: PathBuf) -> anyhow::Result<Self> {
        // Environment is managed internally in ort 2.0
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model)?;
        let tokenizer = Tokenizer::from_file(tok).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
        })
    }
}

async fn health_handler() -> &'static str {
    "OK"
}

async fn embed_handler(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, Error> {
    let start = Instant::now();
    let encoding = state
        .tokenizer
        .encode(req.inputs, true)
        .map_err(|e| Error(StatusCode::BAD_REQUEST, e.to_string()))?;
    let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
    let tokens = ids.len();

    // Get token_type_ids (usually all zeros for single sequence)
    let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&i| i as i64).collect();

    // Get attention_mask (all ones for tokens that should be attended to)
    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&i| i as i64)
        .collect();

    // Create named inputs for the model
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

    // Handle different output shapes:
    // - If shape is [1, 768] or [768], use directly (pooled embedding)
    // - If shape is [1, tokens, 768] or [tokens, 768], we need to pool
    let embedding_vec = embedding.to_vec();
    let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();

    let embedding = match shape_dims.as_slice() {
        // Already pooled: [768] or [1, 768]
        [768] => embedding_vec,
        [1, 768] => embedding_vec,
        // Token-level: [1, tokens, 768] or [tokens, 768]
        [1, num_tokens, 768] | [num_tokens, 768] => {
            // Mean pooling (official nomic-embed-text-v1.5 approach):
            // Average token embeddings weighted by attention mask
            // This matches the official implementation: sum(embeddings * mask) / sum(mask)
            let mut pooled = vec![0.0f32; 768];
            let num_tokens = *num_tokens;
            let mut mask_sum = 0.0f32;
            
            for (i, &mask_val) in attention_mask.iter().enumerate().take(num_tokens) {
                if mask_val > 0 {
                    mask_sum += 1.0;
                    let start_idx = i * 768;
                    for j in 0..768 {
                        pooled[j] += embedding_vec[start_idx + j];
                    }
                }
            }
            // Normalize by number of non-padding tokens (clamp to avoid division by zero)
            if mask_sum > 1e-9 {
                for val in &mut pooled {
                    *val /= mask_sum;
                }
            }
            pooled
        }
        _ => {
            return Err(Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Unexpected embedding shape: {:?}", shape_dims),
            ));
        }
    };

    Ok(Json(EmbedResponse {
        embeddings: embedding,
        tokens,
        time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}
