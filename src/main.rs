//! # nomic-serve
//!
//! A fast multimodal embedding server for nomic-embed-text-v1.5 and nomic-embed-vision-v1.5
//! using ONNX Runtime.
//!
//! ## Endpoints
//!
//! - `POST /txt/embed` - Single text embedding
//! - `POST /txt/batch` - Multiple text embeddings
//! - `POST /query`     - Single text with search_query prefix (convenience)
//! - `POST /img/embed` - Single image embedding
//! - `POST /img/batch` - Multiple image embeddings
//! - `POST /img/stats` - Image statistics (EXIF + colors)
//!
//! ## Why Sequential Processing
//!
//! This server processes each text/image individually rather than batching. This is **required**
//! for the nomic ONNX models because they exhibit cross-sample interference when batched.
//! See README.md for detailed explanation.

mod image_stats;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::{header, Method, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::{DynamicImage, ImageReader};
use ndarray::ShapeError;
#[cfg(feature = "cuda")]
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, ExecutionProviderDispatch,
};
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
    io::Cursor,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::sync::OnceCell;
use tokenizers::Tokenizer;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    trace::TraceLayer,
};
use tracing::{info, warn};
use utoipa::{OpenApi, ToSchema};

// ============================================================================
// Constants
// ============================================================================

/// Maximum input size for images (20 MB)
const MAX_IMAGE_SIZE: usize = 20 * 1024 * 1024;

// ============================================================================
// Path Resolution
// ============================================================================

/// Resolve a path, trying CWD first, then relative to executable directory.
/// This allows double-clicking the executable on macOS (where CWD != exe dir).
fn resolve_model_path<P: AsRef<Path>>(relative: P) -> PathBuf {
    let relative = relative.as_ref();

    // If it's already absolute or exists relative to CWD, use as-is
    if relative.is_absolute() || relative.exists() {
        return relative.to_path_buf();
    }

    // Try relative to executable's directory (for double-click scenarios)
    if let Ok(exe_path) = std::env::current_exe() {
        // Canonicalize to resolve symlinks (important on macOS .app bundles)
        if let Ok(canonical) = exe_path.canonicalize() {
            if let Some(exe_dir) = canonical.parent() {
                let exe_relative = exe_dir.join(relative);
                if exe_relative.exists() {
                    return exe_relative;
                }
            }
        }
    }

    // Fall back to original path (will fail later with proper error)
    relative.to_path_buf()
}

/// Resolve model path with smart fallback: try full precision first, then quantized.
/// If env var is set, use that explicitly. Otherwise, try model.onnx, then model_quantized.onnx.
fn resolve_model_path_with_fallback(
    env_var: &str,
    default_dir: &str,
    default_filename: &str,
) -> PathBuf {
    // If explicit env var is set, use it
    if let Ok(explicit_path) = std::env::var(env_var) {
        return resolve_model_path(explicit_path);
    }

    // Try full precision model first
    let full_path = resolve_model_path(format!("{}/model.onnx", default_dir));
    if full_path.exists() {
        return full_path;
    }

    // Fall back to quantized model
    let quantized_path = resolve_model_path(format!("{}/model_quantized.onnx", default_dir));
    if quantized_path.exists() {
        return quantized_path;
    }

    // Neither exists, return default (will fail later with proper error)
    resolve_model_path(format!("{}/{}", default_dir, default_filename))
}

/// Image preprocessing constants (CLIP-style, from preprocessor_config.json)
const IMAGE_SIZE: usize = 224;
const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGE_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug)]
pub struct Error(pub StatusCode, pub String);

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

/// Model variant detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelVariant {
    /// Full precision (FP32) - supports batching
    Full,
    /// Quantized (INT8) - may have batching restrictions
    Quantized,
}

impl ModelVariant {
    /// Detect model variant from filename
    fn from_path(path: &Path) -> Self {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        if filename.contains("quantized") || filename.contains("int8") || filename.contains("uint8")
        {
            ModelVariant::Quantized
        } else {
            ModelVariant::Full
        }
    }
}

/// Text embedding state (tokenizer + ONNX session)
struct TextState {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    variant: ModelVariant,
    max_batch_size: usize,
}

/// Vision embedding state (ONNX session only, no tokenizer)
pub struct VisionState {
    session: Mutex<Session>,
    max_batch_size: usize,
}

/// Combined application state
#[derive(Clone)]
pub struct AppState {
    text: Arc<OnceCell<TextState>>,
    vision: Arc<OnceCell<VisionState>>,
    default_avg_method: image_stats::AveragingMethod,
    txt_model_path: PathBuf,
    tokenizer_path: PathBuf,
    img_model_path: PathBuf,
    gpu_enabled: bool,
    use_gpu_requested: bool,
}

impl AppState {
    #[allow(unused_variables)]
    async fn new(
        txt_model_path: PathBuf,
        tokenizer_path: PathBuf,
        img_model_path: PathBuf,
        default_avg_method: image_stats::AveragingMethod,
        use_gpu: bool,
    ) -> anyhow::Result<Self> {
        #[cfg(feature = "cuda")]
        let gpu_enabled = use_gpu;
        #[cfg(not(feature = "cuda"))]
        let gpu_enabled = false;

        Ok(Self {
            text: Arc::new(OnceCell::new()),
            vision: Arc::new(OnceCell::new()),
            default_avg_method,
            txt_model_path,
            tokenizer_path,
            img_model_path,
            gpu_enabled,
            use_gpu_requested: use_gpu,
        })
    }

    /// Lazy-load and initialize the text model if not already done
    async fn get_text_state(&self) -> Result<Arc<TextState>, Error> {
        self.text
            .get_or_try_init(|| async {
                // Ensure model and tokenizer exist (download if needed)
                ensure_model_exists(&self.txt_model_path, "txt").await?;
                ensure_model_exists(&self.tokenizer_path, "txt").await?;

                let variant = ModelVariant::from_path(&self.txt_model_path);
                info!(
                    "Initializing text model: {:?} (variant: {:?})",
                    self.txt_model_path, variant
                );

                let model_bytes = std::fs::read(&self.txt_model_path).map_err(|e| {
                    Error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read model file: {}", e),
                    )
                })?;

                let builder = create_session_builder(self.use_gpu_requested)?;
                let session = builder.commit_from_memory(&model_bytes).map_err(|e| {
                    Error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to load model: {}", e),
                    )
                })?;

                let tokenizer = Tokenizer::from_file(&self.tokenizer_path)
                    .map_err(|e| Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

                let txt_max_batch_size = std::env::var("TXT_MAX_BATCH_SIZE")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(16);

                let effective_max_batch = if variant == ModelVariant::Quantized {
                    1
                } else {
                    txt_max_batch_size
                };

                Ok(TextState {
                    session: Mutex::new(session),
                    tokenizer,
                    variant,
                    max_batch_size: effective_max_batch,
                })
            })
            .await
    }

    /// Lazy-load and initialize the vision model if not already done
    async fn get_vision_state(&self) -> Result<&VisionState, Error> {
        self.vision
            .get_or_try_init(|| async {
                // Ensure model exists (download if needed)
                ensure_model_exists(&self.img_model_path, "img").await?;

                let variant = ModelVariant::from_path(&self.img_model_path);
                info!(
                    "Initializing vision model: {:?} (variant: {:?})",
                    self.img_model_path, variant
                );

                let model_bytes = std::fs::read(&self.img_model_path).map_err(|e| {
                    Error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read model file: {}", e),
                    )
                })?;

                let builder = create_session_builder(self.use_gpu_requested)?;
                let session = builder.commit_from_memory(&model_bytes).map_err(|e| {
                    Error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to load model: {}", e),
                    )
                })?;

                let img_max_batch_size = std::env::var("IMG_MAX_BATCH_SIZE")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(4);

                Ok(VisionState {
                    session: Mutex::new(session),
                    max_batch_size: img_max_batch_size,
                })
            })
            .await
    }
}

/// Create a session builder with requested execution providers
fn create_session_builder(use_gpu: bool) -> Result<SessionBuilder, Error> {
    #[cfg(feature = "cuda")]
    let mut builder = SessionBuilder::new()
        .map_err(|e| Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    #[cfg(not(feature = "cuda"))]
    let builder = SessionBuilder::new()
        .map_err(|e| Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    #[cfg(feature = "cuda")]
    if use_gpu {
        let cuda_provider: ExecutionProviderDispatch = CUDAExecutionProvider::default().into();
        let cpu_provider: ExecutionProviderDispatch = CPUExecutionProvider::default().into();
        builder = builder
            .with_execution_providers([cuda_provider, cpu_provider])
            .map_err(|e| {
                warn!("Failed to initialize CUDA: {}. Falling back to CPU.", e);
                Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })?;
    } else {
        let cpu_provider: ExecutionProviderDispatch = CPUExecutionProvider::default().into();
        builder = builder
            .with_execution_providers([cpu_provider])
            .map_err(|e| Error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    Ok(builder)
}

/// Ensure a model file exists locally, downloading it from HuggingFace if missing
async fn ensure_model_exists(path: &Path, category: &str) -> Result<(), Error> {
    if path.exists() {
        return Ok(());
    }

    info!("Model file missing: {:?}. Attempting to download...", path);

    let filename = path
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| Error(StatusCode::INTERNAL_SERVER_ERROR, "Invalid model path".to_string()))?;

    let base_url = if category == "txt" {
        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main"
    } else {
        "https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/resolve/main"
    };

    let url = if filename.ends_with(".onnx") {
        format!("{}/onnx/{}", base_url, filename)
    } else {
        format!("{}/{}", base_url, filename)
    };

    download_file(&url, path).await
}

/// Download a file from URL to local path
async fn download_file(url: &str, dest: &Path) -> Result<(), Error> {
    info!("Downloading {} to {:?}...", url, dest);

    // Create parent directories if they don't exist
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create directory: {}", e),
            )
        })?;
    }

    let response = reqwest::get(url).await.map_err(|e| {
        Error(
            StatusCode::BAD_GATEWAY,
            format!("Failed to download from HF: {}", e),
        )
    })?;

    if !response.status().is_success() {
        return Err(Error(
            StatusCode::NOT_FOUND,
            format!("HF returned {} for {}", response.status(), url),
        ));
    }

    let bytes = response.bytes().await.map_err(|e| {
        Error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read download stream: {}", e),
        )
    })?;

    std::fs::write(dest, bytes).map_err(|e| {
        Error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to save model file: {}", e),
        )
    })?;

    info!("Successfully downloaded {:?}", dest);
    Ok(())
}


// ============================================================================
// Request/Response Types - Text
// ============================================================================

/// Prefix type for nomic-embed-text-v1.5 model
#[derive(Deserialize, Serialize, ToSchema, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[schema(example = "search_query")]
enum Prefix {
    /// For search queries
    SearchQuery,
    /// For search documents
    SearchDocument,
    /// For classification tasks
    Classification,
    /// For clustering tasks
    Clustering,
}

impl Default for Prefix {
    fn default() -> Self {
        Prefix::SearchQuery
    }
}

impl std::fmt::Display for Prefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Prefix::SearchQuery => write!(f, "search_query"),
            Prefix::SearchDocument => write!(f, "search_document"),
            Prefix::Classification => write!(f, "classification"),
            Prefix::Clustering => write!(f, "clustering"),
        }
    }
}

#[derive(Deserialize, ToSchema)]
struct TextEmbedRequest {
    /// Text to embed
    #[schema(example = "ONNX in Rust is fast")]
    input: String,
    /// Embedding dimension (1-768). Supports Matryoshka embeddings.
    #[serde(default = "default_dim")]
    #[schema(example = 768, minimum = 1, maximum = 768)]
    dim: usize,
    /// Prefix to prepend. Defaults to "search_query".
    #[serde(default)]
    #[schema(example = "search_query")]
    prefix: Prefix,
}

#[derive(Deserialize, ToSchema)]
struct TextBatchRequest {
    /// List of texts to embed
    #[schema(example = json!(["Hello world", "Goodbye world"]))]
    #[serde(alias = "input")]
    inputs: Vec<String>,
    /// Embedding dimension (1-768)
    #[serde(default = "default_dim")]
    #[schema(example = 768, minimum = 1, maximum = 768)]
    dim: usize,
    /// Prefix to prepend to each text
    #[serde(default)]
    #[schema(example = "search_document")]
    prefix: Prefix,
}

#[derive(Deserialize, ToSchema)]
struct TextQueryRequest {
    /// Query text to embed (search_query prefix applied automatically)
    #[schema(example = "What is ONNX?")]
    input: String,
    /// Embedding dimension (1-768)
    #[serde(default = "default_dim")]
    #[schema(example = 768, minimum = 1, maximum = 768)]
    dim: usize,
}

#[derive(Serialize, ToSchema)]
struct TextEmbedResponse {
    /// Embedding vector
    #[schema(example = json!([0.123, 0.456, -0.789]))]
    embedding: Vec<f32>,
    /// Number of tokens
    #[schema(example = 6)]
    tokens: usize,
    /// Processing time in milliseconds
    #[schema(example = 12.34)]
    time_ms: f64,
}

#[derive(Serialize, ToSchema)]
struct TextBatchResponse {
    /// List of embedding vectors
    #[schema(example = json!([[0.123, 0.456], [0.789, -0.123]]))]
    embeddings: Vec<Vec<f32>>,
    /// Token count for each input
    #[schema(example = json!([4, 5]))]
    tokens: Vec<usize>,
    /// Total processing time in milliseconds
    #[schema(example = 45.67)]
    time_ms: f64,
}

// ============================================================================
// Request/Response Types - Image
// ============================================================================

#[derive(Deserialize, ToSchema)]
struct ImageEmbedRequest {
    /// Image input: URL (http/https), data URL (data:image/...), or raw base64
    #[schema(example = "https://picsum.photos/400/300")]
    #[serde(alias = "content")]
    input: String,
    /// Embedding dimension (1-768)
    #[serde(default = "default_dim")]
    #[schema(example = 768, minimum = 1, maximum = 768)]
    dim: usize,
}

#[derive(Deserialize, ToSchema)]
struct ImageBatchRequest {
    /// List of image inputs (URLs or base64)
    #[schema(example = json!(["https://picsum.photos/200/200", "https://picsum.photos/300/400", "https://picsum.photos/300/400"]))]
    #[serde(alias = "contents")]
    #[serde(alias = "input")]
    inputs: Vec<String>,
    /// Embedding dimension (1-768)
    #[serde(default = "default_dim")]
    #[schema(example = 768, minimum = 1, maximum = 768)]
    dim: usize,
}

#[derive(Serialize, ToSchema)]
struct ImageEmbedResponse {
    /// Embedding vector
    #[schema(example = json!([0.123, 0.456, -0.789]))]
    embedding: Vec<f32>,
    /// Processing time in milliseconds
    #[schema(example = 45.67)]
    time_ms: f64,
}

#[derive(Serialize, ToSchema)]
struct ImageBatchResponse {
    /// List of embedding vectors
    #[schema(example = json!([[0.123, 0.456], [0.789, -0.123]]))]
    embeddings: Vec<Vec<f32>>,
    /// Total processing time in milliseconds
    #[schema(example = 89.12)]
    time_ms: f64,
}

// ============================================================================
// Common Types
// ============================================================================

pub fn default_dim() -> usize {
    768
}

#[derive(Serialize, ToSchema)]
struct HealthResponse {
    /// Health status
    #[schema(example = "OK")]
    status: String,
    /// Text model loaded
    #[schema(example = true)]
    text_model: bool,
    /// Vision model loaded
    #[schema(example = true)]
    vision_model: bool,
    /// GPU enabled
    #[schema(example = true)]
    gpu_enabled: bool,
}

#[derive(Serialize, ToSchema)]
struct InfoResponse {
    /// Default averaging method for image stats
    #[schema(example = "geometric")]
    averaging: String,
    /// Text model file path (if loaded)
    #[schema(example = "models/txt/model_quantized.onnx")]
    txt_model: Option<String>,
    /// Tokenizer file path (if loaded)
    #[schema(example = "models/txt/tokenizer.json")]
    tokenizer: Option<String>,
    /// Vision model file path (if loaded)
    #[schema(example = "models/img/model_quantized.onnx")]
    img_model: Option<String>,
    /// Maximum batch size for text embeddings
    #[schema(example = 1)]
    txt_max_batch_size: Option<usize>,
    /// Maximum batch size for image embeddings
    #[schema(example = 64)]
    img_max_batch_size: Option<usize>,
    /// Whether GPU is enabled
    #[schema(example = true)]
    gpu_enabled: bool,
}

#[derive(Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error message
    #[schema(example = "Tokenization failed")]
    pub error: String,
}

// ============================================================================
// OpenAPI Documentation
// ============================================================================

#[derive(OpenApi)]
#[openapi(
    info(
        title = "nomic-serve",
        description = "Fast multimodal embedding server for nomic-embed-text-v1.5 and nomic-embed-vision-v1.5",
        version = "0.2.0",
        license(name = "MIT")
    ),
    paths(
        health_handler,
        info_handler,
        txt_embed_handler,
        txt_batch_handler,
        txt_query_handler,
        img_embed_handler,
        img_batch_handler,
        image_stats::img_stats_handler,
    ),
    components(schemas(
        TextEmbedRequest,
        TextEmbedResponse,
        TextBatchRequest,
        TextBatchResponse,
        TextQueryRequest,
        ImageEmbedRequest,
        ImageEmbedResponse,
        ImageBatchRequest,
        ImageBatchResponse,
        HealthResponse,
        InfoResponse,
        ErrorResponse,
        Prefix,
        image_stats::ImageStatsRequest,
        image_stats::ImageStatsResponse,
        image_stats::AveragingMethod,
        image_stats::ColorData,
        image_stats::AverageColorInfo,
        image_stats::ColorInfo,
    ))
)]
struct ApiDoc;

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Model configuration - resolve paths but don't check for existence yet (lazy download)
    let txt_model_path = std::env::var("TXT_MODEL")
        .map(PathBuf::from)
        .map(resolve_model_path)
        .unwrap_or_else(|_| resolve_model_path("models/txt/model_quantized.onnx"));

    let tok_path = std::env::var("TOKENIZER")
        .map(PathBuf::from)
        .map(resolve_model_path)
        .unwrap_or_else(|_| resolve_model_path("models/txt/tokenizer.json"));

    let img_model_path = std::env::var("IMG_MODEL")
        .map(PathBuf::from)
        .map(resolve_model_path)
        .unwrap_or_else(|_| resolve_model_path("models/img/model_quantized.onnx"));

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let disable_cors = std::env::var("DISABLE_CORS")
        .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes" | "y"))
        .unwrap_or(false);

    let use_gpu = std::env::var("USE_GPU")
        .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes" | "y"))
        .unwrap_or(false);

    // Default averaging method for image stats
    let default_avg_method = std::env::var("AVERAGING")
        .ok()
        .and_then(|v| match v.to_lowercase().as_str() {
            "arithmetic" => Some(image_stats::AveragingMethod::Arithmetic),
            "geometric" => Some(image_stats::AveragingMethod::Geometric),
            _ => None,
        })
        .unwrap_or(image_stats::AveragingMethod::Geometric);

    // Initialize application state
    let state = match AppState::new(
        txt_model_path,
        tok_path,
        img_model_path,
        default_avg_method,
        use_gpu,
    )
    .await
    {
        Ok(state) => state,
        Err(e) => {
            eprintln!("Failed to initialize server: {}", e);
            std::process::exit(1);
        }
    };

    let gpu_status = if state.gpu_enabled {
        "✓ (CUDA)"
    } else {
        "✗ (CPU)"
    };

    info!("🚀 Nomic embedding server ready on http://0.0.0.0:{}", port);
    info!("   GPU Support:  {}", gpu_status);
    let body_limit_mb = std::env::var("MAX_BODY_SIZE_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    info!("   Max body size: {} MB", body_limit_mb);
    info!("📚 API docs available at http://0.0.0.0:{}/docs", port);
    info!("💡 Models will be downloaded on first request if missing locally.");

    // Build router - base routes
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/info", get(info_handler))
        // Text endpoints (canonical)
        .route("/txt/embed", post(txt_embed_handler))
        .route("/txt/batch", post(txt_batch_handler))
        .route("/txt/query", post(txt_query_handler))
        // Image endpoints
        .route("/img/embed", post(img_embed_handler))
        .route("/img/batch", post(img_batch_handler))
        .route("/img/stats", post(image_stats::img_stats_handler));

    let mut app = app
        // Legacy aliases (silent, undocumented)
        .route("/embed", post(txt_embed_handler))
        .route("/batch", post(txt_batch_handler))
        .route("/query", post(txt_query_handler))
        // OpenAPI
        .route("/openapi.json", get(openapi_handler))
        .route("/docs/openapi.json", get(openapi_handler))
        .route("/docs", get(docs_handler))
        .layer(DefaultBodyLimit::max(body_limit_mb * 1024 * 1024))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Apply CORS middleware if not disabled
    if disable_cors {
        info!("CORS disabled via DISABLE_CORS env var");
    } else {
        let cors_layer = build_cors_layer();
        app = app.layer(cors_layer);
        info!("CORS enabled");
    }

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
// CORS Configuration
// ============================================================================

const DEFAULT_CORS_ORIGINS: &[&str] = &[
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
];

fn build_cors_layer() -> CorsLayer {
    let env_origins: Vec<String> = std::env::var("CORS_ORIGINS")
        .map(|s| s.split(',').map(|o| o.trim().to_string()).collect())
        .unwrap_or_default();

    let parsed_origins: Vec<_> = if env_origins.is_empty() {
        DEFAULT_CORS_ORIGINS
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect()
    } else {
        let parsed: Vec<_> = env_origins.iter().filter_map(|o| o.parse().ok()).collect();
        if parsed.is_empty() {
            warn!("CORS_ORIGINS contains no valid origins, falling back to localhost defaults");
            DEFAULT_CORS_ORIGINS
                .iter()
                .filter_map(|o| o.parse().ok())
                .collect()
        } else {
            parsed
        }
    };

    info!("CORS: Allowing {} origin(s)", parsed_origins.len());

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(parsed_origins))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::ACCEPT, header::AUTHORIZATION])
        .allow_credentials(true)
}

// ============================================================================
// HTTP Handlers - Status
// ============================================================================

/// Check server health and model availability
#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Server is healthy", body = HealthResponse)
    ),
    tag = "status"
)]
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "OK".to_string(),
        text_model: state.txt_model_path.exists() || state.text.get().is_some(),
        vision_model: state.img_model_path.exists() || state.vision.get().is_some(),
        gpu_enabled: state.gpu_enabled,
    })
}

/// Get server information including model paths and configuration
#[utoipa::path(
    get,
    path = "/info",
    responses(
        (status = 200, description = "Server information", body = InfoResponse)
    ),
    tag = "status"
)]
async fn info_handler(State(state): State<AppState>) -> Json<InfoResponse> {
    Json(InfoResponse {
        averaging: match state.default_avg_method {
            image_stats::AveragingMethod::Arithmetic => "arithmetic".to_string(),
            image_stats::AveragingMethod::Geometric => "geometric".to_string(),
        },
        txt_model: Some(state.txt_model_path.to_string_lossy().to_string()),
        tokenizer: Some(state.tokenizer_path.to_string_lossy().to_string()),
        img_model: Some(state.img_model_path.to_string_lossy().to_string()),
        txt_max_batch_size: state.text.get().map(|t| t.max_batch_size),
        img_max_batch_size: state.vision.get().map(|v| v.max_batch_size),
        gpu_enabled: state.gpu_enabled,
    })
}

// ============================================================================
// HTTP Handlers - Text
// ============================================================================

/// Generate a single text embedding with configurable prefix and dimension
#[utoipa::path(
    post,
    path = "/txt/embed",
    request_body = TextEmbedRequest,
    responses(
        (status = 200, description = "Embedding generated successfully", body = TextEmbedResponse),
        (status = 400, description = "Bad request", body = ErrorResponse),
        (status = 503, description = "Text model not loaded", body = ErrorResponse)
    ),
    tag = "text"
)]
async fn txt_embed_handler(
    State(state): State<AppState>,
    Json(req): Json<TextEmbedRequest>,
) -> Result<Json<TextEmbedResponse>, Error> {
    let start = Instant::now();

    let text_state = state.get_text_state().await?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    let prefixed_text = format!("{}: {}", req.prefix, req.input);
    let (mut embedding, tokens, tokenize_time, inference_time, postprocess_time) =
        embed_text(text_state, &prefixed_text)?;

    if req.dim < embedding.len() {
        embedding.truncate(req.dim);
    }

    let total_time = start.elapsed();
    info!(
        "Text embed timing - tokenize: {:.2}ms, ONNX: {:.2}ms, postprocess: {:.2}ms, total: {:.2}ms",
        tokenize_time.as_secs_f64() * 1000.0,
        inference_time.as_secs_f64() * 1000.0,
        postprocess_time.as_secs_f64() * 1000.0,
        total_time.as_secs_f64() * 1000.0
    );

    Ok(Json(TextEmbedResponse {
        embedding,
        tokens,
        time_ms: total_time.as_secs_f64() * 1000.0,
    }))
}

/// Generate embeddings for multiple texts with configurable prefix and dimension
#[utoipa::path(
    post,
    path = "/txt/batch",
    request_body = TextBatchRequest,
    responses(
        (status = 200, description = "Embeddings generated successfully", body = TextBatchResponse),
        (status = 400, description = "Bad request", body = ErrorResponse),
        (status = 503, description = "Text model not loaded", body = ErrorResponse)
    ),
    tag = "text"
)]
async fn txt_batch_handler(
    State(state): State<AppState>,
    Json(req): Json<TextBatchRequest>,
) -> Result<Json<TextBatchResponse>, Error> {
    let start = Instant::now();

    let text_state = state.get_text_state().await?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    let batch_size = req.inputs.len();

    // Fail fast if quantized model and batch_size > 1 (check this first for better error message)
    if text_state.variant == ModelVariant::Quantized && batch_size > 1 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!(
                "Batching is not supported for quantized text models. This model ({:?}) exhibits severe cross-sample interference when batched (~0.5 max diff, ~50-60%% cosine similarity). Use the full precision (FP32) model for batching, or process texts individually (batch_size=1).",
                state.txt_model_path.file_name()
            ),
        ));
    }

    // Check batch size limit
    if batch_size > text_state.max_batch_size {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!(
                "Batch size {} exceeds maximum allowed batch size of {}",
                batch_size, text_state.max_batch_size
            ),
        ));
    }

    let mut embeddings = Vec::with_capacity(req.inputs.len());
    let mut tokens = Vec::with_capacity(req.inputs.len());
    let mut total_tokenize = std::time::Duration::ZERO;
    let mut total_inference = std::time::Duration::ZERO;
    let mut total_postprocess = std::time::Duration::ZERO;

    for text in &req.inputs {
        let prefixed_text = format!("{}: {}", req.prefix, text);
        let (mut emb, tok, tokenize_time, inference_time, postprocess_time) =
            embed_text(text_state, &prefixed_text)?;
        if req.dim < emb.len() {
            emb.truncate(req.dim);
        }
        embeddings.push(emb);
        tokens.push(tok);
        total_tokenize += tokenize_time;
        total_inference += inference_time;
        total_postprocess += postprocess_time;
    }

    let total_time = start.elapsed();
    let count = req.inputs.len() as f64;
    info!(
        "Text batch timing - count: {}, tokenize: {:.2}ms, ONNX: {:.2}ms, postprocess: {:.2}ms, total: {:.2}ms, avg: {:.2}ms",
        req.inputs.len(),
        total_tokenize.as_secs_f64() * 1000.0,
        total_inference.as_secs_f64() * 1000.0,
        total_postprocess.as_secs_f64() * 1000.0,
        total_time.as_secs_f64() * 1000.0,
        total_time.as_secs_f64() * 1000.0 / count
    );

    Ok(Json(TextBatchResponse {
        embeddings,
        tokens,
        time_ms: total_time.as_secs_f64() * 1000.0,
    }))
}

/// Generate a text embedding optimized for search queries (uses search_query prefix)
#[utoipa::path(
    post,
    path = "/txt/query",
    request_body = TextQueryRequest,
    responses(
        (status = 200, description = "Query embedding generated successfully", body = TextEmbedResponse),
        (status = 400, description = "Bad request", body = ErrorResponse),
        (status = 503, description = "Text model not loaded", body = ErrorResponse)
    ),
    tag = "text"
)]
async fn txt_query_handler(
    State(state): State<AppState>,
    Json(req): Json<TextQueryRequest>,
) -> Result<Json<TextEmbedResponse>, Error> {
    let start = Instant::now();

    let text_state = state.get_text_state().await?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    // Always use search_query prefix for /query endpoint
    let prefixed_text = format!("search_query: {}", req.input);
    let (mut embedding, tokens, tokenize_time, inference_time, postprocess_time) =
        embed_text(text_state, &prefixed_text)?;

    if req.dim < embedding.len() {
        embedding.truncate(req.dim);
    }

    let total_time = start.elapsed();
    info!(
        "Text query timing - tokenize: {:.2}ms, ONNX: {:.2}ms, postprocess: {:.2}ms, total: {:.2}ms",
        tokenize_time.as_secs_f64() * 1000.0,
        inference_time.as_secs_f64() * 1000.0,
        postprocess_time.as_secs_f64() * 1000.0,
        total_time.as_secs_f64() * 1000.0
    );

    Ok(Json(TextEmbedResponse {
        embedding,
        tokens,
        time_ms: total_time.as_secs_f64() * 1000.0,
    }))
}

// ============================================================================
// HTTP Handlers - Image
// ============================================================================

/// Generate a single image embedding from URL or base64-encoded image
#[utoipa::path(
    post,
    path = "/img/embed",
    request_body = ImageEmbedRequest,
    responses(
        (status = 200, description = "Embedding generated successfully", body = ImageEmbedResponse),
        (status = 400, description = "Bad request (invalid image, fetch failed)", body = ErrorResponse),
        (status = 503, description = "Vision model not loaded", body = ErrorResponse)
    ),
    tag = "image"
)]
async fn img_embed_handler(
    State(state): State<AppState>,
    Json(req): Json<ImageEmbedRequest>,
) -> Result<Json<ImageEmbedResponse>, Error> {
    let start = Instant::now();

    let vision_state = state.get_vision_state().await?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    let decode_start = Instant::now();
    let image = decode_image(&req.input).await?;
    let decode_time = decode_start.elapsed();

    let (mut embedding, preprocess_time, onnx_time) = embed_image(vision_state, &image)?;

    info!(
        "Image embed timing - decode: {:.2}ms, preprocess: {:.2}ms, ONNX: {:.2}ms, total: {:.2}ms",
        decode_time.as_secs_f64() * 1000.0,
        preprocess_time.as_secs_f64() * 1000.0,
        onnx_time.as_secs_f64() * 1000.0,
        start.elapsed().as_secs_f64() * 1000.0
    );

    if req.dim < embedding.len() {
        embedding.truncate(req.dim);
    }

    Ok(Json(ImageEmbedResponse {
        embedding,
        time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

/// Generate embeddings for multiple images from URLs or base64-encoded images
#[utoipa::path(
    post,
    path = "/img/batch",
    request_body = ImageBatchRequest,
    responses(
        (status = 200, description = "Embeddings generated successfully", body = ImageBatchResponse),
        (status = 400, description = "Bad request", body = ErrorResponse),
        (status = 503, description = "Vision model not loaded", body = ErrorResponse)
    ),
    tag = "image"
)]
async fn img_batch_handler(
    State(state): State<AppState>,
    Json(req): Json<ImageBatchRequest>,
) -> Result<Json<ImageBatchResponse>, Error> {
    let start = Instant::now();

    let vision_state = state.get_vision_state().await?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    let batch_size = req.inputs.len();

    // Check batch size limit
    if batch_size > vision_state.max_batch_size {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!(
                "Batch size {} exceeds maximum allowed batch size of {}",
                batch_size, vision_state.max_batch_size
            ),
        ));
    }

    // Decode all images first
    let mut images = Vec::with_capacity(req.inputs.len());
    for input in &req.inputs {
        images.push(decode_image(input).await?);
    }

    // Batch inference (more efficient than sequential)
    let mut embeddings = embed_image_batch(vision_state, &images)?;

    // Truncate to requested dimension
    for emb in &mut embeddings {
        if req.dim < emb.len() {
            emb.truncate(req.dim);
        }
    }

    Ok(Json(ImageBatchResponse {
        embeddings,
        time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

// ============================================================================
// HTTP Handlers - OpenAPI
// ============================================================================

async fn openapi_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let openapi = ApiDoc::openapi();
    let mut spec: serde_json::Value =
        serde_json::to_value(&openapi).unwrap_or(serde_json::json!({}));
    if let Some(obj) = spec.as_object_mut() {
        obj.insert("openapi".to_string(), serde_json::json!("3.1.0"));

        // Update the averaging_method example to reflect the runtime default
        if let Some(components) = obj.get_mut("components") {
            if let Some(components_obj) = components.as_object_mut() {
                if let Some(schemas) = components_obj.get_mut("schemas") {
                    if let Some(schemas_obj) = schemas.as_object_mut() {
                        if let Some(image_stats_request) = schemas_obj.get_mut("ImageStatsRequest")
                        {
                            if let Some(props) = image_stats_request.get_mut("properties") {
                                if let Some(averaging_method) = props.get_mut("averaging_method") {
                                    if let Some(averaging_method_obj) =
                                        averaging_method.as_object_mut()
                                    {
                                        let default_example = match state.default_avg_method {
                                            image_stats::AveragingMethod::Arithmetic => {
                                                "arithmetic"
                                            }
                                            image_stats::AveragingMethod::Geometric => "geometric",
                                        };
                                        averaging_method_obj.insert(
                                            "example".to_string(),
                                            serde_json::json!(default_example),
                                        );
                                    }
                                }
                            }
                        }
                        // Also update AverageColorInfo.method example
                        if let Some(avg_color_info) = schemas_obj.get_mut("AverageColorInfo") {
                            if let Some(props) = avg_color_info.get_mut("properties") {
                                if let Some(method) = props.get_mut("method") {
                                    if let Some(method_obj) = method.as_object_mut() {
                                        let default_example = match state.default_avg_method {
                                            image_stats::AveragingMethod::Arithmetic => {
                                                "arithmetic"
                                            }
                                            image_stats::AveragingMethod::Geometric => "geometric",
                                        };
                                        method_obj.insert(
                                            "example".to_string(),
                                            serde_json::json!(default_example),
                                        );
                                    }
                                }
                            }
                        }
                        // Also update InfoResponse.averaging example
                        if let Some(info_response) = schemas_obj.get_mut("InfoResponse") {
                            if let Some(props) = info_response.get_mut("properties") {
                                if let Some(averaging) = props.get_mut("averaging") {
                                    if let Some(averaging_obj) = averaging.as_object_mut() {
                                        let default_example = match state.default_avg_method {
                                            image_stats::AveragingMethod::Arithmetic => {
                                                "arithmetic"
                                            }
                                            image_stats::AveragingMethod::Geometric => "geometric",
                                        };
                                        averaging_obj.insert(
                                            "example".to_string(),
                                            serde_json::json!(default_example),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Json(spec)
}

async fn docs_handler() -> impl IntoResponse {
    const SWAGGER_HTML: &str = include_str!("../static/swagger-ui/index.html");
    (
        StatusCode::OK,
        [("content-type", "text/html")],
        SWAGGER_HTML,
    )
}

// ============================================================================
// Image Decoding
// ============================================================================

/// Decode image from URL, data URL, or raw base64
async fn decode_image(content: &str) -> Result<DynamicImage, Error> {
    let bytes = if content.starts_with("http://") || content.starts_with("https://") {
        // URL - fetch with timeout and size limit
        fetch_image_url(content).await?
    } else if content.starts_with("data:") {
        // Data URL - extract base64 portion
        let parts: Vec<&str> = content.splitn(2, ',').collect();
        if parts.len() != 2 {
            return Err(Error(
                StatusCode::BAD_REQUEST,
                "Invalid data URL format".to_string(),
            ));
        }
        BASE64.decode(parts[1]).map_err(|e| {
            Error(
                StatusCode::BAD_REQUEST,
                format!("Invalid base64 in data URL: {}", e),
            )
        })?
    } else {
        // Raw base64
        BASE64
            .decode(content)
            .map_err(|e| Error(StatusCode::BAD_REQUEST, format!("Invalid base64: {}", e)))?
    };

    // Check size limit
    if bytes.len() > MAX_IMAGE_SIZE {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!(
                "Image too large: {} bytes (max {} MB)",
                bytes.len(),
                MAX_IMAGE_SIZE / 1024 / 1024
            ),
        ));
    }

    // Decode image
    let cursor = Cursor::new(bytes);
    let reader = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| {
            Error(
                StatusCode::BAD_REQUEST,
                format!("Failed to detect image format: {}", e),
            )
        })?;

    reader.decode().map_err(|e| {
        Error(
            StatusCode::BAD_REQUEST,
            format!("Failed to decode image: {}", e),
        )
    })
}

/// Fetch image from URL with timeout and size limit
async fn fetch_image_url(url: &str) -> Result<Vec<u8>, Error> {
    let client_start = Instant::now();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| {
            Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("HTTP client error: {}", e),
            )
        })?;
    let client_time = client_start.elapsed();

    let request_start = Instant::now();
    let response = client.get(url).send().await.map_err(|e| {
        let request_time = request_start.elapsed();
        info!(
            "Image fetch failed after {:.2}ms (client build: {:.2}ms) - {}",
            request_time.as_secs_f64() * 1000.0,
            client_time.as_secs_f64() * 1000.0,
            e
        );
        Error(
            StatusCode::BAD_REQUEST,
            format!("Failed to fetch URL: {}", e),
        )
    })?;
    let request_time = request_start.elapsed();

    if !response.status().is_success() {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("URL returned status {}", response.status()),
        ));
    }

    // Check content-length header if available
    if let Some(len) = response.content_length() {
        if len as usize > MAX_IMAGE_SIZE {
            return Err(Error(
                StatusCode::BAD_REQUEST,
                format!(
                    "Image too large: {} bytes (max {} MB)",
                    len,
                    MAX_IMAGE_SIZE / 1024 / 1024
                ),
            ));
        }
    }

    let body_start = Instant::now();
    let request_time_clone = request_time;
    let result = response
        .bytes()
        .await
        .map(|b| {
            let body_time = body_start.elapsed();
            let total_time = request_time_clone + body_time;
            info!(
                "Image fetch - body downloaded in {:.2}ms (request: {:.2}ms, total: {:.2}ms)",
                body_time.as_secs_f64() * 1000.0,
                request_time_clone.as_secs_f64() * 1000.0,
                total_time.as_secs_f64() * 1000.0
            );
            b.to_vec()
        })
        .map_err(|e| {
            Error(
                StatusCode::BAD_REQUEST,
                format!("Failed to read image data: {}", e),
            )
        });
    result
}

// ============================================================================
// Text Embedding
// ============================================================================

/// Mean pooling for text embeddings
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

/// Embed single text, returns 768-dim embedding and timing info
fn embed_text(
    state: &TextState,
    text: &str,
) -> Result<
    (
        Vec<f32>,
        usize,
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
    ),
    Error,
> {
    let tokenize_start = Instant::now();
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

    let input_shape = vec![1i64, token_count as i64];
    let input_ids_value: Value = Value::from_array((input_shape.clone(), ids))?.into();
    let token_type_ids_value: Value =
        Value::from_array((input_shape.clone(), token_type_ids))?.into();
    let attention_mask_value: Value =
        Value::from_array((input_shape, attention_mask.clone()))?.into();

    let inputs_vec = vec![
        (
            "input_ids".to_string(),
            SessionInputValue::from(input_ids_value),
        ),
        (
            "token_type_ids".to_string(),
            SessionInputValue::from(token_type_ids_value),
        ),
        (
            "attention_mask".to_string(),
            SessionInputValue::from(attention_mask_value),
        ),
    ];
    let tokenize_time = tokenize_start.elapsed();

    let inference_start = Instant::now();
    let mut session_guard = state.session.lock().unwrap();
    let outputs = session_guard.run(SessionInputs::from(inputs_vec))?;
    let inference_time = inference_start.elapsed();

    let (output_shape, raw_embedding) = outputs[0].try_extract_tensor::<f32>()?.to_owned();

    let postprocess_start = Instant::now();
    let embedding_vec = raw_embedding.to_vec();
    let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();

    let mut embedding = match shape_dims.as_slice() {
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

    // L2 normalize for cosine similarity compatibility
    l2_normalize(&mut embedding);
    let postprocess_time = postprocess_start.elapsed();

    Ok((
        embedding,
        token_count,
        tokenize_time,
        inference_time,
        postprocess_time,
    ))
}

// ============================================================================
// Image Embedding
// ============================================================================

/// Preprocess image for CLIP-style model
pub fn preprocess_image(image: &DynamicImage) -> Vec<f32> {
    // Convert to RGB
    let rgb = image.to_rgb8();

    // Resize: shortest edge to IMAGE_SIZE, maintain aspect ratio
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let (new_w, new_h) = if w < h {
        (IMAGE_SIZE, (h * IMAGE_SIZE) / w)
    } else {
        ((w * IMAGE_SIZE) / h, IMAGE_SIZE)
    };

    let resized = image::imageops::resize(
        &rgb,
        new_w as u32,
        new_h as u32,
        image::imageops::FilterType::CatmullRom, // Bicubic
    );

    // Center crop to IMAGE_SIZE x IMAGE_SIZE
    let (rw, rh) = (resized.width() as usize, resized.height() as usize);
    let left = (rw - IMAGE_SIZE) / 2;
    let top = (rh - IMAGE_SIZE) / 2;
    let cropped = image::imageops::crop_imm(
        &resized,
        left as u32,
        top as u32,
        IMAGE_SIZE as u32,
        IMAGE_SIZE as u32,
    );
    let cropped = cropped.to_image();

    // Convert to NCHW float32 tensor with normalization
    let mut tensor = vec![0.0f32; 3 * IMAGE_SIZE * IMAGE_SIZE];

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let pixel = cropped.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - IMAGE_MEAN[c]) / IMAGE_STD[c];
                // NCHW format: tensor[c * H * W + y * W + x]
                tensor[c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] = normalized;
            }
        }
    }

    tensor
}

/// L2 normalize a vector
pub fn l2_normalize(vec: &mut Vec<f32>) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for val in vec.iter_mut() {
            *val /= norm;
        }
    }
}

/// Embed single image, returns 768-dim L2-normalized embedding and timing info
pub fn embed_image(
    state: &VisionState,
    image: &DynamicImage,
) -> Result<(Vec<f32>, std::time::Duration, std::time::Duration), Error> {
    let preprocess_start = Instant::now();
    let tensor = preprocess_image(image);
    let preprocess_time = preprocess_start.elapsed();

    // Create input tensor with shape [1, 3, 224, 224]
    let input_shape = vec![1i64, 3, IMAGE_SIZE as i64, IMAGE_SIZE as i64];
    let pixel_values: Value = Value::from_array((input_shape, tensor))?.into();

    let inputs_vec = vec![(
        "pixel_values".to_string(),
        SessionInputValue::from(pixel_values),
    )];

    let inference_start = Instant::now();
    let mut session_guard = state.session.lock().unwrap();
    let outputs = session_guard.run(SessionInputs::from(inputs_vec))?;
    let inference_time = inference_start.elapsed();

    let (output_shape, raw_output) = outputs[0].try_extract_tensor::<f32>()?.to_owned();

    let output_vec = raw_output.to_vec();
    let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();

    // Extract CLS token (first token) from [batch, num_tokens, hidden_dim]
    let mut embedding = match shape_dims.as_slice() {
        [1, 768] | [768] => output_vec,
        [1, _num_tokens, 768] => {
            // CLS token is the first token
            output_vec[..768].to_vec()
        }
        [_num_tokens, 768] => output_vec[..768].to_vec(),
        _ => {
            return Err(Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Unexpected vision model output shape: {:?}", shape_dims),
            ));
        }
    };

    // L2 normalize
    l2_normalize(&mut embedding);

    Ok((embedding, preprocess_time, inference_time))
}

/// Embed multiple images in a batch, returns list of 768-dim L2-normalized embeddings
///
/// Note: FP32 models batch perfectly (no interference). Quantized models may show
/// ~1% difference (cosine similarity ~0.99) due to quantization artifacts.
pub fn embed_image_batch(
    state: &VisionState,
    images: &[DynamicImage],
) -> Result<Vec<Vec<f32>>, Error> {
    if images.is_empty() {
        return Ok(Vec::new());
    }

    let preprocess_start = Instant::now();
    let mut tensors = Vec::with_capacity(images.len());
    for image in images {
        tensors.push(preprocess_image(image));
    }
    let preprocess_time = preprocess_start.elapsed();

    // Stack tensors: [N, 3, 224, 224]
    let batch_size = images.len();
    let mut batch_tensor = Vec::with_capacity(batch_size * 3 * IMAGE_SIZE * IMAGE_SIZE);
    for tensor in tensors {
        batch_tensor.extend_from_slice(&tensor);
    }

    let input_shape = vec![batch_size as i64, 3, IMAGE_SIZE as i64, IMAGE_SIZE as i64];
    let pixel_values: Value = Value::from_array((input_shape, batch_tensor))?.into();

    let inputs_vec = vec![(
        "pixel_values".to_string(),
        SessionInputValue::from(pixel_values),
    )];

    let inference_start = Instant::now();
    let mut session_guard = state.session.lock().unwrap();
    let outputs = session_guard.run(SessionInputs::from(inputs_vec))?;
    let inference_time = inference_start.elapsed();

    info!(
        "Vision batch inference timing - preprocess: {:.2}ms, ONNX: {:.2}ms, batch_size: {}",
        preprocess_time.as_secs_f64() * 1000.0,
        inference_time.as_secs_f64() * 1000.0,
        batch_size
    );

    let (output_shape, raw_output) = outputs[0].try_extract_tensor::<f32>()?.to_owned();
    let output_vec = raw_output.to_vec();
    let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();

    // Extract CLS token for each image in batch
    let mut embeddings = Vec::with_capacity(batch_size);
    match shape_dims.as_slice() {
        [n, 768] if *n == batch_size => {
            // Output is [N, 768] - already extracted CLS tokens
            for i in 0..batch_size {
                let start = i * 768;
                let end = start + 768;
                embeddings.push(output_vec[start..end].to_vec());
            }
        }
        [n, _num_tokens, 768] if *n == batch_size => {
            // Output is [N, num_tokens, 768] - extract CLS token (first token) for each
            let tokens_per_image = shape_dims[1];
            for i in 0..batch_size {
                let start = i * tokens_per_image * 768;
                let end = start + 768;
                embeddings.push(output_vec[start..end].to_vec());
            }
        }
        _ => {
            return Err(Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Unexpected vision model output shape: {:?} (expected batch_size={})",
                    shape_dims, batch_size
                ),
            ));
        }
    }

    // L2 normalize each embedding
    for embedding in &mut embeddings {
        l2_normalize(embedding);
    }

    Ok(embeddings)
}
