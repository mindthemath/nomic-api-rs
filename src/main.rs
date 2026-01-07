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
    extract::State,
    http::{header, Method, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::{DynamicImage, ImageReader};
use ndarray::ShapeError;
use ort::{
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session, SessionInputValue, SessionInputs,
    },
    value::Value,
    Error as OrtError,
};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

// Check if CUDA feature is compiled in
#[cfg(feature = "cuda")]
fn check_cuda_feature_enabled() -> bool {
    true
}

#[cfg(not(feature = "cuda"))]
fn check_cuda_feature_enabled() -> bool {
    false
}

// Check if CUDA libraries are actually available at runtime
// Returns (available, library_path_if_found)
#[cfg(feature = "cuda")]
fn check_cuda_libraries_available() -> (bool, Option<std::path::PathBuf>) {
    use std::fs;
    use std::path::PathBuf;

    // Check common locations for ONNX Runtime CUDA providers library
    let mut possible_paths = Vec::new();

    // Check target/release/deps (symlink location)
    possible_paths.push(PathBuf::from(
        "target/release/deps/libonnxruntime_providers_shared.so",
    ));

    // Check ~/.cache/ort.pyke.io/dfbin (ort crate cache) - search recursively
    if let Ok(home) = std::env::var("HOME") {
        let cache_dir = PathBuf::from(home).join(".cache/ort.pyke.io/dfbin");
        if cache_dir.exists() {
            // Search for libonnxruntime_providers_shared.so recursively
            if let Ok(entries) = fs::read_dir(&cache_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let lib_path =
                            path.join("onnxruntime/lib/libonnxruntime_providers_shared.so");
                        if lib_path.exists() {
                            possible_paths.push(lib_path);
                        }
                    }
                }
            }
        }
    }

    // Check LD_LIBRARY_PATH
    let ld_library_path = std::env::var("LD_LIBRARY_PATH").ok();
    if let Some(ld_path) = &ld_library_path {
        for p in ld_path.split(':') {
            let path = PathBuf::from(p).join("libonnxruntime_providers_shared.so");
            possible_paths.push(path);
        }
    }

    // Check if any of these paths exist
    for path in possible_paths {
        let resolved_path = if path.is_symlink() {
            fs::read_link(&path)
                .ok()
                .and_then(|p| if p.exists() { Some(p) } else { None })
        } else if path.exists() {
            Some(path.clone())
        } else {
            None
        };

        if let Some(lib_path) = resolved_path {
            // Return the library directory (we'll check if it's in LD_LIBRARY_PATH later)
            let lib_dir = lib_path.parent().map(|p| p.to_path_buf());
            return (true, lib_dir);
        }
    }

    (false, None)
}

#[cfg(not(feature = "cuda"))]
fn check_cuda_libraries_available() -> (bool, Option<std::path::PathBuf>) {
    (false, None)
}

#[cfg(not(feature = "cuda"))]
fn check_cuda_libraries_available() -> bool {
    false
}
use serde::{Deserialize, Serialize};
use std::{
    io::Cursor,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};
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

/// Text embedding state (tokenizer + ONNX session)
struct TextState {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

/// Vision embedding state (ONNX session only, no tokenizer)
struct VisionState {
    session: Mutex<Session>,
}

/// Combined application state
#[derive(Clone)]
pub struct AppState {
    text: Option<Arc<TextState>>,
    vision: Option<Arc<VisionState>>,
    default_avg_method: image_stats::AveragingMethod,
    txt_model_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    img_model_path: Option<PathBuf>,
    use_gpu: bool,
}

impl AppState {
    async fn new(
        txt_model: Option<PathBuf>,
        tokenizer: Option<PathBuf>,
        img_model: Option<PathBuf>,
        use_gpu: bool,
        default_avg_method: image_stats::AveragingMethod,
    ) -> anyhow::Result<Self> {
        let cold_start = Instant::now();

        // Clone paths for storing in state
        let txt_model_path = txt_model.clone();
        let tokenizer_path = tokenizer.clone();
        let img_model_path = img_model.clone();

        // Check if CUDA feature is compiled in and libraries are available
        let cuda_feature_enabled = check_cuda_feature_enabled();
        let (cuda_libraries_available, cuda_lib_dir) = check_cuda_libraries_available();

        if use_gpu && !cuda_feature_enabled {
            warn!("USE_GPU=1 but binary was compiled without CUDA feature. Falling back to CPU.");
        } else if use_gpu && !cuda_libraries_available {
            warn!("USE_GPU=1 but CUDA libraries (libonnxruntime_providers_shared.so) not found. Falling back to CPU.");
            warn!("  Hint: Set LD_LIBRARY_PATH to point to ONNX Runtime CUDA providers library directory.");
        } else if use_gpu && cuda_libraries_available {
            // Check if library directory is in LD_LIBRARY_PATH
            if let Some(lib_dir) = &cuda_lib_dir {
                let ld_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
                let lib_dir_str = lib_dir.to_string_lossy();
                if !ld_path.split(':').any(|p| p == lib_dir_str.as_ref()) {
                    warn!(
                        "CUDA libraries found at {:?} but not in LD_LIBRARY_PATH.",
                        lib_dir
                    );
                    warn!("  ONNX Runtime may fall back to CPU. Set: export LD_LIBRARY_PATH=\"{}\":$LD_LIBRARY_PATH", lib_dir_str);
                }
            }
        }

        // Only use GPU if both feature is enabled AND libraries are available
        // Note: Even with this check, ONNX Runtime may still silently fall back to CPU
        // if CUDA drivers aren't available or GPU isn't accessible
        let mut actual_use_gpu = use_gpu && cuda_feature_enabled && cuda_libraries_available;

        // Load text model if paths provided
        let text = if let (Some(model_path), Some(tok_path)) = (txt_model, tokenizer) {
            info!("Loading text model: {:?}", model_path);
            let model_bytes = std::fs::read(&model_path)
                .map_err(|e| anyhow::anyhow!("Failed to read model file: {}", e))?;

            // Try CUDA if requested, fall back to CPU if it fails
            let session = if actual_use_gpu {
                #[cfg(feature = "cuda")]
                {
                    let builder = SessionBuilder::new()
                        .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
                        .with_optimization_level(GraphOptimizationLevel::Level3)
                        .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?;

                    match builder
                        .with_execution_providers([CUDAExecutionProvider::default().build()])
                    {
                        Ok(cuda_builder) => {
                            match cuda_builder.commit_from_memory(&model_bytes) {
                                Ok(s) => s,
                                Err(e) => {
                                    warn!(
                                        "Failed to load model with CUDA: {}. Falling back to CPU.",
                                        e
                                    );
                                    actual_use_gpu = false;
                                    // Retry without CUDA
                                    let cpu_builder = SessionBuilder::new()
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "Failed to create session builder: {}",
                                                e
                                            )
                                        })?
                                        .with_optimization_level(GraphOptimizationLevel::Level3)
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "Failed to set optimization level: {}",
                                                e
                                            )
                                        })?;
                                    cpu_builder.commit_from_memory(&model_bytes).map_err(|e| {
                                        anyhow::anyhow!("Failed to load model: {}", e)
                                    })?
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to enable CUDA execution provider: {}. Falling back to CPU.", e);
                            actual_use_gpu = false;
                            // Fall back to CPU
                            let cpu_builder = SessionBuilder::new()
                                .map_err(|e| {
                                    anyhow::anyhow!("Failed to create session builder: {}", e)
                                })?
                                .with_optimization_level(GraphOptimizationLevel::Level3)
                                .map_err(|e| {
                                    anyhow::anyhow!("Failed to set optimization level: {}", e)
                                })?;
                            cpu_builder
                                .commit_from_memory(&model_bytes)
                                .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    actual_use_gpu = false;
                    let cpu_builder = SessionBuilder::new()
                        .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
                        .with_optimization_level(GraphOptimizationLevel::Level3)
                        .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?;
                    cpu_builder
                        .commit_from_memory(&model_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?
                }
            } else {
                let builder = SessionBuilder::new()
                    .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?;
                builder
                    .commit_from_memory(&model_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?
            };
            let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!(e))?;
            Some(Arc::new(TextState {
                session: Mutex::new(session),
                tokenizer,
            }))
        } else {
            None
        };

        // Load vision model if path provided
        let vision = if let Some(model_path) = img_model {
            info!("Loading vision model: {:?}", model_path);
            let model_bytes = std::fs::read(&model_path)
                .map_err(|e| anyhow::anyhow!("Failed to read model file: {}", e))?;

            // Try CUDA if requested, fall back to CPU if it fails
            let session = if actual_use_gpu {
                #[cfg(feature = "cuda")]
                {
                    let builder = SessionBuilder::new()
                        .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
                        .with_optimization_level(GraphOptimizationLevel::Level3)
                        .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?;

                    match builder
                        .with_execution_providers([CUDAExecutionProvider::default().build()])
                    {
                        Ok(cuda_builder) => {
                            match cuda_builder.commit_from_memory(&model_bytes) {
                                Ok(s) => s,
                                Err(e) => {
                                    warn!(
                                        "Failed to load vision model with CUDA: {}. Falling back to CPU.",
                                        e
                                    );
                                    actual_use_gpu = false;
                                    // Retry without CUDA
                                    let cpu_builder = SessionBuilder::new()
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "Failed to create session builder: {}",
                                                e
                                            )
                                        })?
                                        .with_optimization_level(GraphOptimizationLevel::Level3)
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "Failed to set optimization level: {}",
                                                e
                                            )
                                        })?;
                                    cpu_builder.commit_from_memory(&model_bytes).map_err(|e| {
                                        anyhow::anyhow!("Failed to load model: {}", e)
                                    })?
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to enable CUDA execution provider for vision model: {}. Falling back to CPU.", e);
                            actual_use_gpu = false;
                            // Fall back to CPU
                            let cpu_builder = SessionBuilder::new()
                                .map_err(|e| {
                                    anyhow::anyhow!("Failed to create session builder: {}", e)
                                })?
                                .with_optimization_level(GraphOptimizationLevel::Level3)
                                .map_err(|e| {
                                    anyhow::anyhow!("Failed to set optimization level: {}", e)
                                })?;
                            cpu_builder
                                .commit_from_memory(&model_bytes)
                                .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    actual_use_gpu = false;
                    let cpu_builder = SessionBuilder::new()
                        .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
                        .with_optimization_level(GraphOptimizationLevel::Level3)
                        .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?;
                    cpu_builder
                        .commit_from_memory(&model_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?
                }
            } else {
                let builder = SessionBuilder::new()
                    .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?;
                builder
                    .commit_from_memory(&model_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?
            };
            Some(Arc::new(VisionState {
                session: Mutex::new(session),
            }))
        } else {
            None
        };

        let cold_start_time = cold_start.elapsed();
        info!(
            "Cold start complete - text model: {}, vision model: {}, averaging: {}, time: {:.2}ms",
            if text.is_some() {
                "loaded"
            } else {
                "not loaded"
            },
            if vision.is_some() {
                "loaded"
            } else {
                "not loaded"
            },
            match default_avg_method {
                image_stats::AveragingMethod::Arithmetic => "arithmetic",
                image_stats::AveragingMethod::Geometric => "geometric",
            },
            cold_start_time.as_secs_f64() * 1000.0
        );

        Ok(Self {
            text,
            vision,
            default_avg_method,
            txt_model_path,
            tokenizer_path,
            img_model_path,
            use_gpu: actual_use_gpu, // Store actual CUDA status, not just env var
        })
    }
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
    /// Image content: URL (http/https), data URL (data:image/...), or raw base64
    #[schema(example = "https://picsum.photos/400/300")]
    content: String,
    /// Embedding dimension (1-768)
    #[serde(default = "default_dim")]
    #[schema(example = 768, minimum = 1, maximum = 768)]
    dim: usize,
}

#[derive(Deserialize, ToSchema)]
struct ImageBatchRequest {
    /// List of image contents (URLs or base64)
    #[schema(example = json!(["https://picsum.photos/200/200", "https://picsum.photos/300/400", "https://picsum.photos/300/400"]))]
    contents: Vec<String>,
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
}

#[derive(Serialize, ToSchema)]
struct InfoResponse {
    /// Default averaging method for image stats
    #[schema(example = "geometric")]
    averaging: String,
    /// CUDA/GPU support enabled
    #[schema(example = true)]
    load_cuda: bool,
    /// Text model file path (if loaded)
    #[schema(example = "models/txt/model_quantized.onnx")]
    txt_model: Option<String>,
    /// Tokenizer file path (if loaded)
    #[schema(example = "models/txt/tokenizer.json")]
    tokenizer: Option<String>,
    /// Vision model file path (if loaded)
    #[schema(example = "models/img/model_quantized.onnx")]
    img_model: Option<String>,
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

    // Text model configuration
    // Priority: TXT_MODEL > MODEL > default path
    // resolve_model_path handles both CLI (CWD) and double-click (exe-relative) scenarios
    let txt_model_path = std::env::var("TXT_MODEL")
        .or_else(|_| std::env::var("MODEL"))
        .map(PathBuf::from)
        .map(resolve_model_path)
        .unwrap_or_else(|_| resolve_model_path("models/txt/model_quantized.onnx"));

    let tok_path = std::env::var("TOKENIZER")
        .map(PathBuf::from)
        .map(resolve_model_path)
        .unwrap_or_else(|_| resolve_model_path("models/txt/tokenizer.json"));

    // Vision model configuration
    let img_model_path = std::env::var("IMG_MODEL")
        .map(PathBuf::from)
        .map(resolve_model_path)
        .unwrap_or_else(|_| resolve_model_path("models/img/model_quantized.onnx"));

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let use_gpu = std::env::var("USE_GPU")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    let disable_cors = std::env::var("DISABLE_CORS")
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

    // Determine which models to load based on file existence
    let txt_model = if txt_model_path.exists() && tok_path.exists() {
        Some(txt_model_path.clone())
    } else {
        warn!(
            "Text model not found at {:?} or tokenizer at {:?}",
            txt_model_path, tok_path
        );
        None
    };

    let tokenizer = if tok_path.exists() {
        Some(tok_path.clone())
    } else {
        None
    };

    let img_model = if img_model_path.exists() {
        Some(img_model_path.clone())
    } else {
        warn!("Vision model not found at {:?}", img_model_path);
        None
    };

    // Initialize application state
    let state =
        match AppState::new(txt_model, tokenizer, img_model, use_gpu, default_avg_method).await {
            Ok(state) => state,
            Err(e) => {
                eprintln!("Failed to initialize server: {}", e);
                std::process::exit(1);
            }
        };

    let device = if state.use_gpu { "GPU" } else { "CPU" };
    let text_status = if state.text.is_some() { "✓" } else { "✗" };
    let vision_status = if state.vision.is_some() { "✓" } else { "✗" };

    info!(
        "🚀 Nomic embedding server ready on http://0.0.0.0:{} ({})",
        port, device
    );
    info!(
        "   Text model:   {} /txt/embed, /txt/batch, /txt/query",
        text_status
    );
    info!(
        "   Vision model: {} /img/embed, /img/batch, /img/stats",
        vision_status
    );
    info!("📚 API docs available at http://0.0.0.0:{}/docs", port);

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
        .route("/docs", get(docs_handler))
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
        text_model: state.text.is_some(),
        vision_model: state.vision.is_some(),
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
        load_cuda: state.use_gpu,
        txt_model: state
            .txt_model_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
        tokenizer: state
            .tokenizer_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
        img_model: state
            .img_model_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
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

    let text_state = state.text.as_ref().ok_or_else(|| {
        Error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Text model not loaded".to_string(),
        )
    })?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    let prefixed_text = format!("{}: {}", req.prefix, req.input);
    let (mut embedding, tokens) = embed_text(text_state, &prefixed_text)?;

    if req.dim < embedding.len() {
        embedding.truncate(req.dim);
    }

    let total_time = start.elapsed();
    info!(
        "Text embed timing - total: {:.2}ms",
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

    let text_state = state.text.as_ref().ok_or_else(|| {
        Error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Text model not loaded".to_string(),
        )
    })?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    let mut embeddings = Vec::with_capacity(req.inputs.len());
    let mut tokens = Vec::with_capacity(req.inputs.len());

    for text in &req.inputs {
        let prefixed_text = format!("{}: {}", req.prefix, text);
        let (mut emb, tok) = embed_text(text_state, &prefixed_text)?;
        if req.dim < emb.len() {
            emb.truncate(req.dim);
        }
        embeddings.push(emb);
        tokens.push(tok);
    }

    let total_time = start.elapsed();
    info!(
        "Text batch timing - count: {}, total: {:.2}ms, avg: {:.2}ms",
        req.inputs.len(),
        total_time.as_secs_f64() * 1000.0,
        total_time.as_secs_f64() * 1000.0 / req.inputs.len() as f64
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

    let text_state = state.text.as_ref().ok_or_else(|| {
        Error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Text model not loaded".to_string(),
        )
    })?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    // Always use search_query prefix for /query endpoint
    let prefixed_text = format!("search_query: {}", req.input);
    let (mut embedding, tokens) = embed_text(text_state, &prefixed_text)?;

    if req.dim < embedding.len() {
        embedding.truncate(req.dim);
    }

    let total_time = start.elapsed();
    info!(
        "Text query timing - total: {:.2}ms",
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

    let vision_state = state.vision.as_ref().ok_or_else(|| {
        Error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Vision model not loaded".to_string(),
        )
    })?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    let decode_start = Instant::now();
    let image = decode_image(&req.content).await?;
    let decode_time = decode_start.elapsed();

    let inference_start = Instant::now();
    let mut embedding = embed_image(vision_state, &image)?;
    let inference_time = inference_start.elapsed();

    info!(
        "Image embed timing - decode: {:.2}ms, inference: {:.2}ms, total: {:.2}ms",
        decode_time.as_secs_f64() * 1000.0,
        inference_time.as_secs_f64() * 1000.0,
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

    let vision_state = state.vision.as_ref().ok_or_else(|| {
        Error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Vision model not loaded".to_string(),
        )
    })?;

    if req.dim == 0 || req.dim > 768 {
        return Err(Error(
            StatusCode::BAD_REQUEST,
            format!("dim must be between 1 and 768, got {}", req.dim),
        ));
    }

    let mut embeddings = Vec::with_capacity(req.contents.len());

    for content in &req.contents {
        let image = decode_image(content).await?;
        let mut emb = embed_image(vision_state, &image)?;
        if req.dim < emb.len() {
            emb.truncate(req.dim);
        }
        embeddings.push(emb);
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
                                // Also update InfoResponse.load_cuda example
                                if let Some(load_cuda) = props.get_mut("load_cuda") {
                                    if let Some(load_cuda_obj) = load_cuda.as_object_mut() {
                                        load_cuda_obj.insert(
                                            "example".to_string(),
                                            serde_json::json!(state.use_gpu),
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

/// Embed single text, returns 768-dim embedding
fn embed_text(state: &TextState, text: &str) -> Result<(Vec<f32>, usize), Error> {
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
    let tokenize_time = tokenize_start.elapsed();

    let prepare_start = Instant::now();
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
    let prepare_time = prepare_start.elapsed();

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

    info!(
        "Text inference timing - tokenize: {:.2}ms, prepare: {:.2}ms, ONNX: {:.2}ms, postprocess: {:.2}ms",
        tokenize_time.as_secs_f64() * 1000.0,
        prepare_time.as_secs_f64() * 1000.0,
        inference_time.as_secs_f64() * 1000.0,
        postprocess_time.as_secs_f64() * 1000.0
    );

    Ok((embedding, token_count))
}

// ============================================================================
// Image Embedding
// ============================================================================

/// Preprocess image for CLIP-style model
fn preprocess_image(image: &DynamicImage) -> Vec<f32> {
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
fn l2_normalize(vec: &mut Vec<f32>) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for val in vec.iter_mut() {
            *val /= norm;
        }
    }
}

/// Embed single image, returns 768-dim L2-normalized embedding
fn embed_image(state: &VisionState, image: &DynamicImage) -> Result<Vec<f32>, Error> {
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

    info!(
        "Vision inference timing - preprocess: {:.2}ms, ONNX: {:.2}ms",
        preprocess_time.as_secs_f64() * 1000.0,
        inference_time.as_secs_f64() * 1000.0
    );
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

    Ok(embedding)
}
