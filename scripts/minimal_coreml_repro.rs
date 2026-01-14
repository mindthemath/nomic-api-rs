// Minimal reproduction example for ort CoreML crash on macOS 15
// Usage: cargo run --bin minimal_coreml_repro -- <path_to_onnx_model.onnx>

use anyhow::Result;
use ort::execution_providers::{CoreMLExecutionProvider, CPUExecutionProvider, ExecutionProviderDispatch};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: cargo run --bin minimal_coreml_repro -- <path_to_onnx_model.onnx>");

    println!("Creating session builder...");
    let mut builder = SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;

    println!("Configuring CoreML execution provider...");
    let coreml_provider: ExecutionProviderDispatch = CoreMLExecutionProvider::default().into();
    let cpu_provider: ExecutionProviderDispatch = CPUExecutionProvider::default().into();

    builder = builder.with_execution_providers([coreml_provider, cpu_provider])?;
    println!("CoreML execution provider configured successfully");

    println!("Loading model from: {}", model_path);
    let model_bytes = std::fs::read(&model_path)?;
    let _session = builder.commit_from_memory(&model_bytes)?;
    println!("Model loaded successfully - this line should not be reached if bug reproduces");

    Ok(())
}
