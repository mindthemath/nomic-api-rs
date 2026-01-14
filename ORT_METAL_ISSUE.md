# ONNX Runtime CoreML Execution Provider Crash on macOS 15

## Issue Summary

When using the `ort` crate (v2.0.0-rc.11) with CoreML execution provider on macOS 15 Sequoia, the application crashes with `dyld: missing symbol called` error immediately after CoreML successfully partitions the ONNX model graph.

## Environment

- **macOS Version**: 15.0.1 (Sequoia)
- **macOS SDK**: 11.3
- **Architecture**: arm64 (Apple Silicon)
- **Rust Version**: 1.92.0
- **ort crate version**: 2.0.0-rc.11
- **ONNX Runtime version**: 1.23 (as bundled with ort 2.0.0-rc.11)
- **Features enabled**: `coreml`, `download-binaries`, `tls-rustls`, `ndarray`, `half`

## Error Details

```
dyld[91799]: missing symbol called
Abort trap: 6
```

The crash occurs **after** CoreML execution provider successfully:
1. Initializes without errors
2. Partitions the ONNX model graph (482/1192 nodes supported by CoreML)
3. Logs warnings about unsupported operations (expected)

The crash happens when CoreML attempts to actually compile/use the partitioned model.

## Steps to Reproduce

### 1. Create a minimal repro project

```bash
cargo new ort-coreml-repro --bin
cd ort-coreml-repro
```

### 2. Add dependencies to `Cargo.toml`

```toml
[package]
name = "ort-coreml-repro"
version = "0.1.0"
edition = "2021"

[dependencies]
ort = { version = "2.0.0-rc.11", default-features = false, features = ["ndarray", "download-binaries", "coreml", "tls-rustls"] }
anyhow = "1"
```

### 3. Create `build.rs` (required for macOS CoreML linking)

```rust
use std::process::Command;

fn macos_rtlib_search_dir() -> Option<String> {
    let output = Command::new(std::env::var("CC").unwrap_or_else(|_| "clang".to_string()))
        .arg("--print-search-dirs")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;
            if !path.is_empty() {
                return Some(format!("{path}/lib/darwin"));
            }
        }
    }

    None
}

fn main() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=objc");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if let Some(dir) = macos_rtlib_search_dir() {
            println!("cargo:rustc-link-search={}", dir);
            println!("cargo:rustc-link-lib=clang_rt.osx");
        }
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
```

### 4. Create minimal reproducible example in `src/main.rs`

```rust
use anyhow::Result;
use ort::execution_providers::{CoreMLExecutionProvider, CPUExecutionProvider, ExecutionProviderDispatch};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

fn main() -> Result<()> {
    // Replace with path to any ONNX model file
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: cargo run -- <path_to_onnx_model.onnx>");

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
    println!("Model loaded successfully - this line should not be reached");

    Ok(())
}
```

### 5. Build and run

```bash
cargo build --release
USE_GPU=true ./target/release/ort-coreml-repro path/to/model.onnx
```

## Expected Behavior

The application should:
1. Initialize CoreML execution provider successfully
2. Load the ONNX model
3. Complete without crashing

## Actual Behavior

The application:
1. Initializes CoreML execution provider successfully
2. Partitions the model graph (logs show CoreML supports some nodes)
3. But crashes with `dyld: missing symbol called` when attempting to load/compile the model

## Full Error Output

```
USE_GPU=true ./target/release/minimal_coreml_repro models/txt/model.onnx
Creating session builder...
Configuring CoreML execution provider...
CoreML execution provider configured successfully
Loading model from: models/txt/model.onnx
2026-01-14 10:02:45.520 minimal_coreml_repro[92780:1266611] 2026-01-14 10:02:45.509085 [W:onnxruntime:, helper.cc:78 IsInputSupported] CoreML does not support shapes with dimension values of 0. Input:/encoder/layers.0/attn/rotary_emb/Slice_5_output_0, shape: {-1,-1,12,0}
...
2026-01-14 10:02:45.521 minimal_coreml_repro[92780:1266611] 2026-01-14 10:02:45.521207 [W:onnxruntime:, helper.cc:78 IsInputSupported] CoreML does not support shapes with dimension values of 0. Input:/encoder/layers.11/attn/rotary_emb/Slice_5_output_0, shape: {-1,-1,12,0}
2026-01-14 10:02:45.521 minimal_coreml_repro[92780:1266611] 2026-01-14 10:02:45.521223 [W:onnxruntime:, helper.cc:78 IsInputSupported] CoreML does not support shapes with dimension values of 0. Input:/encoder/layers.11/attn/rotary_emb/Slice_11_output_0, shape: {-1,-1,12,0}
2026-01-14 10:02:45.522 minimal_coreml_repro[92780:1266611] 2026-01-14 10:02:45.522290 [W:onnxruntime:, coreml_execution_provider.cc:113 GetCapability] CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 97 number of nodes in the graph: 1192 number of nodes supported by CoreML: 482
dyld[92780]: missing symbol called
Abort trap: 6
```

## Additional Context

### Related Issues

- ONNX Runtime issue #22275 was fixed in commit ffca096, addressing a different error (`NSGenericException` about `compute_device_types_mask`). However, this `dyld: missing symbol called` error appears to be a separate issue.

### Build Configuration

The following frameworks are linked (verified with `otool -L`):
- `/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation`
- `/System/Library/Frameworks/CoreML.framework/Versions/A/CoreML`
- `/System/Library/Frameworks/Metal.framework/Versions/A/Metal`
- `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`
- `/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation`

### Workaround

Using CPU-only execution provider works correctly:

```rust
let cpu_provider: ExecutionProviderDispatch = CPUExecutionProvider::default().into();
builder = builder.with_execution_providers([cpu_provider])?;
```

### Investigation Attempts

1. Verified all required frameworks are properly linked
2. Added `clang_rt.osx` runtime library (matching ort-sys configuration)
3. Used `-undefined,dynamic_lookup` linker flag (required for build)
4. Upgraded from ort 2.0.0-rc.10 to 2.0.0-rc.11 (ONNX Runtime 1.22 → 1.23)
5. Added required `tls-rustls` feature for download-binaries

The error persists with all configurations.

### Debugging Output

With `DYLD_PRINT_BINDINGS=1`, the crash occurs during weak symbol binding:

```
dyld[92250]: fixup: *0x00010566C520 = 0x00010513A34C <nomic-serve/weak-bind#8>
dyld[92250]: missing symbol called
```

This suggests a weak symbol that should be provided by the ONNX Runtime CoreML provider library is not being resolved.

## Questions for Maintainers

1. Is this a known issue with ONNX Runtime 1.23 CoreML provider on macOS 15?
2. Are there any additional linking requirements or configuration needed?
3. Should we wait for a newer ONNX Runtime version, or is there a workaround?
4. Is there a way to get more detailed information about which symbol is missing?

## System Information

```bash
$ sw_vers
ProductName:	macOS
ProductVersion:	15.0.1
BuildVersion:	24A348

$ uname -m
arm64

$ rustc --version
rustc 1.92.0

$ cargo --version
cargo 1.92.0
```


## Note to self:

In our project, we reference the script separately:

Appending to `Cargo.toml`:
```toml
...

[[bin]]
name = "minimal_coreml_repro"
path = "scripts/minimal_coreml_repro.rs"
```

Building and running:
```bash
cargo build --release --features coreml --bin minimal_coreml_repro
USE_GPU=true ./target/release/minimal_coreml_repro models/txt/model.onnx
```
