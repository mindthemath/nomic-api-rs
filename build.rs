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
    // Link against C++ standard library and Objective-C runtime on macOS
    // The ONNX Runtime library uses CoreML which requires Objective-C runtime
    if cfg!(target_os = "macos") {
        // Link C++ standard library (needed by ONNX Runtime)
        println!("cargo:rustc-link-lib=c++");
        // Link Objective-C runtime and frameworks (needed for CoreML)
        // Note: These are linked via rustc-link-lib, but we also add them as linker args
        // to ensure they're available when ort_sys links
        println!("cargo:rustc-link-lib=objc");
        println!("cargo:rustc-link-lib=framework=Foundation");
        // Link CoreML framework (required for CoreML execution provider)
        println!("cargo:rustc-link-lib=framework=CoreML");
        // Link Metal framework (CoreML uses Metal for GPU acceleration)
        println!("cargo:rustc-link-lib=framework=Metal");
        // Link Accelerate framework (used by CoreML for optimized operations)
        println!("cargo:rustc-link-lib=framework=Accelerate");
        // Link clang runtime library (ort-sys also links this, helps with symbol resolution)
        if let Some(dir) = macos_rtlib_search_dir() {
            println!("cargo:rustc-link-search={}", dir);
            println!("cargo:rustc-link-lib=clang_rt.osx");
        }
        // Force the linker to include all symbols from the Objective-C runtime
        // This is needed because ONNX Runtime CoreML provider uses Objective-C runtime
        // and some symbols are resolved dynamically at runtime
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
