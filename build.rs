fn main() {
    // Link against C++ standard library and Objective-C runtime on macOS
    // The ONNX Runtime library uses CoreML which requires Objective-C runtime
    if cfg!(target_os = "macos") {
        // Link C++ standard library (needed by ONNX Runtime)
        println!("cargo:rustc-link-lib=c++");
        // Link Objective-C runtime and Foundation framework (needed for CoreML)
        // Note: These are linked via rustc-link-lib, but we also add them as linker args
        // to ensure they're available when ort_sys links
        println!("cargo:rustc-link-lib=objc");
        println!("cargo:rustc-link-lib=framework=Foundation");
        // Force the linker to include all symbols from the Objective-C runtime
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
