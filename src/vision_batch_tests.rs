//! Tests for vision model batching functionality
//!
//! These tests verify that:
//! 1. FP32 models batch perfectly (no interference)
//! 2. Quantized models show acceptable interference (~1% difference)
//! 3. Batching works correctly for various batch sizes

#[cfg(test)]
mod tests {
    use crate::*;
    use image::{DynamicImage, RgbImage};
    use ort::{
        session::{
            builder::{GraphOptimizationLevel, SessionBuilder},
            Session,
        },
        Error as OrtError,
    };
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Create a test image (synthetic pattern)
    fn create_test_image(seed: u32) -> DynamicImage {
        let mut img = RgbImage::new(224, 224);
        for y in 0..224 {
            for x in 0..224 {
                let r = ((x as u32 + seed) % 256) as u8;
                let g = ((y as u32 + seed * 2) % 256) as u8;
                let b = ((x as u32 + y as u32 + seed * 3) % 256) as u8;
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    /// Load a model for testing
    fn load_test_model(path: &str) -> Result<Session, OrtError> {
        let model_path = PathBuf::from(path);
        if !model_path.exists() {
            return Err(OrtError::Msg(format!("Model not found: {}", path)));
        }

        let model_bytes = std::fs::read(&model_path)
            .map_err(|e| OrtError::Msg(format!("Failed to read model: {}", e)))?;

        let builder = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;
        let session = builder.commit_from_memory(&model_bytes)?;
        Ok(session)
    }

    /// Test that same image batched with itself produces identical results
    #[test]
    fn test_same_image_batching() {
        // Try to load quantized model (if available)
        let model_path = "models/img/model_quantized.onnx";
        let session = match load_test_model(model_path) {
            Ok(s) => s,
            Err(_) => {
                // Model not available, skip test
                return;
            }
        };

        let state = VisionState {
            session: Mutex::new(session),
        };

        let image = create_test_image(42);

        // Single inference
        let single_emb = embed_image(&state, &image).expect("Single inference failed");

        // Batch inference (same image × 2)
        let batch_embs = embed_image_batch(&state, &[image.clone(), image])
            .expect("Batch inference failed");

        assert_eq!(batch_embs.len(), 2, "Batch should return 2 embeddings");

        // Both batch results should match single inference
        let diff_0 = single_emb
            .iter()
            .zip(batch_embs[0].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let diff_1 = single_emb
            .iter()
            .zip(batch_embs[1].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            diff_0 < 0.0001,
            "Batch[0] should match single inference (diff: {})",
            diff_0
        );
        assert!(
            diff_1 < 0.0001,
            "Batch[1] should match single inference (diff: {})",
            diff_1
        );
    }

    /// Test batching with multiple different images
    #[test]
    fn test_different_images_batching() {
        // Try to load FP32 model first (should batch perfectly)
        let model_path = if PathBuf::from("models/img/model.onnx").exists() {
            "models/img/model.onnx"
        } else if PathBuf::from("models/img/model_quantized.onnx").exists() {
            "models/img/model_quantized.onnx"
        } else {
            // No model available, skip test
            return;
        };

        let session = match load_test_model(model_path) {
            Ok(s) => s,
            Err(_) => return,
        };

        let state = VisionState {
            session: Mutex::new(session),
        };

        // Create 4 different test images
        let images: Vec<DynamicImage> = (0..4)
            .map(|i| create_test_image(i * 100))
            .collect();

        // Single inference for each
        let single_embs: Vec<Vec<f32>> = images
            .iter()
            .map(|img| embed_image(&state, img).expect("Single inference failed"))
            .collect();

        // Batch inference
        let batch_embs = embed_image_batch(&state, &images).expect("Batch inference failed");

        assert_eq!(
            batch_embs.len(),
            images.len(),
            "Batch should return same number of embeddings as images"
        );

        // Check differences
        let is_fp32 = model_path.contains("model.onnx") && !model_path.contains("quantized");
        let tolerance = if is_fp32 { 0.0001 } else { 0.03 }; // FP32 should be perfect, quantized ~1%

        for (i, (single, batch)) in single_embs.iter().zip(batch_embs.iter()).enumerate() {
            let max_diff = single
                .iter()
                .zip(batch.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            // Calculate cosine similarity
            let dot_product: f32 = single.iter().zip(batch.iter()).map(|(a, b)| a * b).sum();
            let cos_sim = dot_product; // Already normalized

            if is_fp32 {
                assert!(
                    max_diff < tolerance,
                    "FP32 model: Image {} should match perfectly (max_diff: {}, cos_sim: {})",
                    i,
                    max_diff,
                    cos_sim
                );
                assert!(
                    cos_sim > 0.9999,
                    "FP32 model: Image {} cosine similarity should be ~1.0 (got: {})",
                    i,
                    cos_sim
                );
            } else {
                // Quantized model - allow ~1% difference
                assert!(
                    max_diff < tolerance,
                    "Quantized model: Image {} difference acceptable (max_diff: {}, cos_sim: {})",
                    i,
                    max_diff,
                    cos_sim
                );
                assert!(
                    cos_sim > 0.98,
                    "Quantized model: Image {} cosine similarity should be >0.98 (got: {})",
                    i,
                    cos_sim
                );
            }
        }
    }

    /// Test batching with larger batch sizes (8 images)
    #[test]
    fn test_large_batch() {
        let model_path = if PathBuf::from("models/img/model.onnx").exists() {
            "models/img/model.onnx"
        } else if PathBuf::from("models/img/model_quantized.onnx").exists() {
            "models/img/model_quantized.onnx"
        } else {
            return;
        };

        let session = match load_test_model(model_path) {
            Ok(s) => s,
            Err(_) => return,
        };

        let state = VisionState {
            session: Mutex::new(session),
        };

        // Create 8 different images
        let images: Vec<DynamicImage> = (0..8)
            .map(|i| create_test_image(i * 50))
            .collect();

        // Batch inference
        let batch_embs = embed_image_batch(&state, &images).expect("Batch inference failed");

        assert_eq!(
            batch_embs.len(),
            8,
            "Should return 8 embeddings for 8 images"
        );

        // Verify all embeddings are normalized (L2 norm = 1.0)
        for (i, emb) in batch_embs.iter().enumerate() {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.0001,
                "Embedding {} should be L2-normalized (norm: {})",
                i,
                norm
            );
        }

        // Verify embeddings are different (not all identical)
        let first_emb = &batch_embs[0];
        let mut all_same = true;
        for (i, emb) in batch_embs.iter().enumerate().skip(1) {
            let diff: f32 = first_emb
                .iter()
                .zip(emb.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if diff > 0.01 {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "Different images should produce different embeddings");
    }

    /// Test empty batch
    #[test]
    fn test_empty_batch() {
        let model_path = if PathBuf::from("models/img/model_quantized.onnx").exists() {
            "models/img/model_quantized.onnx"
        } else {
            return;
        };

        let session = match load_test_model(model_path) {
            Ok(s) => s,
            Err(_) => return,
        };

        let state = VisionState {
            session: Mutex::new(session),
        };

        let empty_images: Vec<DynamicImage> = Vec::new();
        let batch_embs = embed_image_batch(&state, &empty_images).expect("Empty batch failed");

        assert_eq!(batch_embs.len(), 0, "Empty batch should return empty vector");
    }
}

