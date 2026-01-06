//! Image statistics module for EXIF extraction and color analysis.
//!
//! This module provides functionality to extract metadata and color information from images:
//! - EXIF data extraction with GPS filtering
//! - Average color calculation (arithmetic or geometric mean)
//! - Dominant color detection using HSV clustering
//!
//! Ported from `scripts/api_stats.py`.

use axum::{extract::State, http::StatusCode, response::Json};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Cursor, time::Instant};
use tracing::info;
use utoipa::ToSchema;

use crate::{AppState, Error};

// ============================================================================
// Constants
// ============================================================================

/// Default thumbnail size for color processing (matches Python THUMBNAIL_SIZE)
const THUMBNAIL_SIZE: u32 = 512;

/// Minimum alpha value for a pixel to be considered "valid" (non-transparent)
const ALPHA_THRESHOLD: u8 = 128;

// ============================================================================
// Request/Response Types
// ============================================================================

/// Averaging method for color calculation
#[derive(Debug, Clone, Copy, Deserialize, Serialize, ToSchema, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AveragingMethod {
    /// Arithmetic mean (simple average)
    #[default]
    Arithmetic,
    /// Geometric mean (better for color perception)
    Geometric,
}

impl std::fmt::Display for AveragingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AveragingMethod::Arithmetic => write!(f, "arithmetic"),
            AveragingMethod::Geometric => write!(f, "geometric"),
        }
    }
}

#[derive(Deserialize, ToSchema)]
pub struct ImageStatsRequest {
    /// Image content: URL (http/https), data URL (data:image/...), or raw base64
    #[schema(example = "https://picsum.photos/400/300")]
    pub content: String,
    /// Averaging method for color calculation
    #[serde(default)]
    #[schema(example = "geometric")]
    pub averaging_method: AveragingMethod,
}

/// RGB color with hex representation
#[derive(Serialize, ToSchema, Debug, Clone)]
pub struct ColorInfo {
    /// RGB values in range [0, 1]
    #[schema(example = json!([0.5, 0.3, 0.2]))]
    pub rgb: [f32; 3],
    /// Hex color code
    #[schema(example = "#804d33")]
    pub hex: String,
}

/// Average color information
#[derive(Serialize, ToSchema, Debug, Clone)]
pub struct AverageColorInfo {
    /// RGB values in range [0, 1]
    #[schema(example = json!([0.5, 0.3, 0.2]))]
    pub rgb: [f32; 3],
    /// Hex color code
    #[schema(example = "#804d33")]
    pub hex: String,
    /// Method used for averaging
    #[schema(example = "geometric")]
    pub method: String,
}

/// Color analysis results
#[derive(Serialize, ToSchema, Debug, Clone)]
pub struct ColorData {
    /// Average color of the image
    pub avg_color: AverageColorInfo,
    /// Dominant color (most frequent color cluster)
    pub dominant_color: ColorInfo,
}

#[derive(Serialize, ToSchema)]
pub struct ImageStatsResponse {
    /// EXIF metadata (if available)
    #[schema(example = json!({"Make": "Canon", "Model": "EOS 5D"}))]
    pub exif_data: HashMap<String, serde_json::Value>,
    /// Color analysis results (null if image has no valid pixels)
    pub color_data: Option<ColorData>,
    /// Processing time in milliseconds
    #[schema(example = 12.34)]
    pub time_ms: f64,
}

// ============================================================================
// Color Utilities
// ============================================================================

/// Convert RGB (0-1 range) to hex color code
fn rgb_to_hex(rgb: [f32; 3]) -> String {
    let r = (rgb[0] * 255.0).round() as u8;
    let g = (rgb[1] * 255.0).round() as u8;
    let b = (rgb[2] * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

/// Convert RGB to HSV
/// RGB values should be in range [0, 1]
/// Returns (H, S, V) where H is in [0, 1], S and V are in [0, 1]
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    // Value
    let v = max;

    // Saturation
    let s = if max > 0.0 { delta / max } else { 0.0 };

    // Hue
    let h = if delta < 1e-6 {
        0.0
    } else if (max - r).abs() < 1e-6 {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    (h, s, v)
}

// ============================================================================
// Image Processing
// ============================================================================

/// Resize image for color processing if too large
fn resize_for_processing(image: &DynamicImage) -> DynamicImage {
    let (w, h) = (image.width(), image.height());

    if w.max(h) > THUMBNAIL_SIZE {
        let (new_w, new_h) = if w > h {
            (THUMBNAIL_SIZE, (h * THUMBNAIL_SIZE) / w)
        } else {
            ((w * THUMBNAIL_SIZE) / h, THUMBNAIL_SIZE)
        };

        info!(
            "Resizing large image from {}x{} to {}x{} for color processing",
            w, h, new_w, new_h
        );

        image.resize(new_w, new_h, image::imageops::FilterType::Lanczos3)
    } else {
        image.clone()
    }
}

/// Extract valid (non-transparent) pixels from image
/// Returns Vec of (R, G, B) tuples normalized to [0, 1]
fn extract_valid_pixels(image: &DynamicImage) -> Vec<[f32; 3]> {
    let rgba = image.to_rgba8();
    let mut valid_pixels = Vec::new();

    for pixel in rgba.pixels() {
        // Filter out transparent pixels (alpha < 128)
        if pixel[3] >= ALPHA_THRESHOLD {
            valid_pixels.push([
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
            ]);
        }
    }

    valid_pixels
}

/// Calculate arithmetic mean of RGB values
fn calculate_arithmetic_mean(pixels: &[[f32; 3]]) -> [f32; 3] {
    if pixels.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let mut sum = [0.0f64, 0.0, 0.0];
    for pixel in pixels {
        sum[0] += pixel[0] as f64;
        sum[1] += pixel[1] as f64;
        sum[2] += pixel[2] as f64;
    }

    let n = pixels.len() as f64;
    [
        (sum[0] / n) as f32,
        (sum[1] / n) as f32,
        (sum[2] / n) as f32,
    ]
}

/// Calculate geometric mean of RGB values
fn calculate_geometric_mean(pixels: &[[f32; 3]]) -> [f32; 3] {
    if pixels.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    const EPS: f64 = 1e-8;
    let mut log_sum = [0.0f64, 0.0, 0.0];

    for pixel in pixels {
        log_sum[0] += (pixel[0] as f64).max(EPS).ln();
        log_sum[1] += (pixel[1] as f64).max(EPS).ln();
        log_sum[2] += (pixel[2] as f64).max(EPS).ln();
    }

    let n = pixels.len() as f64;
    [
        (log_sum[0] / n).exp() as f32,
        (log_sum[1] / n).exp() as f32,
        (log_sum[2] / n).exp() as f32,
    ]
}

/// Calculate color average using specified method
fn calculate_color_average(pixels: &[[f32; 3]], method: AveragingMethod) -> [f32; 3] {
    match method {
        AveragingMethod::Arithmetic => calculate_arithmetic_mean(pixels),
        AveragingMethod::Geometric => calculate_geometric_mean(pixels),
    }
}

/// Find dominant color using HSV clustering
/// Quantizes colors in HSV space and finds the most common cluster
fn find_dominant_color(pixels: &[[f32; 3]]) -> [f32; 3] {
    if pixels.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    // Quantize colors in HSV space
    // Using 10 bins for each component (H, S, V)
    let mut color_counts: HashMap<u32, usize> = HashMap::new();
    let mut quantized_to_idx: HashMap<u32, usize> = HashMap::new();

    for (idx, pixel) in pixels.iter().enumerate() {
        let (h, s, v) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);

        // Quantize: H * 10 * 1000 + S * 10 * 10 + V * 10
        // Match Python: (hsv_pixels[:, 0] * 10).astype(int) * 1000 + ...
        // Python's astype(int) truncates towards zero (same as floor for positive numbers)
        // Use direct truncation to match exactly
        let h_q = (h * 10.0) as u32;
        let s_q = (s * 10.0) as u32;
        let v_q = (v * 10.0) as u32;
        let key = h_q * 1000 + s_q * 10 + v_q;

        *color_counts.entry(key).or_insert(0) += 1;
        quantized_to_idx.entry(key).or_insert(idx);
    }

    // Find most common color - collect all entries and sort by count descending
    // This ensures we get the same result as Python's Counter.most_common(1)
    let mut entries: Vec<_> = color_counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0))); // Sort by count desc, then by key asc for tie-breaking

    // Debug: log top 5 for comparison with Python
    if entries.len() >= 5 {
        info!(
            "Top 5 clusters: {:?}",
            entries[..5]
                .iter()
                .map(|(k, c)| (*k, *c))
                .collect::<Vec<_>>()
        );
    }

    let most_common_key = entries.first().map(|(key, _)| **key).unwrap_or(0);

    // Return the actual pixel RGB for this cluster
    let idx = quantized_to_idx.get(&most_common_key).copied().unwrap_or(0);
    pixels[idx]
}

/// Get color analysis for an image
fn get_image_colors(image: &DynamicImage, method: AveragingMethod) -> Option<ColorData> {
    let original_size = (image.width(), image.height());
    let processed = resize_for_processing(image);
    let processed_size = (processed.width(), processed.height());
    let valid_pixels = extract_valid_pixels(&processed);

    info!(
        "Color analysis: original={}x{}, processed={}x{}, valid_pixels={}",
        original_size.0,
        original_size.1,
        processed_size.0,
        processed_size.1,
        valid_pixels.len()
    );

    if valid_pixels.is_empty() {
        return None;
    }

    let avg_rgb = calculate_color_average(&valid_pixels, method);
    let dominant_rgb = find_dominant_color(&valid_pixels);

    Some(ColorData {
        avg_color: AverageColorInfo {
            rgb: avg_rgb,
            hex: rgb_to_hex(avg_rgb),
            method: method.to_string(),
        },
        dominant_color: ColorInfo {
            rgb: dominant_rgb,
            hex: rgb_to_hex(dominant_rgb),
        },
    })
}

// ============================================================================
// EXIF Extraction
// ============================================================================

/// Check if GPS info contains valid data (not default/empty values)
fn is_valid_gps_value(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(s) => {
            // Check for patterns indicating default/empty values
            !s.contains("(0.0, 0.0, 0.0)")
                && !s.contains("'1970:01:01'")
                && !s.contains("0.0, 0.0, 0.0")
        }
        serde_json::Value::Object(map) => {
            // Check if coordinates are non-zero
            if let Some(lat) = map.get("GPSLatitude") {
                if let Some(arr) = lat.as_array() {
                    return arr
                        .iter()
                        .any(|v| v.as_f64().map(|f| f.abs() > 0.0001).unwrap_or(false));
                }
            }
            true // Default to keeping it if we can't determine
        }
        _ => true,
    }
}

/// Extract EXIF data from image bytes
fn extract_exif_data(image_bytes: &[u8]) -> HashMap<String, serde_json::Value> {
    let mut exif_data = HashMap::new();

    // Try to parse EXIF data using the exif crate
    let cursor = Cursor::new(image_bytes);
    let mut bufreader = std::io::BufReader::new(cursor);

    let exif_reader = match exif::Reader::new().read_from_container(&mut bufreader) {
        Ok(reader) => reader,
        Err(_) => return exif_data, // No EXIF data or parsing failed
    };

    for field in exif_reader.fields() {
        let tag_name = format!("{}", field.tag);

        // Convert EXIF value to JSON-compatible value
        let json_value = match &field.value {
            exif::Value::Byte(v) => {
                if v.len() == 1 {
                    serde_json::Value::Number(v[0].into())
                } else {
                    serde_json::Value::Array(v.iter().map(|&b| b.into()).collect())
                }
            }
            exif::Value::Ascii(v) => {
                let strings: Vec<String> = v
                    .iter()
                    .filter_map(|bytes| String::from_utf8(bytes.to_vec()).ok())
                    .collect();
                if strings.len() == 1 {
                    serde_json::Value::String(strings[0].trim_end_matches('\0').to_string())
                } else {
                    serde_json::Value::Array(
                        strings
                            .into_iter()
                            .map(|s| {
                                serde_json::Value::String(s.trim_end_matches('\0').to_string())
                            })
                            .collect(),
                    )
                }
            }
            exif::Value::Short(v) => {
                if v.len() == 1 {
                    serde_json::Value::Number(v[0].into())
                } else {
                    serde_json::Value::Array(v.iter().map(|&n| n.into()).collect())
                }
            }
            exif::Value::Long(v) => {
                if v.len() == 1 {
                    serde_json::Value::Number(v[0].into())
                } else {
                    serde_json::Value::Array(v.iter().map(|&n| n.into()).collect())
                }
            }
            exif::Value::Rational(v) => {
                let floats: Vec<f64> = v.iter().map(|r| r.to_f64()).collect();
                if floats.len() == 1 {
                    serde_json::Number::from_f64(floats[0])
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::String(format!("{}", floats[0])))
                } else {
                    serde_json::Value::Array(
                        floats
                            .iter()
                            .map(|&f| {
                                serde_json::Number::from_f64(f)
                                    .map(serde_json::Value::Number)
                                    .unwrap_or(serde_json::Value::String(format!("{}", f)))
                            })
                            .collect(),
                    )
                }
            }
            exif::Value::SRational(v) => {
                let floats: Vec<f64> = v.iter().map(|r| r.to_f64()).collect();
                if floats.len() == 1 {
                    serde_json::Number::from_f64(floats[0])
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::String(format!("{}", floats[0])))
                } else {
                    serde_json::Value::Array(
                        floats
                            .iter()
                            .map(|&f| {
                                serde_json::Number::from_f64(f)
                                    .map(serde_json::Value::Number)
                                    .unwrap_or(serde_json::Value::String(format!("{}", f)))
                            })
                            .collect(),
                    )
                }
            }
            exif::Value::SByte(v) => {
                if v.len() == 1 {
                    serde_json::Value::Number(v[0].into())
                } else {
                    serde_json::Value::Array(v.iter().map(|&b| b.into()).collect())
                }
            }
            exif::Value::SShort(v) => {
                if v.len() == 1 {
                    serde_json::Value::Number(v[0].into())
                } else {
                    serde_json::Value::Array(v.iter().map(|&n| n.into()).collect())
                }
            }
            exif::Value::SLong(v) => {
                if v.len() == 1 {
                    serde_json::Value::Number(v[0].into())
                } else {
                    serde_json::Value::Array(v.iter().map(|&n| n.into()).collect())
                }
            }
            exif::Value::Float(v) => {
                if v.len() == 1 {
                    serde_json::Number::from_f64(v[0] as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::String(format!("{}", v[0])))
                } else {
                    serde_json::Value::Array(
                        v.iter()
                            .map(|&f| {
                                serde_json::Number::from_f64(f as f64)
                                    .map(serde_json::Value::Number)
                                    .unwrap_or(serde_json::Value::String(format!("{}", f)))
                            })
                            .collect(),
                    )
                }
            }
            exif::Value::Double(v) => {
                if v.len() == 1 {
                    serde_json::Number::from_f64(v[0])
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::String(format!("{}", v[0])))
                } else {
                    serde_json::Value::Array(
                        v.iter()
                            .map(|&f| {
                                serde_json::Number::from_f64(f)
                                    .map(serde_json::Value::Number)
                                    .unwrap_or(serde_json::Value::String(format!("{}", f)))
                            })
                            .collect(),
                    )
                }
            }
            exif::Value::Undefined(v, _) => {
                // For undefined/binary data, convert to hex string if small, otherwise indicate size
                if v.len() <= 32 {
                    serde_json::Value::String(hex::encode(v))
                } else {
                    serde_json::Value::String(format!("<{} bytes>", v.len()))
                }
            }
            _ => serde_json::Value::String(field.display_value().to_string()),
        };

        exif_data.insert(tag_name, json_value);
    }

    // Filter out invalid GPS data
    if let Some(gps_info) = exif_data.get("GPSInfo") {
        if !is_valid_gps_value(gps_info) {
            exif_data.remove("GPSInfo");
            info!("Filtered out empty/default GPS information");
        }
    }

    exif_data
}

// ============================================================================
// HTTP Handler
// ============================================================================

/// Extract image statistics (EXIF data and color analysis)
#[utoipa::path(
    post,
    path = "/img/stats",
    request_body = ImageStatsRequest,
    responses(
        (status = 200, description = "Statistics extracted successfully", body = ImageStatsResponse),
        (status = 400, description = "Bad request (invalid image)", body = crate::ErrorResponse),
    ),
    tag = "image"
)]
pub async fn img_stats_handler(
    State(_state): State<AppState>,
    Json(req): Json<ImageStatsRequest>,
) -> Result<Json<ImageStatsResponse>, Error> {
    let start = Instant::now();

    // Fetch and decode image, keeping the original bytes for EXIF
    let image_bytes = fetch_image_bytes(&req.content).await?;
    let image = decode_image_from_bytes(&image_bytes)?;

    // Extract EXIF data from original bytes
    let exif_data = extract_exif_data(&image_bytes);

    // Perform color analysis
    let color_data = get_image_colors(&image, req.averaging_method);

    let total_time = start.elapsed();
    info!(
        "Image stats timing - total: {:.2}ms, exif_fields: {}, has_colors: {}",
        total_time.as_secs_f64() * 1000.0,
        exif_data.len(),
        color_data.is_some()
    );

    Ok(Json(ImageStatsResponse {
        exif_data,
        color_data,
        time_ms: total_time.as_secs_f64() * 1000.0,
    }))
}

/// Fetch image bytes from URL or decode from base64
async fn fetch_image_bytes(content: &str) -> Result<Vec<u8>, Error> {
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

    if content.starts_with("http://") || content.starts_with("https://") {
        // URL - fetch with timeout
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| {
                Error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("HTTP client error: {}", e),
                )
            })?;

        let response = client.get(content).send().await.map_err(|e| {
            Error(
                StatusCode::BAD_REQUEST,
                format!("Failed to fetch URL: {}", e),
            )
        })?;

        if !response.status().is_success() {
            return Err(Error(
                StatusCode::BAD_REQUEST,
                format!("URL returned status {}", response.status()),
            ));
        }

        response.bytes().await.map(|b| b.to_vec()).map_err(|e| {
            Error(
                StatusCode::BAD_REQUEST,
                format!("Failed to read image data: {}", e),
            )
        })
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
        })
    } else {
        // Raw base64
        BASE64
            .decode(content)
            .map_err(|e| Error(StatusCode::BAD_REQUEST, format!("Invalid base64: {}", e)))
    }
}

/// Decode image from bytes
fn decode_image_from_bytes(bytes: &[u8]) -> Result<DynamicImage, Error> {
    use image::ImageReader;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_hex() {
        assert_eq!(rgb_to_hex([1.0, 1.0, 1.0]), "#ffffff");
        assert_eq!(rgb_to_hex([0.0, 0.0, 0.0]), "#000000");
        assert_eq!(rgb_to_hex([1.0, 0.0, 0.0]), "#ff0000");
        assert_eq!(rgb_to_hex([0.5, 0.5, 0.5]), "#808080");
    }

    #[test]
    fn test_rgb_to_hsv() {
        // Pure red
        let (h, s, v) = rgb_to_hsv(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 0.01);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);

        // Pure green
        let (h, s, v) = rgb_to_hsv(0.0, 1.0, 0.0);
        assert!((h - 0.333).abs() < 0.01);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);

        // Gray (no saturation)
        let (h, s, v) = rgb_to_hsv(0.5, 0.5, 0.5);
        assert!((s - 0.0).abs() < 0.01);
        assert!((v - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_arithmetic_mean() {
        let pixels = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let avg = calculate_arithmetic_mean(&pixels);
        assert!((avg[0] - 0.333).abs() < 0.01);
        assert!((avg[1] - 0.333).abs() < 0.01);
        assert!((avg[2] - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_geometric_mean() {
        let pixels = vec![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]];
        let avg = calculate_geometric_mean(&pixels);
        assert!((avg[0] - 0.5).abs() < 0.01);
        assert!((avg[1] - 0.5).abs() < 0.01);
        assert!((avg[2] - 0.5).abs() < 0.01);
    }
}
