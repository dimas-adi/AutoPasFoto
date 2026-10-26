//! Parity comparison tool between Python reference and Rust pasfoto 3x4 output.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use pasfoto::face::{Face, FaceDetector};
use pasfoto::{assets, config, imageio};

/// Normalized positioning metrics computed from face landmarks and image dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Eye line vertical position as a ratio of image height (`ey / h`).
    pub mata_y: f64,
    /// Face horizontal center position as a ratio of image width (`ex / w`).
    pub tengah_x: f64,
    /// Relative face scale computed as interocular distance over image width (`iod / w`).
    pub skala: f64,
}

/// Absolute and relative differences between two metric sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricDiffs {
    /// Absolute difference in vertical eye position (`|mata_y_a - mata_y_b|`).
    pub diff_mata_y: f64,
    /// Absolute difference in horizontal center position (`|tengah_x_a - tengah_x_b|`).
    pub diff_tengah_x: f64,
    /// Relative difference in scale (`|skala_a - skala_b| / max(skala_a, skala_b)`).
    pub diff_skala: f64,
}

/// Parity evaluation status for a single image pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityStatus {
    /// Framing matches within specified tolerance thresholds.
    Ok,
    /// Difference in one or more metrics exceeds tolerance thresholds.
    Beda,
    /// Face was not detected in one or both images.
    TidakTerdeteksi,
}

impl ParityStatus {
    /// Returns the Indonesian display label for this status.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Beda => "BEDA",
            Self::TidakTerdeteksi => "TIDAK TERDETEKSI",
        }
    }
}

/// Computes normalized framing metrics from a detected face and image dimensions.
pub fn compute_metrics(face: &Face, img_w: f64, img_h: f64) -> Metrics {
    let (ex, ey, iod) = face.eyes();
    let safe_w = if img_w > 0.0 { img_w } else { 1.0 };
    let safe_h = if img_h > 0.0 { img_h } else { 1.0 };

    Metrics {
        mata_y: f64::from(ey) / safe_h,
        tengah_x: f64::from(ex) / safe_w,
        skala: f64::from(iod) / safe_w,
    }
}

/// Computes absolute differences for eye Y and center X, and relative difference for scale.
pub fn compare_metrics(a: &Metrics, b: &Metrics) -> MetricDiffs {
    let diff_mata_y = (a.mata_y - b.mata_y).abs();
    let diff_tengah_x = (a.tengah_x - b.tengah_x).abs();

    let max_skala = a.skala.max(b.skala);
    // If both or either scale is zero, treat as completely divergent (1.0) without division by zero.
    let diff_skala = if max_skala > 0.0 && a.skala > 0.0 && b.skala > 0.0 {
        (a.skala - b.skala).abs() / max_skala
    } else {
        1.0
    };

    MetricDiffs {
        diff_mata_y,
        diff_tengah_x,
        diff_skala,
    }
}

/// Evaluates whether the computed differences satisfy the tolerance thresholds.
pub fn evaluate_status(diffs: &MetricDiffs) -> ParityStatus {
    if diffs.diff_mata_y > config::PARITY_MAX_EYE_Y_DIFF
        || diffs.diff_tengah_x > config::PARITY_MAX_CENTER_X_DIFF
        || diffs.diff_skala > config::PARITY_MAX_SCALE_REL_DIFF
    {
        ParityStatus::Beda
    } else {
        ParityStatus::Ok
    }
}

/// Evaluation result for a matched pair of files.
#[derive(Debug)]
struct PairResult {
    file_name: String,
    diffs: Option<MetricDiffs>,
    status: ParityStatus,
}

/// Scans a directory recursively and collects image paths indexed by file name.
fn collect_image_files(dir: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut map = BTreeMap::new();
    scan_dir_recursive(dir, &mut map)?;
    Ok(map)
}

fn scan_dir_recursive(dir: &Path, map: &mut BTreeMap<String, PathBuf>) -> Result<()> {
    let entries = fs::read_dir(dir)
        .with_context(|| format!("Gagal membaca isi folder: {}", dir.display()))?;

    for entry in entries {
        let entry =
            entry.with_context(|| format!("Gagal membaca entri folder di: {}", dir.display()))?;
        let path = entry.path();

        if path.is_dir() {
            scan_dir_recursive(&path, map)?;
            continue;
        }

        let Some(file_name_os) = path.file_name() else {
            continue;
        };
        let file_name = file_name_os.to_string_lossy().to_string();

        // Skip hidden or temporary files.
        if file_name.starts_with('.') || file_name.starts_with('~') {
            continue;
        }

        // Check image file extensions (.jpg, .jpeg, .png).
        let is_image = path
            .extension()
            .and_then(OsStr::to_str)
            .map(|ext| {
                let lower = ext.to_lowercase();
                lower == "jpg" || lower == "jpeg" || lower == "png"
            })
            .unwrap_or(false);

        if is_image {
            map.insert(file_name, path);
        }
    }

    Ok(())
}

fn run_parity(folder_python: &Path, folder_rust: &Path) -> Result<()> {
    if !folder_python.is_dir() {
        bail!(
            "Folder 3x4 Python tidak ditemukan atau bukan direktori: {}",
            folder_python.display()
        );
    }
    if !folder_rust.is_dir() {
        bail!(
            "Folder 3x4 Rust tidak ditemukan atau bukan direktori: {}",
            folder_rust.display()
        );
    }

    // Locate assets and load face detection model.
    let asset_paths = assets::locate().context("Gagal menemukan folder aset aplikasi")?;
    let detector = FaceDetector::load(&asset_paths.model_onnx)
        .context("Gagal memuat model deteksi wajah YuNet")?;

    let py_files = collect_image_files(folder_python)?;
    let rs_files = collect_image_files(folder_rust)?;

    // Match files present in both directories by file name.
    let mut matched_names: Vec<String> = py_files
        .keys()
        .filter(|k| rs_files.contains_key(*k))
        .cloned()
        .collect();
    matched_names.sort();

    if matched_names.is_empty() {
        bail!(
            "Tidak ditemukan pasangan file yang cocok antara:\n- {}\n- {}",
            folder_python.display(),
            folder_rust.display()
        );
    }

    println!("Membandingkan {} pasangan file 3x4...", matched_names.len());

    let mut results = Vec::with_capacity(matched_names.len());
    let mut count_ok = 0usize;
    let mut count_beda = 0usize;
    let mut count_tidak_terdeteksi = 0usize;

    for name in &matched_names {
        let path_py = py_files
            .get(name)
            .with_context(|| format!("File Python tidak ditemukan di peta: {name}"))?;
        let path_rs = rs_files
            .get(name)
            .with_context(|| format!("File Rust tidak ditemukan di peta: {name}"))?;

        let img_py = imageio::load_image(path_py)
            .with_context(|| format!("Gagal memuat foto Python: {}", path_py.display()))?;
        let img_rs = imageio::load_image(path_rs)
            .with_context(|| format!("Gagal memuat foto Rust: {}", path_rs.display()))?;

        let face_py = detector.detect(&img_py).with_context(|| {
            format!(
                "Deteksi wajah gagal pada foto Python: {}",
                path_py.display()
            )
        })?;
        let face_rs = detector.detect(&img_rs).with_context(|| {
            format!("Deteksi wajah gagal pada foto Rust: {}", path_rs.display())
        })?;

        match (face_py, face_rs) {
            (Some(f_py), Some(f_rs)) => {
                let m_py =
                    compute_metrics(&f_py, f64::from(img_py.width()), f64::from(img_py.height()));
                let m_rs =
                    compute_metrics(&f_rs, f64::from(img_rs.width()), f64::from(img_rs.height()));
                let diffs = compare_metrics(&m_py, &m_rs);
                let status = evaluate_status(&diffs);

                match status {
                    ParityStatus::Ok => count_ok += 1,
                    ParityStatus::Beda => count_beda += 1,
                    ParityStatus::TidakTerdeteksi => count_tidak_terdeteksi += 1,
                }

                results.push(PairResult {
                    file_name: name.clone(),
                    diffs: Some(diffs),
                    status,
                });
            }
            _ => {
                count_tidak_terdeteksi += 1;
                results.push(PairResult {
                    file_name: name.clone(),
                    diffs: None,
                    status: ParityStatus::TidakTerdeteksi,
                });
            }
        }
    }

    // Print comparison table.
    let col_name_w = results
        .iter()
        .map(|r| r.file_name.len())
        .max()
        .unwrap_or(20)
        .max(16);

    let sep_len = col_name_w + 58;
    let separator = "-".repeat(sep_len);

    println!("\n{separator}");
    println!(
        "{:<width$} | {:>12} | {:>14} | {:>14} | {:<16}",
        "Nama File",
        "Δ Mata Y (%)",
        "Δ Tengah X (%)",
        "Δ Skala Rel (%)",
        "Status",
        width = col_name_w
    );
    println!("{separator}");

    for res in &results {
        match res.diffs {
            Some(d) => {
                println!(
                    "{:<width$} | {:>11.2}% | {:>13.2}% | {:>13.2}% | {:<16}",
                    res.file_name,
                    d.diff_mata_y * 100.0,
                    d.diff_tengah_x * 100.0,
                    d.diff_skala * 100.0,
                    res.status.as_str(),
                    width = col_name_w
                );
            }
            None => {
                println!(
                    "{:<width$} | {:>12} | {:>14} | {:>14} | {:<16}",
                    res.file_name,
                    "-",
                    "-",
                    "-",
                    res.status.as_str(),
                    width = col_name_w
                );
            }
        }
    }

    println!("{separator}");

    // Print summary.
    println!("\nRingkasan Uji Kesetaraan:");
    println!("- Total pasangan        : {}", results.len());
    println!("- OK                    : {count_ok}");
    println!("- BEDA                  : {count_beda}");
    println!("- TIDAK TERDETEKSI      : {count_tidak_terdeteksi}");

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args[1] == "-h" || args[1] == "--help" {
        eprintln!(
            "Penggunaan: cargo run --release --bin parity -- <folder_3x4_python> <folder_3x4_rust>"
        );
        std::process::exit(1);
    }

    let folder_python = Path::new(&args[1]);
    let folder_rust = Path::new(&args[2]);

    run_parity(folder_python, folder_rust)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_face(ex: f32, ey: f32, iod: f32) -> Face {
        Face {
            score: 0.95,
            bbox: (ex - 50.0, ey - 50.0, 100.0, 120.0),
            landmarks: [
                (ex - iod / 2.0, ey), // right eye (left in image)
                (ex + iod / 2.0, ey), // left eye
                (ex, ey + 30.0),      // nose
                (ex - 20.0, ey + 50.0),
                (ex + 20.0, ey + 50.0),
            ],
        }
    }

    #[test]
    fn test_compute_metrics() {
        let face = dummy_face(150.0, 280.0, 147.0);
        let m = compute_metrics(&face, 300.0, 400.0);

        // ey = 280, h = 400 => mata_y = 0.70
        assert!((m.mata_y - 0.70).abs() < 1e-6);
        // ex = 150, w = 300 => tengah_x = 0.50
        assert!((m.tengah_x - 0.50).abs() < 1e-6);
        // iod = 147, w = 300 => skala = 0.49
        assert!((m.skala - 0.49).abs() < 1e-6);
    }

    #[test]
    fn test_compare_metrics_normal() {
        let m1 = Metrics {
            mata_y: 0.28,
            tengah_x: 0.50,
            skala: 0.40,
        };
        let m2 = Metrics {
            mata_y: 0.29,
            tengah_x: 0.51,
            skala: 0.42,
        };
        let diffs = compare_metrics(&m1, &m2);

        assert!((diffs.diff_mata_y - 0.01).abs() < 1e-6);
        assert!((diffs.diff_tengah_x - 0.01).abs() < 1e-6);
        // |0.40 - 0.42| / 0.42 = 0.02 / 0.42 ≈ 0.047619
        let expected_scale_diff = 0.02 / 0.42;
        assert!((diffs.diff_skala - expected_scale_diff).abs() < 1e-6);
    }

    #[test]
    fn test_compare_metrics_zero_scale() {
        let m1 = Metrics {
            mata_y: 0.28,
            tengah_x: 0.50,
            skala: 0.0,
        };
        let m2 = Metrics {
            mata_y: 0.28,
            tengah_x: 0.50,
            skala: 0.40,
        };
        let diffs = compare_metrics(&m1, &m2);
        assert!((diffs.diff_skala - 1.0).abs() < 1e-6);

        let m3 = Metrics {
            mata_y: 0.28,
            tengah_x: 0.50,
            skala: 0.0,
        };
        let diffs_both_zero = compare_metrics(&m1, &m3);
        assert!((diffs_both_zero.diff_skala - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_status_boundaries() {
        // All identical -> Ok
        let base = MetricDiffs {
            diff_mata_y: 0.0,
            diff_tengah_x: 0.0,
            diff_skala: 0.0,
        };
        assert_eq!(evaluate_status(&base), ParityStatus::Ok);

        // Boundary: exact 1.5 percentage points (0.015) -> Ok
        let at_boundary_y = MetricDiffs {
            diff_mata_y: 0.015,
            ..base
        };
        assert_eq!(evaluate_status(&at_boundary_y), ParityStatus::Ok);

        // Above boundary: > 0.015 -> Beda
        let above_boundary_y = MetricDiffs {
            diff_mata_y: 0.0150001,
            ..base
        };
        assert_eq!(evaluate_status(&above_boundary_y), ParityStatus::Beda);

        // Boundary: exact 1.5 percentage points (0.015) on center X -> Ok
        let at_boundary_x = MetricDiffs {
            diff_tengah_x: 0.015,
            ..base
        };
        assert_eq!(evaluate_status(&at_boundary_x), ParityStatus::Ok);

        // Above boundary: > 0.015 on center X -> Beda
        let above_boundary_x = MetricDiffs {
            diff_tengah_x: 0.0150001,
            ..base
        };
        assert_eq!(evaluate_status(&above_boundary_x), ParityStatus::Beda);

        // Boundary: exact 8% relative scale (0.08) -> Ok
        let at_boundary_scale = MetricDiffs {
            diff_skala: 0.08,
            ..base
        };
        assert_eq!(evaluate_status(&at_boundary_scale), ParityStatus::Ok);

        // Above boundary: > 0.08 relative scale -> Beda
        let above_boundary_scale = MetricDiffs {
            diff_skala: 0.0800001,
            ..base
        };
        assert_eq!(evaluate_status(&above_boundary_scale), ParityStatus::Beda);
    }
}
