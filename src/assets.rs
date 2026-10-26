//! Asset management: locate Template/ and models/, load template, overlay, and font.

use std::path::PathBuf;

use ab_glyph::FontArc;
use anyhow::{Context, Result, bail};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{ImageReader, RgbaImage};

use crate::config;

/// File paths to all located application assets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetPaths {
    pub template_dir: PathBuf,
    pub models_dir: PathBuf,
    pub template_png: PathBuf,
    pub overlay_png: Option<PathBuf>,
    pub font_ttf: PathBuf,
    pub model_onnx: PathBuf,
}

/// Loaded asset resources in memory.
pub struct Assets {
    pub template: RgbaImage,
    pub overlay: Option<RgbaImage>,
    pub font: FontArc,
}

/// Locates required asset directories by searching the executable's directory first,
/// then falling back to the current working directory.
pub fn locate() -> Result<AssetPaths> {
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        candidates.push(parent.to_path_buf());
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    locate_in(&candidates)
}

/// Locates asset directories and files within the provided candidate base directories.
pub fn locate_in(candidate_dirs: &[PathBuf]) -> Result<AssetPaths> {
    // 1. Locate Template directory
    let template_dir = candidate_dirs
        .iter()
        .map(|p| p.join(config::DIR_TEMPLATE))
        .find(|p| p.is_dir())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Folder \"{}\" tidak ditemukan. Pastikan folder {} ada di samping aplikasi.",
                config::DIR_TEMPLATE,
                config::DIR_TEMPLATE
            )
        })?;

    // 2. Locate models directory
    let models_dir = candidate_dirs
        .iter()
        .map(|p| p.join(config::DIR_MODELS))
        .find(|p| p.is_dir())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Folder \"{}\" tidak ditemukan. Pastikan folder {} ada di samping aplikasi.",
                config::DIR_MODELS,
                config::DIR_MODELS
            )
        })?;

    // 3. Validate mandatory Templates.png
    let template_png = template_dir.join(config::FILE_TEMPLATE_PNG);
    if !template_png.is_file() {
        bail!(
            "File \"{}\" tidak ditemukan di dalam folder {}.",
            config::FILE_TEMPLATE_PNG,
            config::DIR_TEMPLATE
        );
    }

    // 4. Check optional overlay.png
    let overlay_candidate = template_dir.join(config::FILE_OVERLAY_PNG);
    let overlay_png = if overlay_candidate.is_file() {
        Some(overlay_candidate)
    } else {
        None
    };

    // 5. Check font: custom Template/font.ttf if exists, otherwise fallback font path
    let font_candidate = template_dir.join(config::FILE_FONT_TTF);
    let font_ttf = if font_candidate.is_file() {
        font_candidate
    } else {
        let fallback = PathBuf::from(config::FALLBACK_FONT_PATH);
        if fallback.is_file() {
            fallback
        } else {
            bail!(
                "Font label tidak ditemukan. Sediakan file {}/{} atau pastikan font sistem {} tersedia.",
                config::DIR_TEMPLATE,
                config::FILE_FONT_TTF,
                config::FALLBACK_FONT_PATH
            );
        }
    };

    // 6. Validate mandatory model
    let model_onnx = models_dir.join(config::FILE_YUNET_ONNX);
    if !model_onnx.is_file() {
        bail!(
            "File model \"{}\" tidak ditemukan di dalam folder {}.",
            config::FILE_YUNET_ONNX,
            config::DIR_MODELS
        );
    }

    Ok(AssetPaths {
        template_dir,
        models_dir,
        template_png,
        overlay_png,
        font_ttf,
        model_onnx,
    })
}

/// Loads assets from default locations into memory.
pub fn load() -> Result<Assets> {
    let paths = locate()?;
    load_from_paths(&paths)
}

/// Loads assets into memory using specified paths.
pub fn load_from_paths(paths: &AssetPaths) -> Result<Assets> {
    // 1. Load template PNG
    let template_raw = ImageReader::open(&paths.template_png)
        .with_context(|| {
            format!(
                "Gagal membuka file template: {}",
                paths.template_png.display()
            )
        })?
        .with_guessed_format()
        .with_context(|| {
            format!(
                "Gagal mengenali format template: {}",
                paths.template_png.display()
            )
        })?
        .decode()
        .with_context(|| {
            format!(
                "Gagal membaca gambar template dari: {}",
                paths.template_png.display()
            )
        })?
        .into_rgba8();

    if template_raw.dimensions() != (config::CANVAS_4R_WIDTH, config::CANVAS_4R_HEIGHT) {
        bail!(
            "Ukuran file \"{}\" tidak sesuai (harus {}x{} piksel, tetapi ditemukan {}x{}). Pastikan template 4R berukuran tepat.",
            config::FILE_TEMPLATE_PNG,
            config::CANVAS_4R_WIDTH,
            config::CANVAS_4R_HEIGHT,
            template_raw.width(),
            template_raw.height()
        );
    }

    // 2. Load overlay PNG (if present) and resize to 1800x1200 with Lanczos3 if dimensions differ
    let overlay = if let Some(ref overlay_path) = paths.overlay_png {
        let overlay_raw = ImageReader::open(overlay_path)
            .with_context(|| format!("Gagal membuka file overlay: {}", overlay_path.display()))?
            .with_guessed_format()
            .with_context(|| format!("Gagal mengenali format overlay: {}", overlay_path.display()))?
            .decode()
            .with_context(|| {
                format!(
                    "Gagal membaca gambar overlay dari: {}",
                    overlay_path.display()
                )
            })?
            .into_rgba8();

        if overlay_raw.dimensions() != (config::CANVAS_4R_WIDTH, config::CANVAS_4R_HEIGHT) {
            let mut resized = RgbaImage::new(config::CANVAS_4R_WIDTH, config::CANVAS_4R_HEIGHT);
            let mut resizer = Resizer::new();
            let options =
                ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
            resizer
                .resize(&overlay_raw, &mut resized, &options)
                .context("Gagal mengubah ukuran overlay.png ke 1800x1200")?;
            Some(resized)
        } else {
            Some(overlay_raw)
        }
    } else {
        None
    };

    // 3. Load font TTF
    let font_bytes = std::fs::read(&paths.font_ttf)
        .with_context(|| format!("Gagal membaca file font: {}", paths.font_ttf.display()))?;
    let font = FontArc::try_from_vec(font_bytes).map_err(|_| {
        anyhow::anyhow!(
            "File font \"{}\" tidak valid atau rusak.",
            paths.font_ttf.display()
        )
    })?;

    Ok(Assets {
        template: template_raw,
        overlay,
        font,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{File, create_dir_all};
    use tempfile::tempdir;

    #[test]
    fn test_locate_in_missing_template_dir() {
        let dir = tempdir().expect("temp dir");
        let err = locate_in(&[dir.path().to_path_buf()]).unwrap_err();
        assert!(
            err.to_string()
                .contains("Folder \"Template\" tidak ditemukan"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_locate_in_missing_models_dir() {
        let dir = tempdir().expect("temp dir");
        create_dir_all(dir.path().join("Template")).expect("create Template");
        let err = locate_in(&[dir.path().to_path_buf()]).unwrap_err();
        assert!(
            err.to_string()
                .contains("Folder \"models\" tidak ditemukan"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_locate_in_missing_templates_png() {
        let dir = tempdir().expect("temp dir");
        create_dir_all(dir.path().join("Template")).expect("create Template");
        create_dir_all(dir.path().join("models")).expect("create models");
        let err = locate_in(&[dir.path().to_path_buf()]).unwrap_err();
        assert!(
            err.to_string()
                .contains("File \"Templates.png\" tidak ditemukan"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_locate_in_missing_model_file() {
        let dir = tempdir().expect("temp dir");
        let template_dir = dir.path().join("Template");
        create_dir_all(&template_dir).expect("create Template");
        File::create(template_dir.join("Templates.png")).expect("create dummy Templates.png");

        let models_dir = dir.path().join("models");
        create_dir_all(&models_dir).expect("create models");

        let err = locate_in(&[dir.path().to_path_buf()]).unwrap_err();
        assert!(
            err.to_string()
                .contains("File model \"face_detection_yunet_2023mar.onnx\" tidak ditemukan"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_locate_in_success_with_fallback_or_custom_font() {
        let dir = tempdir().expect("temp dir");
        let template_dir = dir.path().join("Template");
        create_dir_all(&template_dir).expect("create Template");
        File::create(template_dir.join("Templates.png")).expect("create dummy Templates.png");

        let models_dir = dir.path().join("models");
        create_dir_all(&models_dir).expect("create models");
        File::create(models_dir.join("face_detection_yunet_2023mar.onnx"))
            .expect("create dummy onnx");

        // Check locate result
        let fallback_exists = PathBuf::from(config::FALLBACK_FONT_PATH).is_file();
        let res = locate_in(&[dir.path().to_path_buf()]);
        if fallback_exists {
            let paths = res.expect("locate should succeed when fallback font exists");
            assert_eq!(paths.template_png, template_dir.join("Templates.png"));
            assert_eq!(paths.overlay_png, None);
            assert_eq!(paths.font_ttf, PathBuf::from(config::FALLBACK_FONT_PATH));
        } else {
            // If fallback font does not exist, provide custom Template/font.ttf
            File::create(template_dir.join("font.ttf")).expect("create dummy font.ttf");
            let paths = locate_in(&[dir.path().to_path_buf()])
                .expect("locate should succeed with font.ttf");
            assert_eq!(paths.font_ttf, template_dir.join("font.ttf"));
        }
    }
}
