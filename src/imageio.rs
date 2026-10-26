//! Image I/O: load with EXIF orientation, save JPEG at quality 100 and 300 DPI.

use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

/// Loads an image from disk, applies EXIF orientation, and converts it to RGB8.
pub fn load_image(path: &Path) -> Result<RgbImage> {
    let reader = ImageReader::open(path)
        .with_context(|| format!("Gagal membuka file foto: {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("Gagal mengenali format file foto: {}", path.display()))?;

    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("Format foto tidak didukung atau rusak: {}", path.display()))?;

    let orientation = decoder
        .orientation()
        .with_context(|| format!("Gagal membaca orientasi EXIF foto: {}", path.display()))?;

    let mut img = DynamicImage::from_decoder(decoder)
        .with_context(|| format!("Gagal membaca isi gambar dari foto: {}", path.display()))?;

    img.apply_orientation(orientation);
    Ok(img.into_rgb8())
}
