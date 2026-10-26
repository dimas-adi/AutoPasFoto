//! Image I/O: load with EXIF orientation, save JPEG at quality 100 and 300 DPI.

use std::fs::{File, create_dir_all};
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};
use image::codecs::jpeg::{JpegEncoder, PixelDensity};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

use crate::config;

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

/// Saves an RGB image as a JPEG file with JPEG_QUALITY (100) and OUTPUT_DPI (300).
/// Creates parent directories automatically if they do not exist.
pub fn save_jpeg(img: &RgbImage, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        create_dir_all(parent)
            .with_context(|| format!("Gagal membuat folder: {}", parent.display()))?;
    }

    let file = File::create(path)
        .with_context(|| format!("Gagal membuat file foto: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    let mut encoder = JpegEncoder::new_with_quality(&mut writer, config::JPEG_QUALITY);
    encoder.set_pixel_density(PixelDensity::dpi(config::OUTPUT_DPI as u16));
    encoder
        .encode_image(img)
        .with_context(|| format!("Gagal mengompresi dan menyimpan foto: {}", path.display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::read;
    use tempfile::tempdir;

    #[test]
    fn test_save_jpeg_jfif_dpi_and_tolerance() {
        let width = 16;
        let height = 16;
        let mut img = RgbImage::new(width, height);
        for x in 0..width {
            for y in 0..height {
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([(x * 15) as u8, (y * 15) as u8, ((x + y) * 7) as u8]),
                );
            }
        }

        let dir = tempdir().expect("create temp dir");
        let nested_path = dir.path().join("sub").join("nested").join("test_out.jpg");

        save_jpeg(&img, &nested_path).expect("save jpeg");
        assert!(nested_path.exists());

        // Validate raw JFIF APP0 header
        let raw = read(&nested_path).expect("read saved jpeg bytes");
        assert!(raw.len() >= 20, "File too small to contain JFIF header");
        assert_eq!(&raw[0..2], &[0xFF, 0xD8], "Missing SOI marker");
        assert_eq!(&raw[2..4], &[0xFF, 0xE0], "Missing APP0 marker");

        let seg_len = u16::from_be_bytes([raw[4], raw[5]]);
        assert!(seg_len >= 16, "APP0 segment length too short");
        assert_eq!(&raw[6..11], b"JFIF\0", "Missing JFIF identifier");

        let density_unit = raw[13];
        assert_eq!(density_unit, 1, "Expected density unit 1 (DPI)");

        let x_density = u16::from_be_bytes([raw[14], raw[15]]);
        let y_density = u16::from_be_bytes([raw[16], raw[17]]);
        assert_eq!(x_density, config::OUTPUT_DPI as u16, "X DPI mismatch");
        assert_eq!(y_density, config::OUTPUT_DPI as u16, "Y DPI mismatch");

        // Decode back and check pixel tolerance
        let decoded = load_image(&nested_path).expect("load decoded image");
        assert_eq!(decoded.dimensions(), (width, height));

        let mut max_diff: u8 = 0;
        for (orig_p, dec_p) in img.pixels().zip(decoded.pixels()) {
            for c in 0..3 {
                let diff = orig_p.0[c].abs_diff(dec_p.0[c]);
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        assert!(
            max_diff <= 5,
            "Maximum pixel difference {max_diff} exceeded tolerance of 5"
        );
    }
}
