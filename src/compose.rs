//! 4R layout composition: slot fitting (335x445), canvas (1800x1200), and text rendering.

use ab_glyph::{Font, PxScale};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, RgbImage, Rgba, RgbaImage, imageops};
use imageproc::drawing::draw_text_mut;

use crate::assets::Assets;
use crate::config;

/// Fits an arbitrary input image into a 335x445 pasfoto slot with centering (0.5, 0.0) and Lanczos3.
/// Port of reference/app.py:576-578 (ImageOps.fit).
pub fn fit_to_slot(img: &RgbImage) -> RgbImage {
    let mut dst = RgbImage::new(config::SLOT_WIDTH, config::SLOT_HEIGHT);
    if img.width() == 0 || img.height() == 0 {
        return dst;
    }

    let mut resizer = Resizer::new();
    let options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
        .fit_into_destination(Some((0.5, 0.0)));

    if resizer.resize(img, &mut dst, &options).is_err() {
        return dst;
    }

    dst
}

/// Composes a 1800x1200 4R sheet for up to two students.
/// Pure function without disk I/O.
pub fn compose_sheet(
    assets: &Assets,
    first: Option<(&RgbImage, &str)>,
    second: Option<(&RgbImage, &str)>,
    class_name: &str,
) -> RgbImage {
    // 1. Initialize white RGBA canvas
    let mut canvas = RgbaImage::from_pixel(
        config::CANVAS_4R_WIDTH,
        config::CANVAS_4R_HEIGHT,
        Rgba([255, 255, 255, 255]),
    );

    // 2. Paste first student photo across row 1
    if let Some((img_1, _)) = first {
        let scaled_1 = fit_to_slot(img_1);
        let scaled_rgba = DynamicImage::ImageRgb8(scaled_1).into_rgba8();
        for &col_x in &config::SLOT_COLS_X {
            imageops::overlay(
                &mut canvas,
                &scaled_rgba,
                col_x as i64,
                config::SLOT_ROW1_Y as i64,
            );
        }
    }

    // 3. Paste second student photo across row 2 (if present)
    if let Some((img_2, _)) = second {
        let scaled_2 = fit_to_slot(img_2);
        let scaled_rgba = DynamicImage::ImageRgb8(scaled_2).into_rgba8();
        for &col_x in &config::SLOT_COLS_X {
            imageops::overlay(
                &mut canvas,
                &scaled_rgba,
                col_x as i64,
                config::SLOT_ROW2_Y as i64,
            );
        }
    }

    // 4. Alpha composite overlay.png (if present)
    if let Some(ref overlay) = assets.overlay {
        imageops::overlay(&mut canvas, overlay, 0, 0);
    }

    // 5. Alpha composite Templates.png
    imageops::overlay(&mut canvas, &assets.template, 0, 0);

    // 6. Draw labels if class_name is not empty
    if !class_name.is_empty() {
        let text_color = Rgba([
            config::LABEL_TEXT_COLOR[0],
            config::LABEL_TEXT_COLOR[1],
            config::LABEL_TEXT_COLOR[2],
            255,
        ]);
        let units_per_em = assets
            .font
            .units_per_em()
            .unwrap_or(config::DEFAULT_UNITS_PER_EM);
        let scale = PxScale::from(
            config::LABEL_FONT_SIZE_EM * assets.font.height_unscaled() / units_per_em,
        );

        if let Some((_, absen_1)) = first {
            let label_1 = format!("{class_name} / {absen_1}");
            draw_text_mut(
                &mut canvas,
                text_color,
                config::LABEL_ROW1_POS.0,
                config::LABEL_ROW1_POS.1,
                scale,
                &assets.font,
                &label_1,
            );
        }

        if let Some((_, absen_2)) = second {
            let label_2 = format!("{class_name} / {absen_2}");
            draw_text_mut(
                &mut canvas,
                text_color,
                config::LABEL_ROW2_POS.0,
                config::LABEL_ROW2_POS.1,
                scale,
                &assets.font,
                &label_2,
            );
        }
    }

    // 7. Convert canvas to RGB
    DynamicImage::ImageRgba8(canvas).into_rgb8()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ab_glyph::FontArc;
    use image::Rgb;
    use std::path::Path;

    #[test]
    fn test_fit_to_slot_discards_bottom() {
        let width = 670; // 2 * SLOT_WIDTH
        let height = 1000;
        let mut img = RgbImage::new(width, height);

        let color_a = Rgb([10, 20, 30]);
        let color_b = Rgb([200, 210, 220]);

        // Keep region is y in 0..890. Region y >= 900 is discarded bottom.
        for y in 0..height {
            for x in 0..width {
                if y < 900 {
                    img.put_pixel(x, y, color_a);
                } else {
                    img.put_pixel(x, y, color_b);
                }
            }
        }

        let result = fit_to_slot(&img);
        assert_eq!(
            result.dimensions(),
            (config::SLOT_WIDTH, config::SLOT_HEIGHT)
        );

        // Check every single row and pixel across the entire result
        for y in 0..config::SLOT_HEIGHT {
            for x in 0..config::SLOT_WIDTH {
                let p = result.get_pixel(x, y);
                assert_eq!(
                    *p, color_a,
                    "Pixel at ({x}, {y}) must be color A, found {:?}",
                    p
                );
                assert_ne!(
                    *p, color_b,
                    "Discarded bottom color B must not appear at row {y}"
                );
            }
        }
    }

    #[test]
    fn test_compose_sheet_synthetic() {
        let font_path = Path::new(config::FALLBACK_FONT_PATH);
        if !font_path.exists() {
            println!(
                "Skipping test_compose_sheet_synthetic: font fallback tidak ditemukan di {:?}",
                font_path
            );
            return;
        }

        let font_bytes = std::fs::read(font_path).expect("read font");
        let font = FontArc::try_from_vec(font_bytes).expect("parse font");

        let template = RgbaImage::new(config::CANVAS_4R_WIDTH, config::CANVAS_4R_HEIGHT);
        let assets = Assets {
            template,
            overlay: None,
            font,
        };

        let student_1 =
            RgbImage::from_pixel(config::SLOT_WIDTH, config::SLOT_HEIGHT, Rgb([0, 255, 0]));
        let student_2 =
            RgbImage::from_pixel(config::SLOT_WIDTH, config::SLOT_HEIGHT, Rgb([0, 0, 255]));

        // 1. When class_name is empty, slots are filled but label area remains pure white
        let sheet_empty_class = compose_sheet(
            &assets,
            Some((&student_1, "01")),
            Some((&student_2, "02")),
            "",
        );
        assert_eq!(
            sheet_empty_class.dimensions(),
            (config::CANVAS_4R_WIDTH, config::CANVAS_4R_HEIGHT)
        );

        // Check slot 1 in row 1
        let slot1_p =
            sheet_empty_class.get_pixel(config::SLOT_COLS_X[0] + 10, config::SLOT_ROW1_Y + 10);
        assert_eq!(*slot1_p, Rgb([0, 255, 0]));

        // Check slot 2 in row 2
        let slot2_p =
            sheet_empty_class.get_pixel(config::SLOT_COLS_X[0] + 10, config::SLOT_ROW2_Y + 10);
        assert_eq!(*slot2_p, Rgb([0, 0, 255]));

        // Check label areas: must remain pure white
        let label1_p = sheet_empty_class.get_pixel(
            config::LABEL_ROW1_POS.0 as u32 + 5,
            config::LABEL_ROW1_POS.1 as u32 + 5,
        );
        assert_eq!(
            *label1_p,
            Rgb([255, 255, 255]),
            "Label area 1 must remain white when class_name is empty"
        );
        let label2_p = sheet_empty_class.get_pixel(
            config::LABEL_ROW2_POS.0 as u32 + 5,
            config::LABEL_ROW2_POS.1 as u32 + 5,
        );
        assert_eq!(
            *label2_p,
            Rgb([255, 255, 255]),
            "Label area 2 must remain white when class_name is empty"
        );

        // 2. When class_name is provided, label text is drawn
        let sheet_with_class = compose_sheet(
            &assets,
            Some((&student_1, "01")),
            Some((&student_2, "02")),
            "12A",
        );

        // Find colored pixels in row 1 label area
        let mut found_text_pixel_1 = false;
        for dy in 0..50 {
            for dx in 0..200 {
                let px = sheet_with_class.get_pixel(
                    config::LABEL_ROW1_POS.0 as u32 + dx,
                    config::LABEL_ROW1_POS.1 as u32 + dy,
                );
                if *px != Rgb([255, 255, 255]) {
                    found_text_pixel_1 = true;
                    break;
                }
            }
            if found_text_pixel_1 {
                break;
            }
        }
        assert!(
            found_text_pixel_1,
            "Label text 1 must be drawn when class_name is provided"
        );

        // 3. When student 2 is None, row 2 slots and label 2 remain white
        let sheet_single = compose_sheet(&assets, Some((&student_1, "01")), None, "12A");
        let empty_slot2_p =
            sheet_single.get_pixel(config::SLOT_COLS_X[0] + 10, config::SLOT_ROW2_Y + 10);
        assert_eq!(
            *empty_slot2_p,
            Rgb([255, 255, 255]),
            "Row 2 slot must remain white when student 2 is None"
        );

        let mut found_text_pixel_2 = false;
        for dy in 0..50 {
            for dx in 0..200 {
                let px = sheet_single.get_pixel(
                    config::LABEL_ROW2_POS.0 as u32 + dx,
                    config::LABEL_ROW2_POS.1 as u32 + dy,
                );
                if *px != Rgb([255, 255, 255]) {
                    found_text_pixel_2 = true;
                    break;
                }
            }
            if found_text_pixel_2 {
                break;
            }
        }
        assert!(
            !found_text_pixel_2,
            "Label text 2 must not be drawn when student 2 is None"
        );
    }
}
