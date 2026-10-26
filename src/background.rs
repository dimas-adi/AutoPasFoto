//! Background replacement using chroma keying.

use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::distance_transform::Norm;
use imageproc::filter::gaussian_blur_f32;
use imageproc::morphology::dilate;
use imageproc::region_labelling::{Connectivity, connected_components};

use crate::config;

/// Replaces studio red background with a solid target color using chroma keying.
///
/// Follows PRD section 3.5 steps:
/// 1. Ratio `q = max(g, b) / max(r, 1)`.
/// 2. Hard background mask with 8-connected components touching top, left, or right edges.
/// 3. Edge transition band by dilating hard background with `Norm::LInf`.
/// 4. Alpha ramp with hard background priority, followed by Gaussian blur on alpha map.
/// 5. Despill on original red channel using blurred alpha.
/// 6. Composite despilled foreground over target color.
pub fn replace_background(img: &RgbImage, color: [u8; 3]) -> RgbImage {
    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return img.clone();
    }

    let total_pixels = (width as usize).saturating_mul(height as usize);
    let mut q_map = Vec::with_capacity(total_pixels);
    let mut hard_raw = GrayImage::new(width, height);

    // Step 1 & candidate filtering for Step 2
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y).0;
            let r_f = pixel[0] as f32;
            let g_f = pixel[1] as f32;
            let b_f = pixel[2] as f32;

            let max_gb = g_f.max(b_f);
            let max_r = r_f.max(1.0);
            let q = max_gb / max_r;
            q_map.push(q);

            if r_f > config::CHROMA_R_MIN && q < config::CHROMA_Q_MAX_HARD {
                hard_raw.put_pixel(x, y, Luma([255u8]));
            }
        }
    }

    // Step 2: 8-connected components touching top, left, or right edges (bottom excluded)
    let labels = connected_components(&hard_raw, Connectivity::Eight, Luma([0u8]));
    let mut max_label = 0u32;
    for p in labels.pixels() {
        if p.0[0] > max_label {
            max_label = p.0[0];
        }
    }

    let mut is_border_label = vec![false; (max_label as usize).saturating_add(1)];
    for x in 0..width {
        let l_top = labels.get_pixel(x, 0).0[0] as usize;
        if l_top > 0
            && let Some(slot) = is_border_label.get_mut(l_top)
        {
            *slot = true;
        }
    }
    for y in 0..height {
        let l_left = labels.get_pixel(0, y).0[0] as usize;
        if l_left > 0
            && let Some(slot) = is_border_label.get_mut(l_left)
        {
            *slot = true;
        }
        let l_right = labels.get_pixel(width - 1, y).0[0] as usize;
        if l_right > 0
            && let Some(slot) = is_border_label.get_mut(l_right)
        {
            *slot = true;
        }
    }

    let mut latar_keras = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let label = labels.get_pixel(x, y).0[0] as usize;
            if label > 0 && is_border_label.get(label).copied().unwrap_or(false) {
                latar_keras.put_pixel(x, y, Luma([255u8]));
            }
        }
    }

    // Step 3: Edge band dilation with Norm::LInf
    let h_f = height as f32;
    let edge_radius = ((h_f * config::CHROMA_EDGE_RADIUS_FACTOR).round() as u32)
        .max(config::CHROMA_EDGE_MIN_RADIUS);
    let edge_radius_u8 = edge_radius.min(255) as u8;
    let dilated = dilate(&latar_keras, Norm::LInf, edge_radius_u8);

    // Step 4: Alpha map (hard background overrides edge band) and Gaussian blur
    let mut alpha_map = ImageBuffer::<Luma<f32>, Vec<f32>>::new(width, height);
    for y in 0..height {
        let row_offset = (y as usize) * (width as usize);
        for x in 0..width {
            let idx = row_offset + (x as usize);
            let q_val = q_map.get(idx).copied().unwrap_or(1.0);

            let a = if latar_keras.get_pixel(x, y).0[0] == 255 {
                1.0f32
            } else if dilated.get_pixel(x, y).0[0] == 255 {
                ((config::CHROMA_ALPHA_UPPER_Q - q_val) / config::CHROMA_ALPHA_RANGE_Q)
                    .clamp(0.0, 1.0)
            } else {
                0.0f32
            };
            alpha_map.put_pixel(x, y, Luma([a]));
        }
    }

    let blur_sigma = (h_f * config::CHROMA_BLUR_SIGMA_FACTOR).max(config::CHROMA_BLUR_SIGMA_MIN);
    let blurred_alpha = gaussian_blur_f32(&alpha_map, blur_sigma);

    // Step 5 & Step 6: Despill and Composite
    let mut out = RgbImage::new(width, height);
    let target_r = color[0] as f32;
    let target_g = color[1] as f32;
    let target_b = color[2] as f32;

    for y in 0..height {
        for x in 0..width {
            let orig = img.get_pixel(x, y).0;
            let r_f = orig[0] as f32;
            let g_f = orig[1] as f32;
            let b_f = orig[2] as f32;

            let alpha = blurred_alpha.get_pixel(x, y).0[0].clamp(0.0, 1.0);

            // Step 5: despill on original red channel using blurred alpha
            let excess_r = (r_f - g_f.max(b_f)).max(0.0);
            let r_despill = r_f - excess_r * alpha;
            let g_despill = g_f;
            let b_despill = b_f;

            // Step 6: composite using despilled foreground and target color
            let out_r = (r_despill * (1.0 - alpha) + target_r * alpha)
                .round()
                .clamp(0.0, 255.0) as u8;
            let out_g = (g_despill * (1.0 - alpha) + target_g * alpha)
                .round()
                .clamp(0.0, 255.0) as u8;
            let out_b = (b_despill * (1.0 - alpha) + target_b * alpha)
                .round()
                .clamp(0.0, 255.0) as u8;

            out.put_pixel(x, y, Rgb([out_r, out_g, out_b]));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn test_replace_background_synthetic_blue() {
        let width = 100u32;
        let height = 100u32;
        let bg_red = [190u8, 15, 30];
        let target_blue = [31u8, 79, 191];
        let skin_color = [220u8, 150, 130];
        let white_shirt = [255u8, 255, 255];
        let dark_red_tie = [140u8, 20, 30];

        let mut img = RgbImage::new(width, height);
        // Base canvas: red studio background
        for p in img.pixels_mut() {
            *p = Rgb(bg_red);
        }

        // Skin box: x in [35, 64], y in [20, 49]
        for y in 20..=49 {
            for x in 35..=64 {
                img.put_pixel(x, y, Rgb(skin_color));
            }
        }

        // White shirt separator: x in [25, 74], y in [50, 99]
        for y in 50..=99 {
            for x in 25..=74 {
                img.put_pixel(x, y, Rgb(white_shirt));
            }
        }

        // Dark red tie touching bottom: x in [45, 54], y in [60, 99]
        for y in 60..=99 {
            for x in 45..=54 {
                img.put_pixel(x, y, Rgb(dark_red_tie));
            }
        }

        let out = replace_background(&img, target_blue);

        // 1. Deep background x in [0, 20], y in [0, 20] must match target color exactly
        let mut mismatched_bg_count = 0u32;
        for y in 0..=20 {
            for x in 0..=20 {
                if out.get_pixel(x, y).0 != target_blue {
                    mismatched_bg_count += 1;
                }
            }
        }
        assert_eq!(mismatched_bg_count, 0);

        // 2. Dark red tie x in [45, 54], y in [60, 99] must remain exactly (140, 20, 30)
        for y in 60..=99 {
            for x in 45..=54 {
                assert_eq!(
                    out.get_pixel(x, y).0,
                    dark_red_tie,
                    "Tie pixel at ({x}, {y}) was altered",
                );
            }
        }

        // 3. Skin interior x in [37, 62], y in [22, 47] must not deviate by > 5 per channel
        let mut max_skin_diff = 0u8;
        for y in 22..=47 {
            for x in 37..=62 {
                let pixel = out.get_pixel(x, y).0;
                for c in 0..3 {
                    let diff = pixel[c].abs_diff(skin_color[c]);
                    if diff > max_skin_diff {
                        max_skin_diff = diff;
                    }
                    assert!(
                        diff <= 5,
                        "Skin channel {c} at ({x}, {y}) changed by {diff} > 5 (got {pixel:?})",
                    );
                }
            }
        }

        assert!(max_skin_diff <= 5);
    }

    #[test]
    fn test_replace_background_synthetic_shaded_red() {
        let width = 100u32;
        let height = 100u32;
        let target_default = config::DEFAULT_BG_COLOR_RGB;

        let mut img = RgbImage::new(width, height);
        // Shaded red background: r varying from 90 to 230 across x
        for y in 0..height {
            for x in 0..width {
                let r = (90.0 + (140.0 * (x as f32) / 99.0).round()) as u8;
                let g = 15u8;
                let b = 25u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        let out = replace_background(&img, target_default);

        // All pixels are hard background connected to edges, so shadows disappear entirely
        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    out.get_pixel(x, y).0,
                    target_default,
                    "Shaded background at ({x}, {y}) did not match default target",
                );
            }
        }
    }
}
