//! 3x4 framing: pasfoto crop, chest logo detection, banner overlap check.

use image::{GrayImage, Luma, RgbImage, RgbaImage};
use imageproc::distance_transform::Norm;
use imageproc::morphology::{dilate, erode};
use imageproc::region_labelling::{Connectivity, connected_components};

use crate::config;

/// Parameters controlling 3x4 pasfoto framing.
#[derive(Debug, Clone, Copy)]
pub struct FramingParams {
    /// Target headroom above the crown as a fraction of crop height.
    pub headroom: f64,
    /// Target eye line position from the top as a fraction of crop height.
    pub eye_line: f64,
    /// Bottom fraction of the photo slot covered by the 4R template/overlay banner.
    pub banner_cover: f64,
}

impl Default for FramingParams {
    fn default() -> Self {
        Self {
            headroom: config::HEADROOM_DEFAULT,
            eye_line: config::EYE_LINE_DEFAULT,
            banner_cover: config::BANNER_COVER_DEFAULT,
        }
    }
}

/// Output of framing a single photo to 3x4 aspect ratio.
#[derive(Debug, Clone)]
pub struct Framed {
    /// Cropped 3x4 RGB image at original pixel resolution (no resizing).
    pub image: RgbImage,
    /// Reasons why this photo should be manually reviewed, if any.
    pub reasons: Vec<&'static str>,
}

/// Computes the bottom fraction (`0.0..=1.0`) of a slot covered by overlay/template banners.
// port of hitung_tutup_bawah()
pub fn banner_cover(layers: &[&RgbaImage], x: u32, y: u32, w: u32, h: u32) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }

    let mut covered = vec![false; h as usize];
    for layer in layers {
        let (lw, lh) = layer.dimensions();
        for (ry, row_cov) in covered.iter_mut().enumerate() {
            let py = y.saturating_add(ry as u32);
            if py >= lh {
                continue;
            }
            let mut opaque_count = 0u32;
            for dx in 0..w {
                let px = x.saturating_add(dx);
                if px < lw && layer.get_pixel(px, py).0[3] > 20 {
                    opaque_count += 1;
                }
            }
            if f64::from(opaque_count) / f64::from(w) > 0.5 {
                *row_cov = true;
            }
        }
    }

    let mut n = 0usize;
    while n < h as usize {
        let idx = h as usize - 1 - n;
        if covered.get(idx).copied().unwrap_or(false) {
            n += 1;
        } else {
            break;
        }
    }
    n as f32 / h as f32
}

/// Frames a photo to 3x4 aspect ratio based on eye coordinates, head silhouette, and chest logo.
///
/// Note: `frame_photo` only crops (with border replication when extending edges) and never resizes,
/// preserving exact source pixel values.
pub fn frame_photo(
    img: &RgbImage,
    eyes: Option<(f32, f32, f32)>,
    params: &FramingParams,
) -> Framed {
    match eyes {
        Some((ex, ey, iod)) if iod > config::MODEL_MIN_IOD => {
            crop_pasfoto_inner(img, ex, ey, iod, params)
        }
        // port of process_smart_crop() else branch
        _ => Framed {
            image: crop_center_fallback(img),
            reasons: vec!["wajah tidak terdeteksi"],
        },
    }
}

/// Center-crops `img` to `config::RATIO` (3:4) when no valid face is detected.
fn crop_center_fallback(img: &RgbImage) -> RgbImage {
    let (w_img, h_img) = img.dimensions();
    if w_img == 0 || h_img == 0 {
        return RgbImage::new(0, 0);
    }

    let mut crop_w = w_img;
    let mut crop_h = (f64::from(w_img) / config::RATIO) as u32;
    if crop_h > h_img {
        crop_h = h_img;
        crop_w = (f64::from(h_img) * config::RATIO) as u32;
    }
    crop_w = crop_w.max(1).min(w_img);
    crop_h = crop_h.max(1).min(h_img);

    let crop_x = (w_img - crop_w) / 2;
    let crop_y = (h_img - crop_h) / 2;

    let mut out = RgbImage::new(crop_w, crop_h);
    for dy in 0..crop_h {
        for dx in 0..crop_w {
            out.put_pixel(dx, dy, *img.get_pixel(crop_x + dx, crop_y + dy));
        }
    }
    out
}

/// Searches for a colored school emblem/logo on the student's left chest (right side of image).
// port of cari_logo()
fn find_logo(
    img: &RgbImage,
    cx: f64,
    ey: f32,
    iod: f32,
    latar_tepi: &[bool],
) -> Option<(u32, u32, u32, u32)> {
    let (w_img, h_img) = img.dimensions();
    let iod_f = f64::from(iod);
    let ey_f = f64::from(ey);

    let y1 = (ey_f + 3.0 * iod_f) as i32;
    let x1 = ((cx + 0.9 * iod_f) as i32).max(0);
    let x2 = (cx + 3.8 * iod_f).min(f64::from(w_img)) as i32;

    if y1 >= h_img as i32 || x1 >= x2 {
        return None;
    }
    let y1_u = y1.max(0) as u32;
    let x1_u = x1 as u32;
    let x2_u = (x2 as u32).min(w_img);

    let mut area = GrayImage::new(w_img, h_img);
    for y in y1_u..h_img {
        let row_offset = (y as usize) * (w_img as usize);
        for x in x1_u..x2_u {
            let idx = row_offset + (x as usize);
            if latar_tepi.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let [r, g, b] = img.get_pixel(x, y).0;
            let v = r.max(g).max(b);
            let min_rgb = r.min(g).min(b);
            let s = if v == 0 {
                0
            } else {
                (255 * u32::from(v - min_rgb) / u32::from(v)) as u8
            };
            if s > config::LOGO_S_MIN && v > config::LOGO_V_MIN {
                area.put_pixel(x, y, Luma([255u8]));
            }
        }
    }

    // MORPH_CLOSE with k x k box kernel = dilate then erode with Norm::LInf and radius k / 2
    let k = ((iod_f * 0.08) as u32).max(3);
    let radius = (k / 2).min(u32::from(u8::MAX)) as u8;
    let dilated = dilate(&area, Norm::LInf, radius);
    let closed = erode(&dilated, Norm::LInf, radius);

    let labels = connected_components(&closed, Connectivity::Eight, Luma([0u8]));
    let mut max_label = 0u32;
    for p in labels.pixels() {
        if p.0[0] > max_label {
            max_label = p.0[0];
        }
    }
    if max_label == 0 {
        return None;
    }

    let num_labels = (max_label as usize).saturating_add(1);
    let mut min_x = vec![u32::MAX; num_labels];
    let mut max_x = vec![0u32; num_labels];
    let mut min_y = vec![u32::MAX; num_labels];
    let mut max_y = vec![0u32; num_labels];
    let mut counts = vec![0u32; num_labels];

    for y in 0..h_img {
        for x in 0..w_img {
            let lab = labels.get_pixel(x, y).0[0] as usize;
            if lab == 0 {
                continue;
            }
            if let (Some(lx0), Some(lx1), Some(ly0), Some(ly1), Some(cnt)) = (
                min_x.get_mut(lab),
                max_x.get_mut(lab),
                min_y.get_mut(lab),
                max_y.get_mut(lab),
                counts.get_mut(lab),
            ) {
                *lx0 = (*lx0).min(x);
                *lx1 = (*lx1).max(x);
                *ly0 = (*ly0).min(y);
                *ly1 = (*ly1).max(y);
                *cnt += 1;
            }
        }
    }

    let min_area = 0.15 * iod_f * iod_f;
    let max_w = 2.0 * iod_f;
    let max_h = 2.2 * iod_f;

    let mut best: Option<(u32, u32, u32, u32, u32)> = None;
    for i in 1..num_labels {
        let a = counts.get(i).copied().unwrap_or(0);
        if a == 0 {
            continue;
        }
        let x0 = min_x.get(i).copied().unwrap_or(0);
        let x1 = max_x.get(i).copied().unwrap_or(0);
        let y0 = min_y.get(i).copied().unwrap_or(0);
        let y1 = max_y.get(i).copied().unwrap_or(0);
        let w = x1.saturating_sub(x0) + 1;
        let h = y1.saturating_sub(y0) + 1;

        if f64::from(a) < min_area || f64::from(w) > max_w || f64::from(h) > max_h {
            continue;
        }
        if best.is_none_or(|b| a > b.4) {
            best = Some((x0, y0, w, h, a));
        }
    }

    best.map(|(x, y, w, h, _)| (x, y, w, h))
}

/// Core 3x4 framing algorithm ported 1:1 from `crop_pasfoto()`.
fn crop_pasfoto_inner(
    img: &RgbImage,
    ex: f32,
    ey: f32,
    iod: f32,
    params: &FramingParams,
) -> Framed {
    let (w_img, h_img) = img.dimensions();
    if w_img == 0 || h_img == 0 {
        return Framed {
            image: RgbImage::new(0, 0),
            reasons: Vec::new(),
        };
    }

    let mut latar_mask = GrayImage::new(w_img, h_img);
    let mut fg_raw = GrayImage::new(w_img, h_img);

    for y in 0..h_img {
        for x in 0..w_img {
            let [r, g, b] = img.get_pixel(x, y).0;
            let (r, g, b) = (i16::from(r), i16::from(g), i16::from(b));
            let is_red = r > config::RED_BG_R_MIN
                && (r - g) > config::RED_BG_R_MINUS_G_MIN
                && (r - b) > config::RED_BG_R_MINUS_B_MIN;
            if is_red {
                latar_mask.put_pixel(x, y, Luma([255u8]));
            } else {
                fg_raw.put_pixel(x, y, Luma([255u8]));
            }
        }
    }

    // MORPH_OPEN with 9x9 box kernel = erode then dilate with Norm::LInf and radius 4
    let open_radius = config::MORPH_OPEN_RADIUS as u8;
    let fg_eroded = erode(&fg_raw, Norm::LInf, open_radius);
    let fg = dilate(&fg_eroded, Norm::LInf, open_radius);

    // 8-connected background components touching top, left, or right borders (bottom excluded)
    let lab = connected_components(&latar_mask, Connectivity::Eight, Luma([0u8]));
    let mut max_lab = 0u32;
    for p in lab.pixels() {
        if p.0[0] > max_lab {
            max_lab = p.0[0];
        }
    }
    let mut is_border_label = vec![false; (max_lab as usize).saturating_add(1)];
    for x in 0..w_img {
        let l_top = lab.get_pixel(x, 0).0[0] as usize;
        if l_top > 0
            && let Some(slot) = is_border_label.get_mut(l_top)
        {
            *slot = true;
        }
    }
    for y in 0..h_img {
        let l_left = lab.get_pixel(0, y).0[0] as usize;
        if l_left > 0
            && let Some(slot) = is_border_label.get_mut(l_left)
        {
            *slot = true;
        }
        let l_right = lab.get_pixel(w_img - 1, y).0[0] as usize;
        if l_right > 0
            && let Some(slot) = is_border_label.get_mut(l_right)
        {
            *slot = true;
        }
    }

    let mut latar_tepi = vec![false; (w_img as usize) * (h_img as usize)];
    for y in 0..h_img {
        let row_offset = (y as usize) * (w_img as usize);
        for x in 0..w_img {
            let l = lab.get_pixel(x, y).0[0] as usize;
            if is_border_label.get(l).copied().unwrap_or(false)
                && let Some(slot) = latar_tepi.get_mut(row_offset + (x as usize))
            {
                *slot = true;
            }
        }
    }

    let mut reasons: Vec<&'static str> = Vec::new();
    let ex_f = f64::from(ex);
    let ey_f = f64::from(ey);
    let iod_f = f64::from(iod);

    // port of crop_pasfoto() step 1
    let x1 = ((ex_f - 0.6 * iod_f) as i32).max(0) as u32;
    let x2 = ((ex_f + 0.6 * iod_f) as i32).clamp(0, w_img as i32) as u32;
    let mut pita = vec![false; h_img as usize];
    if x1 < x2 {
        for (y, has_fg) in pita.iter_mut().enumerate() {
            for x in x1..x2 {
                if fg.get_pixel(x, y as u32).0[0] != 0 {
                    *has_fg = true;
                    break;
                }
            }
        }
    }
    let mut crown = (ey as i32).clamp(0, h_img as i32) as u32;
    while crown > 0 && pita.get((crown - 1) as usize).copied().unwrap_or(false) {
        crown -= 1;
    }
    if crown == 0 {
        reasons.push("kepala menyentuh tepi atas foto mentah");
    }

    // port of crop_pasfoto() step 2
    let mut pusat: Vec<f64> = Vec::with_capacity(config::HEAD_SILHOUETTE_SAMPLES);
    let y_start = f64::from(crown) + 0.25 * (ey_f - f64::from(crown));
    let y_end = ey_f;
    let ex_u = (ex as i32).clamp(0, w_img as i32 - 1) as u32;

    for i in 0..config::HEAD_SILHOUETTE_SAMPLES {
        let t = if config::HEAD_SILHOUETTE_SAMPLES > 1 {
            i as f64 / (config::HEAD_SILHOUETTE_SAMPLES - 1) as f64
        } else {
            0.0
        };
        let y_sample = ((y_start + t * (y_end - y_start)) as i32).clamp(0, h_img as i32 - 1) as u32;
        if fg.get_pixel(ex_u, y_sample).0[0] == 0 {
            continue;
        }
        let mut l = ex_u;
        while l > 0 && fg.get_pixel(l - 1, y_sample).0[0] != 0 {
            l -= 1;
        }
        let mut r = ex_u;
        while r + 1 < w_img && fg.get_pixel(r + 1, y_sample).0[0] != 0 {
            r += 1;
        }
        // Only count rows bounded by red background inside the image (PRD 3.4 silhouette rule)
        if l > 0 && r + 1 < w_img {
            pusat.push(f64::from(l + r) / 2.0);
        }
    }

    let cx = if pusat.len() >= config::MIN_HEAD_POINTS {
        median(&mut pusat)
    } else {
        reasons.push("latar bukan merah, posisi pakai tengah wajah");
        ex_f
    };

    // port of crop_pasfoto() step 3
    let headroom = params.headroom;
    let eye_line = params.eye_line;
    let h_mata = iod_f / config::IOD_FRAC / config::RATIO;
    let ideal = ((ey_f - f64::from(crown)) / (eye_line - headroom).max(0.05))
        .clamp(0.8 * h_mata, 1.25 * h_mata);
    let mut h = (ey_f + (1.0 - eye_line) * ideal - f64::from(crown)) / (1.0 - headroom);

    // port of crop_pasfoto() step 4
    let tutup_bawah = params.banner_cover;
    let logo = find_logo(img, cx, ey, iod, &latar_tepi);
    let (bawah_logo, logo_terpotong_dari_raw) = if let Some((_, ly, _, lh)) = logo {
        let bottom = ly.saturating_add(lh);
        (f64::from(bottom), bottom >= h_img.saturating_sub(2))
    } else {
        reasons.push("logo di dada tidak terdeteksi");
        (ey_f + 6.7 * iod_f, false)
    };

    let h_logo = (bawah_logo + config::LOGO_MARGIN * iod_f - f64::from(crown))
        / (1.0 - headroom - tutup_bawah).max(0.5);
    h = h.max(h_logo);

    let h_maks = (f64::from(h_img) - f64::from(crown)) / (1.0 - headroom);
    h = h.min(h_maks);
    let y0_f = f64::from(crown) - headroom * h;

    if let Some((_, _, _, lh)) = logo
        && !logo_terpotong_dari_raw
    {
        let tertutup = bawah_logo - (y0_f + (1.0 - tutup_bawah) * h);
        if tertutup > config::BANNER_COVER_THRESHOLD * f64::from(lh) {
            reasons.push("logo tertutup banner, foto mentah terlalu mepet di bawah");
        }
    }
    if h < ideal * config::ZOOM_MIN_FACTOR || h > ideal * config::ZOOM_MAX_FACTOR {
        reasons.push("ukuran wajah jauh dari ideal");
    }

    // port of crop_pasfoto() step 5
    let w_f = h * config::RATIO;
    let x0 = (cx - w_f / 2.0).round() as i32;
    let y0 = y0_f.round() as i32;
    let w = w_f.round().max(1.0) as i32;
    let h_i = h.round().max(1.0) as i32;

    let pt = (-y0).max(0);
    let pl = (-x0).max(0);
    let pr = (x0 + w - w_img as i32).max(0);
    let _pb = (y0 + h_i - h_img as i32).max(0);

    if f64::from(pt.max(pl).max(pr)) > config::MAX_EXTENSION_RATIO * f64::from(h_i) {
        reasons.push("foto mentah terlalu mepet, latar diperpanjang");
    }

    let cropped = crop_replicate(img, x0, y0, w as u32, h_i as u32);
    Framed {
        image: cropped,
        reasons,
    }
}

/// Computes the median of a non-empty slice of `f64` values (matching `np.median`).
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    let mid = n / 2;
    if n % 2 == 1 {
        values.get(mid).copied().unwrap_or(0.0)
    } else {
        let a = values.get(mid.saturating_sub(1)).copied().unwrap_or(0.0);
        let b = values.get(mid).copied().unwrap_or(0.0);
        (a + b) / 2.0
    }
}

/// Crops `(x0, y0, w, h)` from `img` using border replication (`cv2.BORDER_REPLICATE`).
fn crop_replicate(img: &RgbImage, x0: i32, y0: i32, w: u32, h: u32) -> RgbImage {
    let (w_img, h_img) = img.dimensions();
    let max_x = w_img as i32 - 1;
    let max_y = h_img as i32 - 1;
    let mut out = RgbImage::new(w, h);

    for dy in 0..h {
        let sy = (y0 + dy as i32).clamp(0, max_y) as u32;
        for dx in 0..w {
            let sx = (x0 + dx as i32).clamp(0, max_x) as u32;
            out.put_pixel(dx, dy, *img.get_pixel(sx, sy));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, Rgba};

    #[test]
    fn test_frame_photo_synthetic_red_bg() {
        let (w, h) = (1200u32, 1600u32);
        let red_bg = Rgb([190u8, 15u8, 30u8]);
        let head_color = Rgb([30u8, 30u8, 30u8]);
        let body_color = Rgb([240u8, 240u8, 240u8]);
        let logo_color = Rgb([30u8, 60u8, 200u8]);

        let mut img = RgbImage::from_pixel(w, h, red_bg);

        // White body from y = 800 downward
        for y in 800..h {
            for x in 360..760 {
                img.put_pixel(x, y, body_color);
            }
        }

        // Dark head ellipse centered at x = 560, y = 600 (rx = 90, ry = 120 -> crown at y = 480)
        for y in 480..=720 {
            for x in 470..=650 {
                let dx = (f64::from(x) - 560.0) / 90.0;
                let dy = (f64::from(y) - 600.0) / 120.0;
                if dx * dx + dy * dy <= 1.0 {
                    img.put_pixel(x, y, head_color);
                }
            }
        }

        // Blue chest logo 50x60 at x = 650, y = 1050
        for y in 1050..(1050 + 60) {
            for x in 650..(650 + 50) {
                img.put_pixel(x, y, logo_color);
            }
        }

        // Eyes at (530, 600) and (590, 600) -> ex = 560.0, ey = 600.0, iod = 60.0
        let params = FramingParams {
            headroom: config::HEADROOM_DEFAULT,
            eye_line: config::EYE_LINE_DEFAULT,
            banner_cover: 0.10,
        };
        let framed = frame_photo(&img, Some((560.0, 600.0, 60.0)), &params);

        let (w_crop, h_crop) = framed.image.dimensions();
        assert_eq!((w_crop, h_crop), (577, 769));

        // Scan head pixels to measure horizontal center and top crown in cropped output
        let mut head_min_x = u32::MAX;
        let mut head_max_x = 0u32;
        let mut head_min_y = u32::MAX;
        let mut logo_max_y = 0u32;

        for y in 0..h_crop {
            for x in 0..w_crop {
                let px = *framed.image.get_pixel(x, y);
                if px == head_color {
                    head_min_x = head_min_x.min(x);
                    head_max_x = head_max_x.max(x);
                    head_min_y = head_min_y.min(y);
                } else if px == logo_color {
                    logo_max_y = logo_max_y.max(y);
                }
            }
        }

        let center_x_crop = f64::from(head_min_x + head_max_x) / 2.0;
        let target_center_x = f64::from(w_crop) / 2.0;
        let crown_y_crop = f64::from(head_min_y);
        let crown_ratio = crown_y_crop / f64::from(h_crop);
        let logo_bottom_y_crop = f64::from(logo_max_y + 1);
        let banner_line_y = (1.0 - params.banner_cover) * f64::from(h_crop);

        // Exact deterministic measurements on the synthetic image
        assert_eq!(center_x_crop, 288.0);
        assert_eq!(crown_y_crop, 53.0);
        assert_eq!(logo_bottom_y_crop, 683.0);

        // PRD tolerance checks:
        // 1) Horizontal center within ±1% of crop width
        assert!(
            (center_x_crop - target_center_x).abs() <= 0.01 * f64::from(w_crop),
            "center_x_crop={center_x_crop}, target={target_center_x}, w_crop={w_crop}"
        );
        // 2) Crown at 7% ± 0.5% of crop height
        assert!(
            (crown_ratio - 0.07).abs() <= 0.005,
            "crown_ratio={crown_ratio}"
        );
        // 3) Logo bottom strictly above the banner line
        assert!(
            logo_bottom_y_crop < banner_line_y,
            "logo_bottom_y_crop={logo_bottom_y_crop}, banner_line_y={banner_line_y}"
        );
        // 4) No manual review reasons
        assert!(
            framed.reasons.is_empty(),
            "expected empty reasons, got: {:?}",
            framed.reasons
        );
    }

    #[test]
    fn test_frame_photo_white_bg_flags_non_red() {
        let img = RgbImage::from_pixel(1200, 1600, Rgb([255u8, 255u8, 255u8]));
        let params = FramingParams::default();
        let framed = frame_photo(&img, Some((560.0, 600.0, 60.0)), &params);

        assert!(
            framed
                .reasons
                .contains(&"latar bukan merah, posisi pakai tengah wajah"),
            "reasons did not contain non-red background warning: {:?}",
            framed.reasons
        );
    }

    #[test]
    fn test_frame_photo_no_face_fallback() {
        let img = RgbImage::from_pixel(1200, 1600, Rgb([190u8, 15u8, 30u8]));
        let params = FramingParams::default();
        let framed = frame_photo(&img, None, &params);

        assert_eq!(framed.image.dimensions(), (1200, 1600));
        assert_eq!(framed.reasons, vec!["wajah tidak terdeteksi"]);
    }

    #[test]
    fn test_banner_cover() {
        let mut layer1 = RgbaImage::from_pixel(100, 200, Rgba([0, 0, 0, 0]));
        let mut layer2 = RgbaImage::from_pixel(100, 200, Rgba([0, 0, 0, 0]));

        // Slot at (x=10, y=20, w=50, h=100) -> rows y = 20..120
        // Layer 1 covers bottom 10 rows (y = 110..120) across 30/50 columns (60% > 50%)
        for y in 110..120 {
            for x in 10..40 {
                layer1.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        // Layer 2 covers 5 rows above that (y = 105..110) across 50/50 columns (100% > 50%)
        for y in 105..110 {
            for x in 10..60 {
                layer2.put_pixel(x, y, Rgba([0, 0, 0, 100]));
            }
        }
        // Row y = 104 only has 20/50 columns opaque (40% <= 50%, not counted as covered)
        for x in 10..30 {
            layer2.put_pixel(x, 104, Rgba([0, 0, 0, 255]));
        }

        let cov = banner_cover(&[&layer1, &layer2], 10, 20, 50, 100);
        assert!((cov - 0.15).abs() < 1e-6, "expected 0.15, got {cov}");
    }
}
