//! YuNet face detection via tract: preprocessing, inference, and output decoding.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::RgbImage;
use tract_onnx::prelude::*;

use crate::config;

type RunnableModel = Arc<TypedRunnableModel>;

/// Detected face in original image coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Face {
    /// Detection confidence score in `0.0..=1.0`.
    pub score: f32,
    /// Bounding box `(x, y, w, h)` in original image coordinates.
    pub bbox: (f32, f32, f32, f32),
    /// Five facial landmarks `(x, y)` in original image coordinates:
    /// 0 = right eye (left in image), 1 = left eye, 2 = nose, 3 = right mouth, 4 = left mouth.
    pub landmarks: [(f32, f32); 5],
}

impl Face {
    /// Returns eye midpoint `(ex, ey)` and interocular distance `iod` in original image coordinates.
    pub fn eyes(&self) -> (f32, f32, f32) {
        let [(x0, y0), (x1, y1), _, _, _] = self.landmarks;
        let ex = 0.5 * (x0 + x1);
        let ey = 0.5 * (y0 + y1);
        let iod = (x0 - x1).abs();
        (ex, ey, iod)
    }
}

/// Output tensor indices for a single feature pyramid stride.
#[derive(Debug, Clone, Copy)]
struct StrideOutputIndices {
    stride: usize,
    cls: usize,
    obj: usize,
    bbox: usize,
    kps: usize,
}

/// Raw tensor slices for a single feature pyramid stride.
pub(crate) struct StrideTensors<'a> {
    pub stride: usize,
    pub cls: &'a [f32],
    pub obj: &'a [f32],
    pub bbox: &'a [f32],
    pub kps: &'a [f32],
}

/// Thread-safe YuNet face detector backed by an optimized `tract` runnable plan.
pub struct FaceDetector {
    model: RunnableModel,
    outputs: [StrideOutputIndices; 3],
}

const fn _assert_send_sync<T: Send + Sync>() {}
const _: () = _assert_send_sync::<FaceDetector>();

impl FaceDetector {
    /// Loads and optimizes the YuNet ONNX model with fixed `[1, 3, 640, 640]` `f32` input.
    pub fn load(model_path: &Path) -> Result<Self> {
        let side = config::MODEL_INPUT_SIZE;
        let mut model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| {
                format!(
                    "Gagal memuat file model deteksi wajah: {}",
                    model_path.display()
                )
            })?;

        model
            .set_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec![1, 3, side, side]),
            )
            .context("Gagal mengatur ukuran input model deteksi wajah")?;

        let optimized = model
            .into_typed()
            .context("Gagal memeriksa tipe graf model deteksi wajah")?
            .into_optimized()
            .context("Gagal mengoptimasi model deteksi wajah")?;

        let outputs = map_output_indices(&optimized)?;
        let runnable = optimized
            .into_runnable()
            .context("Gagal menyiapkan eksekusi model deteksi wajah")?;

        Ok(Self {
            model: runnable,
            outputs,
        })
    }

    /// Runs face detection on an RGB image and returns the highest-scoring valid face.
    pub fn detect(&self, img: &RgbImage) -> Result<Option<Face>> {
        let w = img.width();
        let h = img.height();
        if w == 0 || h == 0 {
            return Ok(None);
        }

        let (input_tensor, scale) = preprocess(img)?;
        let raw_outputs = self
            .model
            .run(tvec![input_tensor.into()])
            .context("Gagal menjalankan deteksi wajah pada foto")?;

        let mut stride_slices = Vec::with_capacity(self.outputs.len());
        for idx_group in &self.outputs {
            let cls = extract_f32_slice(&raw_outputs, idx_group.cls, "cls")?;
            let obj = extract_f32_slice(&raw_outputs, idx_group.obj, "obj")?;
            let bbox = extract_f32_slice(&raw_outputs, idx_group.bbox, "bbox")?;
            let kps = extract_f32_slice(&raw_outputs, idx_group.kps, "kps")?;
            stride_slices.push(StrideTensors {
                stride: idx_group.stride,
                cls,
                obj,
                bbox,
                kps,
            });
        }

        decode_strides(&stride_slices, scale)
    }
}

/// Resolves the 12 YuNet output tensor indices by name from the optimized model.
fn map_output_indices(optimized: &TypedModel) -> Result<[StrideOutputIndices; 3]> {
    let side = config::MODEL_INPUT_SIZE;
    let [s0, s1, s2] = config::MODEL_STRIDES;
    Ok([
        resolve_stride_outputs(optimized, side, s0)?,
        resolve_stride_outputs(optimized, side, s1)?,
        resolve_stride_outputs(optimized, side, s2)?,
    ])
}

/// Finds and validates `cls_{st}`, `obj_{st}`, `bbox_{st}`, and `kps_{st}` on the optimized model.
fn resolve_stride_outputs(
    optimized: &TypedModel,
    side: usize,
    stride: usize,
) -> Result<StrideOutputIndices> {
    let n = (side / stride) * (side / stride);
    let cls_name = format!("cls_{stride}");
    let obj_name = format!("obj_{stride}");
    let bbox_name = format!("bbox_{stride}");
    let kps_name = format!("kps_{stride}");

    let cls = find_output_by_name_and_shape(optimized, &cls_name, [1, n, 1])?;
    let obj = find_output_by_name_and_shape(optimized, &obj_name, [1, n, 1])?;
    let bbox = find_output_by_name_and_shape(optimized, &bbox_name, [1, n, 4])?;
    let kps = find_output_by_name_and_shape(optimized, &kps_name, [1, n, 10])?;

    Ok(StrideOutputIndices {
        stride,
        cls,
        obj,
        bbox,
        kps,
    })
}

/// Looks up an output index by its label or node name on the optimized model and checks its shape.
fn find_output_by_name_and_shape(
    optimized: &TypedModel,
    target_name: &str,
    expected_shape: [usize; 3],
) -> Result<usize> {
    let mut available = Vec::with_capacity(optimized.outputs.len());
    for (idx, &outlet) in optimized.outputs.iter().enumerate() {
        let label = optimized.outlet_label(outlet);
        let node_name = optimized.node(outlet.node).name.as_str();
        let effective_name = label.unwrap_or(node_name);
        available.push(format!("{idx}:{effective_name}"));

        if label == Some(target_name) || node_name == target_name {
            let fact = optimized
                .output_fact(idx)
                .with_context(|| format!("Gagal membaca metadata output {target_name}"))?;
            let dims: Vec<usize> = fact
                .shape
                .as_concrete()
                .with_context(|| format!("Bentuk tensor output {target_name} tidak konkret"))?
                .to_vec();
            if dims.as_slice() != expected_shape {
                bail!(
                    "Bentuk tensor output {target_name} tidak sesuai: diharapkan {:?}, ditemukan {:?}",
                    expected_shape,
                    dims
                );
            }
            return Ok(idx);
        }
    }

    bail!(
        "Output model '{target_name}' tidak ditemukan. Output tersedia: {}",
        available.join(", ")
    )
}

/// Extracts an `f32` slice from a `tract` output vector at `index`.
fn extract_f32_slice<'a>(
    outputs: &'a TVec<TValue>,
    index: usize,
    label: &str,
) -> Result<&'a [f32]> {
    let val = outputs
        .get(index)
        .with_context(|| format!("Indeks output model {label} ({index}) di luar batas"))?;
    let view = val
        .to_plain_array_view::<f32>()
        .with_context(|| format!("Gagal membaca tensor output {label} sebagai f32"))?;
    view.to_slice()
        .with_context(|| format!("Tensor output {label} tidak kontigu di memori"))
}

/// Resizes the image so its longest side is 640 and places it at the top-left of a BGR `[1,3,640,640]` tensor.
fn preprocess(img: &RgbImage) -> Result<(Tensor, f32)> {
    let w = img.width();
    let h = img.height();
    let side = config::MODEL_INPUT_SIZE;
    let max_dim = w.max(h) as f32;
    let scale = (side as f32) / max_dim;

    let new_w = ((w as f32) * scale).round().clamp(1.0, side as f32) as u32;
    let new_h = ((h as f32) * scale).round().clamp(1.0, side as f32) as u32;

    let mut dst_img = Image::new(new_w, new_h, PixelType::U8x3);
    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Box));
    resizer
        .resize(img, &mut dst_img, &options)
        .context("Gagal mengubah ukuran foto untuk deteksi wajah")?;

    let plane_len = side * side;
    let mut nchw = vec![0.0_f32; 3 * plane_len];
    let (b_plane, rest) = nchw.split_at_mut(plane_len);
    let (g_plane, r_plane) = rest.split_at_mut(plane_len);

    let new_w_usize = new_w as usize;
    for (idx, &[r, g, b]) in dst_img.buffer().as_chunks::<3>().0.iter().enumerate() {
        let y = idx / new_w_usize;
        let x = idx % new_w_usize;
        let dst_idx = y * side + x;
        if let (Some(bp), Some(gp), Some(rp)) = (
            b_plane.get_mut(dst_idx),
            g_plane.get_mut(dst_idx),
            r_plane.get_mut(dst_idx),
        ) {
            *bp = f32::from(b);
            *gp = f32::from(g);
            *rp = f32::from(r);
        }
    }

    let tensor = tract_ndarray::Array4::from_shape_vec((1, 3, side, side), nchw)
        .context("Gagal membentuk tensor input deteksi wajah")?
        .into_tensor();

    Ok((tensor, scale))
}

/// Decodes the 12 YuNet stride outputs, converts coordinates to original image space, and checks IOD.
pub(crate) fn decode_strides(strides: &[StrideTensors<'_>], scale: f32) -> Result<Option<Face>> {
    if scale <= 0.0 || !scale.is_finite() {
        return Ok(None);
    }

    let side = config::MODEL_INPUT_SIZE;
    let inv_s = 1.0 / scale;
    let mut best: Option<Face> = None;

    for head in strides {
        let st = head.stride;
        if st == 0 {
            bail!("Stride model tidak boleh nol");
        }
        let cols = side / st;
        let n = cols * cols;

        if head.cls.len() != n
            || head.obj.len() != n
            || head.bbox.len() != n * 4
            || head.kps.len() != n * 10
        {
            bail!("Ukuran tensor output model untuk stride {st} tidak sesuai");
        }

        let st_f = st as f32;
        for (
            i,
            (((&cls_val, &obj_val), &[b0, b1, b2, b3]), &[k0, k1, k2, k3, k4, k5, k6, k7, k8, k9]),
        ) in head
            .cls
            .iter()
            .zip(head.obj.iter())
            .zip(head.bbox.as_chunks::<4>().0.iter())
            .zip(head.kps.as_chunks::<10>().0.iter())
            .enumerate()
        {
            let score = (cls_val.clamp(0.0, 1.0) * obj_val.clamp(0.0, 1.0)).sqrt();
            if score < config::MODEL_SCORE_THRESHOLD {
                continue;
            }
            if let Some(ref current_best) = best
                && score <= current_best.score
            {
                continue;
            }

            let row = (i / cols) as f32;
            let col = (i % cols) as f32;

            let cx = (col + b0) * st_f;
            let cy = (row + b1) * st_f;
            let w = b2.exp() * st_f;
            let h = b3.exp() * st_f;
            let x = cx - 0.5 * w;
            let y = cy - 0.5 * h;

            // Scale coordinates back to original photo space before computing IOD.
            let bbox = (x * inv_s, y * inv_s, w * inv_s, h * inv_s);
            let landmarks = [
                ((col + k0) * st_f * inv_s, (row + k1) * st_f * inv_s),
                ((col + k2) * st_f * inv_s, (row + k3) * st_f * inv_s),
                ((col + k4) * st_f * inv_s, (row + k5) * st_f * inv_s),
                ((col + k6) * st_f * inv_s, (row + k7) * st_f * inv_s),
                ((col + k8) * st_f * inv_s, (row + k9) * st_f * inv_s),
            ];

            best = Some(Face {
                score,
                bbox,
                landmarks,
            });
        }
    }

    if let Some(face) = best {
        let (_, _, iod) = face.eyes();
        if iod <= config::MODEL_MIN_IOD {
            return Ok(None);
        }
        return Ok(Some(face));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::imageio;

    struct SyntheticBuffers {
        stride: usize,
        cls: Vec<f32>,
        obj: Vec<f32>,
        bbox: Vec<f32>,
        kps: Vec<f32>,
    }

    impl SyntheticBuffers {
        fn new(stride: usize) -> Self {
            let cols = config::MODEL_INPUT_SIZE / stride;
            let n = cols * cols;
            Self {
                stride,
                cls: vec![0.0; n],
                obj: vec![0.0; n],
                bbox: vec![0.0; n * 4],
                kps: vec![0.0; n * 10],
            }
        }

        fn as_tensors(&self) -> StrideTensors<'_> {
            StrideTensors {
                stride: self.stride,
                cls: &self.cls,
                obj: &self.obj,
                bbox: &self.bbox,
                kps: &self.kps,
            }
        }
    }

    #[test]
    fn test_decode_synthetic() {
        let [s0, s1, s2] = config::MODEL_STRIDES;
        let mut b8 = SyntheticBuffers::new(s0);
        let b16 = SyntheticBuffers::new(s1);
        let b32 = SyntheticBuffers::new(s2);

        let cols8 = config::MODEL_INPUT_SIZE / s0;
        let row = 10_usize;
        let col = 20_usize;
        let idx = row * cols8 + col;

        // Score = sqrt(0.81 * 0.64) = 0.72
        b8.cls[idx] = 0.81;
        b8.obj[idx] = 0.64;

        // Box offsets at (col=20, row=10), stride=8:
        // cx = (20 + 0.5) * 8 = 164.0, cy = (10 + 0.25) * 8 = 82.0
        // w = exp(ln(10)) * 8 = 80.0, h = exp(ln(12)) * 8 = 96.0
        // x = 164 - 40 = 124.0, y = 82 - 48 = 34.0
        b8.bbox[idx * 4] = 0.5;
        b8.bbox[idx * 4 + 1] = 0.25;
        b8.bbox[idx * 4 + 2] = 10.0_f32.ln();
        b8.bbox[idx * 4 + 3] = 12.0_f32.ln();

        // Eye0 at col - 2.0 => x = 18 * 8 = 144.0, y = 10 * 8 = 80.0
        // Eye1 at col + 2.0 => x = 22 * 8 = 176.0, y = 10 * 8 = 80.0
        // In 640-space IOD = 32.0; with scale = 0.5, original IOD = 64.0 (> 10.0).
        b8.kps[idx * 10] = -2.0;
        b8.kps[idx * 10 + 1] = 0.0;
        b8.kps[idx * 10 + 2] = 2.0;
        b8.kps[idx * 10 + 3] = 0.0;
        b8.kps[idx * 10 + 4] = 0.0;
        b8.kps[idx * 10 + 5] = 1.0;
        b8.kps[idx * 10 + 6] = -1.5;
        b8.kps[idx * 10 + 7] = 2.5;
        b8.kps[idx * 10 + 8] = 1.5;
        b8.kps[idx * 10 + 9] = 2.5;

        let scale = 0.5_f32;
        let slices = [b8.as_tensors(), b16.as_tensors(), b32.as_tensors()];
        let face = decode_strides(&slices, scale)
            .expect("decode should succeed")
            .expect("face should be detected");

        let eps = 1e-4_f32;
        assert!((face.score - 0.72).abs() < eps);
        assert!((face.bbox.0 - 248.0).abs() < eps);
        assert!((face.bbox.1 - 68.0).abs() < eps);
        assert!((face.bbox.2 - 160.0).abs() < eps);
        assert!((face.bbox.3 - 192.0).abs() < eps);

        assert!((face.landmarks[0].0 - 288.0).abs() < eps);
        assert!((face.landmarks[0].1 - 160.0).abs() < eps);
        assert!((face.landmarks[1].0 - 352.0).abs() < eps);
        assert!((face.landmarks[1].1 - 160.0).abs() < eps);

        let (ex, ey, iod) = face.eyes();
        assert!((ex - 320.0).abs() < eps);
        assert!((ey - 160.0).abs() < eps);
        assert!((iod - 64.0).abs() < eps);
    }

    #[test]
    fn test_min_iod_threshold() {
        let [s0, s1, s2] = config::MODEL_STRIDES;
        let mut b8 = SyntheticBuffers::new(s0);
        let b16 = SyntheticBuffers::new(s1);
        let b32 = SyntheticBuffers::new(s2);

        let idx = 0_usize;
        b8.cls[idx] = 0.9;
        b8.obj[idx] = 0.9;

        // Eye0 x = 0, Eye1 x = 1.0 * 8 = 8.0 px in 640-space.
        b8.kps[idx * 10] = 0.0;
        b8.kps[idx * 10 + 2] = 1.0;

        // At scale = 1.0, original-space IOD is 8.0 <= MODEL_MIN_IOD (10.0) -> rejected.
        let slices = [b8.as_tensors(), b16.as_tensors(), b32.as_tensors()];
        let rejected = decode_strides(&slices, 1.0).expect("decode should succeed");
        assert!(
            rejected.is_none(),
            "IOD <= 10.0 in original space must return None"
        );

        // At scale = 0.5, original-space IOD is 8.0 / 0.5 = 16.0 > MODEL_MIN_IOD (10.0) -> accepted.
        let accepted = decode_strides(&slices, 0.5).expect("decode should succeed");
        assert!(
            accepted.is_some(),
            "IOD > 10.0 in original space must return Some"
        );
    }

    #[test]
    fn test_yunet_syalva() {
        let model_path = PathBuf::from(config::DIR_MODELS).join(config::FILE_YUNET_ONNX);
        if !model_path.exists() {
            println!(
                "SKIP test_yunet_syalva: file model tidak ditemukan di {}",
                model_path.display()
            );
            return;
        }

        let detector = FaceDetector::load(&model_path).expect("FaceDetector::load should succeed");

        let photo_path = PathBuf::from("testdata/raw/35_Syalva.jpg");
        if !photo_path.exists() {
            println!(
                "SKIP test_yunet_syalva: file foto uji tidak ditemukan di {}",
                photo_path.display()
            );
            return;
        }

        let img = imageio::load_image(&photo_path).expect("load_image should succeed");
        let max_dim = img.width().max(img.height()) as f32;
        let scale = (config::MODEL_INPUT_SIZE as f32) / max_dim;

        let face = detector
            .detect(&img)
            .expect("detect should succeed")
            .expect("face should be detected on 35_Syalva.jpg");

        // Convert back to 640-space for reference comparison (PRD 3.3).
        let bx = face.bbox.0 * scale;
        let by = face.bbox.1 * scale;
        let bw = face.bbox.2 * scale;
        let bh = face.bbox.3 * scale;
        let lm_640: Vec<(f32, f32)> = face
            .landmarks
            .iter()
            .map(|&(x, y)| (x * scale, y * scale))
            .collect();

        println!(
            "Syalva (ruang 640): score={:.4}, bbox=({:.1}, {:.1}, {:.1}, {:.1}), mata0=({:.1}, {:.1}), mata1=({:.1}, {:.1}), hidung=({:.1}, {:.1}), mulut_kanan=({:.1}, {:.1}), mulut_kiri=({:.1}, {:.1})",
            face.score,
            bx,
            by,
            bw,
            bh,
            lm_640[0].0,
            lm_640[0].1,
            lm_640[1].0,
            lm_640[1].1,
            lm_640[2].0,
            lm_640[2].1,
            lm_640[3].0,
            lm_640[3].1,
            lm_640[4].0,
            lm_640[4].1,
        );

        let score_tol = 0.03_f32;
        let px_tol = 3.0_f32;

        assert!(
            (face.score - 0.937).abs() <= score_tol,
            "score {:.4} di luar toleransi terhadap 0.937",
            face.score
        );
        assert!((bx - 277.1).abs() <= px_tol, "bbox.x {bx:.1} != 277.1");
        assert!((by - 138.8).abs() <= px_tol, "bbox.y {by:.1} != 138.8");
        assert!((bw - 82.2).abs() <= px_tol, "bbox.w {bw:.1} != 82.2");
        assert!((bh - 109.5).abs() <= px_tol, "bbox.h {bh:.1} != 109.5");

        let expected_lm = [
            (297.3_f32, 183.9_f32),
            (337.6, 183.7),
            (319.3, 206.7),
            (302.4, 219.4),
            (337.7, 219.7),
        ];
        for (j, (&(gx, gy), &(ex, ey))) in lm_640.iter().zip(expected_lm.iter()).enumerate() {
            assert!(
                (gx - ex).abs() <= px_tol && (gy - ey).abs() <= px_tol,
                "landmark[{j}] ({gx:.1}, {gy:.1}) di luar toleransi terhadap ({ex:.1}, {ey:.1})"
            );
        }
    }
}
