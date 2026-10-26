//! Batch processing pipeline, rayon parallelism, cancellation, and PERLU_DICEK.txt.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use image::RgbImage;
use rayon::prelude::*;

use crate::assets;
use crate::background;
use crate::compose;
use crate::config;
use crate::face::FaceDetector;
use crate::files::{self, Photo, Skipped};
use crate::framing::{self, FramingParams};
use crate::imageio;

/// Configuration for a single batch processing job.
#[derive(Debug, Clone)]
pub struct Job {
    /// Root input directory containing class subfolders or raw photos directly.
    pub input: PathBuf,
    /// Root output directory where `Output_3x4` and `Output_4R` will be created.
    pub output: PathBuf,
    /// Optional replacement background RGB color (`None` keeps original studio background).
    pub bg_color: Option<[u8; 3]>,
    /// Target headroom fraction above crown.
    pub headroom: f64,
    /// Target eye line fraction from top of 3x4 frame.
    pub eye_line: f64,
}

/// Summary statistics returned when a batch job finishes.
#[derive(Debug, Clone)]
pub struct Summary {
    /// Total 4R sheets generated and saved.
    pub sheets: usize,
    /// Total individual 3x4 photos generated and saved.
    pub photos_3x4: usize,
    /// Total files/photos flagged in `PERLU_DICEK.txt`.
    pub flagged: usize,
    /// Total wall-clock duration of the batch job.
    pub duration: Duration,
    /// Whether the batch job was cancelled before completion.
    pub cancelled: bool,
}

/// Progress and status events emitted by [`run`].
#[derive(Debug, Clone)]
pub enum Event {
    /// Emitted once per non-empty class after upfront planning across all classes.
    ClassPlanned {
        class: String,
        photos: usize,
        pairs: usize,
        summary: String,
    },
    /// Emitted whenever a 4R sheet finishes processing.
    SheetDone { done: usize, total: usize },
    /// Emitted when a photo or skipped file is flagged for manual review.
    Flagged {
        class: String,
        file: String,
        reasons: Vec<String>,
    },
    /// Informational log message in Indonesian.
    Log(String),
    /// Final event emitted when the pipeline finishes.
    Finished(Summary),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FlagEntry {
    class: String,
    file: String,
    reasons: Vec<String>,
}

struct PlannedClass {
    name: String,
    label: String,
    photos: Vec<Photo>,
    skipped: Vec<Skipped>,
    pairs: Vec<(Photo, Option<Photo>)>,
    summary: String,
}

/// Sends an event to the channel, ignoring receiver disconnection errors.
fn send_event(events: &Sender<Event>, event: Event) {
    let _ = events.send(event);
}

/// Processes a single student photo: load -> detect -> frame -> optional bg -> save 3x4 -> fit to slot.
// port of proses_satu() in reference/app.py:580-591
fn process_single_photo(
    photo: &Photo,
    out_3x4_dir: &Path,
    detector: &FaceDetector,
    framing_params: &FramingParams,
    bg_color: Option<[u8; 3]>,
) -> Result<(RgbImage, Vec<String>)> {
    let img = imageio::load_image(&photo.path)?;
    let face = detector.detect(&img)?;
    let eyes = face.map(|f| f.eyes());
    let framed = framing::frame_photo(&img, eyes, framing_params);
    let img_3x4 = match bg_color {
        Some(rgb) => background::replace_background(&framed.image, rgb),
        None => framed.image,
    };

    let out_3x4_path = out_3x4_dir.join(format!("{}_3x4.jpg", photo.stem));
    imageio::save_jpeg(&img_3x4, &out_3x4_path)?;

    let slot_img = compose::fit_to_slot(&img_3x4);
    let reasons = framed.reasons.into_iter().map(String::from).collect();
    Ok((slot_img, reasons))
}

struct PairContext<'a> {
    detector: &'a FaceDetector,
    framing_params: &'a FramingParams,
    bg_color: Option<[u8; 3]>,
    events: &'a Sender<Event>,
    flagged_mutex: &'a Mutex<Vec<FlagEntry>>,
    photos_3x4_done: &'a AtomicUsize,
}

/// Evaluates one photo inside a pair, emitting flags on failure or framing warnings without panicking.
fn evaluate_photo_in_pair(
    photo: &Photo,
    class_label: &str,
    out_3x4_dir: &Path,
    ctx: &PairContext<'_>,
) -> Option<RgbImage> {
    let file_name = photo
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&photo.stem)
        .to_string();

    match process_single_photo(
        photo,
        out_3x4_dir,
        ctx.detector,
        ctx.framing_params,
        ctx.bg_color,
    ) {
        Ok((slot_img, reasons)) => {
            ctx.photos_3x4_done.fetch_add(1, Ordering::Relaxed);
            if !reasons.is_empty() {
                send_event(
                    ctx.events,
                    Event::Log(format!("Perlu dicek: {file_name} ({})", reasons.join("; "))),
                );
                send_event(
                    ctx.events,
                    Event::Flagged {
                        class: class_label.to_string(),
                        file: file_name.clone(),
                        reasons: reasons.clone(),
                    },
                );
                let mut guard = ctx
                    .flagged_mutex
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.push(FlagEntry {
                    class: class_label.to_string(),
                    file: file_name,
                    reasons,
                });
            }
            Some(slot_img)
        }
        Err(e) => {
            let reason = format!("gagal diproses: {e}");
            let reasons = vec![reason];
            send_event(
                ctx.events,
                Event::Log(format!("GAGAL memproses {file_name}: {e}")),
            );
            send_event(
                ctx.events,
                Event::Flagged {
                    class: class_label.to_string(),
                    file: file_name.clone(),
                    reasons: reasons.clone(),
                },
            );
            let mut guard = ctx
                .flagged_mutex
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.push(FlagEntry {
                class: class_label.to_string(),
                file: file_name,
                reasons,
            });
            None
        }
    }
}

/// Formats sorted `PERLU_DICEK.txt` content according to PRD 3.7.
fn format_perlu_dicek(entries: &mut [FlagEntry]) -> String {
    entries.sort_by(|a, b| {
        a.class
            .to_lowercase()
            .cmp(&b.class.to_lowercase())
            .then_with(|| a.class.cmp(&b.class))
            .then_with(|| a.file.to_lowercase().cmp(&b.file.to_lowercase()))
            .then_with(|| a.file.cmp(&b.file))
    });

    let mut content = String::from("Foto berikut sebaiknya dicek manual:\n");
    for entry in entries.iter() {
        content.push_str(&format!(
            "[{}] {}: {}\n",
            entry.class,
            entry.file,
            entry.reasons.join("; ")
        ));
    }
    content
}

/// Runs the batch processing pipeline across all classes in `job.input`.
// port of _run_automation_engine() in reference/app.py:546-672
pub fn run(job: Job, events: Sender<Event>, cancel: Arc<AtomicBool>) -> Result<Summary> {
    let start = Instant::now();

    let asset_paths = assets::locate()?;
    let loaded_assets = assets::load_from_paths(&asset_paths)?;
    let detector = FaceDetector::load(&asset_paths.model_onnx)?;

    // Compute bottom banner coverage once on the first slot (reference/app.py:573-574).
    let layers: Vec<&_> = match &loaded_assets.overlay {
        Some(ov) => vec![ov, &loaded_assets.template],
        None => vec![&loaded_assets.template],
    };
    let [first_col_x, ..] = config::SLOT_COLS_X;
    let cover = framing::banner_cover(
        &layers,
        first_col_x,
        config::SLOT_ROW1_Y,
        config::SLOT_WIDTH,
        config::SLOT_HEIGHT,
    );
    send_event(
        &events,
        Event::Log(format!(
            "Banner template menutupi {:.0}% bagian bawah foto; logo dijaga tetap di atasnya.",
            cover * 100.0
        )),
    );

    let framing_params = FramingParams {
        headroom: job.headroom,
        eye_line: job.eye_line,
        banner_cover: f64::from(cover),
    };

    // Scan all classes upfront to calculate accurate total sheets across the entire batch.
    let class_dirs = files::collect_classes(&job.input)?;
    let mut planned_classes = Vec::with_capacity(class_dirs.len());
    for class_dir in class_dirs {
        let (photos, skipped) = files::collect_photos(&class_dir.path)?;
        let pairs = files::pair_photos(&photos);
        let summary = files::pairing_summary(&pairs);
        let label = if class_dir.name.is_empty() {
            "folder input".to_string()
        } else {
            class_dir.name.clone()
        };
        planned_classes.push(PlannedClass {
            name: class_dir.name,
            label,
            photos,
            skipped,
            pairs,
            summary,
        });
    }

    let total_sheets: usize = planned_classes.iter().map(|c| c.pairs.len()).sum();
    let flagged_mutex: Mutex<Vec<FlagEntry>> = Mutex::new(Vec::new());

    // Emit skipped files and ClassPlanned events upfront (reference/app.py:603-615).
    for c in &planned_classes {
        for sk in &c.skipped {
            send_event(
                &events,
                Event::Log(format!("[{}] {}: {}", c.label, sk.filename, sk.reason)),
            );
            let reasons = vec![sk.reason.clone()];
            send_event(
                &events,
                Event::Flagged {
                    class: c.label.clone(),
                    file: sk.filename.clone(),
                    reasons: reasons.clone(),
                },
            );
            let mut guard = flagged_mutex
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.push(FlagEntry {
                class: c.label.clone(),
                file: sk.filename.clone(),
                reasons,
            });
        }

        if !c.photos.is_empty() {
            send_event(
                &events,
                Event::ClassPlanned {
                    class: c.name.clone(),
                    photos: c.photos.len(),
                    pairs: c.pairs.len(),
                    summary: c.summary.clone(),
                },
            );
            let display_class = if c.name.is_empty() { "-" } else { &c.name };
            send_event(
                &events,
                Event::Log(format!(
                    "Kelas {display_class}: {} foto, pasangan 4R: {}",
                    c.photos.len(),
                    c.summary
                )),
            );
        }
    }

    let out_3x4_root = job.output.join(config::DIR_OUTPUT_3X4);
    let out_4r_root = job.output.join(config::DIR_OUTPUT_4R);
    fs::create_dir_all(&out_3x4_root).with_context(|| {
        format!(
            "Gagal membuat folder output 3x4: {}",
            out_3x4_root.display()
        )
    })?;
    fs::create_dir_all(&out_4r_root)
        .with_context(|| format!("Gagal membuat folder output 4R: {}", out_4r_root.display()))?;

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .context("Gagal menginisialisasi thread pool pemrosesan paralel")?;

    let sheets_done = AtomicUsize::new(0);
    let photos_3x4_done = AtomicUsize::new(0);

    for c in &planned_classes {
        if c.pairs.is_empty() {
            continue;
        }
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        let out_4r_dir = if c.name.is_empty() {
            out_4r_root.clone()
        } else {
            out_4r_root.join(format!("{}_4R", c.name))
        };
        let out_3x4_dir = if c.name.is_empty() {
            out_3x4_root.clone()
        } else {
            out_3x4_root.join(format!("{}_3x4", c.name))
        };

        fs::create_dir_all(&out_4r_dir)
            .with_context(|| format!("Gagal membuat folder kelas 4R: {}", out_4r_dir.display()))?;
        fs::create_dir_all(&out_3x4_dir).with_context(|| {
            format!("Gagal membuat folder kelas 3x4: {}", out_3x4_dir.display())
        })?;

        let pair_ctx = PairContext {
            detector: &detector,
            framing_params: &framing_params,
            bg_color: job.bg_color,
            events: &events,
            flagged_mutex: &flagged_mutex,
            photos_3x4_done: &photos_3x4_done,
        };

        pool.install(|| {
            c.pairs.par_iter().for_each(|(p1, p2_opt)| {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }

                let absen_2_text = p2_opt
                    .as_ref()
                    .map(|p2| p2.label.as_str())
                    .unwrap_or("(Kosong)");
                send_event(
                    &events,
                    Event::Log(format!(
                        "Memproses Pasangan Absen {} dan {absen_2_text}...",
                        p1.label
                    )),
                );

                let slot1 = evaluate_photo_in_pair(p1, &c.label, &out_3x4_dir, &pair_ctx);

                let slot2 = match p2_opt.as_ref() {
                    Some(p2) => evaluate_photo_in_pair(p2, &c.label, &out_3x4_dir, &pair_ctx),
                    None => None,
                };

                let first_arg = slot1.as_ref().map(|img| (img, p1.label.as_str()));
                let second_arg = match (&slot2, p2_opt.as_ref()) {
                    (Some(img), Some(p2)) => Some((img, p2.label.as_str())),
                    _ => None,
                };

                let sheet = compose::compose_sheet(&loaded_assets, first_arg, second_arg, &c.name);
                // port of reference/app.py:653
                let output_name = match p2_opt.as_ref() {
                    Some(p2) => format!("{}_Absen_{}-{}_4R.jpg", c.name, p1.label, p2.label),
                    None => format!("{}_Absen_{}_4R.jpg", c.name, p1.label),
                };
                let out_4r_path = out_4r_dir.join(&output_name);

                match imageio::save_jpeg(&sheet, &out_4r_path) {
                    Ok(()) => {
                        let done = sheets_done.fetch_add(1, Ordering::Relaxed) + 1;
                        send_event(
                            &events,
                            Event::SheetDone {
                                done,
                                total: total_sheets,
                            },
                        );
                    }
                    Err(e) => {
                        send_event(
                            &events,
                            Event::Log(format!("GAGAL menyimpan lembar 4R {output_name}: {e}")),
                        );
                    }
                }
            });
        });
    }

    let mut flagged_entries = flagged_mutex
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let perlu_dicek_path = out_4r_root.join(config::FILE_PERLU_DICEK);

    if !flagged_entries.is_empty() {
        let content = format_perlu_dicek(&mut flagged_entries);
        fs::write(&perlu_dicek_path, content).with_context(|| {
            format!(
                "Gagal menulis file daftar cek manual: {}",
                perlu_dicek_path.display()
            )
        })?;
        send_event(
            &events,
            Event::Log(format!(
                "Selesai. {} foto perlu dicek, daftarnya ada di {}",
                flagged_entries.len(),
                config::FILE_PERLU_DICEK
            )),
        );
    } else {
        if perlu_dicek_path.exists() {
            let _ = fs::remove_file(&perlu_dicek_path);
        }
        send_event(
            &events,
            Event::Log("Proses otomatisasi pas foto selesai!".to_string()),
        );
    }

    let summary = Summary {
        sheets: sheets_done.load(Ordering::Relaxed),
        photos_3x4: photos_3x4_done.load(Ordering::Relaxed),
        flagged: flagged_entries.len(),
        duration: start.elapsed(),
        cancelled: cancel.load(Ordering::Relaxed),
    };
    send_event(&events, Event::Finished(summary.clone()));

    Ok(summary)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    input: PathBuf,
    output: PathBuf,
    bg_color: Option<[u8; 3]>,
}

/// Returns the Indonesian CLI usage string.
fn cli_usage() -> &'static str {
    "Cara pakai mode CLI:\n  PasFoto.exe --cli --input <folder_input> --output <folder_output> [--bg <#RRGGBB>]"
}

/// Parses a 6-digit hex color string (`#RRGGBB` or `RRGGBB`) into `[r, g, b]`.
fn parse_hex_color(raw: &str) -> Result<[u8; 3]> {
    let hex = raw.trim().strip_prefix('#').unwrap_or(raw.trim());
    if hex.len() != 6 || !hex.is_ascii() {
        bail!(
            "Format warna latar tidak valid: '{raw}'. Gunakan format heksadesimal 6 digit, misalnya #BE1220 atau #1F4FBF.\n{}",
            cli_usage()
        );
    }
    let r = u8::from_str_radix(&hex[0..2], 16).with_context(|| {
        format!(
            "Komponen merah tidak valid pada warna '{raw}'. Gunakan format #RRGGBB.\n{}",
            cli_usage()
        )
    })?;
    let g = u8::from_str_radix(&hex[2..4], 16).with_context(|| {
        format!(
            "Komponen hijau tidak valid pada warna '{raw}'. Gunakan format #RRGGBB.\n{}",
            cli_usage()
        )
    })?;
    let b = u8::from_str_radix(&hex[4..6], 16).with_context(|| {
        format!(
            "Komponen biru tidak valid pada warna '{raw}'. Gunakan format #RRGGBB.\n{}",
            cli_usage()
        )
    })?;
    Ok([r, g, b])
}

/// Parses CLI arguments manually without external crates.
fn parse_cli_args<I, S>(args: I) -> Result<CliArgs>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut bg_color: Option<[u8; 3]> = None;

    let mut iter = args.into_iter().skip(1);
    while let Some(arg_item) = iter.next() {
        let arg = arg_item.as_ref();
        match arg {
            "--cli" => {}
            "--input" => {
                let val = iter.next().with_context(|| {
                    format!("Argumen --input membutuhkan jalur folder.\n{}", cli_usage())
                })?;
                input = Some(PathBuf::from(val.as_ref()));
            }
            "--output" => {
                let val = iter.next().with_context(|| {
                    format!(
                        "Argumen --output membutuhkan jalur folder.\n{}",
                        cli_usage()
                    )
                })?;
                output = Some(PathBuf::from(val.as_ref()));
            }
            "--bg" => {
                let val = iter.next().with_context(|| {
                    format!(
                        "Argumen --bg membutuhkan kode warna #RRGGBB.\n{}",
                        cli_usage()
                    )
                })?;
                bg_color = Some(parse_hex_color(val.as_ref())?);
            }
            other => {
                bail!("Argumen tidak dikenal: '{other}'.\n{}", cli_usage());
            }
        }
    }

    let input = input.with_context(|| {
        format!(
            "Folder input belum ditentukan (--input <folder>).\n{}",
            cli_usage()
        )
    })?;
    let output = output.with_context(|| {
        format!(
            "Folder output belum ditentukan (--output <folder>).\n{}",
            cli_usage()
        )
    })?;

    Ok(CliArgs {
        input,
        output,
        bg_color,
    })
}

/// Formats a pipeline event as a single human-readable line for CLI output and `log_cli.txt`.
fn format_cli_event(event: &Event) -> String {
    match event {
        Event::ClassPlanned {
            class,
            photos,
            pairs,
            summary,
        } => {
            let disp = if class.is_empty() { "-" } else { class };
            format!("[RENCANA] Kelas {disp}: {photos} foto, {pairs} lembar 4R ({summary})")
        }
        Event::SheetDone { done, total } => {
            format!("[PROGRES] Lembar 4R selesai: {done}/{total}")
        }
        Event::Flagged {
            class,
            file,
            reasons,
        } => {
            let disp = if class.is_empty() {
                "folder input"
            } else {
                class
            };
            format!("[PERLU DICEK] [{disp}] {file}: {}", reasons.join("; "))
        }
        Event::Log(msg) => format!("[LOG] {msg}"),
        Event::Finished(summary) => format!(
            "[SELESAI] Lembar 4R: {}, Foto 3x4: {}, Perlu dicek: {}, Durasi: {:.2} detik",
            summary.sheets,
            summary.photos_3x4,
            summary.flagged,
            summary.duration.as_secs_f64()
        ),
    }
}

/// Writes a line to stdout (ignoring broken/unavailable stdout) and to `log_cli.txt` if provided.
fn write_cli_line(log_writer: &mut Option<BufWriter<File>>, line: &str) {
    let _ = writeln!(std::io::stdout(), "{line}");
    if let Some(writer) = log_writer.as_mut() {
        let _ = writeln!(writer, "{line}");
        let _ = writer.flush();
    }
}

/// Runs the developer CLI mode (`--cli --input <folder> --output <folder> [--bg <#RRGGBB>]`).
pub fn run_cli() -> Result<()> {
    let cli_args = match parse_cli_args(std::env::args()) {
        Ok(args) => args,
        Err(e) => {
            let _ = writeln!(std::io::stdout(), "{e}");
            let _ = writeln!(std::io::stderr(), "{e}");
            return Err(e);
        }
    };

    fs::create_dir_all(&cli_args.output)
        .with_context(|| format!("Gagal membuat folder output: {}", cli_args.output.display()))?;

    let log_path = cli_args.output.join("log_cli.txt");
    let mut log_writer = File::create(&log_path).ok().map(BufWriter::new);

    let job = Job {
        input: cli_args.input,
        output: cli_args.output,
        bg_color: cli_args.bg_color,
        headroom: config::HEADROOM_DEFAULT,
        eye_line: config::EYE_LINE_DEFAULT,
    };

    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));

    let run_result = std::thread::scope(|s| {
        let handle = s.spawn(move || run(job, tx, cancel));
        for event in rx {
            let line = format_cli_event(&event);
            write_cli_line(&mut log_writer, &line);
        }
        handle.join()
    });

    match run_result {
        Ok(Ok(_summary)) => Ok(()),
        Ok(Err(e)) => {
            let err_line = format!("[ERROR] {e:#}");
            write_cli_line(&mut log_writer, &err_line);
            Err(e)
        }
        Err(_) => {
            let err_line = "[ERROR] Thread pemrosesan berhenti secara tidak terduga.";
            write_cli_line(&mut log_writer, err_line);
            bail!("{err_line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_jpegs_recursive(dir: &Path) -> usize {
        let Ok(entries) = fs::read_dir(dir) else {
            return 0;
        };
        let mut count = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                count += count_jpegs_recursive(&path);
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg"))
            {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn test_parse_hex_color() {
        assert_eq!(parse_hex_color("#1F4FBF").ok(), Some([31, 79, 191]));
        assert_eq!(parse_hex_color("be1220").ok(), Some([190, 18, 32]));
        assert!(parse_hex_color("#12345").is_err());
        assert!(parse_hex_color("#GGGGGG").is_err());
    }

    #[test]
    fn test_parse_cli_args() {
        let parsed = parse_cli_args([
            "PasFoto.exe",
            "--cli",
            "--input",
            "testdata/raw",
            "--output",
            "target/uji",
            "--bg",
            "#1F4FBF",
        ])
        .expect("valid CLI args should parse");
        assert_eq!(parsed.input, PathBuf::from("testdata/raw"));
        assert_eq!(parsed.output, PathBuf::from("target/uji"));
        assert_eq!(parsed.bg_color, Some([31, 79, 191]));

        assert!(parse_cli_args(["PasFoto.exe", "--cli", "--input", "a"]).is_err());
        assert!(parse_cli_args(["PasFoto.exe", "--cli", "--unknown"]).is_err());
    }

    #[test]
    fn test_format_perlu_dicek_sorting_and_header() {
        let mut entries = vec![
            FlagEntry {
                class: "XII-B".to_string(),
                file: "02_Budi.jpg".to_string(),
                reasons: vec!["wajah tidak terdeteksi".to_string()],
            },
            FlagEntry {
                class: "XII-A".to_string(),
                file: "03_Citra.jpg".to_string(),
                reasons: vec!["zoom terlalu dekat".to_string(), "logo mepet".to_string()],
            },
            FlagEntry {
                class: "XII-A".to_string(),
                file: "01_Andi.jpg".to_string(),
                reasons: vec!["latar bukan merah".to_string()],
            },
        ];

        let formatted = format_perlu_dicek(&mut entries);
        let expected = "\
Foto berikut sebaiknya dicek manual:
[XII-A] 01_Andi.jpg: latar bukan merah
[XII-A] 03_Citra.jpg: zoom terlalu dekat; logo mepet
[XII-B] 02_Budi.jpg: wajah tidak terdeteksi
";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_output_folder_and_filename_structure_single_class() {
        let temp_dir = tempfile::tempdir().expect("create tempdir");
        let class_input = temp_dir.path().join("X-1");
        std::fs::create_dir(&class_input).expect("create class input");

        // Verify collect_classes identifies "X-1" as class name
        let classes = files::collect_classes(&class_input).expect("collect_classes");
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].name, "X-1");

        // Verify output paths logic
        let out_root = temp_dir.path().join("output");
        let out_3x4_root = out_root.join(config::DIR_OUTPUT_3X4);
        let out_4r_root = out_root.join(config::DIR_OUTPUT_4R);

        let out_3x4_dir = out_3x4_root.join(format!("{}_3x4", classes[0].name));
        let out_4r_dir = out_4r_root.join(format!("{}_4R", classes[0].name));

        assert_eq!(out_3x4_dir, out_root.join("Output_3x4").join("X-1_3x4"));
        assert_eq!(out_4r_dir, out_root.join("Output_4R").join("X-1_4R"));

        // Verify 4R output file naming
        let output_name_pair = format!("{}_Absen_{}-{}_4R.jpg", classes[0].name, "01", "02");
        assert_eq!(output_name_pair, "X-1_Absen_01-02_4R.jpg");

        let output_name_single = format!("{}_Absen_{}_4R.jpg", classes[0].name, "01");
        assert_eq!(output_name_single, "X-1_Absen_01_4R.jpg");
    }

    #[test]
    fn test_pipeline_integration_testdata_raw() {
        let raw_dir = Path::new("testdata/raw");
        let template_dir = Path::new(config::DIR_TEMPLATE);
        let models_dir = Path::new(config::DIR_MODELS);

        if !raw_dir.exists() || !template_dir.exists() || !models_dir.exists() {
            let _ = writeln!(
                std::io::stdout(),
                "Skipping test_pipeline_integration_testdata_raw: testdata/raw, Template, or models not found"
            );
            return;
        }

        let classes = files::collect_classes(raw_dir).expect("collect_classes should succeed");
        let mut expected_photos = 0usize;
        let mut expected_sheets = 0usize;
        for c in &classes {
            let (photos, _skipped) =
                files::collect_photos(&c.path).expect("collect_photos should succeed");
            let pairs = files::pair_photos(&photos);
            expected_photos += photos.len();
            expected_sheets += pairs.len();
        }

        let temp_out = tempfile::tempdir().expect("tempdir should succeed");
        let job = Job {
            input: raw_dir.to_path_buf(),
            output: temp_out.path().to_path_buf(),
            bg_color: None,
            headroom: config::HEADROOM_DEFAULT,
            eye_line: config::EYE_LINE_DEFAULT,
        };

        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let summary = run(job, tx, cancel).expect("pipeline::run should succeed");
        let events: Vec<Event> = rx.try_iter().collect();

        assert_eq!(summary.photos_3x4, expected_photos);
        assert_eq!(summary.sheets, expected_sheets);
        assert!(!summary.cancelled);
        assert!(events.iter().any(|e| matches!(e, Event::Finished(_))));

        let actual_3x4 = count_jpegs_recursive(&temp_out.path().join(config::DIR_OUTPUT_3X4));
        let actual_4r = count_jpegs_recursive(&temp_out.path().join(config::DIR_OUTPUT_4R));
        assert_eq!(actual_3x4, expected_photos);
        assert_eq!(actual_4r, expected_sheets);

        // Verify folder structure for input without subfolders:
        // Output_3x4/raw_3x4/ and Output_4R/raw_4R/
        let raw_3x4_dir = temp_out.path().join(config::DIR_OUTPUT_3X4).join("raw_3x4");
        let raw_4r_dir = temp_out.path().join(config::DIR_OUTPUT_4R).join("raw_4R");
        assert!(
            raw_3x4_dir.is_dir(),
            "Folder Output_3x4/raw_3x4 harus dibuat"
        );
        assert!(raw_4r_dir.is_dir(), "Folder Output_4R/raw_4R harus dibuat");

        // Verify 4R sheet filenames begin with "raw_Absen_"
        let r4_entries = std::fs::read_dir(&raw_4r_dir).expect("read raw_4r_dir");
        let mut count_4r_files = 0;
        for entry in r4_entries {
            let entry = entry.expect("valid entry");
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".jpg") {
                count_4r_files += 1;
                assert!(
                    name.starts_with("raw_Absen_"),
                    "File 4R harus diawali dengan raw_Absen_, ditemukan: {name}"
                );
            }
        }
        assert_eq!(count_4r_files, expected_sheets);

        // Verify PERLU_DICEK.txt uses [raw] prefix if flagged entries exist
        let perlu_dicek_path = temp_out
            .path()
            .join(config::DIR_OUTPUT_4R)
            .join(config::FILE_PERLU_DICEK);
        if perlu_dicek_path.is_file() {
            let content = std::fs::read_to_string(&perlu_dicek_path).expect("read PERLU_DICEK.txt");
            assert!(
                !content.contains("[folder input]"),
                "PERLU_DICEK.txt tidak boleh memuat '[folder input]'"
            );
            assert!(
                content.contains("[raw]"),
                "PERLU_DICEK.txt harus menggunakan prefix '[raw]'"
            );
        }
    }
}
