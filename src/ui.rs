//! egui/eframe GUI implementation.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;

use eframe::egui;
use serde::{Deserialize, Serialize};

use crate::assets;
use crate::config;
use crate::files::{self, Skipped};
use crate::pipeline::{self, Event, Job, Summary};

/// Minimum allowed eye line slider percentage.
pub const EYE_LINE_MIN_PCT: u32 = 20;
/// Maximum allowed eye line slider percentage.
pub const EYE_LINE_MAX_PCT: u32 = 40;
/// Default eye line slider percentage (28%).
pub const EYE_LINE_DEFAULT_PCT: u32 = 28;

/// Minimum allowed headroom slider percentage.
pub const HEADROOM_MIN_PCT: u32 = 2;
/// Maximum allowed headroom slider percentage.
pub const HEADROOM_MAX_PCT: u32 = 15;
/// Default headroom slider percentage (7%).
pub const HEADROOM_DEFAULT_PCT: u32 = 7;

/// Persistent user settings saved across sessions via eframe storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub raw_dir: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub replace_bg: bool,
    pub bg_color: [u8; 3],
    pub eye_line_pct: u32,
    pub headroom_pct: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            raw_dir: None,
            output_dir: default_output_dir(),
            replace_bg: false,
            bg_color: config::DEFAULT_BG_COLOR_RGB,
            eye_line_pct: EYE_LINE_DEFAULT_PCT,
            headroom_pct: HEADROOM_DEFAULT_PCT,
        }
    }
}

impl Settings {
    /// Resets advanced framing sliders to their default values.
    pub fn reset_advanced(&mut self) {
        self.eye_line_pct = EYE_LINE_DEFAULT_PCT;
        self.headroom_pct = HEADROOM_DEFAULT_PCT;
    }

    /// Converts the eye line percentage slider to a fraction (e.g. 28 -> 0.28).
    pub fn eye_line_fraction(&self) -> f64 {
        pct_to_fraction(self.eye_line_pct)
    }

    /// Converts the headroom percentage slider to a fraction (e.g. 7 -> 0.07).
    pub fn headroom_fraction(&self) -> f64 {
        pct_to_fraction(self.headroom_pct)
    }
}

/// Returns the default output directory (directory containing the executable, or current dir).
pub fn default_output_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        return parent.to_path_buf();
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Converts a percentage integer (`0..=100`) to a floating-point fraction (`0.0..=1.0`).
pub fn pct_to_fraction(pct: u32) -> f64 {
    f64::from(pct) / 100.0
}

/// Returns the display label for a class name, matching `src/pipeline.rs` lines 277-281.
pub fn class_display_label(name: &str) -> String {
    if name.is_empty() {
        "folder input".to_string()
    } else {
        name.to_string()
    }
}

/// Formats the preview summary text after scanning the raw folder (e.g. "3 kelas, 105 foto").
pub fn format_preview_summary(classes: usize, photos: usize) -> String {
    format!("{classes} kelas, {photos} foto")
}

/// Formats a single class pairing plan line for the preview section.
pub fn format_class_pairing_line(label: &str, photo_count: usize, pairing_summary: &str) -> String {
    if pairing_summary.is_empty() {
        format!("Kelas {label}: {photo_count} foto")
    } else {
        format!("Kelas {label}: {photo_count} foto, pasangan 4R: {pairing_summary}")
    }
}

/// Formats the sheet progress counter text (e.g. "12 dari 18 lembar").
pub fn format_sheet_progress(done: usize, total: usize) -> String {
    format!("{done} dari {total} lembar")
}

/// Formats the final summary message when a batch finishes or is cancelled.
pub fn format_finished_summary(summary: &Summary) -> String {
    if summary.cancelled {
        format!(
            "Dibatalkan. Sebagian selesai: {} lembar 4R dan {} foto 3x4. {} foto perlu dicek.",
            summary.sheets, summary.photos_3x4, summary.flagged
        )
    } else {
        format!(
            "Selesai: {} lembar 4R dan {} foto 3x4. {} foto perlu dicek.",
            summary.sheets, summary.photos_3x4, summary.flagged
        )
    }
}

/// Computes the label and enabled state of the main action button.
pub fn main_button_state(
    is_running: bool,
    is_cancelling: bool,
    has_valid_preview: bool,
    has_asset_error: bool,
) -> (&'static str, bool) {
    if is_running {
        ("Batalkan", !is_cancelling)
    } else {
        ("Mulai Proses", has_valid_preview && !has_asset_error)
    }
}

/// Preview information for a single class folder.
#[derive(Debug, Clone)]
pub struct ClassPreview {
    pub label: String,
    pub photo_count: usize,
    pub pairing_summary: String,
    pub skipped: Vec<Skipped>,
}

/// Complete preview data computed on a background thread for the selected raw folder.
#[derive(Debug, Clone)]
pub struct FolderPreview {
    pub class_count: usize,
    pub total_photos: usize,
    pub classes: Vec<ClassPreview>,
}

/// State of the raw photo folder preview.
enum PreviewState {
    None,
    Loading(Receiver<Result<FolderPreview, String>>),
    Ready(FolderPreview),
    Error(String),
}

/// Flagged photo entry recorded during batch execution.
#[derive(Debug, Clone)]
struct FlaggedItem {
    class: String,
    file: String,
    reasons: Vec<String>,
}

/// Scans a raw input directory and builds the pairing preview.
fn compute_folder_preview(input: &Path) -> Result<FolderPreview, String> {
    let class_dirs = files::collect_classes(input).map_err(|e| format!("{e:#}"))?;
    let mut classes = Vec::with_capacity(class_dirs.len());
    let mut total_photos = 0usize;

    for class_dir in class_dirs {
        let (photos, skipped) =
            files::collect_photos(&class_dir.path).map_err(|e| format!("{e:#}"))?;
        let pairs = files::pair_photos(&photos);
        let pairing_summary = files::pairing_summary(&pairs);
        let label = class_display_label(&class_dir.name);
        let photo_count = photos.len();
        total_photos += photo_count;

        classes.push(ClassPreview {
            label,
            photo_count,
            pairing_summary,
            skipped,
        });
    }

    let class_count = classes.len();
    Ok(FolderPreview {
        class_count,
        total_photos,
        classes,
    })
}

/// Checks whether required asset files exist using `assets::locate()`.
fn check_assets() -> Option<String> {
    assets::locate().err().map(|e| format!("{e:#}"))
}

/// Main application state for egui/eframe.
pub struct App {
    settings: Settings,
    asset_error: Option<String>,
    process_error: Option<String>,
    preview: PreviewState,
    is_running: bool,
    is_cancelling: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    event_rx: Option<Receiver<Event>>,
    result_rx: Option<Receiver<Result<Summary, String>>>,
    sheets_done: usize,
    sheets_total: usize,
    status_text: String,
    logs: Vec<String>,
    flagged_items: Vec<FlaggedItem>,
    last_summary: Option<Summary>,
}

impl Default for App {
    fn default() -> Self {
        Self::from_settings(Settings::default())
    }
}

impl App {
    /// Creates the application, restoring saved settings from `cc.storage` if available.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let settings = cc
            .storage
            .and_then(|storage| eframe::get_value::<Settings>(storage, eframe::APP_KEY))
            .unwrap_or_default();
        Self::from_settings(settings)
    }

    fn from_settings(mut settings: Settings) -> Self {
        settings.eye_line_pct = settings
            .eye_line_pct
            .clamp(EYE_LINE_MIN_PCT, EYE_LINE_MAX_PCT);
        settings.headroom_pct = settings
            .headroom_pct
            .clamp(HEADROOM_MIN_PCT, HEADROOM_MAX_PCT);

        let asset_error = check_assets();
        let mut app = Self {
            settings,
            asset_error,
            process_error: None,
            preview: PreviewState::None,
            is_running: false,
            is_cancelling: false,
            cancel_flag: None,
            event_rx: None,
            result_rx: None,
            sheets_done: 0,
            sheets_total: 0,
            status_text: String::new(),
            logs: Vec::new(),
            flagged_items: Vec::new(),
            last_summary: None,
        };

        if let Some(raw_dir) = app.settings.raw_dir.clone() {
            app.start_preview_scan(raw_dir);
        }

        app
    }

    fn start_preview_scan(&mut self, dir: PathBuf) {
        self.asset_error = check_assets();
        let (tx, rx) = mpsc::channel();
        self.preview = PreviewState::Loading(rx);
        thread::spawn(move || {
            let res = compute_folder_preview(&dir);
            let _ = tx.send(res);
        });
    }

    fn poll_preview(&mut self, ctx: &egui::Context) {
        let next_state = match &self.preview {
            PreviewState::Loading(rx) => match rx.try_recv() {
                Ok(Ok(data)) => Some(PreviewState::Ready(data)),
                Ok(Err(err)) => Some(PreviewState::Error(err)),
                Err(TryRecvError::Empty) => {
                    ctx.request_repaint();
                    None
                }
                Err(TryRecvError::Disconnected) => Some(PreviewState::Error(
                    "Gagal membaca folder foto mentah.".to_string(),
                )),
            },
            _ => None,
        };

        if let Some(state) = next_state {
            self.preview = state;
        }
    }

    fn start_process(&mut self) {
        self.asset_error = check_assets();
        if self.asset_error.is_some() {
            return;
        }
        let Some(input) = self.settings.raw_dir.clone() else {
            return;
        };

        self.process_error = None;
        self.logs.clear();
        self.flagged_items.clear();
        self.last_summary = None;
        self.sheets_done = 0;
        self.sheets_total = 0;
        self.status_text = "Menyiapkan proses…".to_string();
        self.is_running = true;
        self.is_cancelling = false;

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_flag = Some(Arc::clone(&cancel));

        let (event_tx, event_rx) = mpsc::channel();
        let (res_tx, res_rx) = mpsc::channel();
        self.event_rx = Some(event_rx);
        self.result_rx = Some(res_rx);

        let job = Job {
            input,
            output: self.settings.output_dir.clone(),
            bg_color: self.settings.replace_bg.then_some(self.settings.bg_color),
            headroom: self.settings.headroom_fraction(),
            eye_line: self.settings.eye_line_fraction(),
        };

        thread::spawn(move || {
            let result = pipeline::run(job, event_tx, cancel).map_err(|e| format!("{e:#}"));
            let _ = res_tx.send(result);
        });
    }

    fn poll_process(&mut self, ctx: &egui::Context) {
        if !self.is_running {
            return;
        }

        let mut finished_from_event = false;
        if let Some(rx) = &self.event_rx {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    Event::ClassPlanned {
                        class,
                        photos,
                        pairs,
                        ..
                    } => {
                        self.sheets_total += pairs;
                        if !self.is_cancelling {
                            self.status_text = format!("Menyiapkan kelas {class} ({photos} foto)…");
                        }
                    }
                    Event::SheetDone { done, total } => {
                        self.sheets_done = done;
                        self.sheets_total = total;
                        if !self.is_cancelling {
                            self.status_text =
                                format!("Memproses lembar {}…", format_sheet_progress(done, total));
                        }
                    }
                    Event::Flagged {
                        class,
                        file,
                        reasons,
                    } => {
                        self.flagged_items.push(FlaggedItem {
                            class,
                            file,
                            reasons,
                        });
                    }
                    Event::Log(line) => {
                        self.logs.push(line);
                    }
                    Event::Finished(summary) => {
                        self.status_text = if summary.cancelled {
                            "Dibatalkan".to_string()
                        } else {
                            "Selesai".to_string()
                        };
                        self.last_summary = Some(summary);
                        self.is_running = false;
                        self.is_cancelling = false;
                        finished_from_event = true;
                    }
                }
            }
        }

        if finished_from_event {
            self.event_rx = None;
            self.result_rx = None;
            self.cancel_flag = None;
            return;
        }

        if let Some(res_rx) = &self.result_rx {
            match res_rx.try_recv() {
                Ok(Ok(summary)) => {
                    self.status_text = if summary.cancelled {
                        "Dibatalkan".to_string()
                    } else {
                        "Selesai".to_string()
                    };
                    self.last_summary = Some(summary);
                    self.is_running = false;
                    self.is_cancelling = false;
                    self.event_rx = None;
                    self.result_rx = None;
                    self.cancel_flag = None;
                }
                Ok(Err(err_msg)) => {
                    self.process_error = Some(err_msg);
                    self.status_text.clear();
                    self.is_running = false;
                    self.is_cancelling = false;
                    self.event_rx = None;
                    self.result_rx = None;
                    self.cancel_flag = None;
                }
                Err(TryRecvError::Empty) => {
                    ctx.request_repaint();
                }
                Err(TryRecvError::Disconnected) => {
                    self.process_error = Some("Proses berhenti secara tidak terduga.".to_string());
                    self.status_text.clear();
                    self.is_running = false;
                    self.is_cancelling = false;
                    self.event_rx = None;
                    self.result_rx = None;
                    self.cancel_flag = None;
                }
            }
        }
    }

    fn open_output_folder(&mut self) {
        if let Err(err) = Command::new("explorer")
            .arg(&self.settings.output_dir)
            .spawn()
        {
            self.process_error = Some(format!(
                "Gagal membuka folder hasil ({}): {}",
                self.settings.output_dir.display(),
                err
            ));
        }
    }

    fn render_folder_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Folder");
        ui.add_space(4.0);

        egui::Grid::new("folder_grid")
            .num_columns(3)
            .spacing([8.0, 8.0])
            .show(ui, |ui| {
                ui.label("Folder foto mentah:");
                let raw_text = self
                    .settings
                    .raw_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "Belum dipilih".to_string());
                ui.label(raw_text);

                if ui
                    .add_enabled(!self.is_running, egui::Button::new("Pilih…"))
                    .clicked()
                    && let Some(picked) = rfd::FileDialog::new().pick_folder()
                {
                    self.settings.raw_dir = Some(picked.clone());
                    self.start_preview_scan(picked);
                }
                ui.end_row();

                ui.label("Folder hasil:");
                ui.label(self.settings.output_dir.display().to_string());
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!self.is_running, egui::Button::new("Pilih…"))
                        .clicked()
                        && let Some(picked) = rfd::FileDialog::new().pick_folder()
                    {
                        self.settings.output_dir = picked;
                        self.asset_error = check_assets();
                    }
                    if ui.button("Buka Folder").clicked() {
                        self.open_output_folder();
                    }
                });
                ui.end_row();
            });

        ui.add_space(6.0);
        match &self.preview {
            PreviewState::None => {}
            PreviewState::Loading(_) => {
                ui.label("Membaca folder…");
            }
            PreviewState::Error(err) => {
                render_error_box(ui, err);
            }
            PreviewState::Ready(preview) => {
                ui.strong(format_preview_summary(
                    preview.class_count,
                    preview.total_photos,
                ));

                egui::CollapsingHeader::new("Rencana pasangan per kelas")
                    .default_open(false)
                    .show(ui, |ui| {
                        for cls in &preview.classes {
                            ui.label(format_class_pairing_line(
                                &cls.label,
                                cls.photo_count,
                                &cls.pairing_summary,
                            ));
                            for sk in &cls.skipped {
                                ui.label(format!(
                                    "  - [{}] {}: {}",
                                    cls.label, sk.filename, sk.reason
                                ));
                            }
                        }
                    });
            }
        }
    }

    fn render_settings_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Pengaturan");
        ui.add_space(4.0);

        ui.add_enabled_ui(!self.is_running, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.settings.replace_bg, "Ganti warna latar");
                ui.add_enabled_ui(self.settings.replace_bg, |ui| {
                    ui.color_edit_button_srgb(&mut self.settings.bg_color);
                });
            });

            ui.add_space(4.0);
            egui::CollapsingHeader::new("Pengaturan lanjutan")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Posisi mata:");
                        ui.add(
                            egui::Slider::new(
                                &mut self.settings.eye_line_pct,
                                EYE_LINE_MIN_PCT..=EYE_LINE_MAX_PCT,
                            )
                            .suffix("%"),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Ruang di atas kepala:");
                        ui.add(
                            egui::Slider::new(
                                &mut self.settings.headroom_pct,
                                HEADROOM_MIN_PCT..=HEADROOM_MAX_PCT,
                            )
                            .suffix("%"),
                        );
                    });

                    if ui.button("Kembalikan ke awal").clicked() {
                        self.settings.reset_advanced();
                    }
                });
        });
    }

    fn render_action_and_status(&mut self, ui: &mut egui::Ui) {
        if let Some(err) = &self.asset_error {
            render_error_box(ui, err);
            ui.add_space(6.0);
        }

        if let Some(err) = &self.process_error {
            render_error_box(ui, err);
            ui.add_space(6.0);
        }

        let has_valid_preview = matches!(
            &self.preview,
            PreviewState::Ready(p) if p.total_photos > 0
        );
        let (btn_label, btn_enabled) = main_button_state(
            self.is_running,
            self.is_cancelling,
            has_valid_preview,
            self.asset_error.is_some(),
        );

        let button = egui::Button::new(btn_label).min_size(egui::vec2(140.0, 32.0));
        if ui.add_enabled(btn_enabled, button).clicked() {
            if self.is_running {
                if let Some(cancel) = &self.cancel_flag {
                    cancel.store(true, Ordering::Relaxed);
                }
                self.is_cancelling = true;
                self.status_text = "Membatalkan…".to_string();
            } else {
                self.start_process();
            }
        }

        if self.is_running || self.sheets_total > 0 {
            ui.add_space(8.0);
            let progress = if self.sheets_total > 0 {
                (self.sheets_done as f32 / self.sheets_total as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let progress_text = format_sheet_progress(self.sheets_done, self.sheets_total);
            ui.add(egui::ProgressBar::new(progress).text(progress_text));

            if !self.status_text.is_empty() {
                ui.label(&self.status_text);
            }
        }

        let mut open_result_clicked = false;
        if let Some(summary) = &self.last_summary {
            ui.add_space(8.0);
            ui.strong(format_finished_summary(summary));

            if !self.flagged_items.is_empty() {
                ui.add_space(4.0);
                ui.label("Daftar foto yang perlu dicek:");
                egui::ScrollArea::vertical()
                    .id_salt("flagged_scroll")
                    .max_height(140.0)
                    .show(ui, |ui| {
                        for item in &self.flagged_items {
                            ui.label(format!(
                                "[{}] {}: {}",
                                item.class,
                                item.file,
                                item.reasons.join("; ")
                            ));
                        }
                    });
            }

            ui.add_space(4.0);
            if ui.button("Buka Folder Hasil").clicked() {
                open_result_clicked = true;
            }
        }

        if open_result_clicked {
            self.open_output_folder();
        }

        ui.add_space(8.0);
        egui::CollapsingHeader::new("Catatan proses")
            .default_open(false)
            .show(ui, |ui| {
                if self.logs.is_empty() {
                    ui.label("Belum ada catatan.");
                } else {
                    egui::ScrollArea::vertical()
                        .id_salt("logs_scroll")
                        .max_height(160.0)
                        .show(ui, |ui| {
                            for line in &self.logs {
                                ui.label(line);
                            }
                        });
                }
            });
    }
}

fn render_error_box(ui: &mut egui::Ui, message: &str) {
    let bg = egui::Color32::from_rgb(255, 228, 228);
    let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 80, 80));
    let text_color = egui::Color32::from_rgb(140, 20, 20);

    egui::Frame::new()
        .fill(bg)
        .stroke(stroke)
        .inner_margin(egui::Margin::same(8))
        .corner_radius(egui::CornerRadius::same(4))
        .show(ui, |ui| {
            ui.colored_label(text_color, message);
        });
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.settings);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_preview(&ctx);
        self.poll_process(&ctx);

        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_folder_section(ui);
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);
                self.render_settings_section(ui);
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);
                self.render_action_and_status(ui);
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_format_preview_summary() {
        assert_eq!(format_preview_summary(3, 105), "3 kelas, 105 foto");
        assert_eq!(format_preview_summary(1, 20), "1 kelas, 20 foto");
        assert_eq!(format_preview_summary(0, 0), "0 kelas, 0 foto");
    }

    #[test]
    fn test_format_class_pairing_line() {
        assert_eq!(
            format_class_pairing_line("1B", 20, "1&2, 3&4"),
            "Kelas 1B: 20 foto, pasangan 4R: 1&2, 3&4"
        );
        assert_eq!(
            format_class_pairing_line("folder input", 0, ""),
            "Kelas folder input: 0 foto"
        );
    }

    #[test]
    fn test_class_display_label() {
        assert_eq!(class_display_label(""), "folder input");
        assert_eq!(class_display_label("X-1"), "X-1");
    }

    #[test]
    fn test_format_sheet_progress() {
        assert_eq!(format_sheet_progress(12, 18), "12 dari 18 lembar");
        assert_eq!(format_sheet_progress(0, 5), "0 dari 5 lembar");
    }

    #[test]
    fn test_format_finished_summary_completed_and_cancelled() {
        let completed = Summary {
            sheets: 18,
            photos_3x4: 35,
            flagged: 2,
            duration: Duration::from_secs(5),
            cancelled: false,
        };
        assert_eq!(
            format_finished_summary(&completed),
            "Selesai: 18 lembar 4R dan 35 foto 3x4. 2 foto perlu dicek."
        );

        let cancelled = Summary {
            sheets: 4,
            photos_3x4: 8,
            flagged: 1,
            duration: Duration::from_secs(2),
            cancelled: true,
        };
        assert_eq!(
            format_finished_summary(&cancelled),
            "Dibatalkan. Sebagian selesai: 4 lembar 4R dan 8 foto 3x4. 1 foto perlu dicek."
        );
    }

    #[test]
    fn test_slider_pct_to_fraction_and_reset() {
        let mut settings = Settings::default();
        assert_eq!(settings.eye_line_pct, 28);
        assert_eq!(settings.headroom_pct, 7);
        assert_eq!(settings.bg_color, config::DEFAULT_BG_COLOR_RGB);
        assert!(!settings.replace_bg);
        assert!((settings.eye_line_fraction() - 0.28).abs() < 1e-9);
        assert!((settings.headroom_fraction() - 0.07).abs() < 1e-9);

        settings.eye_line_pct = 35;
        settings.headroom_pct = 12;
        assert!((settings.eye_line_fraction() - 0.35).abs() < 1e-9);
        assert!((settings.headroom_fraction() - 0.12).abs() < 1e-9);

        settings.reset_advanced();
        assert_eq!(settings.eye_line_pct, 28);
        assert_eq!(settings.headroom_pct, 7);
    }

    #[test]
    fn test_main_button_state() {
        // Ready to start
        assert_eq!(
            main_button_state(false, false, true, false),
            ("Mulai Proses", true)
        );
        // Preview not ready
        assert_eq!(
            main_button_state(false, false, false, false),
            ("Mulai Proses", false)
        );
        // Asset error present
        assert_eq!(
            main_button_state(false, false, true, true),
            ("Mulai Proses", false)
        );
        // Running, cancel not clicked yet
        assert_eq!(
            main_button_state(true, false, true, false),
            ("Batalkan", true)
        );
        // Running, cancel already clicked (waiting for Finished)
        assert_eq!(
            main_button_state(true, true, true, false),
            ("Batalkan", false)
        );
    }
}
