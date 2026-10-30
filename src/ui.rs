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

/// Friendly Indonesian error message when hex color format is invalid.
pub const HEX_COLOR_ERROR_MSG: &str = "Gunakan format #RRGGBB, misalnya #BE1220";

/// Formats an RGB color array as an uppercase 6-digit hex string with `#` prefix (e.g. `[190, 18, 32]` -> `"#BE1220"`).
pub fn format_hex_color(rgb: [u8; 3]) -> String {
    format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2])
}

/// Validates and parses a hex color input string using `pipeline::parse_hex_color`.
///
/// Accepts 6-digit hex with or without `#`, case-insensitive.
/// Returns friendly Indonesian error message if invalid.
pub fn validate_hex_color(raw: &str) -> Result<[u8; 3], &'static str> {
    pipeline::parse_hex_color(raw).map_err(|_| HEX_COLOR_ERROR_MSG)
}

/// Joins flag reasons into one line, separated by `; ` (same as `PERLU_DICEK.txt`).
pub fn format_flag_reasons(reasons: &[String]) -> String {
    reasons.join("; ")
}

/// One row of the flagged photo table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlagRow {
    pub no: usize,
    pub class: String,
    pub file: String,
    pub reason: String,
}

/// Builds the table rows: sorted by class, then file name (ignoring case), numbered from 1.
fn flag_table_rows(items: &[FlaggedItem]) -> Vec<FlagRow> {
    let mut sorted: Vec<&FlaggedItem> = items.iter().collect();
    sorted.sort_by_key(|item| {
        (
            item.class.to_lowercase(),
            item.class.clone(),
            item.file.to_lowercase(),
            item.file.clone(),
        )
    });
    sorted
        .into_iter()
        .enumerate()
        .map(|(index, item)| FlagRow {
            no: index + 1,
            class: item.class.clone(),
            file: item.file.clone(),
            reason: format_flag_reasons(&item.reasons),
        })
        .collect()
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
        let label = class_dir.name.clone();
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
    bg_color_hex: String,
    bg_color_error: Option<String>,
}

impl Default for App {
    fn default() -> Self {
        Self::from_settings(Settings::default())
    }
}

impl App {
    /// Creates the application, restoring saved settings from `cc.storage` if available.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Cards are light, so widgets must use the light theme even if Windows is in dark mode.
        cc.egui_ctx.set_theme(egui::Theme::Light);
        cc.egui_ctx.set_fonts(load_fonts());
        cc.egui_ctx
            .style_mut_of(egui::Theme::Light, apply_light_style);

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
        let bg_color_hex = format_hex_color(settings.bg_color);
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
            bg_color_hex,
            bg_color_error: None,
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
        if self.settings.replace_bg {
            match validate_hex_color(&self.bg_color_hex) {
                Ok(rgb) => {
                    self.settings.bg_color = rgb;
                    self.bg_color_hex = format_hex_color(rgb);
                    self.bg_color_error = None;
                }
                Err(err) => {
                    self.bg_color_error = Some(err.to_string());
                    return;
                }
            }
        }
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
        card(ui, Some("Folder"), |ui| {
            let raw_text = self
                .settings
                .raw_dir
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "Belum dipilih".to_string());
            folder_row(ui, "Folder foto mentah:", &raw_text, |ui| {
                if ui
                    .add_enabled(!self.is_running, egui::Button::new("Pilih…"))
                    .clicked()
                    && let Some(picked) = rfd::FileDialog::new().pick_folder()
                {
                    self.settings.raw_dir = Some(picked.clone());
                    self.start_preview_scan(picked);
                }
            });

            let output_text = self.settings.output_dir.display().to_string();
            folder_row(ui, "Folder hasil:", &output_text, |ui| {
                // Right-to-left layout: the first button added is the rightmost one.
                if ui.button("Buka Folder").clicked() {
                    self.open_output_folder();
                }
                if ui
                    .add_enabled(!self.is_running, egui::Button::new("Pilih…"))
                    .clicked()
                    && let Some(picked) = rfd::FileDialog::new().pick_folder()
                {
                    self.settings.output_dir = picked;
                    self.asset_error = check_assets();
                }
            });

            match &self.preview {
                PreviewState::None => {}
                PreviewState::Loading(_) => {
                    ui.add_space(config::UI_SMALL_GAP);
                    ui.label(
                        egui::RichText::new("Membaca folder…")
                            .color(rgb(config::UI_TEXT_SECONDARY_RGB)),
                    );
                }
                PreviewState::Error(err) => {
                    ui.add_space(config::UI_SMALL_GAP);
                    render_error_box(ui, err);
                }
                PreviewState::Ready(preview) => {
                    ui.add_space(config::UI_SMALL_GAP);
                    ui.label(
                        egui::RichText::new(format_preview_summary(
                            preview.class_count,
                            preview.total_photos,
                        ))
                        .font(semibold_font(config::UI_BODY_TEXT_SIZE))
                        .color(rgb(config::UI_TEXT_PRIMARY_RGB)),
                    );

                    collapsing_panel(
                        ui,
                        "Rencana pasangan per kelas",
                        config::UI_BOX_RADIUS,
                        |ui| {
                            egui::ScrollArea::vertical()
                                .id_salt("preview_scroll")
                                .max_height(config::UI_PREVIEW_LIST_MAX_HEIGHT)
                                .auto_shrink([false, true])
                                .show(ui, |ui| {
                                    for cls in &preview.classes {
                                        wrapped_label(
                                            ui,
                                            egui::RichText::new(format_class_pairing_line(
                                                &cls.label,
                                                cls.photo_count,
                                                &cls.pairing_summary,
                                            ))
                                            .color(rgb(config::UI_TEXT_PRIMARY_RGB)),
                                        );
                                        for sk in &cls.skipped {
                                            wrapped_label(
                                                ui,
                                                egui::RichText::new(format!(
                                                    "  - [{}] {}: {}",
                                                    cls.label, sk.filename, sk.reason
                                                ))
                                                .color(rgb(config::UI_TEXT_SECONDARY_RGB)),
                                            );
                                        }
                                    }
                                });
                        },
                    );
                }
            }
        });
    }

    fn render_settings_section(&mut self, ui: &mut egui::Ui) {
        card(ui, Some("Pengaturan"), |ui| {
            ui.add_enabled_ui(!self.is_running, |ui| {
                ui.horizontal(|ui| {
                    let chk = ui.checkbox(&mut self.settings.replace_bg, "Ganti warna latar");
                    if chk.changed() && !self.settings.replace_bg {
                        self.bg_color_error = None;
                        self.bg_color_hex = format_hex_color(self.settings.bg_color);
                    }
                    ui.add_enabled_ui(self.settings.replace_bg, |ui| {
                        let color_res = ui.color_edit_button_srgb(&mut self.settings.bg_color);
                        if color_res.changed() {
                            self.bg_color_hex = format_hex_color(self.settings.bg_color);
                            self.bg_color_error = None;
                        }

                        let text_edit =
                            egui::TextEdit::singleline(&mut self.bg_color_hex).desired_width(75.0);
                        let text_res = ui.add(text_edit);
                        if text_res.changed() {
                            self.bg_color_error = None;
                        }

                        let enter_pressed =
                            text_res.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        let lost_focus = text_res.lost_focus();

                        if lost_focus || enter_pressed {
                            match validate_hex_color(&self.bg_color_hex) {
                                Ok(rgb) => {
                                    self.settings.bg_color = rgb;
                                    self.bg_color_hex = format_hex_color(rgb);
                                    self.bg_color_error = None;
                                }
                                Err(err_msg) => {
                                    self.bg_color_error = Some(err_msg.to_string());
                                }
                            }
                        }
                    });
                });

                if self.settings.replace_bg
                    && let Some(err) = &self.bg_color_error
                {
                    ui.add_space(config::UI_SMALL_GAP);
                    ui.colored_label(rgb(config::UI_ERROR_TEXT_RGB), err);
                }

                collapsing_panel(ui, "Pengaturan lanjutan", config::UI_BOX_RADIUS, |ui| {
                    ui.spacing_mut().slider_width = config::UI_SLIDER_WIDTH;
                    egui::Grid::new("advanced_grid")
                        .num_columns(2)
                        .min_col_width(config::UI_SETTINGS_LABEL_WIDTH)
                        .spacing([config::UI_ROW_GAP, config::UI_SMALL_GAP])
                        .show(ui, |ui| {
                            ui.label("Posisi mata:");
                            ui.add(
                                egui::Slider::new(
                                    &mut self.settings.eye_line_pct,
                                    EYE_LINE_MIN_PCT..=EYE_LINE_MAX_PCT,
                                )
                                .suffix("%"),
                            );
                            ui.end_row();

                            ui.label("Ruang di atas kepala:");
                            ui.add(
                                egui::Slider::new(
                                    &mut self.settings.headroom_pct,
                                    HEADROOM_MIN_PCT..=HEADROOM_MAX_PCT,
                                )
                                .suffix("%"),
                            );
                            ui.end_row();
                        });

                    ui.add_space(config::UI_SMALL_GAP);
                    if ui.button("Kembalikan ke awal").clicked() {
                        self.settings.reset_advanced();
                    }
                });
            });
        });
    }

    /// Status pill colors (text, background) derived from the current run state.
    fn status_tone(&self) -> ([u8; 3], [u8; 3]) {
        if self.is_running {
            return (config::UI_TEXT_SECONDARY_RGB, config::UI_HOVER_BG_RGB);
        }
        match &self.last_summary {
            Some(summary) if summary.cancelled => {
                (config::UI_WARNING_TEXT_RGB, config::UI_WARNING_BG_RGB)
            }
            Some(_) => (config::UI_SUCCESS_TEXT_RGB, config::UI_SUCCESS_BG_RGB),
            None => (config::UI_TEXT_SECONDARY_RGB, config::UI_HOVER_BG_RGB),
        }
    }

    fn render_process_card(&mut self, ui: &mut egui::Ui) {
        let mut main_clicked = false;
        let mut open_result_clicked = false;

        card(ui, None, |ui| {
            if let Some(err) = &self.asset_error {
                render_error_box(ui, err);
            }

            if let Some(err) = &self.process_error {
                render_error_box(ui, err);
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
            let show_progress = self.is_running || self.sheets_total > 0;

            ui.horizontal(|ui| {
                main_clicked = ui
                    .add_enabled(btn_enabled, main_button(btn_label, self.is_running))
                    .clicked();
                if show_progress && !self.status_text.is_empty() {
                    let (text_rgb, bg_rgb) = self.status_tone();
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        pill(
                            ui,
                            egui::RichText::new(&self.status_text)
                                .font(semibold_font(config::UI_BODY_TEXT_SIZE))
                                .color(rgb(text_rgb)),
                            bg_rgb,
                        );
                    });
                }
            });

            if show_progress {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    ui.label(
                        egui::RichText::new(format_sheet_progress(
                            self.sheets_done,
                            self.sheets_total,
                        ))
                        .color(rgb(config::UI_TEXT_SECONDARY_RGB)),
                    );
                });
                let progress = if self.sheets_total > 0 {
                    (self.sheets_done as f32 / self.sheets_total as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(progress)
                        .fill(rgb(config::UI_ACCENT_RGB))
                        .desired_height(config::UI_PROGRESS_HEIGHT),
                );
            }

            if let Some(summary) = &self.last_summary {
                ui.add_space(config::UI_SMALL_GAP);
                let (text_rgb, bg_rgb) = if !summary.cancelled && summary.flagged == 0 {
                    (config::UI_SUCCESS_TEXT_RGB, config::UI_SUCCESS_BG_RGB)
                } else {
                    (config::UI_WARNING_TEXT_RGB, config::UI_WARNING_BG_RGB)
                };
                egui::Frame::new()
                    .fill(rgb(bg_rgb))
                    .stroke(egui::Stroke::new(1.0, rgb(config::UI_CARD_BORDER_RGB)))
                    .corner_radius(egui::CornerRadius::same(config::UI_BOX_RADIUS))
                    .inner_margin(egui::Margin::same(config::UI_BOX_PADDING))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        egui::Sides::new().shrink_left().show(
                            ui,
                            |ui| {
                                wrapped_label(
                                    ui,
                                    egui::RichText::new(format_finished_summary(summary))
                                        .font(semibold_font(config::UI_BODY_TEXT_SIZE))
                                        .color(rgb(text_rgb)),
                                );
                            },
                            |ui| {
                                open_result_clicked = ui.button("Buka Folder Hasil").clicked();
                            },
                        );
                    });
            }
        });

        if main_clicked {
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

        if open_result_clicked {
            self.open_output_folder();
        }
    }

    fn render_flagged_card(&self, ui: &mut egui::Ui) {
        if self.last_summary.is_none() || self.flagged_items.is_empty() {
            return;
        }

        ui.add_space(config::UI_CARD_GAP);
        card(ui, Some("Daftar foto yang perlu dicek"), |ui| {
            egui::Frame::new()
                .stroke(egui::Stroke::new(1.0, rgb(config::UI_CARD_BORDER_RGB)))
                .corner_radius(egui::CornerRadius::same(config::UI_BOX_RADIUS))
                .show(ui, |ui| {
                    ui.set_min_width(ui.available_width());
                    egui::ScrollArea::vertical()
                        .id_salt("flagged_scroll")
                        .min_scrolled_height(config::FLAG_LIST_HEIGHT)
                        .max_height(config::FLAG_LIST_HEIGHT)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            render_flagged_table(ui, &flag_table_rows(&self.flagged_items));
                        });
                });
        });
    }

    fn render_log_card(&self, ui: &mut egui::Ui) {
        collapsing_panel(ui, "Catatan proses", config::UI_CARD_RADIUS, |ui| {
            if self.logs.is_empty() {
                ui.label(
                    egui::RichText::new("Belum ada catatan.")
                        .color(rgb(config::UI_TEXT_SECONDARY_RGB)),
                );
            } else {
                egui::ScrollArea::vertical()
                    .id_salt("logs_scroll")
                    .max_height(config::UI_LOG_MAX_HEIGHT)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        for line in &self.logs {
                            wrapped_label(
                                ui,
                                egui::RichText::new(line).color(rgb(config::UI_TEXT_SECONDARY_RGB)),
                            );
                        }
                    });
            }
        });
    }
}

/// Font family name registered by [`load_fonts`] for semibold text.
const SEMIBOLD_FAMILY: &str = "semibold";

/// Builds font definitions with Windows UI fonts in front of egui's defaults.
/// Missing font files are skipped, so egui's own fonts remain as fallback.
fn load_fonts() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    if let Ok(bytes) = std::fs::read(config::UI_FONT_REGULAR_PATH) {
        insert_font_front(&mut fonts, "segoeui", bytes, egui::FontFamily::Proportional);
    }

    // The semibold family always exists so `semibold_font` never refers to a missing family.
    let proportional = fonts
        .families
        .get(&egui::FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    let semibold = egui::FontFamily::Name(SEMIBOLD_FAMILY.into());
    fonts.families.insert(semibold.clone(), proportional);
    if let Ok(bytes) = std::fs::read(config::UI_FONT_SEMIBOLD_PATH) {
        insert_font_front(&mut fonts, "seguisb", bytes, semibold);
    }

    if let Ok(bytes) = std::fs::read(config::UI_FONT_MONO_PATH) {
        insert_font_front(&mut fonts, "consola", bytes, egui::FontFamily::Monospace);
    }
    fonts
}

fn insert_font_front(
    fonts: &mut egui::FontDefinitions,
    name: &str,
    bytes: Vec<u8>,
    family: egui::FontFamily,
) {
    fonts.font_data.insert(
        name.to_string(),
        Arc::new(egui::FontData::from_owned(bytes)),
    );
    fonts
        .families
        .entry(family)
        .or_default()
        .insert(0, name.to_string());
}

/// Light theme style: larger text, taller controls, white bordered buttons.
fn apply_light_style(style: &mut egui::Style) {
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::proportional(config::UI_BODY_TEXT_SIZE),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::proportional(config::UI_BODY_TEXT_SIZE),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::monospace(config::UI_MONO_TEXT_SIZE),
    );
    style.interaction.selectable_labels = false;

    style.spacing.item_spacing = egui::Vec2::from(config::UI_ITEM_SPACING);
    style.spacing.button_padding = egui::Vec2::from(config::UI_BUTTON_PADDING);
    style.spacing.interact_size.y = config::UI_CONTROL_HEIGHT;

    let visuals = &mut style.visuals;
    let primary = rgb(config::UI_TEXT_PRIMARY_RGB);
    let border = egui::Stroke::new(1.0, rgb(config::UI_CARD_BORDER_RGB));
    visuals.indent_has_left_vline = false;
    visuals.extreme_bg_color = rgb(config::UI_PRESSED_BG_RGB);
    visuals.widgets.noninteractive.fg_stroke.color = primary;
    visuals.widgets.noninteractive.bg_stroke = border;

    let widgets = &mut visuals.widgets;
    for (widget, fill) in [
        (&mut widgets.inactive, config::UI_CARD_BG_RGB),
        (&mut widgets.hovered, config::UI_HOVER_BG_RGB),
        (&mut widgets.active, config::UI_PRESSED_BG_RGB),
        (&mut widgets.open, config::UI_PRESSED_BG_RGB),
    ] {
        widget.weak_bg_fill = rgb(fill);
        widget.bg_stroke = border;
        widget.fg_stroke.color = primary;
        widget.corner_radius = egui::CornerRadius::same(config::UI_WIDGET_RADIUS);
        widget.expansion = 0.0;
    }
}

/// Converts an RGB constant from `config` to an egui color.
fn rgb(color: [u8; 3]) -> egui::Color32 {
    let [r, g, b] = color;
    egui::Color32::from_rgb(r, g, b)
}

/// Semibold font at the given size.
fn semibold_font(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name(SEMIBOLD_FAMILY.into()))
}

/// Card title text: semibold and in the accent color.
fn section_title(text: &str) -> egui::RichText {
    egui::RichText::new(text)
        .font(semibold_font(config::UI_SECTION_TITLE_SIZE))
        .color(rgb(config::UI_ACCENT_RGB))
}

/// Adds a label that always wraps instead of overflowing horizontally.
fn wrapped_label(ui: &mut egui::Ui, text: egui::RichText) {
    ui.add(egui::Label::new(text).wrap());
}

/// Draws a full-width white card with a border and an optional title with a divider.
fn card(ui: &mut egui::Ui, title: Option<&str>, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(rgb(config::UI_CARD_BG_RGB))
        .stroke(egui::Stroke::new(1.0, rgb(config::UI_CARD_BORDER_RGB)))
        .corner_radius(egui::CornerRadius::same(config::UI_CARD_RADIUS))
        .inner_margin(egui::Margin::same(config::UI_CARD_PADDING))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            if let Some(title) = title {
                ui.label(section_title(title));
                ui.add(egui::Separator::default().spacing(config::UI_ROW_GAP));
            }
            add_contents(ui);
        });
}

/// Full-width bordered panel with a collapsible header, closed by default.
fn collapsing_panel(
    ui: &mut egui::Ui,
    title: &str,
    corner_radius: u8,
    add_body: impl FnOnce(&mut egui::Ui),
) {
    egui::Frame::new()
        .fill(rgb(config::UI_CARD_BG_RGB))
        .stroke(egui::Stroke::new(1.0, rgb(config::UI_CARD_BORDER_RGB)))
        .corner_radius(egui::CornerRadius::same(corner_radius))
        .inner_margin(egui::Margin::symmetric(config::UI_BOX_PADDING, 0))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            egui::CollapsingHeader::new(
                egui::RichText::new(title).color(rgb(config::UI_TEXT_PRIMARY_RGB)),
            )
            .default_open(false)
            .show(ui, |ui| {
                add_body(ui);
                ui.add_space(config::UI_ROW_GAP);
            });
        });
}

/// Draws a full-width colored message box.
fn message_box(ui: &mut egui::Ui, text: egui::RichText, bg_rgb: [u8; 3]) {
    egui::Frame::new()
        .fill(rgb(bg_rgb))
        .corner_radius(egui::CornerRadius::same(config::UI_BOX_RADIUS))
        .inner_margin(egui::Margin::same(config::UI_BOX_PADDING))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            wrapped_label(ui, text);
        });
}

fn render_error_box(ui: &mut egui::Ui, message: &str) {
    message_box(
        ui,
        egui::RichText::new(message).color(rgb(config::UI_ERROR_TEXT_RGB)),
        config::UI_ERROR_BG_RGB,
    );
}

/// Small rounded label used for the status and for flagged reasons.
fn pill(ui: &mut egui::Ui, text: egui::RichText, bg_rgb: [u8; 3]) {
    let [pad_x, pad_y] = config::UI_CHIP_PADDING;
    egui::Frame::new()
        .fill(rgb(bg_rgb))
        .corner_radius(egui::CornerRadius::same(config::UI_PILL_RADIUS))
        .inner_margin(egui::Margin::symmetric(pad_x, pad_y))
        .show(ui, |ui| {
            wrapped_label(ui, text);
        });
}

/// Main action button: filled accent for "Mulai Proses", outlined for "Batalkan".
fn main_button(label: &'static str, is_running: bool) -> egui::Button<'static> {
    let text = egui::RichText::new(label).font(semibold_font(config::UI_BODY_TEXT_SIZE));
    let button = if is_running {
        egui::Button::new(text.color(rgb(config::UI_ACCENT_RGB)))
            .fill(rgb(config::UI_CARD_BG_RGB))
            .stroke(egui::Stroke::new(1.0, rgb(config::UI_CARD_BORDER_RGB)))
    } else {
        egui::Button::new(text.color(rgb(config::UI_ON_ACCENT_TEXT_RGB)))
            .fill(rgb(config::UI_ACCENT_RGB))
            .stroke(egui::Stroke::NONE)
    };
    button.min_size(egui::Vec2::from(config::UI_PRIMARY_BUTTON_SIZE))
}

/// One folder row: fixed-width label, path box filling the middle, buttons on the right.
/// `add_buttons` runs in a right-to-left layout.
fn folder_row(ui: &mut egui::Ui, label: &str, path: &str, add_buttons: impl FnOnce(&mut egui::Ui)) {
    // Top-aligned so a wrapped (multi-line) path grows downward without overlapping.
    ui.horizontal_top(|ui| {
        let label_size = egui::vec2(config::UI_FOLDER_LABEL_WIDTH, config::UI_CONTROL_HEIGHT);
        ui.allocate_ui_with_layout(
            label_size,
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_min_size(label_size);
                wrapped_label(
                    ui,
                    egui::RichText::new(label).color(rgb(config::UI_TEXT_PRIMARY_RGB)),
                );
            },
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            add_buttons(ui);
            path_box(ui, path);
        });
    });
}

/// Input-like box showing a folder path in monospace, filling the remaining width.
fn path_box(ui: &mut egui::Ui, path: &str) {
    let padding = config::UI_PATH_BOX_PADDING;
    egui::Frame::new()
        .fill(rgb(config::UI_APP_BG_RGB))
        .stroke(egui::Stroke::new(1.0, rgb(config::UI_CARD_BORDER_RGB)))
        .corner_radius(egui::CornerRadius::same(config::UI_WIDGET_RADIUS))
        .inner_margin(egui::Margin::same(padding))
        .show(ui, |ui| {
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_min_width(ui.available_width());
                // Subtract padding and the 1 px border on both sides to match control height.
                ui.set_min_height(config::UI_CONTROL_HEIGHT - 2.0 * f32::from(padding) - 2.0);
                wrapped_label(
                    ui,
                    egui::RichText::new(path)
                        .monospace()
                        .color(rgb(config::UI_TEXT_PRIMARY_RGB)),
                );
            });
        });
}

/// Width of the widest text among `texts` (and the header) when laid out on one line.
fn max_text_width<'a>(
    ui: &egui::Ui,
    header: (&str, egui::FontId),
    texts: impl Iterator<Item = &'a str>,
    font: egui::FontId,
) -> f32 {
    let measure = |text: &str, font: egui::FontId| {
        ui.painter()
            .layout_no_wrap(text.to_string(), font, egui::Color32::PLACEHOLDER)
            .size()
            .x
    };
    texts
        .map(|text| measure(text, font.clone()))
        .fold(measure(header.0, header.1), f32::max)
}

/// Flagged photo table: No., Kelas, Nama File, Keterangan.
/// The first three columns fit their content; "Keterangan" gets the remaining width
/// and wraps, so the table never grows wider than the available space.
fn render_flagged_table(ui: &mut egui::Ui, rows: &[FlagRow]) {
    let header_font = semibold_font(config::UI_BODY_TEXT_SIZE);
    let body_font = egui::TextStyle::Body.resolve(ui.style());
    let mono_font = egui::TextStyle::Monospace.resolve(ui.style());

    // the last row has the highest number, so it is the widest
    let last_no = rows
        .last()
        .map(|row| row.no.to_string())
        .unwrap_or_default();
    let no_width = max_text_width(
        ui,
        ("No.", header_font.clone()),
        std::iter::once(last_no.as_str()),
        body_font.clone(),
    );
    let class_width = max_text_width(
        ui,
        ("Kelas", header_font.clone()),
        rows.iter().map(|row| row.class.as_str()),
        body_font,
    );
    let file_width = max_text_width(
        ui,
        ("Nama File", header_font.clone()),
        rows.iter().map(|row| row.file.as_str()),
        mono_font,
    );
    let spacing = ui.spacing().item_spacing.x;
    let reason_width = (ui.available_width() - no_width - class_width - file_width - 3.0 * spacing)
        .max(config::FLAG_REASON_MIN_WIDTH);

    egui::Grid::new("flagged_table")
        .num_columns(4)
        .striped(true)
        .show(ui, |ui| {
            for title in ["No.", "Kelas", "Nama File", "Keterangan"] {
                ui.label(
                    egui::RichText::new(title)
                        .font(header_font.clone())
                        .color(rgb(config::UI_TEXT_PRIMARY_RGB)),
                );
            }
            ui.end_row();

            for row in rows {
                ui.label(
                    egui::RichText::new(row.no.to_string()).color(rgb(config::UI_TEXT_PRIMARY_RGB)),
                );
                ui.label(egui::RichText::new(&row.class).color(rgb(config::UI_TEXT_PRIMARY_RGB)));
                ui.label(
                    egui::RichText::new(&row.file)
                        .monospace()
                        .color(rgb(config::UI_TEXT_PRIMARY_RGB)),
                );
                ui.scope(|ui| {
                    ui.set_max_width(reason_width);
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&row.reason)
                                .color(rgb(config::UI_WARNING_TEXT_RGB)),
                        )
                        .wrap(),
                    );
                });
                ui.end_row();
            }
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

        let page_frame = egui::Frame::new()
            .fill(rgb(config::UI_APP_BG_RGB))
            .inner_margin(egui::Margin::same(config::UI_PAGE_MARGIN));
        egui::CentralPanel::default()
            .frame(page_frame)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // Center a column of at most UI_CONTENT_MAX_WIDTH.
                        let full_width = ui.available_width();
                        let width = full_width.min(config::UI_CONTENT_MAX_WIDTH);
                        ui.horizontal_top(|ui| {
                            ui.add_space(((full_width - width) / 2.0).max(0.0));
                            ui.vertical(|ui| {
                                ui.set_width(width);
                                self.render_folder_section(ui);
                                ui.add_space(config::UI_CARD_GAP);
                                self.render_settings_section(ui);
                                ui.add_space(config::UI_CARD_GAP);
                                self.render_process_card(ui);
                                self.render_flagged_card(ui);
                                ui.add_space(config::UI_CARD_GAP);
                                self.render_log_card(ui);
                            });
                        });
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
        assert_eq!(format_class_pairing_line("raw", 0, ""), "Kelas raw: 0 foto");
    }

    fn flagged(class: &str, file: &str, reasons: &[&str]) -> FlaggedItem {
        FlaggedItem {
            class: class.to_string(),
            file: file.to_string(),
            reasons: reasons.iter().map(|r| r.to_string()).collect(),
        }
    }

    fn sample_flagged() -> Vec<FlaggedItem> {
        vec![
            flagged("X-8", "02_Budi.jpg", &["a"]),
            flagged("X8", "03_Citra.jpg", &["b", "c"]),
            flagged("X-8", "01_andi.jpg", &["d"]),
            flagged("x-1", "05_Eka.jpg", &["e"]),
        ]
    }

    #[test]
    fn test_flag_table_rows_sorted_by_class_then_file() {
        let rows = flag_table_rows(&sample_flagged());
        let order: Vec<(&str, &str)> = rows
            .iter()
            .map(|r| (r.class.as_str(), r.file.as_str()))
            .collect();
        assert_eq!(
            order,
            vec![
                ("x-1", "05_Eka.jpg"),
                ("X-8", "01_andi.jpg"),
                ("X-8", "02_Budi.jpg"),
                ("X8", "03_Citra.jpg"),
            ]
        );
    }

    #[test]
    fn test_flag_table_rows_numbered_across_classes() {
        let numbers: Vec<usize> = flag_table_rows(&sample_flagged())
            .iter()
            .map(|r| r.no)
            .collect();
        assert_eq!(numbers, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_flag_table_rows_joins_reasons() {
        let rows = flag_table_rows(&sample_flagged());
        let reason = rows
            .iter()
            .find(|r| r.file == "03_Citra.jpg")
            .map(|r| r.reason.as_str());
        assert_eq!(reason, Some("b; c"));
        assert!(flag_table_rows(&[]).is_empty());
    }

    #[test]
    fn test_format_flag_reasons() {
        let reasons = vec![
            "logo tidak terdeteksi".to_string(),
            "kepala miring".to_string(),
        ];
        assert_eq!(
            format_flag_reasons(&reasons),
            "logo tidak terdeteksi; kepala miring"
        );
        assert_eq!(format_flag_reasons(&[]), "");
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

    #[test]
    fn test_format_hex_color() {
        assert_eq!(format_hex_color([190, 18, 32]), "#BE1220");
        assert_eq!(format_hex_color([31, 79, 191]), "#1F4FBF");
        assert_eq!(format_hex_color([0, 0, 0]), "#000000");
        assert_eq!(format_hex_color([255, 255, 255]), "#FFFFFF");
        assert_eq!(format_hex_color([10, 11, 12]), "#0A0B0C");
    }

    #[test]
    fn test_validate_hex_color_valid() {
        // With hash, uppercase and lowercase
        assert_eq!(validate_hex_color("#BE1220"), Ok([190, 18, 32]));
        assert_eq!(validate_hex_color("#be1220"), Ok([190, 18, 32]));
        assert_eq!(validate_hex_color("#1F4FBF"), Ok([31, 79, 191]));
        assert_eq!(validate_hex_color("#1f4fbf"), Ok([31, 79, 191]));

        // Without hash, uppercase and lowercase
        assert_eq!(validate_hex_color("BE1220"), Ok([190, 18, 32]));
        assert_eq!(validate_hex_color("be1220"), Ok([190, 18, 32]));
        assert_eq!(validate_hex_color("1F4FBF"), Ok([31, 79, 191]));
        assert_eq!(validate_hex_color("1f4fbf"), Ok([31, 79, 191]));

        // With surrounding whitespace
        assert_eq!(validate_hex_color("  #1F4FBF  "), Ok([31, 79, 191]));
    }

    #[test]
    fn test_validate_hex_color_invalid() {
        // Empty or whitespace only
        assert_eq!(validate_hex_color(""), Err(HEX_COLOR_ERROR_MSG));
        assert_eq!(validate_hex_color("   "), Err(HEX_COLOR_ERROR_MSG));

        // Wrong length (too short)
        assert_eq!(validate_hex_color("#12345"), Err(HEX_COLOR_ERROR_MSG));
        assert_eq!(validate_hex_color("12345"), Err(HEX_COLOR_ERROR_MSG));
        assert_eq!(validate_hex_color("#"), Err(HEX_COLOR_ERROR_MSG));

        // Wrong length (too long)
        assert_eq!(validate_hex_color("#1234567"), Err(HEX_COLOR_ERROR_MSG));
        assert_eq!(validate_hex_color("1234567"), Err(HEX_COLOR_ERROR_MSG));
        assert_eq!(validate_hex_color("##123456"), Err(HEX_COLOR_ERROR_MSG));

        // Non-hex characters
        assert_eq!(validate_hex_color("xyz"), Err(HEX_COLOR_ERROR_MSG));
        assert_eq!(validate_hex_color("#xyz123"), Err(HEX_COLOR_ERROR_MSG));
        assert_eq!(validate_hex_color("#GGGGGG"), Err(HEX_COLOR_ERROR_MSG));
    }
}
