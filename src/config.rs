//! Global configuration constants.

// --- Framing constants (PRD 3.4) ---
/// Headroom above crown (7% of crop height).
pub const HEADROOM_DEFAULT: f64 = 0.07;
/// Eye line position from top (28% of crop height).
pub const EYE_LINE_DEFAULT: f64 = 0.28;
/// Interocular distance fraction of crop height.
pub const IOD_FRAC: f64 = 0.147;
/// Margin around detected chest logo.
pub const LOGO_MARGIN: f64 = 0.15;
/// Standard 3x4 pasfoto aspect ratio (width / height).
pub const RATIO: f64 = 3.0 / 4.0;

/// Minimum red channel value for red studio background.
pub const RED_BG_R_MIN: i16 = 100;
/// Minimum (r - g) difference for red studio background.
pub const RED_BG_R_MINUS_G_MIN: i16 = 60;
/// Minimum (r - b) difference for red studio background.
pub const RED_BG_R_MINUS_B_MIN: i16 = 45;

/// Morphological open radius (9x9 box, LInf norm).
pub const MORPH_OPEN_RADIUS: u32 = 4;

/// Minimum saturation for chest logo in HSV space (OpenCV u8 scale).
pub const LOGO_S_MIN: u8 = 90;
/// Minimum value for chest logo in HSV space (OpenCV u8 scale).
pub const LOGO_V_MIN: u8 = 50;

/// Number of sample points across face width for silhouette detection.
pub const HEAD_SILHOUETTE_SAMPLES: usize = 12;
/// Minimum required silhouette boundary points before falling back to face center.
pub const MIN_HEAD_POINTS: usize = 4;

/// Minimum zoom factor relative to ideal before flagging manual check.
pub const ZOOM_MIN_FACTOR: f64 = 0.85;
/// Maximum zoom factor relative to ideal before flagging manual check.
pub const ZOOM_MAX_FACTOR: f64 = 1.45;
/// Maximum border replication extension ratio before flagging manual check.
pub const MAX_EXTENSION_RATIO: f64 = 0.06;
/// Maximum allowed banner coverage of chest logo before flagging manual check.
pub const BANNER_COVER_THRESHOLD: f64 = 0.25;
/// Default estimated bottom banner coverage ratio.
pub const BANNER_COVER_DEFAULT: f64 = 0.10;

// --- Background color (PRD 3.5, 3.8) ---
/// Default studio red background replacement color in hex.
pub const DEFAULT_BG_COLOR_HEX: &str = "#BE1220";
/// Default studio red background replacement color in RGB.
pub const DEFAULT_BG_COLOR_RGB: [u8; 3] = [190, 18, 32];

// --- Chroma key constants (PRD 3.5) ---
/// Minimum red channel value for hard chroma background.
pub const CHROMA_R_MIN: f32 = 60.0;
/// Maximum ratio q = max(g,b)/max(r,1) for hard chroma background.
pub const CHROMA_Q_MAX_HARD: f32 = 0.30;
/// Minimum dilation radius for edge transition band.
pub const CHROMA_EDGE_MIN_RADIUS: u32 = 2;
/// Height scaling factor for edge transition band dilation.
pub const CHROMA_EDGE_RADIUS_FACTOR: f32 = 0.0012;
/// Upper bound of q for alpha ramp transition.
pub const CHROMA_ALPHA_UPPER_Q: f32 = 0.50;
/// Divisor for alpha ramp transition: (0.50 - q) / 0.20.
pub const CHROMA_ALPHA_RANGE_Q: f32 = 0.20;
/// Minimum Gaussian blur sigma for alpha matte smoothing.
pub const CHROMA_BLUR_SIGMA_MIN: f32 = 0.8;
/// Height scaling factor for alpha blur sigma.
pub const CHROMA_BLUR_SIGMA_FACTOR: f32 = 0.0004;

// --- 4R sheet layout constants (PRD 3.6, 3.7) ---
/// 4R print canvas width in pixels.
pub const CANVAS_4R_WIDTH: u32 = 1800;
/// 4R print canvas height in pixels.
pub const CANVAS_4R_HEIGHT: u32 = 1200;
/// Photo slot width in 4R sheet.
pub const SLOT_WIDTH: u32 = 335;
/// Photo slot height in 4R sheet.
pub const SLOT_HEIGHT: u32 = 445;
/// X coordinates for the 4 photo columns on the 4R sheet.
pub const SLOT_COLS_X: [u32; 4] = [35, 400, 767, 1133];
/// Y coordinate for row 1 photo slots (first student).
pub const SLOT_ROW1_Y: u32 = 35;
/// Y coordinate for row 2 photo slots (second student).
pub const SLOT_ROW2_Y: u32 = 523;

/// Top-left position (x, y) for row 1 label text.
pub const LABEL_ROW1_POS: (i32, i32) = (1533, 200);
/// Top-left position (x, y) for row 2 label text.
pub const LABEL_ROW2_POS: (i32, i32) = (1533, 678);
/// Dark red RGB color for label text.
pub const LABEL_TEXT_COLOR: [u8; 3] = [139, 0, 0];
/// Label font size in points/em (matches PIL size 60).
pub const LABEL_FONT_SIZE_EM: f32 = 60.0;
/// Fallback font design units per em when not provided by font metadata (used to scale font size to match PIL 60px).
pub const DEFAULT_UNITS_PER_EM: f32 = 2048.0;

/// Output JPEG compression quality (100 = maximum).
pub const JPEG_QUALITY: u8 = 100;
/// Target DPI metadata for output JPEG images.
pub const OUTPUT_DPI: u32 = 300;

// --- Asset paths and filenames (PRD 3.1, 3.7) ---
/// Template directory name.
pub const DIR_TEMPLATE: &str = "Template";
/// Models directory name.
pub const DIR_MODELS: &str = "models";
/// Mandatory template overlay PNG filename.
pub const FILE_TEMPLATE_PNG: &str = "Templates.png";
/// Optional banner overlay PNG filename.
pub const FILE_OVERLAY_PNG: &str = "overlay.png";
/// Optional label font TTF filename.
pub const FILE_FONT_TTF: &str = "font.ttf";
/// Fallback Windows font path if custom font is missing.
pub const FALLBACK_FONT_PATH: &str = r"C:\Windows\Fonts\arialbd.ttf";
/// Output directory for individual 3x4 cropped photos.
pub const DIR_OUTPUT_3X4: &str = "Output_3x4";
/// Output directory for 4R printable sheets.
pub const DIR_OUTPUT_4R: &str = "Output_4R";
/// Summary file listing photos flagged for manual review.
pub const FILE_PERLU_DICEK: &str = "PERLU_DICEK.txt";

// --- Face detection model constants (PRD 3.1, 3.3) ---
/// YuNet square input dimension in pixels (640x640).
pub const MODEL_INPUT_SIZE: usize = 640;
/// Minimum confidence score threshold for face detection.
pub const MODEL_SCORE_THRESHOLD: f32 = 0.6;
/// Minimum interocular distance in pixels required for valid face detection.
pub const MODEL_MIN_IOD: f32 = 10.0;
/// Feature pyramid strides for YuNet detection heads.
pub const MODEL_STRIDES: [usize; 3] = [8, 16, 32];
/// YuNet ONNX model filename.
pub const FILE_YUNET_ONNX: &str = "face_detection_yunet_2023mar.onnx";
/// Expected SHA-256 hash of the YuNet ONNX model file.
pub const YUNET_SHA256: &str = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4";

// --- GUI window constants (PRD 3.8) ---
/// Initial application window width.
pub const UI_INIT_WIDTH: f32 = 900.0;
/// Initial application window height.
pub const UI_INIT_HEIGHT: f32 = 700.0;
/// Minimum allowed application window width.
pub const UI_MIN_WIDTH: f32 = 760.0;
/// Minimum allowed application window height.
pub const UI_MIN_HEIGHT: f32 = 600.0;
/// Application title displayed in the window header.
pub const APP_TITLE: &str = "Pas Foto Otomatis";

// --- GUI color palette (PRD 3.8, G1.10) ---
/// Application background color (#F5F6F8).
pub const UI_APP_BG_RGB: [u8; 3] = [245, 246, 248];
/// Card background color (#FFFFFF).
pub const UI_CARD_BG_RGB: [u8; 3] = [255, 255, 255];
/// Card border color (#E4E7EC).
pub const UI_CARD_BORDER_RGB: [u8; 3] = [228, 231, 236];
/// Primary text color (#1D2939).
pub const UI_TEXT_PRIMARY_RGB: [u8; 3] = [29, 41, 57];
/// Secondary text color (#667085).
pub const UI_TEXT_SECONDARY_RGB: [u8; 3] = [102, 112, 133];
/// Accent color for main button, section titles, and progress bar (#BE1220).
pub const UI_ACCENT_RGB: [u8; 3] = [190, 18, 32];
/// Text color on top of the accent color (#FFFFFF).
pub const UI_ON_ACCENT_TEXT_RGB: [u8; 3] = [255, 255, 255];
/// Success message text color (#067647).
pub const UI_SUCCESS_TEXT_RGB: [u8; 3] = [6, 118, 71];
/// Success message background color (#ECFDF3).
pub const UI_SUCCESS_BG_RGB: [u8; 3] = [236, 253, 243];
/// Warning message text color, used for photos that need checking (#B54708).
pub const UI_WARNING_TEXT_RGB: [u8; 3] = [181, 71, 8];
/// Warning message background color (#FFFAEB).
pub const UI_WARNING_BG_RGB: [u8; 3] = [255, 250, 235];
/// Error message text color, used for asset errors (#B42318).
pub const UI_ERROR_TEXT_RGB: [u8; 3] = [180, 35, 24];
/// Error message background color (#FEF3F2).
pub const UI_ERROR_BG_RGB: [u8; 3] = [254, 243, 242];
/// Widget background color on hover (#F2F4F7).
pub const UI_HOVER_BG_RGB: [u8; 3] = [242, 244, 247];
/// Widget background color while pressed (#EAECF0).
pub const UI_PRESSED_BG_RGB: [u8; 3] = [234, 236, 240];

// --- GUI fonts (PRD 3.8, G1.10) ---
/// Windows UI font for body text; egui default font is used if missing.
pub const UI_FONT_REGULAR_PATH: &str = r"C:\Windows\Fonts\segoeui.ttf";
/// Windows semibold UI font for titles and the main button.
pub const UI_FONT_SEMIBOLD_PATH: &str = r"C:\Windows\Fonts\seguisb.ttf";
/// Windows monospace font for paths and file names.
pub const UI_FONT_MONO_PATH: &str = r"C:\Windows\Fonts\consola.ttf";

// --- GUI spacing and sizes (PRD 3.8, G1.10) ---
/// Outer margin around the whole page in pixels.
pub const UI_PAGE_MARGIN: i8 = 16;
/// Inner padding of each card in pixels.
pub const UI_CARD_PADDING: i8 = 16;
/// Corner radius of each card in pixels.
pub const UI_CARD_RADIUS: u8 = 8;
/// Inner padding of message boxes and path boxes in pixels.
pub const UI_BOX_PADDING: i8 = 10;
/// Corner radius of message boxes and path boxes in pixels.
pub const UI_BOX_RADIUS: u8 = 6;
/// Vertical gap between cards.
pub const UI_CARD_GAP: f32 = 12.0;
/// Vertical gap between rows inside a card.
pub const UI_ROW_GAP: f32 = 8.0;
/// Small vertical gap between closely related elements.
pub const UI_SMALL_GAP: f32 = 4.0;
/// Maximum width of the centered content column.
pub const UI_CONTENT_MAX_WIDTH: f32 = 1120.0;
/// Font size of card titles.
pub const UI_SECTION_TITLE_SIZE: f32 = 15.0;
/// Font size of body and button text.
pub const UI_BODY_TEXT_SIZE: f32 = 14.0;
/// Font size of monospace text (paths and file names).
pub const UI_MONO_TEXT_SIZE: f32 = 13.0;
/// Height of buttons, path boxes, and other controls.
pub const UI_CONTROL_HEIGHT: f32 = 34.0;
/// Corner radius of buttons and other widgets in pixels.
pub const UI_WIDGET_RADIUS: u8 = 6;
/// Inner padding (x, y) of buttons.
pub const UI_BUTTON_PADDING: [f32; 2] = [12.0, 6.0];
/// Spacing (x, y) between widgets.
pub const UI_ITEM_SPACING: [f32; 2] = [10.0, 8.0];
/// Width of the folder label column.
pub const UI_FOLDER_LABEL_WIDTH: f32 = 140.0;
/// Inner padding of the folder path box in pixels.
pub const UI_PATH_BOX_PADDING: i8 = 8;
/// Width of the label column in advanced settings.
pub const UI_SETTINGS_LABEL_WIDTH: f32 = 160.0;
/// Width of the advanced settings sliders.
pub const UI_SLIDER_WIDTH: f32 = 220.0;
/// Minimum size (width, height) of the main action button.
pub const UI_PRIMARY_BUTTON_SIZE: [f32; 2] = [160.0, 38.0];
/// Height of the progress bar.
pub const UI_PROGRESS_HEIGHT: f32 = 8.0;
/// Inner padding (x, y) of status pills and reason chips in pixels.
pub const UI_CHIP_PADDING: [i8; 2] = [8, 3];
/// Corner radius of status pills and reason chips in pixels.
pub const UI_PILL_RADIUS: u8 = 12;
/// Maximum height of the pairing plan list before it scrolls.
pub const UI_PREVIEW_LIST_MAX_HEIGHT: f32 = 160.0;
/// Fixed height of the flagged photo list; it scrolls inside this height.
pub const FLAG_LIST_HEIGHT: f32 = 360.0;
/// Maximum height of the process log before it scrolls.
pub const UI_LOG_MAX_HEIGHT: f32 = 160.0;

// --- Parity comparison constants (PRD 3.3, G1.8) ---
/// Maximum allowed eye Y position difference in percentage points (1.5%).
pub const PARITY_MAX_EYE_Y_DIFF: f64 = 0.015;
/// Maximum allowed center X position difference in percentage points (1.5%).
pub const PARITY_MAX_CENTER_X_DIFF: f64 = 0.015;
/// Maximum allowed relative scale difference (8%).
pub const PARITY_MAX_SCALE_REL_DIFF: f64 = 0.08;
