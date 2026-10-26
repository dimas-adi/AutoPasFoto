# Pas Foto Otomatis

A Windows desktop tool that turns raw student portrait photos into print-ready sheets:

- **3x4 portraits**: each photo is cropped to a 3:4 passport frame, with the face positioned consistently (eye line and headroom are adjustable).
- **4R sheets**: two students per 4R sheet, placed on a school template with class name and student number labels.
- **Optional background replacement**: the red background can be replaced with a chosen color.
- **Flagging**: photos with problems (for example, the logo hidden by the banner, or a face too small) are listed in `PERLU_DICEK.txt` for manual review.

The application is written in pure Rust. Face detection uses the YuNet model through `tract-onnx`, so no Python, OpenCV, or ONNX Runtime is required at runtime. It ships as a single Windows `.exe` with a graphical interface (egui) and a command-line mode for developers.

> **Status:** in development. The core pipeline and interface are implemented. Equivalence testing against the original Python application is still in progress.

## Requirements

- Windows 10 or 11
- Rust stable (edition 2024) and Cargo, to build from source

## Project layout

```
.
├── src/                 Rust source code
│   ├── bin/parity.rs    Developer tool: compare 3x4 framing against Python output
│   ├── config.rs        All rules, constants, and colors
│   ├── pipeline.rs      Batch processing and CLI mode
│   └── ...
├── reference/app.py     Original Python application (behavior reference, read-only)
├── docs/                PRD and agent working rules
├── Template/            Template assets (not in version control, see below)
└── models/              YuNet model (not in version control, see below)
```

## Setup

### 1. Template assets

Place these files in a `Template/` folder next to the executable (or in the working directory when running from source):

| File | Required | Notes |
|---|---|---|
| `Templates.png` | Yes | Must be exactly 1800 x 1200 pixels, RGBA. Contains the 4R frame and school banner. |
| `overlay.png` | No | Drawn between the photo and the template. Resized to 1800 x 1200 if the size differs. |
| `font.ttf` | No | Font for the class and student number labels. If missing, `C:\Windows\Fonts\arialbd.ttf` is used. |

### 2. Face detection model

Place the YuNet model in a `models/` folder next to the executable:

```
models/face_detection_yunet_2023mar.onnx
```

Download it from the [opencv_zoo repository](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (`face_detection_yunet_2023mar.onnx`). Verify the file with:

- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- Size: 232,589 bytes

If the application cannot find these assets, it shows a message in the interface and disables the start button.

## Build

```powershell
cargo build --release
```

The executable is written to `target\release\PasFoto.exe`. In release builds, the console window is hidden.

## Usage

### Graphical interface

1. Double-click `PasFoto.exe`, or run it from the terminal.
2. Choose the **raw photo folder** (the input folder).
3. Check the preview: the number of classes and photos, the planned pairs per class, and any files that will be skipped with the reason.
4. Choose the **output folder** (defaults to the folder containing the executable).
5. Optionally, enable **Ganti warna latar** to replace the background color. The default is `#BE1220`.
6. Press **Mulai Proses**. Progress is shown per sheet. You can press **Batalkan** to stop after the current pair.
7. When finished, review the list of flagged photos and open the output folder.

Folder selections and settings are remembered between sessions.

### Command line (developer mode)

```powershell
PasFoto.exe --cli --input <input folder> --output <output folder> [--bg <#RRGGBB>]
```

Example:

```powershell
cargo run --release -- --cli --input testdata/raw --output target/uji
```

In CLI mode, progress is written to `log_cli.txt` inside the output folder, because release builds have no console window.

## Input folders

The input can be either:

- **A folder of photos**, where each file name starts with the student number, for example `2_Adreena Hana Khairunnisa.JPG`. Files that have no number at the start of the name are skipped. Files already processed (ending in `_3x4` or `_4R`) are skipped too.
- **A folder with class subfolders**, for example `Input_Raw\X-1\...`. Each subfolder is treated as one class.

When the input folder has no class subfolders, its name is used as the class name.

If several photos have the same number, the most recently modified file is used.

## Output

```
<output folder>/
├── Output_3x4/
│   └── <class>_3x4/
│       └── <photo name>_3x4.jpg          one portrait per student
├── Output_4R/
│   ├── <class>_4R/
│   │   └── <class>_Absen_<n1>-<n2>_4R.jpg   one 4R sheet, two students
│   └── PERLU_DICEK.txt                   list of photos to review manually (only if any)
└── log_cli.txt                           (CLI mode only)
```

- A 4R sheet holds two students. A sheet with only one student is named `<class>_Absen_<n1>_4R.jpg`.
- `PERLU_DICEK.txt` is created only when at least one photo is flagged, and removed when there are none.

## Developer tools

### Parity check

`parity` compares the 3x4 framing of the Rust output against the Python output. Photos are matched by file name. For each pair, the tool measures the eye line, horizontal center, and relative face scale, and reports `OK` or `BEDA` (different).

```powershell
cargo run --release --bin parity -- <3x4 folder from Python> <3x4 folder from Rust>
```

The tolerance thresholds are defined in `src/config.rs`.

### Tests and checks

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Some tests use photos from `testdata/`. When those files are not present, the tests are skipped with a message.

## Privacy

Student photos are not stored in this repository. `testdata/`, `Output_3x4/`, `Output_4R/`, and `target/` are excluded by `.gitignore`. Do not commit photos of students.

## Dependencies

- [eframe](https://crates.io/crates/eframe) / egui: graphical interface
- [rfd](https://crates.io/crates/rfd): folder selection dialogs
- [image](https://crates.io/crates/image), [imageproc](https://crates.io/crates/imageproc), [ab_glyph](https://crates.io/crates/ab_glyph), [fast_image_resize](https://crates.io/crates/fast_image_resize): image processing and text
- [tract-onnx](https://crates.io/crates/tract-onnx): YuNet face detection
- [rayon](https://crates.io/crates/rayon): parallel processing
- [anyhow](https://crates.io/crates/anyhow), [serde](https://crates.io/crates/serde): errors and saved settings
