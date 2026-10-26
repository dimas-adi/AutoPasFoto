# PRD — Pas Foto Otomatis (Rust) v1.0

**Status:** Aktif.
**Acuan perilaku:** `reference/app.py` (versi Python yang sudah dipakai produksi). Hanya dibaca.
**Aturan agent:** `AGENTS.md`.
**Branch kerja:** `main` (satu commit per tugas, di-commit developer).

---

## 1. Tujuan

Menulis ulang aplikasi pas foto dari Python ke **Rust murni** menjadi satu `.exe` Windows yang cepat dibuka, ringan, dan mudah dipakai oleh orang awam (operator studio). **Hasil foto harus sama** dengan versi Python yang sekarang sudah dianggap benar.

**Alur pengguna:**
1. Pilih folder foto mentah. Setiap subfolder adalah satu kelas, contoh `1B/`, `X-1/`.
2. Pilih folder hasil.
3. Tekan **Mulai Proses**.
4. Aplikasi membuat:
   - foto 3x4 per siswa;
   - lembar 4R berisi 2 siswa (masing-masing 4 foto) dengan label `Kelas / Absen`;
   - daftar foto yang perlu dicek manual.

**Di luar lingkup v1.0:** penghapus latar berbasis AI, editor manual, cetak langsung ke printer, versi macOS/Linux, auto-update, jaringan atau telemetri apa pun.

---

## 2. Keputusan (sudah final, jangan diubah agent)

| ID | Keputusan |
|---|---|
| K1 | **Rust murni.** Inferensi model memakai `tract-onnx`. Tanpa OpenCV, tanpa ONNX Runtime (`ort`), tanpa Python. |
| K2 | Deteksi wajah memakai **YuNet** (`face_detection_yunet_2023mar.onnx`, lisensi MIT), menggantikan MediaPipe. |
| K3 | Ganti warna latar memakai **chroma key** (latar studio selalu kain merah), menggantikan rembg/u2net. |
| K4 | Aset ada di **folder di samping .exe**: `Template/` dan `models/`. Tidak ditanam di .exe. |
| K5 | Perilaku crop, penomoran, dan susunan 4R **setara** dengan `reference/app.py`. Perbedaan hanya boleh karena detektor wajah berbeda (toleransi di G1.8). |
| K6 | Agent boleh menjalankan `cargo check/test/clippy/fmt/build`. Developer yang menjalankan GUI dan `git commit`. |
| K7 | Foto siswa untuk uji ada di `testdata/` dan **tidak di-commit** (privasi, foto anak di bawah umur). |

---

## 3. Spesifikasi Fungsional

### 3.1 Struktur folder distribusi
```
PasFoto/
  PasFoto.exe
  Template/Templates.png        (wajib, 1800x1200 RGBA)
  Template/overlay.png          (opsional, di-resize ke 1800x1200 jika ukurannya beda)
  Template/font.ttf             (opsional, font label; jika tidak ada pakai C:\Windows\Fonts\arialbd.ttf)
  models/face_detection_yunet_2023mar.onnx
```
- Folder aset dicari di **folder tempat .exe berada**, lalu di **current working directory** (supaya `cargo run` di root repo bekerja).
- Jika aset wajib tidak ada, tampilkan pesan ramah, contoh: `Folder "Template" tidak ditemukan. Pastikan folder Template ada di samping aplikasi.` Proses tidak dimulai.
- Model YuNet diunduh developer dari repo `opencv/opencv_zoo` (`models/face_detection_yunet/face_detection_yunet_2023mar.onnx`). SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` (232.589 byte).

### 3.2 Input dan penomoran (port `kumpulkan_foto` dari reference)
- **Folder input:**
  - jika punya subfolder, setiap subfolder adalah satu kelas (nama kelas = nama folder), diurutkan berdasarkan nama tanpa memperhatikan huruf besar/kecil;
  - jika tidak punya subfolder, folder itu sendiri diproses dengan nama kelas kosong.
- **File foto:** ekstensi `.jpg`, `.jpeg`, `.png` (huruf besar/kecil apa saja). Abaikan nama yang diawali `.` atau `~`.
- **Dilewati dan dicatat:**
  - nama yang berakhiran `_3x4` atau `_4R` (huruf besar/kecil apa saja), dengan alasan `dilewati: ini file hasil olahan, bukan foto mentah`;
  - nama yang tidak diawali angka, dengan alasan `dilewati: nama file tidak diawali nomor absen`.
- **Nomor absen** = angka di **awal** nama file (spasi di depan boleh). Label ditulis persis seperti di nama file: `04_...` menjadi label `04`, nilai urut 4.
- **Nomor dobel:** pakai file dengan waktu modifikasi paling baru. File lain dilewati dengan alasan `dilewati: nomor absen N dobel, yang dipakai <nama file>`.
- **Urutan:** berdasarkan nilai angka (1, 2, 3, …, 10, 20).
- **Pasangan:** berurutan dari daftar: [0,1], [2,3], … Jumlah ganjil berarti lembar terakhir berisi satu siswa.
- **Rencana pasangan dicatat** sebelum proses, contoh: `Kelas 1B: 20 foto, pasangan 4R: 1&2, 3&4, …`

### 3.3 Deteksi wajah (YuNet via tract)
- **Input model:** `input`, bentuk `[1,3,640,640]`, `f32`, urutan kanal **BGR**, nilai 0..255 (tanpa normalisasi), NCHW.
- **Praproses:**
  - foto di-resize supaya sisi terpanjang menjadi 640 (`s = 640 / max(W,H)`, filter area atau bilinear);
  - hasilnya ditempel di **pojok kiri atas** kanvas 640x640 berisi nol.
- **Output (12 tensor):** `cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32`. Untuk stride `st` ∈ {8,16,32}, `cols = 640/st`, dan jumlah titik `N = cols²`:
  - `cls_st`, `obj_st`: `[1,N,1]`; `bbox_st`: `[1,N,4]`; `kps_st`: `[1,N,10]`.
  - Indeks `i`: `row = i / cols`, `col = i % cols`.
  - `score = sqrt(clamp(cls,0,1) * clamp(obj,0,1))`.
  - Kotak: `cx = (col + b0)*st`, `cy = (row + b1)*st`, `w = exp(b2)*st`, `h = exp(b3)*st`.
  - Landmark `j` (0..4): `x = (col + k[2j])*st`, `y = (row + k[2j+1])*st`. Urutan landmark: 0 = mata kanan siswa (sisi **kiri** gambar), 1 = mata kiri siswa, 2 = hidung, 3 = mulut kanan, 4 = mulut kiri.
- **Pilih wajah:** ambil satu deteksi dengan score tertinggi dan score ≥ 0.6. NMS tidak wajib karena hanya satu wajah yang dipakai.
- **Kembali ke koordinat asli:** bagi dengan `s`. Ke framing diberikan `ex, ey` = titik tengah kedua mata, dan `iod = |x_mata0 − x_mata1|`.
- Jika tidak ada wajah atau `iod ≤ 10`: crop tengah (port cabang `else` di `process_smart_crop`), lalu catat alasan `wajah tidak terdeteksi`.
- Model dimuat **sekali**, dioptimasi (`into_optimized().into_runnable()`), lalu dipakai bersama oleh semua thread.
- **Nilai acuan uji** (`testdata/raw/35_Syalva.jpg`, 5472x3648, ruang 640; dihitung dengan OpenCV `FaceDetectorYN`, resize INTER_AREA). Toleransi ±3 px dan score ±0.03:
  - score 0.937;
  - kotak x,y,w,h = 277.1, 138.8, 82.2, 109.5;
  - mata0 (297.3, 183.9), mata1 (337.6, 183.7), hidung (319.3, 206.7), mulut kanan (302.4, 219.4), mulut kiri (337.7, 219.7).

### 3.4 Framing 3x4 (port `crop_pasfoto`, `cari_logo`, `hitung_tutup_bawah`)
- **Port 1:1** dari reference, termasuk ambang warna, urutan langkah, dan kondisi "perlu dicek". Aturan dan prioritasnya:
  1. kepala di tengah, berdasarkan garis tepi rambut atau hijab;
  2. logo di dada utuh dan tidak tertutup banner template;
  3. ruang di atas kepala tetap.
- **Konstanta** di `config.rs` (nilai dari reference):

| Konstanta | Nilai |
|---|---|
| `HEADROOM_DEFAULT` | 0.07 |
| `EYE_LINE_DEFAULT` | 0.28 |
| `IOD_FRAC` | 0.147 |
| `LOGO_MARGIN` | 0.15 |
| `RATIO` | 3/4 |

- **Pemetaan OpenCV/NumPy ke Rust** (wajib diikuti agar hasil setara):
  - Latar merah: `r>100 && r−g>60 && r−b>45` (i16).
  - `MORPH_OPEN` kernel kotak 9x9 = erode lalu dilate `Norm::LInf` radius 4 (`imageproc::morphology`).
  - `MORPH_CLOSE` kernel kxk = dilate lalu erode `Norm::LInf` radius `k/2`.
  - `connectedComponents` = konektivitas 8. "Latar tepi" = komponen latar yang menyentuh baris atas, kolom kiri, atau kolom kanan. **Baris bawah tidak dihitung.**
  - HSV OpenCV (u8): `V = max(r,g,b)`; `S = 0` jika `V = 0`, selain itu `255*(V − min)/V`. Logo: `S>90 && V>50 && bukan latar tepi`.
  - `np.linspace(a, b, 12).astype(int)` = 12 titik rata dari a sampai b (inklusif), dipotong ke bawah (trunc).
  - `round()` Python = pembulatan ke bilangan genap terdekat; dalam praktik `f64::round()` cukup.
  - `cv2.copyMakeBorder(..., BORDER_REPLICATE)` = piksel di luar gambar mengambil piksel tepi terdekat (clamp koordinat).
- **Alasan "perlu dicek"** dicatat sebagai teks, bukan sekadar bool. Satu foto bisa punya lebih dari satu alasan:

| Kondisi di reference | Teks alasan |
|---|---|
| `crown == 0` | `kepala menyentuh tepi atas foto mentah` |
| siluet kepala < 4 titik | `latar bukan merah, posisi pakai tengah wajah` |
| logo tidak ketemu | `logo di dada tidak terdeteksi` |
| logo tertutup banner > 1/4 | `logo tertutup banner, foto mentah terlalu mepet di bawah` |
| zoom < 0.85 atau > 1.45 × ideal | `ukuran wajah jauh dari ideal` |
| perpanjangan > 6% | `foto mentah terlalu mepet, latar diperpanjang` |

- **Banner:** `tutup_bawah` dihitung sekali per proses dari `overlay.png` dan `Templates.png` pada slot pertama (port `hitung_tutup_bawah`). Nilai yang diharapkan untuk template sekarang ≈ 0.10.

### 3.5 Ganti warna latar (chroma key, opsional)
Hanya dijalankan jika pengguna mencentang "Ganti warna latar". Dijalankan pada foto 3x4 hasil crop, sebelum disimpan. Warna default `#BE1220`.
1. `q = max(g,b) / max(r,1)` (f32).
2. Latar keras = `r>60 && q<0.30`, lalu hanya komponen (konektivitas 8) yang menyentuh baris atas, kolom kiri, atau kolom kanan. Baris bawah tidak dihitung, supaya dasi merah tidak ikut terhapus.
3. Pita tepi = dilate latar keras dengan `Norm::LInf` radius `max(2, round(h*0.0012))`.
4. Alpha:
   - latar keras: 1;
   - di dalam pita: `clamp((0.50 − q)/0.20, 0, 1)`;
   - selain itu: 0.
   
   Lalu Gaussian blur dengan sigma `max(0.8, h*0.0004)`.
5. Despill: `r' = r − max(0, r − max(g,b)) * alpha`.
6. Hasil = `fg*(1−alpha) + warna*alpha`, per kanal, dibulatkan ke u8.

Sudah diuji di prototipe: telinga, dasi, logo, hijab putih, dan hijab hitam tetap utuh.

### 3.6 Komposisi 4R (port bagian 4R di `_run_automation_engine`)
- **Kanvas:** 1800x1200 RGBA putih.
- **Slot:** ukuran 335x445. Kolom x = 35, 400, 767, 1133. Baris 1 y = 35 (siswa pertama), baris 2 y = 523 (siswa kedua).
- **Foto ke slot** (setara `ImageOps.fit(img, (335,445), LANCZOS, centering=(0.5, 0.0))`):
  - potong foto ke rasio 335:445 dengan posisi tengah horizontal dan rata atas (yang terbuang bagian bawah);
  - resize Lanczos3 ke 335x445.
- **Urutan tempel:** foto → `overlay.png` (alpha) → `Templates.png` (alpha) → teks.
- **Teks** (hanya jika nama kelas tidak kosong):
  - `"{kelas} / {absen}"` di (1533, 200) untuk siswa pertama dan (1533, 678) untuk siswa kedua;
  - warna (139,0,0);
  - posisi (x,y) = kiri atas kotak teks, seperti `ImageDraw.text` anchor `la`;
  - ukuran font setara PIL size 60, yaitu em 60 px. Di ab_glyph: `PxScale::from(60.0 * font.height_unscaled() / font.units_per_em().unwrap_or(2048.0))`.
- **Slot siswa kedua** kosong jika jumlah ganjil, atau jika foto gagal diproses.

### 3.7 Output
- **Foto 3x4:** `<folder hasil>/Output_3x4/<kelas>_3x4/<nama file tanpa ekstensi>_3x4.jpg`. Kelas kosong: langsung di `Output_3x4/`. Ukuran asli hasil crop, tidak di-resize.
- **Lembar 4R:** `<folder hasil>/Output_4R/<kelas>_4R/<kelas>_Absen_<a1>-<a2>_4R.jpg`, atau `<kelas>_Absen_<a1>_4R.jpg` jika sendiri. Kelas kosong: langsung di `Output_4R/`.
- **Format semua JPEG:** kualitas 100 dan DPI 300 (`JpegEncoder::new_with_quality` + `set_pixel_density(PixelDensity::dpi(300))`).
- **File baca:** orientasi EXIF diterapkan (`ImageDecoder::orientation()` + `DynamicImage::apply_orientation`). Foto PNG dengan alpha diubah ke RGB.
- **`Output_4R/PERLU_DICEK.txt`:**
  - berisi satu baris per foto: `[kelas] nama_file: alasan1; alasan2`, diurutkan per kelas lalu nama file;
  - baris pertama: `Foto berikut sebaiknya dicek manual:`;
  - jika tidak ada yang perlu dicek, file lama dihapus.
- **File yang sudah ada ditimpa.**

### 3.8 Tampilan (egui), bahasa Indonesia awam
- **Judul jendela:** `Pas Foto Otomatis`. Ukuran awal 900x700, minimum 760x600.
- **Bagian "Folder"**
  - `Folder foto mentah` dengan tombol `Pilih…`. Setelah dipilih, tampilkan ringkasan, contoh `3 kelas, 105 foto`, dan daftar rencana pasangan per kelas yang bisa dibuka-tutup.
  - `Folder hasil` dengan tombol `Pilih…` dan `Buka Folder`. Default: folder tempat aplikasi berada.
- **Bagian "Pengaturan"**
  - Centang `Ganti warna latar`, dengan pemilih warna yang aktif hanya jika dicentang.
  - Bagian lipat `Pengaturan lanjutan`:
    - slider `Posisi mata` 20–40% (default 28%);
    - slider `Ruang di atas kepala` 2–15% (default 7%);
    - tombol `Kembalikan ke awal`.
- **Tombol utama:** `Mulai Proses`. Saat berjalan berubah menjadi `Batalkan`.
- **Status:** progress bar, teks `12 dari 18 lembar`, dan kalimat status singkat.
- **Selesai:** ringkasan, contoh `Selesai: 18 lembar 4R dan 35 foto 3x4. 2 foto perlu dicek.`, lalu daftar foto perlu dicek beserta alasannya, dan tombol `Buka Folder Hasil`.
- **Bagian lipat "Catatan proses"** berisi log (tertutup secara default).
- Folder terakhir dan pengaturan **diingat** antar sesi (persistence eframe).
- **Dilarang:** teks teknis atau promosi (ONNX, tract, engine, offline, AI, versi build), emoji, dan jendela konsol di build release.

---

## 4. Spesifikasi Teknis

### 4.1 Dependency (versi stabil terbaru saat G1.0, tercatat di `Cargo.lock`)
| Crate | Fungsi |
|---|---|
| `eframe` (fitur `persistence`) | GUI |
| `rfd` | Dialog pilih folder |
| `image` (fitur `jpeg`, `png`) | Baca/tulis gambar, EXIF |
| `imageproc` | Morfologi, connected components, blur, teks |
| `ab_glyph` | Font. **Versi harus sama** dengan yang dipakai `imageproc` (cek `cargo tree -i ab_glyph`) |
| `fast_image_resize` (fitur `image`) | Resize cepat |
| `tract-onnx` | Inferensi YuNet |
| `rayon` | Paralel |
| `anyhow` | Error |
| `serde` (fitur `derive`) | Simpan pengaturan |
| `tempfile` (dev) | Test folder sementara |

### 4.2 Struktur kode
```
src/
  main.rs        // entry: GUI by default, `--cli` for developer testing; windows_subsystem in release
  lib.rs         // pub mod ...
  config.rs      // all constants (framing, thresholds, template layout, file names)
  assets.rs      // locate Template/ and models/, load template, overlay, font
  files.rs       // collect photos, absen parsing, dedupe, pairing
  imageio.rs     // load with EXIF orientation, save JPEG q100 300 dpi
  face.rs        // YuNet via tract: preprocess, run, decode
  framing.rs     // crop_pasfoto, find_logo, banner cover (pure functions)
  background.rs  // chroma key (pure function)
  compose.rs     // fit-to-slot, 4R canvas, text (pure function)
  pipeline.rs    // batch per class, rayon, progress events, cancel, PERLU_DICEK.txt
  ui.rs          // egui app
src/bin/parity.rs  // developer tool, G1.8
```

### 4.3 Kinerja dan stabilitas
- **Paralel per pasangan:** satu job = proses 2 foto, simpan 3x4, susun dan simpan 4R. Jumlah thread `min(jumlah core, 4)` supaya RAM aman (foto mentah 20 MP).
- **UI tidak pernah membeku.** Pipeline berjalan di thread lain dan mengirim event lewat `std::sync::mpsc`. UI memanggil `request_repaint`.
- **Batalkan:** `Arc<AtomicBool>` dicek sebelum setiap pasangan.
- **Satu foto gagal tidak menghentikan proses.** Foto itu dicatat dan slotnya dikosongkan.
- **Profil build:**
  - `[profile.release]`: `lto = "fat"`, `codegen-units = 1`, `strip = true`, `opt-level = 3`;
  - `[profile.dev]`: `opt-level = 1`, ditambah `[profile.dev.package."*"] opt-level = 3`, supaya uji CLI di mode debug tidak lambat.

### 4.4 Mode CLI (untuk developer, bukan untuk pengguna)
```
PasFoto.exe --cli --input <folder> --output <folder> [--bg <#RRGGBB>]
```
Menjalankan pipeline yang sama tanpa GUI. Log ditulis ke stdout. Exit code 0 jika selesai, walaupun ada foto yang perlu dicek.

---

## 5. Gerbang dan Tugas

**Aturan eksekusi:** 1 tugas = 1 chat baru = 1 model. Prompt dimulai dengan `/plan`. Revisi plan maksimal 1 kali. Developer meng-commit setelah verifikasi manual lolos.

**Persiapan developer (sebelum G1.0, manual):**
1. Buat repo baru `pasfoto-rs`, letakkan `AGENTS.md` dan `PRD.md` di root.
2. Salin `app.py` produksi ke `reference/app.py`.
3. Salin `Template/` (Templates.png, overlay.png) ke root repo.
4. Unduh model YuNet ke `models/` dan cek SHA-256 (bagian 3.1).
5. Isi `testdata/raw/` dengan foto mentah uji:
   - `35_Syalva.jpg` (wajib untuk test G1.1);
   - beberapa foto lain di subfolder kelas, contoh `testdata/raw/X-1/`.

---

### GERBANG 1: Implementasi

**Kriteria lolos G1:**
- Semua tugas G1.0 sampai G1.8 selesai.
- `cargo test` lulus dan `cargo clippy --all-targets -- -D warnings` bersih.
- Uji kesetaraan G1.8 lolos.
- Developer sudah memproses satu kelas nyata lewat GUI dan hasilnya diterima.

#### G1.0 — Kerangka Proyek
- **Rekomendasi Model:** Gemini Flash 3.8 (Effort: High)
- **File dibaca:** `AGENTS.md`, `PRD.md`
- **File dibuat:** `Cargo.toml`, `.gitignore`, `src/main.rs`, `src/lib.rs`, `src/config.rs`, serta modul kosong lain dari bagian 4.2 (cukup `//! <tujuan modul>`).
- **Spesifikasi:**
  1. Paket `pasfoto`, binary bernama `PasFoto`, edition 2024. Dependency dari bagian 4.1 ditambah dengan `cargo add`. Profil build dari bagian 4.3.
  2. `.gitignore`: `target/`, `testdata/`, `Output_3x4/`, `Output_4R/`.
  3. `config.rs` berisi semua konstanta di bagian 3.2 sampai 3.7: framing, ambang warna, chroma key, layout 4R, nama folder dan file aset, ukuran input model 640, ambang score 0.6.
  4. `main.rs`:
     - `#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]`;
     - argumen `--cli` diteruskan ke stub `pipeline::run_cli` yang untuk sementara mengembalikan `Ok(())`;
     - tanpa argumen, membuka jendela eframe kosong berjudul `Pas Foto Otomatis`.
  5. Pastikan versi `ab_glyph` sama dengan yang dipakai `imageproc`.
- **Selesai jika:** `cargo build` dan `cargo clippy --all-targets -- -D warnings` bersih.
- **Verifikasi manual:** `cargo run` membuka jendela kosong berjudul benar.
- **Pesan commit:** `chore: scaffold rust project with dependencies and config`

#### G1.1 — Deteksi Wajah YuNet
- **Rekomendasi Model:** Gemini Argon 4 (Effort: High)
- **Prasyarat:** G1.0
- **File dibaca:** `PRD.md` bagian 3.3, `src/config.rs`
- **File dibuat/diedit:** `src/face.rs`, `src/imageio.rs` (hanya fungsi `load_image`)
- **Spesifikasi:**
  1. `imageio::load_image(path) -> Result<RgbImage>` dengan orientasi EXIF diterapkan.
  2. `FaceDetector::load(model_path) -> Result<Self>` memuat ONNX dengan input tetap `[1,3,640,640]` f32, dioptimasi, dan `Send + Sync`.
  3. `FaceDetector::detect(&self, img: &RgbImage) -> Result<Option<Face>>`. `Face { score, bbox, landmarks: [(f32,f32);5] }` dalam koordinat foto asli. Ikuti praproses dan decode di bagian 3.3 persis.
  4. Fungsi kecil `Face::eyes() -> (ex, ey, iod)`.
  5. Test:
     - `testdata/raw/35_Syalva.jpg` dibandingkan dengan nilai acuan bagian 3.3. Bandingkan dalam ruang 640, yaitu koordinat asli dikali `s`. Test di-skip jika file tidak ada.
     - Unit test decode dengan tensor buatan kecil.
  6. Jika `tract` gagal memuat atau menjalankan model: **berhenti**, laporkan error (maks 20 baris) dan operator ONNX yang bermasalah. Jangan ganti ke `ort` atau OpenCV (K1).
- **Selesai jika:** test acuan lulus dalam toleransi.
- **Verifikasi manual:** `cargo test face -- --nocapture` menampilkan score dan posisi mata Syalva.
- **Pesan commit:** `feat(face): add YuNet face detection with tract`

#### G1.2 — Daftar Foto, Penomoran, dan Simpan JPEG
- **Rekomendasi Model:** Gemini Flash 3.8 (Effort: Medium)
- **Prasyarat:** G1.0
- **File dibaca:** `PRD.md` bagian 3.2 dan 3.7, fungsi `kumpulkan_foto` di `reference/app.py`
- **File dibuat/diedit:** `src/files.rs`, `src/imageio.rs` (tambah `save_jpeg`)
- **Spesifikasi:**
  1. `collect_classes(input) -> Result<Vec<ClassDir>>` (aturan subfolder di 3.2).
  2. `collect_photos(dir) -> Result<(Vec<Photo>, Vec<Skipped>)>`, port `kumpulkan_foto`. `Photo { number: u32, label: String, path, stem }`.
  3. `pair_photos(&[Photo]) -> Vec<(Photo, Option<Photo>)>` dan `pairing_summary(...) -> String`, contoh `1&2, 3&4, 5`.
  4. `save_jpeg(img, path)`: kualitas 100, DPI 300, membuat folder induk jika belum ada.
  5. Test dengan `tempfile` memakai nama-nama berikut:
     - `1_A.jpg`, `2_B.JPG`, `3_C.jpg`, `4_D.jpg`, lalu `4_D (ulang).jpg` yang dibuat lebih baru;
     - `5_E.jpeg`, `10_F.png`, `11_G.jpg`, `20_H.jpg`, `04_X.jpg` di folder lain;
     - `1_A_3x4.jpg`, `IMG_1234.jpg`, `desktop.ini`.
     
     Hasil yang diharapkan: urutan 1,2,3,4(ulang),5,10,11,20; pasangan 1&2, 3&4, 5&10, 11&20; tiga file dilewati dengan alasan yang benar; `04_X` berlabel `04`.
  6. Test `save_jpeg` lalu baca ulang: DPI 300 tersimpan (cek header JFIF).
- **Selesai jika:** semua test lulus.
- **Verifikasi manual:** tidak ada (cukup test).
- **Pesan commit:** `feat(files): collect photos, parse absen numbers, and pair for 4R`

#### G1.3 — Framing 3x4
- **Rekomendasi Model:** Gemini Argon 4 (Effort: High)
- **Prasyarat:** G1.0
- **File dibaca:** `PRD.md` bagian 3.4; fungsi `crop_pasfoto`, `cari_logo`, `hitung_tutup_bawah`, serta cabang `else` (tanpa wajah) di `process_smart_crop` pada `reference/app.py`
- **File dibuat/diedit:** `src/framing.rs`
- **Spesifikasi:**
  1. `frame_photo(img: &RgbImage, eyes: Option<(f32,f32,f32)>, params: &FramingParams) -> Framed`. `FramingParams { headroom, eye_line, banner_cover }`. `Framed { image: RgbImage, reasons: Vec<&'static str> }`. Alasan diambil dari tabel bagian 3.4.
  2. `banner_cover(layers: &[&RgbaImage], x, y, w, h) -> f32`, port `hitung_tutup_bawah`.
  3. Port 1:1 dengan tabel pemetaan OpenCV/NumPy di bagian 3.4. Beri komentar `// port of crop_pasfoto() step N`.
  4. Tanpa wajah: crop tengah 3:4 dengan alasan `wajah tidak terdeteksi`.
  5. Test dengan gambar sintetis (tanpa file):
     - latar merah (190,15,30) 1200x1600;
     - kepala elips gelap, tengah x = 560 (sengaja tidak di tengah gambar);
     - mata di (530,600) dan (590,600), sehingga iod = 60;
     - badan putih dari y = 800 ke bawah;
     - logo biru (30,60,200) 50x60 di x = 650, y = 1050;
     - `banner_cover` 0.10.
     
     Yang diharapkan:
     - tengah horizontal hasil crop = tengah elips ±1% lebar;
     - puncak kepala di 7% ±0.5% tinggi;
     - bawah logo di atas garis banner;
     - `reasons` kosong.
  6. Test kedua: latar putih polos, sehingga muncul alasan "latar bukan merah…".
- **Selesai jika:** semua test lulus.
- **Verifikasi manual:** tidak ada (diuji lewat G1.6 dan G1.8).
- **Pesan commit:** `feat(framing): port head-centered, logo-aware 3x4 framing`

#### G1.4 — Ganti Warna Latar (Chroma Key)
- **Rekomendasi Model:** Gemini Flash 3.8 (Effort: High)
- **Prasyarat:** G1.0
- **File dibaca:** `PRD.md` bagian 3.5
- **File dibuat/diedit:** `src/background.rs`
- **Spesifikasi:**
  1. `replace_background(img: &RgbImage, color: [u8;3]) -> RgbImage` sesuai 6 langkah di bagian 3.5.
  2. Test sintetis:
     - latar merah;
     - kotak dasi merah tua (140,20,30) yang menyentuh tepi bawah;
     - kotak kulit (220,150,130) yang menyentuh latar.
     
     Yang diharapkan: latar menjadi warna tujuan, dasi tetap merah, kulit tidak berubah lebih dari 5 per kanal kecuali 2 px di tepinya.
- **Selesai jika:** test lulus.
- **Verifikasi manual:** dilakukan di G1.6, dengan `--bg #1F4FBF` pada testdata.
- **Pesan commit:** `feat(background): add chroma key background replacement`

#### G1.5 — Komposisi 4R
- **Rekomendasi Model:** Gemini Flash 3.8 (Effort: High)
- **Prasyarat:** G1.0
- **File dibaca:** `PRD.md` bagian 3.1 dan 3.6, bagian 4R di `_run_automation_engine` pada `reference/app.py`
- **File dibuat/diedit:** `src/compose.rs`, `src/assets.rs`
- **Spesifikasi:**
  1. `assets::locate() -> Result<AssetPaths>` dengan aturan pencarian dan pesan ramah di bagian 3.1. `assets::load() -> Result<Assets { template, overlay: Option, font }>`. Overlay di-resize ke 1800x1200 jika ukurannya beda. Template yang ukurannya bukan 1800x1200 dianggap error ramah.
  2. `fit_to_slot(img) -> RgbImage` (setara `ImageOps.fit` di bagian 3.6).
  3. `compose_sheet(assets, first: Option<(&RgbImage, &str)>, second: Option<(&RgbImage, &str)>, class_name: &str) -> RgbImage`. Kedua argumen foto berisi gambar dan label absen.
  4. Teks persis sesuai bagian 3.6, termasuk rumus skala font.
  5. Test:
     - `fit_to_slot` menghasilkan 335x445 dan yang terbuang bagian bawah;
     - `compose_sheet` dengan template sintetis transparan: piksel slot terisi, label tidak digambar jika kelas kosong.
- **Selesai jika:** test lulus.
- **Verifikasi manual:** dilakukan di G1.6, dengan membandingkan satu lembar 4R berdampingan dengan hasil Python.
- **Pesan commit:** `feat(compose): build 4R sheet from template, overlay, and labels`

#### G1.6 — Pipeline Batch dan Mode CLI
- **Rekomendasi Model:** Gemini Argon 4 (Effort: Medium)
- **Prasyarat:** G1.1 sampai G1.5
- **File dibaca:** `PRD.md` bagian 3.7, 4.3, dan 4.4; API publik dari `files.rs`, `face.rs`, `framing.rs`, `background.rs`, `compose.rs`, dan `assets.rs` (signature saja)
- **File dibuat/diedit:** `src/pipeline.rs`, `src/main.rs`
- **Spesifikasi:**
  1. `run(job: Job, events: Sender<Event>, cancel: Arc<AtomicBool>) -> Result<Summary>`.
     - `Job { input, output, bg_color: Option<[u8;3]>, headroom, eye_line }`.
     - `Event`: `ClassPlanned { class, photos, pairs, summary }`, `SheetDone { done, total }`, `Flagged { class, file, reasons }`, `Log(String)`, `Finished(Summary)`.
  2. Hitung total lembar di semua kelas di awal, supaya progress akurat.
  3. Thread pool rayon lokal dengan `min(core, 4)` thread. Satu job = satu pasangan (bagian 4.3).
  4. Per foto: load → detect → frame → (bg) → simpan 3x4 → `fit_to_slot`.
  5. Kesalahan per foto dicatat sebagai flag `gagal diproses: <pesan>`, lalu proses lanjut.
  6. Tulis atau hapus `PERLU_DICEK.txt` (bagian 3.7).
  7. `run_cli` di `main.rs`:
     - parsing argumen manual, tanpa crate tambahan;
     - mencetak event ke stdout;
     - mencetak durasi total di akhir.
  8. Test integrasi dengan `testdata/raw/` jika ada (skip jika tidak): jumlah lembar dan file 3x4 sesuai jumlah foto.
- **Selesai jika:** `cargo run --release -- --cli --input testdata/raw --output target/uji` selesai tanpa error. Agent boleh menjalankannya.
- **Verifikasi manual (developer):**
  - buka `target/uji/Output_4R`, bandingkan dengan hasil Python untuk foto yang sama;
  - jalankan sekali lagi dengan `--bg #1F4FBF`, lalu cek tepi rambut dan hijab.
- **Pesan commit:** `feat(pipeline): add parallel batch pipeline and developer CLI`

#### G1.7 — Tampilan (egui)
- **Rekomendasi Model:** Gemini Argon 4 (Effort: Medium)
- **Prasyarat:** G1.6
- **File dibaca:** `PRD.md` bagian 3.8 dan 4.3; API publik `pipeline.rs` dan `files.rs` (signature saja)
- **File dibuat/diedit:** `src/ui.rs`, `src/main.rs`
- **Spesifikasi:**
  1. Semua elemen dan teks di bagian 3.8, persis dalam bahasa Indonesia.
  2. **Pratinjau rencana pasangan:** setelah folder input dipilih, panggil `files::collect_classes` dan `collect_photos` di thread terpisah, lalu tampilkan ringkasan dan pasangan per kelas, serta file yang dilewati beserta alasannya.
  3. **Proses:**
     - `pipeline::run` dijalankan di `std::thread::spawn`;
     - event dibaca di `update()` dengan `try_recv`;
     - tombol berubah menjadi `Batalkan` selama proses berjalan.
  4. **Pengaturan disimpan** (folder input, folder hasil, warna latar, centang, dua slider) lewat eframe persistence.
  5. **Error aset** (bagian 3.1) tampil sebagai kotak pesan merah muda di atas tombol, lalu tombol `Mulai Proses` dinonaktifkan.
  6. `Buka Folder` memakai `std::process::Command::new("explorer")`.
- **Selesai jika:** `cargo build --release` dan clippy bersih. **Agent tidak menjalankan GUI.**
- **Verifikasi manual (developer):**
  1. Buka aplikasi, pilih `testdata/raw`, lalu cek ringkasan dan pasangan.
  2. Proses, lalu batalkan di tengah. Proses ulang sampai selesai.
  3. Tutup dan buka lagi: folder dan pengaturan masih diingat.
  4. Ganti nama folder `Template` sementara: muncul pesan ramah dan tombol nonaktif.
- **Pesan commit:** `feat(ui): add egui interface with progress, preview, and settings`

#### G1.8 — Uji Kesetaraan dengan Versi Python
- **Rekomendasi Model:** Gemini Flash 3.8 (Effort: Medium)
- **Prasyarat:** G1.7
- **File dibaca:** `PRD.md` bagian 2 (K5) dan 3.3; API publik `face.rs`
- **File dibuat/diedit:** `src/bin/parity.rs`
- **Spesifikasi:**
  1. `cargo run --release --bin parity -- <folder_3x4_python> <folder_3x4_rust>` mencocokkan file berdasarkan nama.
  2. Untuk setiap pasangan foto, jalankan YuNet pada kedua hasil 3x4, lalu hitung dalam persen tinggi atau lebar foto masing-masing:
     - `mata_y = ey/h`;
     - `tengah_x = ex/w`;
     - `skala = iod/w`.
  3. Cetak tabel: nama file, selisih tiap metrik, dan status `OK` atau `BEDA`. `BEDA` jika selisih `mata_y` atau `tengah_x` > 1.5 poin persen, atau `skala` > 8% relatif.
  4. Ringkasan di akhir: jumlah OK dan BEDA.
- **Selesai jika:** tool berjalan pada dua folder contoh.
- **Verifikasi manual (developer):**
  1. Proses satu kelas penuh dengan `reference/app.py` (Windows) dan dengan Rust CLI.
  2. Jalankan `parity`.
  3. Target: ≥ 90% foto `OK`. Setiap `BEDA` dicek mata, lalu dilaporkan ke reviewer jika hasil Rust lebih buruk.
- **Pesan commit:** `test(parity): add tool to compare 3x4 framing with python reference`

---

### GERBANG 2: Audit

**Kriteria lolos G2:**
- Target kinerja tercapai (G2.0).
- Tidak ada panic di skenario rusak (G2.1).
- Teks UI lolos audit (G2.2).
- Build rilis v1.0.0 dibuat (G2.3).

#### G2.0 — Audit Kinerja
- **Rekomendasi Model:** Gemini Argon 4 (Effort: Medium)
- **Prasyarat:** Gerbang 1 lolos
- **File dibaca:** `src/pipeline.rs`, `src/framing.rs`, `src/face.rs`, `src/imageio.rs`
- **File diedit:** file yang sama (sebutkan di plan)
- **Spesifikasi:**
  1. Tambah pengukuran waktu per tahap (load, detect, frame, bg, simpan 3x4, compose, simpan 4R) di mode CLI dengan flag `--profil`. Cetak rata-rata per tahap.
  2. Plan wajib memuat hasil pengukuran awal pada `testdata/raw`, sebelum ada optimasi.
  3. Optimasi yang boleh diusulkan:
     - analisis framing pada salinan yang diperkecil (sisi panjang 2000 px), sedangkan crop tetap dari resolusi penuh;
     - memakai ulang buffer;
     - resize lebih cepat untuk input model.
  4. **Setiap optimasi wajib lolos `parity`** (G1.8) terhadap hasil sebelum optimasi.
  5. Target di laptop 4 core:
     - aplikasi terbuka < 1 detik;
     - satu kelas 36 foto 20 MP < 20 detik;
     - RAM puncak < 1.5 GB.
- **Verifikasi manual:** `PasFoto.exe --cli --profil ...` sebelum dan sesudah, ditambah `parity`.
- **Pesan commit:** `perf: profile pipeline stages and optimize hot paths`

#### G2.1 — Audit Ketahanan dan Keamanan
- **Rekomendasi Model:** Gemini Argon 4 (Effort: Low)
- **Prasyarat:** G2.0
- **File dibaca:** semua `src/*.rs`. Mulai dengan `grep` untuk `unwrap(`, `expect(`, `panic!`, `[` indexing, dan `as usize`, lalu baca hanya bagian yang relevan.
- **File diedit:** sesuai temuan (sebutkan di plan)
- **Spesifikasi:**
  1. Plan berisi tabel temuan: file, baris, risiko, perbaikan.
  2. Skenario yang wajib aman, yaitu tidak panic dan menampilkan pesan ramah:
     - file JPEG rusak atau 0 byte;
     - foto sangat kecil (100x100);
     - foto tanpa wajah;
     - nama file Unicode dan spasi;
     - folder hasil read-only atau tidak bisa ditulis;
     - folder input kosong;
     - template berukuran salah;
     - model hilang.
  3. Tambah test untuk skenario yang bisa diotomasi.
  4. Pastikan tidak ada akses jaringan dan tidak ada `unsafe`.
- **Verifikasi manual:** `cargo test`. Developer menjalankan `cargo audit` (opsional, perlu `cargo install cargo-audit`).
- **Pesan commit:** `fix: harden error handling for corrupt input and missing assets`

#### G2.2 — Audit Teks dan Kemudahan Pakai
- **Rekomendasi Model:** Gemini Flash 3.8 (Effort: Low)
- **Prasyarat:** G2.1
- **File dibaca:** `src/ui.rs`, `src/pipeline.rs`, `src/assets.rs` (hanya string yang tampil ke pengguna)
- **File diedit:** file yang sama
- **Spesifikasi:**
  1. Daftar semua string yang tampil ke pengguna, lalu periksa terhadap bagian 3.8: bahasa Indonesia awam, tanpa istilah teknis, dan tanpa teks promosi.
  2. Plan berupa tabel `sekarang → usulan`. Hanya string yang diubah, logika tidak disentuh.
- **Verifikasi manual:** developer membuka GUI dan membaca semua teks.
- **Pesan commit:** `fix(ui): simplify user-facing texts`

#### G2.3 — Rilis v1.0.0 (manual developer, tanpa agent)
1. `cargo build --release`.
2. Susun folder `PasFoto/` sesuai bagian 3.1 dan zip.
3. Uji di komputer studio: buka, proses satu kelas, lalu cetak satu lembar 4R untuk mengecek ukuran dan DPI.
4. `git tag v1.0.0`, lalu `git push --tags`.
