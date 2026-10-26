# AGENTS.md — Aturan Kerja Agent (Pas Foto Otomatis, Rust)

Dokumen ini wajib dibaca di awal setiap chat. Isinya **cara** bekerja. **Apa** yang dibangun ada di `PRD.md`.

## 1. Peran

| Pihak | Tugas |
|---|---|
| Developer | Memberi prompt, menyetujui plan, menjalankan GUI, `git commit`, dan uji manual |
| Reviewer (Claude) | Menulis prompt per tugas dan mereview plan |
| Agent (kamu) | Membaca `AGENTS.md` dan `PRD.md`, membuat plan, lalu implementasi setelah disetujui |

## 2. Sumber Kebenaran

Urutan prioritas jika ada yang bertentangan:
1. Instruksi developer di chat saat ini.
2. `PRD.md`: spesifikasi dan tugas (apa yang dibangun).
3. `AGENTS.md`: aturan kerja (dokumen ini).
4. `reference/app.py`: implementasi Python yang sudah dipakai produksi. Ini acuan **perilaku** (aturan crop, penomoran, susunan 4R). **Hanya dibaca, tidak pernah diedit.**

Jika tetap ragu, tulis pertanyaan di plan. Jangan menebak.

## 3. Alur Kerja Setiap Tugas

1. **Satu chat = satu tugas.** Kerjakan hanya tugas yang disebut di prompt.
2. **Plan dulu.** Jangan mengedit file sebelum developer membalas "OK" atau "lanjut".
3. Setelah disetujui, implementasi sesuai plan. Jika di tengah jalan plan harus berubah, berhenti dan jelaskan dalam 1 sampai 3 kalimat.
4. Selesai implementasi: kirim laporan akhir (format bagian 5).

### Format plan (singkat, tanpa basa-basi)
```
## Plan G1.x — <judul>
File dibaca: ...
File dibuat/diedit: ...
Langkah:
1. ...
Uji yang akan dijalankan: ...
Risiko / asumsi: ... (maks 3)
Pertanyaan: ... (maks 3, tulis "tidak ada" jika tidak ada)
```

### Revisi plan
- Revisi hanya poin yang diminta developer. Kirim ulang plan lengkap dalam format yang sama, tandai poin yang berubah dengan `(revisi)`.

## 4. Batas Lingkup

- Baca dan edit **hanya** file yang tertulis di tugas. Jika butuh file lain, sebutkan di plan beserta alasannya.
- Jangan refactor, rename, atau merapikan kode di luar tugas.
- Jangan menambah dependency di luar daftar `PRD.md` bagian 4 tanpa persetujuan di plan.
- Jangan menambah fitur yang tidak ada di PRD.
- Dilarang mengedit: `reference/`, `testdata/`, `models/`, `Template/`, `PRD.md`, `AGENTS.md`.
- Dilarang membaca: `target/`, `Cargo.lock` (kecuali diminta), file gambar atau biner.
- Untuk `reference/app.py`: jangan membaca seluruh file. Cari fungsi yang disebut tugas (`def nama_fungsi`), lalu baca rentang baris fungsi itu saja.

## 5. Laporan Akhir (setelah implementasi)

```
## Selesai G1.x
Diubah: <file> (<1 kalimat>), ...
Uji: cargo test -> <lulus/gagal, jumlah test>; cargo clippy -> <bersih/ada peringatan>
Belum dikerjakan / catatan: ... (tulis "tidak ada" jika tidak ada)
Verifikasi manual untuk developer: <salin dari PRD>
Pesan commit: <salin dari PRD>
```
Jangan menempel isi file lengkap atau diff panjang di laporan.

## 6. Terminal

**Boleh dijalankan agent:**
- `cargo check`, `cargo build`, `cargo build --release`
- `cargo test`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt`
- `cargo tree`, `cargo add <crate>` (hanya crate di PRD bagian 4)
- `cargo run --release -- --cli ...` dengan input dari `testdata/` (mode CLI, bukan GUI)

**Dilarang:**
- `git commit`, `git push`, `git reset`, `git checkout`, atau perintah git yang mengubah riwayat. `git status` dan `git diff --stat` boleh.
- Menghapus file atau folder (`rm`, `del`, `Remove-Item`) di luar file yang dibuat sendiri pada tugas yang sama.
- Mengunduh apa pun selain crate lewat cargo.
- Menjalankan GUI (developer yang menguji tampilan).

**Hemat token di terminal:**
- Selalu potong output: `cargo test -q 2>&1 | tail -n 40`, `cargo clippy ... 2>&1 | tail -n 40`.
- Maksimal **3 putaran** perbaikan error kompilasi atau test. Jika masih gagal, berhenti dan laporkan error terakhir (maks 20 baris) beserta dugaan penyebabnya.

## 7. Aturan Kode Rust

- Rust stable, edition 2024. Tanpa `unsafe`.
- Tanpa `unwrap()`, `expect()`, `panic!()`, atau indexing yang bisa panic di kode non-test, kecuali invariant yang dijamin dan diberi komentar `// invariant: ...`.
- Error: `anyhow::Result` di lapisan aplikasi, dengan `.context("...")` berbahasa Indonesia yang ramah pengguna (pesan ini bisa tampil di UI).
- Semua konstanta aturan (persentase, ambang warna, koordinat template) ada di `src/config.rs`, tidak ditulis langsung di tengah kode.
- Nama fungsi dan variabel dalam bahasa Inggris. Teks yang tampil ke pengguna dalam bahasa Indonesia yang awam (lihat PRD bagian 3.6).
- **Komentar inline dalam bahasa Inggris, singkat**, hanya untuk logika yang tidak jelas dari kodenya. Port dari Python diberi komentar asal fungsinya, contoh `// port of crop_pasfoto() step 2`.
- Fungsi pemrosesan gambar dibuat **murni** (input gambar dan angka, output hasil) supaya mudah di-test. Tidak ada I/O di `framing.rs`, `background.rs`, dan `compose.rs`.
- Setiap modul logika punya unit test di file yang sama (`#[cfg(test)]`). Test yang butuh foto dari `testdata/` harus **skip dengan pesan** jika filenya tidak ada, bukan gagal.
- Harus bersih dari `cargo clippy --all-targets -- -D warnings` dan sudah di-`cargo fmt`.

## 8. Komunikasi

- **Selalu bahasa Indonesia** saat berbicara dengan developer. Singkat dan langsung ke poin.
- Jangan mengulang isi PRD atau isi file kepada developer.
- Jangan menulis teks promosi atau teknis berlebihan di UI maupun di laporan (contoh yang dilarang: "100% offline", "Optimized Engine", "AI-powered", "blazing fast").
