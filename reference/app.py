import os
import glob
import threading
import re
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import customtkinter as ctk
from tkinter import filedialog, messagebox, colorchooser
import mediapipe.python.solutions.face_detection as mp_face_detection
from rembg import remove, new_session

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


# ============================================================================
# ATURAN FRAMING PAS FOTO (patokan diambil dari contoh 35_Syalva)
# Prioritas: 1) kepala di tengah  2) logo di dada utuh  3) ada ruang di atas kepala
# ============================================================================
HEADROOM_DEFAULT = 0.07   # ruang di atas kepala: 7% dari tinggi foto
EYE_LINE_DEFAULT = 0.28   # garis mata ideal: 28% dari atas
IOD_FRAC = 0.147          # jarak antar mata ≈ 14,7% lebar foto (untuk pengaman zoom)
LOGO_MARGIN = 0.15        # jarak antara bawah logo dan banner, dalam satuan jarak antar mata


def hitung_tutup_bawah(layers, x, y, size):
    """Berapa bagian bawah slot foto (0..1) yang tertutup banner overlay/template."""
    w, h = size
    tertutup = np.zeros(h, bool)
    for layer in layers:
        if layer is None:
            continue
        a = np.array(layer)[y:y + h, x:x + w, 3]
        tertutup |= (a > 20).mean(axis=1) > 0.5
    n = 0
    while n < h and tertutup[h - 1 - n]:
        n += 1
    return n / h


EKSTENSI_FOTO = {".jpg", ".jpeg", ".png"}


def kumpulkan_foto(folder):
    """Ambil semua foto siswa di satu folder kelas, urut berdasarkan nomor absen.

    Aturan:
    - Nomor absen = angka di AWAL nama file ("4_Alvira Nada.jpg" -> 4, "04_..." -> 04).
    - Diurutkan sebagai angka: 1, 2, 3, ..., 10, 20 (bukan 1, 10, 2, 20, 3).
    - Satu nomor absen hanya dipakai sekali. Kalau ada 2 file dengan nomor sama
      (mis. foto ulang), yang dipakai file yang paling baru, sisanya dilewati.
    - File hasil olahan (berakhiran _3x4 / _4R) dan file tanpa nomor di awal dilewati.
    Return: (daftar_foto, peringatan). daftar_foto berisi dict: no, label, path, name.
    """
    per_nomor, peringatan = {}, []
    with os.scandir(folder) as it:
        entries = sorted((e for e in it if e.is_file()), key=lambda e: e.name.lower())
    for e in entries:
        stem, ext = os.path.splitext(e.name)
        if ext.lower() not in EKSTENSI_FOTO or e.name.startswith((".", "~")):
            continue
        if re.search(r'_(3x4|4r)$', stem, re.IGNORECASE):
            peringatan.append(f"{e.name}  (dilewati: ini file hasil olahan, bukan foto mentah)")
            continue
        m = re.match(r'\s*(\d+)', stem)
        if not m:
            peringatan.append(f"{e.name}  (dilewati: nama file tidak diawali nomor absen)")
            continue
        foto = {"no": int(m.group(1)), "label": m.group(1), "path": e.path, "name": stem,
                "mtime": e.stat().st_mtime}
        lama = per_nomor.get(foto["no"])
        if lama is None:
            per_nomor[foto["no"]] = foto
        else:
            baru, dibuang = (foto, lama) if foto["mtime"] > lama["mtime"] else (lama, foto)
            per_nomor[foto["no"]] = baru
            peringatan.append(f"{os.path.basename(dibuang['path'])}  (dilewati: nomor absen {foto['no']} "
                              f"dobel, yang dipakai {os.path.basename(baru['path'])})")
    return [per_nomor[n] for n in sorted(per_nomor)], peringatan


def cari_logo(img, cx, ey, iod, latar_tepi):
    """Mencari logo di dada kiri siswa (sisi kanan foto). Return (x, y, w, h) atau None."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
    berwarna = (S > 90) & (V > 50) & ~latar_tepi          # logo berwarna, seragam putih/krem
    y1, x1, x2 = int(ey + 3.0 * iod), max(0, int(cx + 0.9 * iod)), int(min(W, cx + 3.8 * iod))
    if y1 >= H or x1 >= x2:
        return None
    area = np.zeros(berwarna.shape, np.uint8)
    area[y1:, x1:x2] = berwarna[y1:, x1:x2]
    k = max(3, int(iod * 0.08))
    area = cv2.morphologyEx(area, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(area)
    terbaik = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.15 * iod * iod or w > 2.0 * iod or h > 2.2 * iod:
            continue
        if terbaik is None or a > terbaik[4]:
            terbaik = (int(x), int(y), int(w), int(h), int(a))
    return terbaik[:4] if terbaik else None


def crop_pasfoto(img, ex, ey, iod, ratio=3 / 4, headroom=HEADROOM_DEFAULT, eye_line=EYE_LINE_DEFAULT,
                 tutup_bawah=0.0):
    """img: foto BGR. ex, ey: titik tengah kedua mata (piksel). iod: jarak antar mata (piksel).
    tutup_bawah: bagian bawah foto (0..1) yang nanti tertutup banner di template 4R.
    Mengembalikan (foto_crop_bgr, perlu_dicek)."""
    H, W = img.shape[:2]
    b, g, r = [img[:, :, i].astype(np.int16) for i in range(3)]
    latar = (r > 100) & (r - g > 60) & (r - b > 45)                  # kain merah
    fg = cv2.morphologyEx((~latar).astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((9, 9), np.uint8)).astype(bool)
    _, lab = cv2.connectedComponents(latar.astype(np.uint8))
    tepi = np.unique(np.concatenate([lab[0], lab[:, 0], lab[:, -1]]))
    latar_tepi = np.isin(lab, tepi[tepi > 0])                        # latar yang menyambung ke tepi foto
    perlu_dicek = False

    # 1) Puncak kepala: naik dari garis mata selama masih ada rambut/hijab di sekitar wajah
    x1, x2 = max(0, int(ex - 0.6 * iod)), min(W, int(ex + 0.6 * iod))
    pita = fg[:, x1:x2].any(axis=1)
    crown = int(ey)
    while crown > 0 and pita[crown - 1]:
        crown -= 1
    if crown == 0:
        perlu_dicek = True                          # kepala menyentuh tepi atas foto mentah

    # 2) Tengah kiri-kanan = tengah garis tepi kepala (rambut/hijab) antara dahi dan mata,
    #    bukan tengah wajah, supaya siswa yang menoleh/badannya miring tetap terlihat di tengah
    pusat = []
    for y in np.linspace(crown + 0.25 * (ey - crown), ey, 12).astype(int):
        row, x = fg[y], int(ex)
        if not row[x]:
            continue
        L = x
        while L > 0 and row[L - 1]:
            L -= 1
        R = x
        while R < W - 1 and row[R + 1]:
            R += 1
        pusat.append((L + R) / 2)
    if len(pusat) >= 4:
        cx = float(np.median(pusat))
    else:
        cx, perlu_dicek = ex, True                  # latar bukan merah -> pakai tengah mata

    # 3) Ukuran ideal: puncak kepala di `headroom`, mata di `eye_line`; dijaga oleh jarak antar mata
    h_mata = iod / IOD_FRAC / ratio
    ideal = float(np.clip((ey - crown) / max(eye_line - headroom, 0.05), 0.8 * h_mata, 1.25 * h_mata))
    # Ruang di atas kepala selalu tetap, jadi tinggi foto h menentukan semuanya:
    # atas foto = crown - headroom*h, bawah foto = atas + h
    h = (ey + (1 - eye_line) * ideal - crown) / (1 - headroom)

    # 4) Logo di dada harus utuh DAN tidak tertutup banner: kalau perlu zoom out
    logo = cari_logo(img, cx, ey, iod, latar_tepi)
    if logo:
        bawah_logo = logo[1] + logo[3]
        logo_terpotong_dari_raw = bawah_logo >= H - 2
    else:
        bawah_logo = ey + 6.7 * iod                 # logo tidak ketemu: pakai perkiraan posisinya
        logo_terpotong_dari_raw = False
        perlu_dicek = True
    h_logo = (bawah_logo + LOGO_MARGIN * iod - crown) / max(1 - headroom - tutup_bawah, 0.5)
    h = max(h, h_logo)

    # Foto mentah tidak bisa diperpanjang ke bawah (badan tidak bisa dikarang)
    h_maks = (H - crown) / (1 - headroom)
    h = min(h, h_maks)
    y0 = crown - headroom * h
    if logo and not logo_terpotong_dari_raw:
        tertutup = bawah_logo - (y0 + (1 - tutup_bawah) * h)    # piksel logo di bawah garis banner
        if tertutup > 0.25 * logo[3]:
            perlu_dicek = True                      # lebih dari 1/4 logo tertutup banner
    if h < ideal * 0.85 or h > ideal * 1.45:
        perlu_dicek = True                          # jauh lebih zoom in / zoom out dibanding ideal

    # 5) Potong. Kekurangan di atas/kiri/kanan diisi perpanjangan latar merah
    w = h * ratio
    x0, y0, w, h = int(round(cx - w / 2)), int(round(y0)), int(round(w)), int(round(h))
    pt, pl = max(0, -y0), max(0, -x0)
    pr, pb = max(0, x0 + w - W), max(0, y0 + h - H)
    if max(pt, pl, pr) > 0.06 * h:
        perlu_dicek = True
    besar = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return besar[y0 + pt:y0 + pt + h, x0 + pl:x0 + pl + w], perlu_dicek


def muat_font(ukuran):
    for nama in ("arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(nama, ukuran)
        except OSError:
            continue
    return ImageFont.load_default(size=ukuran)

class PasFotoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pas Foto Automation - Smart Crop V3.8.2")
        self.geometry("1280x920")
        self.minsize(1100, 850)
        self.configure(fg_color="#F8F9FA")

        self.input_dir = ctk.StringVar(value=os.path.abspath("Input_Raw"))
        self.output_dir = ctk.StringVar(value=os.path.abspath("Output_4R"))
        self.eye_anchor_val = ctk.DoubleVar(value=EYE_LINE_DEFAULT)    # garis mata
        self.face_scale_val = ctk.DoubleVar(value=HEADROOM_DEFAULT)    # ruang di atas kepala
        self.replace_bg_var = ctk.BooleanVar(value=False) 
        self.bg_hex_val = ctk.StringVar(value="#be1220") 
        self.is_processing = False

        self.camera_presets = {
            "Kamera Utama (Studio)": {"eye": EYE_LINE_DEFAULT, "scale": HEADROOM_DEFAULT},
            "Kamera Cadangan / Lensa Lain": {"eye": EYE_LINE_DEFAULT, "scale": HEADROOM_DEFAULT}
        }

        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.rembg_session = new_session("u2net_human_seg", providers=['CPUExecutionProvider'])

        self.create_header()
        self.create_config_section()
        self.create_status_section()
        self.create_action_and_console_section()
        self.create_footer()

        self.log_message("Sistem siap. Mode V3.8.2 Optimized aktif.")

    def create_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="#FFFFFF", height=80, corner_radius=0)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)

        title_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=24, pady=16)

        lbl_title = ctk.CTkLabel(title_box, text="Pas Foto Automation V3.8.2", font=("Inter", 22, "bold"), text_color="#191C1D")
        lbl_title.pack(anchor="w")

        lbl_sub = ctk.CTkLabel(title_box, text="Batch Pairing Processing with Optimized C++ Engine", font=("Inter", 12), text_color="#6C757D")
        lbl_sub.pack(anchor="w")

        right_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        right_box.pack(side="right", padx=24, pady=16)

        badge = ctk.CTkLabel(right_box, text="OFFLINE / LOCAL PROCESSING", font=("Inter", 12, "bold"), 
                             fg_color="#D1E4FF", text_color="#003F6D", corner_radius=12)
        badge.pack(side="left", padx=12)

    def create_config_section(self):
        config_frame = ctk.CTkFrame(self, fg_color="transparent")
        config_frame.pack(fill="x", padx=24, pady=16)
        config_frame.grid_columnconfigure((0, 1), weight=1, uniform="col")

        card_dir = ctk.CTkFrame(config_frame, fg_color="#FFFFFF", corner_radius=8, border_width=1, border_color="#DEE2E6")
        card_dir.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        lbl_dir_title = ctk.CTkLabel(card_dir, text="Directory Configuration", font=("Inter", 15, "bold"), text_color="#191C1D")
        lbl_dir_title.pack(anchor="w", padx=16, pady=(16, 12))

        lbl_in = ctk.CTkLabel(card_dir, text="Input Directory", font=("Inter", 12, "bold"), text_color="#404752")
        lbl_in.pack(anchor="w", padx=16)

        row_in = ctk.CTkFrame(card_dir, fg_color="transparent")
        row_in.pack(fill="x", padx=16, pady=(4, 12))
        
        self.entry_in = ctk.CTkEntry(row_in, textvariable=self.input_dir, height=36, font=("Inter", 12))
        self.entry_in.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        btn_in = ctk.CTkButton(row_in, text="Pilih Folder", width=110, height=36, font=("Inter", 12, "bold"),
                               fg_color="#E7E8E9", text_color="#212529", command=self.browse_input)
        btn_in.pack(side="right")

        lbl_out = ctk.CTkLabel(card_dir, text="Output Directory (4R)", font=("Inter", 12, "bold"), text_color="#404752")
        lbl_out.pack(anchor="w", padx=16)

        row_out = ctk.CTkFrame(card_dir, fg_color="transparent")
        row_out.pack(fill="x", padx=16, pady=(4, 16))
        
        self.entry_out = ctk.CTkEntry(row_out, textvariable=self.output_dir, height=36, font=("Inter", 12))
        self.entry_out.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        btn_out = ctk.CTkButton(row_out, text="Buka Folder", width=110, height=36, font=("Inter", 12, "bold"),
                                fg_color="#E7E8E9", text_color="#212529", command=self.open_output_folder)
        btn_out.pack(side="right")

        card_cam = ctk.CTkFrame(config_frame, fg_color="#FFFFFF", corner_radius=8, border_width=1, border_color="#DEE2E6")
        card_cam.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        top_cam_row = ctk.CTkFrame(card_cam, fg_color="transparent")
        top_cam_row.pack(fill="x", padx=16, pady=(16, 8))

        lbl_cam_title = ctk.CTkLabel(top_cam_row, text="Camera Profile & Tuning", font=("Inter", 15, "bold"), text_color="#191C1D")
        lbl_cam_title.pack(side="left")

        prof_ctrl_row = ctk.CTkFrame(card_cam, fg_color="transparent")
        prof_ctrl_row.pack(fill="x", padx=16, pady=(0, 12))
        prof_ctrl_row.grid_columnconfigure(0, weight=1)

        self.combo_cam = ctk.CTkComboBox(prof_ctrl_row, values=list(self.camera_presets.keys()), height=34, font=("Inter", 12),
                                         command=self.on_camera_profile_selected)
        self.combo_cam.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.combo_cam.set("Kamera Utama (Studio)")

        slider_box = ctk.CTkFrame(card_cam, fg_color="transparent")
        slider_box.pack(fill="x", padx=16, pady=(0, 8))
        slider_box.grid_columnconfigure((0, 1), weight=1)

        lbl_e = ctk.CTkLabel(slider_box, text="Posisi Mata", font=("Inter", 11, "bold"), text_color="#404752")
        lbl_e.grid(row=0, column=0, sticky="w")
        self.lbl_val_e = ctk.CTkLabel(slider_box, text=f"{self.eye_anchor_val.get():.0%}", font=("Inter", 11, "bold"), text_color="#0078D4")
        self.lbl_val_e.grid(row=0, column=0, sticky="e", padx=10)
        self.slider_e = ctk.CTkSlider(slider_box, from_=0.20, to=0.40, number_of_steps=20, variable=self.eye_anchor_val, command=lambda val: self.lbl_val_e.configure(text=f"{val:.0%}"))
        self.slider_e.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(4, 2))

        lbl_f = ctk.CTkLabel(slider_box, text="Ruang di Atas Kepala", font=("Inter", 11, "bold"), text_color="#404752")
        lbl_f.grid(row=0, column=1, sticky="w")
        self.lbl_val_f = ctk.CTkLabel(slider_box, text=f"{self.face_scale_val.get():.0%}", font=("Inter", 11, "bold"), text_color="#0078D4")
        self.lbl_val_f.grid(row=0, column=1, sticky="e", padx=0)
        self.slider_f = ctk.CTkSlider(slider_box, from_=0.02, to=0.15, number_of_steps=13, variable=self.face_scale_val, command=lambda val: self.lbl_val_f.configure(text=f"{val:.0%}"))
        self.slider_f.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(4, 2))
        
        bg_box = ctk.CTkFrame(card_cam, fg_color="#F0F4F8", corner_radius=6, border_width=1, border_color="#D1E4FF")
        bg_box.pack(fill="x", padx=16, pady=(8, 16))
        
        self.cb_bg = ctk.CTkCheckBox(bg_box, text="Ganti Background Latar (High-Res Rembg)", 
                                     variable=self.replace_bg_var, font=("Inter", 12, "bold"), text_color="#003F6D",
                                     hover_color="#005A9E", border_color="#0078D4", command=self.toggle_bg_controls)
        self.cb_bg.pack(padx=12, pady=(12, 6), anchor="w")
        
        self.color_controls = ctk.CTkFrame(bg_box, fg_color="transparent")
        self.color_controls.pack(fill="x", padx=12, pady=(0, 12))
        
        self.btn_pick_color = ctk.CTkButton(self.color_controls, text="Pilih Warna", width=100, height=30,
                                            command=self.open_color_picker, fg_color="#0078D4")
        self.btn_pick_color.pack(side="left", padx=(0, 8))

        self.entry_hex = ctk.CTkEntry(self.color_controls, textvariable=self.bg_hex_val, width=80, height=30, font=("Inter", 12))
        self.entry_hex.pack(side="left", padx=(0, 8))
        self.entry_hex.bind("<KeyRelease>", self.update_color_preview)

        self.color_preview = ctk.CTkFrame(self.color_controls, width=30, height=30, fg_color=self.bg_hex_val.get(), 
                                          corner_radius=4, border_width=1, border_color="#333333")
        self.color_preview.pack(side="left")
        
        self.toggle_bg_controls()

    def toggle_bg_controls(self):
        state = "normal" if self.replace_bg_var.get() else "disabled"
        self.btn_pick_color.configure(state=state)
        self.entry_hex.configure(state=state)

    def open_color_picker(self):
        color_tuple = colorchooser.askcolor(title="Pilih Warna Latar Studio", initialcolor=self.bg_hex_val.get())
        if color_tuple[1]:
            self.bg_hex_val.set(color_tuple[1])
            self.color_preview.configure(fg_color=color_tuple[1])

    def update_color_preview(self, event=None):
        val = self.bg_hex_val.get().strip()
        if val.startswith("#"):
            clean_val = val[1:]
            if len(clean_val) == 3:
                val = "#" + "".join([c * 2 for c in clean_val])
            if len(val) == 7:
                self.color_preview.configure(fg_color=val)

    def on_camera_profile_selected(self, choice):
        preset = self.camera_presets[choice]
        self.eye_anchor_val.set(preset["eye"])
        self.face_scale_val.set(preset["scale"])
        self.slider_e.set(preset["eye"])
        self.slider_f.set(preset["scale"])
        self.lbl_val_e.configure(text=f"{preset['eye']:.0%}")
        self.lbl_val_f.configure(text=f"{preset['scale']:.0%}")

    def create_status_section(self):
        status_card = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=8, border_width=1, border_color="#DEE2E6")
        status_card.pack(fill="x", padx=24, pady=(0, 16))
        top_st = ctk.CTkFrame(status_card, fg_color="transparent")
        top_st.pack(fill="x", padx=16, pady=(12, 4))
        self.lbl_status_text = ctk.CTkLabel(top_st, text="Sistem siap memproses foto", font=("Inter", 13, "bold"), text_color="#28A745")
        self.lbl_status_text.pack(side="left")
        self.lbl_progress_count = ctk.CTkLabel(top_st, text="0 / 0 pasang", font=("Inter", 13, "bold"), text_color="#212529")
        self.lbl_progress_count.pack(side="right")
        self.progress_bar = ctk.CTkProgressBar(status_card, height=8, corner_radius=2, fg_color="#E9ECEF", progress_color="#0078D4")
        self.progress_bar.pack(fill="x", padx=16, pady=(4, 16))
        self.progress_bar.set(0.0)

    def create_action_and_console_section(self):
        main_bot = ctk.CTkFrame(self, fg_color="transparent")
        main_bot.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        main_bot.grid_columnconfigure(1, weight=1)
        main_bot.grid_rowconfigure(0, weight=1)

        self.action_card = ctk.CTkFrame(main_bot, fg_color="#0078D4", corner_radius=8, width=320)
        self.action_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.action_card.grid_propagate(False)

        action_inner = ctk.CTkFrame(self.action_card, fg_color="transparent")
        action_inner.place(relx=0.5, rely=0.5, anchor="center")
        self.btn_action = ctk.CTkButton(action_inner, text="Mulai Proses", font=("Inter", 16, "bold"), fg_color="#FFFFFF", text_color="#0078D4", hover_color="#F0F1F2", width=260, height=54, corner_radius=6, command=self.start_processing_thread)
        self.btn_action.pack(pady=(0, 12))
        self.lbl_action_sub = ctk.CTkLabel(action_inner, text="Otomatisasi 3x4 & 4R", font=("Inter", 12), text_color="#D1E4FF")
        self.lbl_action_sub.pack()

        console_card = ctk.CTkFrame(main_bot, fg_color="#1E1E1E", corner_radius=8, border_width=1, border_color="#333333")
        console_card.grid(row=0, column=1, sticky="nsew")
        console_card.grid_rowconfigure(1, weight=1)
        console_card.grid_columnconfigure(0, weight=1)
        con_head = ctk.CTkFrame(console_card, fg_color="#252526", height=36, corner_radius=0)
        con_head.grid(row=0, column=0, sticky="ew")
        con_head.grid_propagate(False)
        lbl_con_title = ctk.CTkLabel(con_head, text="System Log", font=("Inter", 12, "bold"), text_color="#D4D4D4")
        lbl_con_title.pack(side="left", padx=12)
        btn_clear = ctk.CTkButton(con_head, text="Clear", width=50, height=24, font=("Inter", 11), fg_color="#333333", text_color="#D4D4D4", command=self.clear_console)
        btn_clear.pack(side="right", padx=12)
        self.textbox_log = ctk.CTkTextbox(console_card, fg_color="#1E1E1E", text_color="#D4D4D4", font=("Consolas", 12), corner_radius=0)
        self.textbox_log.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.textbox_log.configure(state="disabled")

    def create_footer(self):
        footer_frame = ctk.CTkFrame(self, fg_color="#E9ECEF", height=32, corner_radius=0)
        footer_frame.pack(fill="x", side="bottom")
        footer_frame.pack_propagate(False)
        lbl_foot_l = ctk.CTkLabel(footer_frame, text="v3.8.2 Build 1018 | ONNX CPU Optimized", font=("Inter", 11), text_color="#6C757D")
        lbl_foot_l.pack(side="left", padx=16)
        lbl_foot_r = ctk.CTkLabel(footer_frame, text="System Status: Ready", font=("Inter", 11), text_color="#28A745")
        lbl_foot_r.pack(side="right", padx=16)

    def browse_input(self):
        folder = filedialog.askdirectory(title="Pilih Folder Input Raw")
        if folder:
            self.input_dir.set(folder)
            self.log_message(f"Folder input diatur ke: {folder}")

    def open_output_folder(self):
        out = self.output_dir.get()
        if os.path.exists(out): os.startfile(out)

    def clear_console(self):
        self.textbox_log.configure(state="normal")
        self.textbox_log.delete("0.0", "end")
        self.textbox_log.configure(state="disabled")

    def ui(self, fn, *args, **kwargs):
        """Jalankan perubahan tampilan di thread utama (tkinter tidak aman diubah dari thread lain)."""
        if threading.current_thread() is threading.main_thread():
            fn(*args, **kwargs)
        else:
            self.after(0, lambda: fn(*args, **kwargs))

    def log_message(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.ui(self._tulis_log, f"{timestamp} {message}\n")

    def _tulis_log(self, formatted_msg):
        self.textbox_log.configure(state="normal")
        self.textbox_log.insert("end", formatted_msg)
        self.textbox_log.see("end")
        self.textbox_log.configure(state="disabled")

    def start_processing_thread(self):
        if self.is_processing: return
        self.is_processing = True
        self.btn_action.configure(text="Memproses...", state="disabled")
        threading.Thread(target=self.run_automation_engine, daemon=True).start()

    def process_smart_crop(self, file_path, target_ratio, headroom, eye_line, tutup_bawah=0.0):
        """Mengembalikan (foto_PIL, perlu_dicek)."""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("file foto tidak bisa dibaca")
        h_img, w_img = img.shape[:2]
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.face_detection.process(rgb_img)
        iod = 0
        if results.detections:
            detection = max(results.detections, key=lambda d: d.score[0])
            right_eye = detection.location_data.relative_keypoints[0]
            left_eye = detection.location_data.relative_keypoints[1]
            ex = (right_eye.x + left_eye.x) / 2 * w_img
            ey = (right_eye.y + left_eye.y) / 2 * h_img
            iod = abs(right_eye.x - left_eye.x) * w_img

        if iod > 10:
            cropped, perlu_dicek = crop_pasfoto(img, ex, ey, iod, target_ratio, headroom, eye_line, tutup_bawah)
        else:
            perlu_dicek = True                     # wajah tidak terdeteksi: potong di tengah
            crop_w = w_img
            crop_h = int(w_img / target_ratio)
            if crop_h > h_img:
                crop_h = h_img
                crop_w = int(h_img * target_ratio)
            crop_x = (w_img - crop_w) // 2
            crop_y = (h_img - crop_h) // 2
            cropped = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        if self.replace_bg_var.get():
            hex_raw = self.bg_hex_val.get().strip().lstrip('#')
            if len(hex_raw) == 3:
                hex_raw = "".join([c * 2 for c in hex_raw])
            
            if len(hex_raw) == 6:
                r, g, b = tuple(int(hex_raw[i:i+2], 16) for i in (0, 2, 4))
            else:
                r, g, b = (178, 0, 0)

            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(cropped_rgb)
            fg_rgba = remove(pil_crop, session=self.rembg_session)
            
            data = np.array(fg_rgba)
            r_ch, g_ch, b_ch, a_ch = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
            
            kernel = np.ones((2, 2), np.uint8)
            a_ch = cv2.erode(a_ch, kernel, iterations=1)
            a_ch[a_ch < 60] = 0
            
            clean_data = np.stack([r_ch, g_ch, b_ch, a_ch], axis=-1)
            fg_clean = Image.fromarray(clean_data, mode="RGBA")
            
            bg_solid = Image.new("RGBA", pil_crop.size, (r, g, b, 255))
            bg_solid.paste(fg_clean, (0, 0), fg_clean)

            return bg_solid.convert("RGB"), perlu_dicek

        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_rgb), perlu_dicek

    def run_automation_engine(self):
        try:
            self._run_automation_engine()
        except Exception as e:
            self.log_message(f"Proses berhenti karena error: {e}")
            self.ui(self.lbl_status_text.configure, text="Proses berhenti karena error, lihat catatan di bawah", text_color="#DC3545")
        finally:
            self.is_processing = False
            self.ui(self.btn_action.configure, text="Mulai Proses", state="normal")

    def _run_automation_engine(self):
        input_root = self.input_dir.get()
        output_4r_root = self.output_dir.get()
        output_3x4_root = "Output_3x4"
        template_dir = "Template"

        os.makedirs(output_4r_root, exist_ok=True)
        os.makedirs(output_3x4_root, exist_ok=True)

        master_template = Image.open(os.path.join(template_dir, "Templates.png")).convert("RGBA")
        template_size = (1800, 1200)

        overlay_path = os.path.join(template_dir, "overlay.png")
        overlay = Image.open(overlay_path).convert("RGBA").resize(template_size, Image.Resampling.LANCZOS) if os.path.exists(overlay_path) else None

        class_folders = sorted((f.path for f in os.scandir(input_root) if f.is_dir()), key=lambda p: os.path.basename(p).lower()) or [input_root]

        size_3x4 = (335, 445)
        coords_row_1 = [(x, 35) for x in [35, 400, 767, 1133]]
        coords_row_2 = [(x, 523) for x in [35, 400, 767, 1133]]

        current_headroom = self.face_scale_val.get()
        current_eye_line = self.eye_anchor_val.get()
        target_ratio = 3 / 4
        font = muat_font(60)
        daftar_dicek = []
        # bagian bawah foto yang tertutup banner (mis. "SMK - SMAK BOGOR"), dibaca dari template
        tutup_bawah = hitung_tutup_bawah([overlay, master_template], coords_row_1[0][0], coords_row_1[0][1], size_3x4)
        self.log_message(f"Banner template menutupi {tutup_bawah:.0%} bagian bawah foto; logo dijaga tetap di atasnya.")

        def scale_exact(p_img, box):
            # pas persis ke kotak template; kalau rasio beda sedikit, yang terpotong bagian bawah
            return ImageOps.fit(p_img, box, Image.Resampling.LANCZOS, centering=(0.5, 0.0))

        def proses_satu(file_path, name):
            try:
                pil, dicek = self.process_smart_crop(file_path, target_ratio, current_headroom, current_eye_line, tutup_bawah)
            except Exception as e:
                self.log_message(f"GAGAL memproses {os.path.basename(file_path)}: {e}")
                daftar_dicek.append(f"{os.path.basename(file_path)}  (gagal diproses: {e})")
                return None
            if dicek:
                self.log_message(f"Perlu dicek: {os.path.basename(file_path)}")
                daftar_dicek.append(os.path.basename(file_path))
            pil.save(os.path.join(current_out_3x4, f"{name}_3x4.jpg"), quality=100, dpi=(300, 300))
            return pil

        for folder in class_folders:
            class_name = os.path.basename(folder) if folder != input_root else ""
            
            current_out_4r = os.path.join(output_4r_root, f"{class_name}_4R") if class_name else output_4r_root
            current_out_3x4 = os.path.join(output_3x4_root, f"{class_name}_3x4") if class_name else output_3x4_root
                
            os.makedirs(current_out_4r, exist_ok=True)
            os.makedirs(current_out_3x4, exist_ok=True)

            daftar_foto, peringatan = kumpulkan_foto(folder)
            for p in peringatan:
                self.log_message(f"[{class_name or 'folder input'}] {p}")
                daftar_dicek.append(f"[{class_name or 'folder input'}] {p}")
            if not daftar_foto:
                continue

            total_pairs = (len(daftar_foto) + 1) // 2
            processed_pairs = 0
            rencana = ", ".join(
                f"{daftar_foto[i]['label']}&{daftar_foto[i + 1]['label']}" if i + 1 < len(daftar_foto)
                else daftar_foto[i]['label'] for i in range(0, len(daftar_foto), 2))
            self.log_message(f"Kelas {class_name or '-'}: {len(daftar_foto)} foto, pasangan 4R: {rencana}")

            for i in range(0, len(daftar_foto), 2):
                foto_1 = daftar_foto[i]
                foto_2 = daftar_foto[i + 1] if i + 1 < len(daftar_foto) else None
                file_1, name_1, absen_1 = foto_1["path"], foto_1["name"], foto_1["label"]
                if foto_2:
                    file_2, name_2, absen_2 = foto_2["path"], foto_2["name"], foto_2["label"]
                else:
                    file_2, name_2, absen_2 = None, None, None

                self.ui(self.lbl_status_text.configure, text=f"Memproses: {class_name} | Absen {absen_1} & {absen_2 if file_2 else '(Ganjil)'}", text_color="#FFC107")
                self.log_message(f"Memproses Pasangan Absen {absen_1} dan {absen_2 if file_2 else '(Kosong)'}...")

                canvas = Image.new("RGBA", template_size, (255, 255, 255, 255))

                pil_1 = proses_satu(file_1, name_1)
                if pil_1:
                    img_1_scaled = scale_exact(pil_1, size_3x4)
                    for coord in coords_row_1:
                        canvas.paste(img_1_scaled, coord)

                if file_2:
                    pil_2 = proses_satu(file_2, name_2)
                    if pil_2:
                        img_2_scaled = scale_exact(pil_2, size_3x4)
                        for coord in coords_row_2:
                            canvas.paste(img_2_scaled, coord)

                if overlay:
                    canvas.paste(overlay, (0, 0), overlay)
                canvas.paste(master_template, (0, 0), master_template)

                if class_name:
                    draw = ImageDraw.Draw(canvas)
                    draw.text((1533, 200), f"{class_name} / {absen_1}", font=font, fill=(139, 0, 0, 255))
                    if file_2:
                        draw.text((1533, 678), f"{class_name} / {absen_2}", font=font, fill=(139, 0, 0, 255))

                output_name = f"{class_name}_Absen_{absen_1}-{absen_2}_4R.jpg" if file_2 else f"{class_name}_Absen_{absen_1}_4R.jpg"
                canvas.convert("RGB").save(os.path.join(current_out_4r, output_name), quality=100, dpi=(300, 300))

                processed_pairs += 1
                self.ui(self.progress_bar.set, processed_pairs / total_pairs)
                self.ui(self.lbl_progress_count.configure, text=f"{processed_pairs} / {total_pairs} pasang")

        path_dicek = os.path.join(output_4r_root, "PERLU_DICEK.txt")
        if daftar_dicek:
            with open(path_dicek, "w", encoding="utf-8") as f:
                f.write("Foto berikut sebaiknya dicek manual (framing mepet, logo/wajah tidak terdeteksi, dll):\n\n")
                f.write("\n".join(daftar_dicek) + "\n")
            self.ui(self.lbl_status_text.configure, text=f"Selesai. {len(daftar_dicek)} foto perlu dicek (lihat PERLU_DICEK.txt)", text_color="#E67E22")
            self.log_message(f"Selesai. {len(daftar_dicek)} foto perlu dicek, daftarnya ada di PERLU_DICEK.txt")
        else:
            if os.path.exists(path_dicek):
                os.remove(path_dicek)
            self.ui(self.lbl_status_text.configure, text="Semua foto selesai diproses!", text_color="#28A745")
            self.log_message("Proses otomatisasi pas foto selesai!")

if __name__ == "__main__":
    app = PasFotoApp()
    app.mainloop()