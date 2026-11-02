"""
Pelacakan Objek Berwarna pada Video - antarmuka Streamlit
UTS Visi Komputer, Tema 10.

Fungsi pengolahan citra (resize_frame, praproses, rentang_hsv, segmentasi,
luas_acuan, deteksi, gambar_hasil) diambil dari notebook UTS_Kelompok_NN_v3.ipynb.
Yang baru: pemilihan titik warna manual untuk satu atau beberapa bola,
pemrosesan video dengan progress bar, dan tampilan Streamlit.

Jalankan:  streamlit run app.py
"""
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- konfigurasi
UKURAN_MAX = 960
PANJANG_JEJAK = 30
RASIO_AREA_MIN = 0.15
RASIO_AREA_MAX = 4.0
BULAT_MIN = 0.5

# pengaturan di belakang layar (tidak ditampilkan di website)
BLUR_K = 9
TOL_H, TOL_S, TOL_V = 10, 60, 60
AREA_MIN = 80
MAKS_BOLA = 5
WARNA_BOLA = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255)]  # BGR


# ------------------------------------------------- fungsi dari notebook (P1-P4)
def resize_frame(frame, ukuran_max=UKURAN_MAX):
    h, w = frame.shape[:2]
    skala = ukuran_max / max(h, w)
    if skala >= 1:
        return frame
    return cv2.resize(frame, (int(w * skala), int(h * skala)), interpolation=cv2.INTER_AREA)


def praproses(frame, blur_k):
    if blur_k is None or blur_k <= 1:
        return frame
    if blur_k % 2 == 0:
        blur_k += 1
    return cv2.GaussianBlur(frame, (blur_k, blur_k), 0)


def ambil_patch(frame, x, y, r):
    h, w = frame.shape[:2]
    return frame[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]


def rentang_hsv(frame_bgr, x, y, dh, ds, dv, r=5):
    patch = ambil_patch(frame_bgr, x, y, r)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = np.median(hsv, axis=0).astype(int)
    lower = np.array([h - dh, max(s - ds, 0), max(v - dv, 0)])
    upper = np.array([h + dh, min(s + ds, 255), min(v + dv, 255)])
    return lower, upper, (int(h), int(s), int(v))


def segmentasi(frame_bgr, lower, upper):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h_lo, h_hi = int(lower[0]), int(upper[0])
    sv_lo = [int(lower[1]), int(lower[2])]
    sv_hi = [int(upper[1]), int(upper[2])]
    if h_lo < 0:  # rentang hue melewati 0 (merah/oranye)
        m1 = cv2.inRange(hsv, np.array([0] + sv_lo, np.uint8), np.array([h_hi] + sv_hi, np.uint8))
        m2 = cv2.inRange(hsv, np.array([180 + h_lo] + sv_lo, np.uint8), np.array([179] + sv_hi, np.uint8))
        mask = cv2.bitwise_or(m1, m2)
    elif h_hi > 179:
        m1 = cv2.inRange(hsv, np.array([h_lo] + sv_lo, np.uint8), np.array([179] + sv_hi, np.uint8))
        m2 = cv2.inRange(hsv, np.array([0] + sv_lo, np.uint8), np.array([h_hi - 180] + sv_hi, np.uint8))
        mask = cv2.bitwise_or(m1, m2)
    else:
        mask = cv2.inRange(hsv, np.array([h_lo] + sv_lo, np.uint8), np.array([h_hi] + sv_hi, np.uint8))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def luas_acuan(mask, x, y):
    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(kontur) == 0:
        return None
    jarak = [cv2.pointPolygonTest(c, (float(x), float(y)), True) for c in kontur]
    return cv2.contourArea(kontur[int(np.argmax(jarak))])


def deteksi(mask, luas_ref, area_min):
    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    terbaik, luas_terbaik = None, 0
    for c in kontur:
        a = cv2.contourArea(c)
        if a < area_min:
            continue
        if luas_ref and not (RASIO_AREA_MIN * luas_ref <= a <= RASIO_AREA_MAX * luas_ref):
            continue
        keliling = cv2.arcLength(c, True)
        if keliling == 0 or 4 * np.pi * a / (keliling ** 2) < BULAT_MIN:
            continue
        if a > luas_terbaik:
            terbaik, luas_terbaik = c, a
    if terbaik is None:
        return None
    (cx, cy), radius = cv2.minEnclosingCircle(terbaik)
    bbox = cv2.boundingRect(terbaik)
    return (int(cx), int(cy)), int(radius), bbox


def gambar_hasil(frame, daftar_hasil, daftar_jejak):
    out = frame.copy()
    for k, (hasil, jejak) in enumerate(zip(daftar_hasil, daftar_jejak)):
        warna = WARNA_BOLA[k % len(WARNA_BOLA)]
        n = len(jejak)
        for i in range(1, n):
            if jejak[i - 1] is None or jejak[i] is None:
                continue
            cv2.line(out, jejak[i - 1], jejak[i], warna, 1 + int(3 * i / n))
        if hasil is not None:
            pusat, radius, (bx, by, bw, bh) = hasil
            cv2.circle(out, pusat, radius, warna, 2)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), warna, 2)
            cv2.putText(out, "Bola %d" % (k + 1), (bx, max(by - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)
    n_det = sum(h is not None for h in daftar_hasil)
    cv2.putText(out, "Terdeteksi %d / %d" % (n_det, len(daftar_hasil)), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


# ------------------------------------------------------------- fungsi tambahan
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def info_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Video tidak bisa dibuka. Jika .MOV/HEVC, konversi ke H.264 .mp4 terlebih dahulu.")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    return n, fps


def baca_frame(path, idx=0):
    """Baca berurutan sampai frame ke-idx (lebih akurat untuk video HEVC)."""
    cap = cv2.VideoCapture(path)
    frame, i = None, 0
    while i <= idx:
        ok, f = cap.read()
        if not ok:
            break
        frame, i = f, i + 1
    cap.release()
    if frame is None:
        raise IOError("Tidak ada frame yang bisa dibaca.")
    return resize_frame(frame)


def ke_h264(path_in, path_out):
    """Re-encode ke H.264 agar bisa diputar di browser. Mengembalikan path yang dipakai."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return path_in
    cmd = [exe, "-y", "-loglevel", "error", "-i", path_in, "-vcodec", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path_out]
    try:
        subprocess.run(cmd, check=True)
        return path_out
    except Exception:
        return path_in


def proses_video(path_in, dir_kerja, objek, progress):
    cap = cv2.VideoCapture(path_in)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    raw_asli = os.path.join(dir_kerja, "asli_raw.mp4")
    raw_hasil = os.path.join(dir_kerja, "hasil_raw.mp4")
    w_asli = w_hasil = None
    baris, i = [], 0
    jejak = [[] for _ in objek]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_frame(frame)
        if w_asli is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w_asli = cv2.VideoWriter(raw_asli, fourcc, fps, (w, h))
            w_hasil = cv2.VideoWriter(raw_hasil, fourcc, fps, (w, h))
        frame_b = praproses(frame, BLUR_K)
        daftar_hasil = []
        for k, o in enumerate(objek):
            mask = segmentasi(frame_b, o["lower"], o["upper"])
            hasil = deteksi(mask, o["luas_ref"], AREA_MIN)
            daftar_hasil.append(hasil)
            jejak[k].append(hasil[0] if hasil is not None else None)
            jejak[k] = jejak[k][-PANJANG_JEJAK:]
            baris.append({
                "frame": i,
                "waktu_s": round(i / fps, 3),
                "bola": k + 1,
                "terdeteksi": hasil is not None,
                "x": hasil[0][0] if hasil else None,
                "y": hasil[0][1] if hasil else None,
                "radius_px": hasil[1] if hasil else None,
            })
        w_asli.write(frame)
        w_hasil.write(gambar_hasil(frame, daftar_hasil, jejak))
        i += 1
        if i % 5 == 0:
            progress.progress(min(i / n_total, 1.0), text="Memproses frame %d / %d" % (i, n_total))
    cap.release()
    if w_asli is None:
        raise IOError("Video tidak berisi frame yang bisa dibaca.")
    w_asli.release()
    w_hasil.release()

    progress.progress(1.0, text="Menyiapkan video untuk ditampilkan...")
    f_asli = ke_h264(raw_asli, os.path.join(dir_kerja, "asli.mp4"))
    f_hasil = ke_h264(raw_hasil, os.path.join(dir_kerja, "hasil.mp4"))
    with open(f_asli, "rb") as f:
        b_asli = f.read()
    with open(f_hasil, "rb") as f:
        b_hasil = f.read()
    return pd.DataFrame(baris), b_asli, b_hasil, fps


# ------------------------------------------------------------------ antarmuka
def main():
    st.set_page_config(page_title="Pelacakan Objek Berwarna", page_icon="", layout="wide")
    st.title("Pelacakan Objek (BOLA) Berwarna pada Video")
    st.caption("UTS Visi Komputer · Tema 10 · segmentasi HSV, kontur, dan jejak lintasan")

    file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if file is None:
        st.info("Upload video yang berisi satu atau beberapa bola berwarna untuk memulai.")
        return

    # simpan upload ke file sementara (hanya diganti bila file baru)
    if st.session_state.get("nama_file") != (file.name, file.size):
        st.session_state.clear()
        st.session_state["nama_file"] = (file.name, file.size)
        dir_kerja = tempfile.mkdtemp()
        path = os.path.join(dir_kerja, "input" + os.path.splitext(file.name)[1])
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        st.session_state["dir_kerja"], st.session_state["path"] = dir_kerja, path
    dir_kerja, path = st.session_state["dir_kerja"], st.session_state["path"]

    try:
        n_frame, fps = info_video(path)
    except IOError as e:
        st.error(str(e))
        return

    st.subheader("1. Pilih bola yang dilacak")
    kol_a, kol_b = st.columns([1, 1])
    with kol_a:
        n_bola = int(st.number_input("Jumlah bola", 1, MAKS_BOLA, 1))
        idx = st.number_input("Frame acuan (semua bola harus terlihat)", 0, max(n_frame - 1, 0), 0)
        try:
            frame_ref = baca_frame(path, int(idx))
        except IOError as e:
            st.error(str(e))
            return
        h, w = frame_ref.shape[:2]
        titik = []
        for k in range(n_bola):
            st.markdown("**Bola %d**" % (k + 1))
            x = st.slider("Koordinat x bola %d" % (k + 1), 0, w - 1,
                          int(w * (k + 1) / (n_bola + 1)), key="x%d" % k)
            y = st.slider("Koordinat y bola %d" % (k + 1), 0, h - 1, h // 2, key="y%d" % k)
            titik.append((x, y))

    with kol_b:
        prev = frame_ref.copy()
        for k, t in enumerate(titik):
            warna = WARNA_BOLA[k % len(WARNA_BOLA)]
            cv2.drawMarker(prev, t, warna, cv2.MARKER_CROSS, 30, 2)
            cv2.putText(prev, str(k + 1), (t[0] + 12, t[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)
        st.image(bgr2rgb(prev), caption="Frame acuan (tanda + = titik warna tiap bola)",
                 use_container_width=True)

    st.subheader("2. Proses")
    if st.button("Jalankan pelacakan", type="primary"):
        frame_b = praproses(frame_ref, BLUR_K)
        objek = []
        for k, (x, y) in enumerate(titik):
            lower, upper, hsv_med = rentang_hsv(frame_ref, x, y, TOL_H, TOL_S, TOL_V)
            luas_ref = luas_acuan(segmentasi(frame_b, lower, upper), x, y)
            if luas_ref is None:
                st.warning("Bola %d tidak ditemukan di sekitar titiknya; hasil mungkin kosong. Coba titik lain." % (k + 1))
            objek.append(dict(lower=lower, upper=upper, luas_ref=luas_ref, hsv=hsv_med))
        progress = st.progress(0.0, text="Memulai...")
        try:
            df, b_asli, b_hasil, fps_out = proses_video(path, dir_kerja, objek, progress)
        except IOError as e:
            st.error(str(e))
            return
        progress.empty()
        st.session_state["res"] = dict(df=df, asli=b_asli, hasil=b_hasil, fps=fps_out, objek=objek)

    res = st.session_state.get("res")
    if res is None:
        return

    st.subheader("3. Hasil")
    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown("**Video asli**")
        st.video(res["asli"])
    with kanan:
        st.markdown("**Video hasil pelacakan**")
        st.video(res["hasil"])

    df = res["df"]
    n_frame_total = df["frame"].nunique()

    st.markdown("**Informasi hasil**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Objek dilacak", len(res["objek"]))
    m2.metric("Frame diproses", n_frame_total)
    m3.metric("Durasi", "%.1f detik" % (n_frame_total / res["fps"]))

    for k, o in enumerate(res["objek"]):
        d = df[df["bola"] == k + 1]
        ada = d[d["terdeteksi"]]
        persen = 100 * len(ada) / max(len(d), 1)
        with st.expander("Bola %d" % (k + 1), expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                h_, s_, v_ = o["hsv"]
                st.write("Frame terdeteksi: %d / %d (%.1f%%)" % (len(ada), len(d), persen))
                st.write("Warna acuan (HSV median): H=%d, S=%d, V=%d" % (h_, s_, v_))
                st.write("Rentang HSV: %s sampai %s" % (o["lower"].tolist(), o["upper"].tolist()))
                if len(ada):
                    st.write("Radius rata-rata: %.1f px" % ada["radius_px"].mean())
            with c2:
                if len(ada):
                    st.write("Terdeteksi pertama kali pada frame %d" % ada["frame"].iloc[0])
                    st.write("Terakhir terlihat pada frame %d" % ada["frame"].iloc[-1])
                    st.line_chart(ada.set_index("frame")[["x", "y"]], height=180)
                else:
                    st.write("Bola ini tidak terdeteksi pada frame mana pun.")

    with st.expander("Data per frame"):
        st.dataframe(df, use_container_width=True)
        st.download_button("Unduh CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="hasil_pelacakan.csv", mime="text/csv")


if __name__ == "__main__":
    main()