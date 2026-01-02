import os
import re
import base64
import mimetypes
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory, StopWordRemover, ArrayDictionary
)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob      


# ============================================================
#  Avatar bulat 
# ============================================================
def img_to_html_circle(path: str, size: int = 220, alt: str = "") -> str:
    """
    Render gambar lokal (path) sebagai <img> base64 berbentuk lingkaran.
    - size: lebar gambar (px)
    - alt : teks alternatif (SEO/aksesibilitas)
    """
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # CSS: bulat penuh + border merah + shadow lembut
    return (
        f"<img src='data:{mime};base64,{b64}' alt='{alt}' width='{size}' "
        "style='border-radius:50%; border:3px solid #ff4b4b; "
        "box-shadow:0 4px 16px rgba(0,0,0,.25);'/>"
    )


# ============================================================
# ADD 1: Scraping ulasan Google Play
# ============================================================
def _ensure_gps() -> bool:
    """
    Pastikan modul google_play_scraper tersedia.
    Jika tidak ada, tampilkan instruksi pemasangan.
    """
    try:
        import google_play_scraper  # noqa: F401
        from google_play_scraper import Sort, reviews  # noqa: F401
        return True
    except Exception as e:
        st.error(
            "Modul `google_play_scraper` belum terpasang.\n"
            "Jalankan:\n    pip install google-play-scraper\n\n"
            f"Detail: {e}"
        )
        return False


@st.cache_data(show_spinner=True)
def scrape_gplay_reviews(
    package_id: str,
    lang: str = "id",
    country: str = "id",
    count: int = 500,
    sort_key: str = "NEWEST",
    filter_score: int | None = None
) -> pd.DataFrame:
    """
    Ambil ulasan Google Play → DataFrame kolom: ['user','rating','date','text'].
    Param:
      - package_id  : nama paket aplikasi (mis. blibli.mobile.commerce)
      - lang/country: bahasa & negara (kode ISO)
      - count       : jumlah ulasan
      - sort_key    : NEWEST atau MOST_RELEVANT
      - filter_score: None atau skor 1..5 untuk filter
    """
    if not _ensure_gps():
        return pd.DataFrame(columns=["user", "rating", "date", "text"])

    from google_play_scraper import reviews, Sort

    sort = Sort.NEWEST if str(sort_key).upper() == "NEWEST" else Sort.MOST_RELEVANT

    result, _ = reviews(
        package_id,
        lang=lang,
        country=country,
        sort=sort,
        count=int(count),
        filter_score_with=filter_score
    )

    if not result:
        return pd.DataFrame(columns=["user", "rating", "date", "text"])

    # Normalisasi kolom agar konsisten
    df = pd.DataFrame(result)
    keep = ["userName", "score", "at", "content"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.rename(
        columns={"userName": "user", "score": "rating", "at": "date", "content": "text"},
        inplace=True
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", ascending=False, inplace=True, ignore_index=True)
    return df

# ============================================================
# 1) Kamus Normalisasi & Stopword
# ============================================================
norm = {
    "nggak": "tidak", "gak": "tidak", "ga": "tidak", "ngga": "tidak", "tdk": "tidak",
    "yg": "yang", "dr": "dari", "dgn": "dengan", "utk": "untuk", "tp": "tapi",
    "bgt": "banget", "bngt": "banget", "bkn": "bukan", "aja": "saja", "aj": "saja",
    "krn": "karena", "krna": "karena", "sm": "sama", "bgus": "bagus", "trnyata": "ternyata",
    "km": "kamu", "kmu": "kamu", "sy": "saya", "gw": "saya", "ane": "saya",
}

# Tambahan stopword (dipertahankan sesuai kode asli)
more_stop_words = [
    "ada", "adalah", "adanya", "akan", "amat", "an", "anda", "andalah", "antara", "apa", "apaan",
    "apakah", "apalagi", "bagi", "bahkan", "bagaimana", "bahwa", "bahwasanya", "baik", "beberapa",
    "bagian", "banyak", "baru", "bawah", "berikut", "berbagai", "bersama", "bersama-sama", "bisa",
    "boleh", "bukan", "kepada", "kalian", "kami", "kamu", "karena", "dari", "daripada", "dalam",
    "dengan", "di", "dia", "dirimu", "juga", "jika", "lagi", "lain", "lalu", "mana", "maka", "atau",
    "telah", "kemudian", "kalau", "sedang", "dan", "tapi", "dapat", "itu", "saja", "hanya", "lebih",
    "setiap", "sangat", "sudah", "ini", "pada", "lebih", "saja", "lagi", "maka", "sangat", "atau",
    "kami", "atau", "sangat", "hanya", "lebih", "dan", "saja", "maka", "telah", "tetapi", "baru"
]

stop_words = StopWordRemoverFactory().get_stop_words()
stop_words.extend(more_stop_words)
new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)


# ============================================================
# 2) Fungsi Pre-processing (6 langkah)
# ============================================================
def cleaning(text):
    # Hapus mention, hashtag, link
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Hapus angka
    text = re.sub(r'\d+', ' ', text)
    # Hapus ekspresi tidak penting seperti 'hehe', 'wkwk', 'haha', dan kata promosi
    text = re.sub(r'\b(hehe|wkwk|haha|promosi|promo|diskon|gratis ongkir|voucher|cashback)\b', ' ', text)
    # Hapus simbol, tanda baca, dan emoji
    text = re.sub(r'[^\w\s]', ' ', text)
    # Ganti huruf berulang (baguuuus → bagus)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Hapus spasi berlebih dan ubah ke huruf kecil
    text = re.sub(r'\s+', ' ', text).lower().strip()
    return text

def normalisasi(str_text: str) -> str:
    """Ganti slang → baku berdasar kamus norm."""
    for k, v in norm.items():
        str_text = str_text.replace(k, v)
    return str_text

def stopword(str_text: str) -> str:
    """Hapus stopword (Sastrawi + tambahan)."""
    return stop_words_remover_new.remove(str_text)

def stemming(tokens: list[str]) -> str:
    """Stemming token menggunakan Sastrawi kemudian gabung kembali sebagai string."""
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stems = [stemmer.stem(w) for w in tokens]
    return " ".join(stems)

# ============================================================
# 3) Deteksi Sentimen (leksikon sederhana)
# ============================================================
POS = [
    "bagus", "mantap", "cepat", "murah", "puas", "baik", "keren", "nyaman", "top", "recommended",
    "memuaskan", "senang", "lancar", "hebat", "ok", "oke", "terbaik", "worth", "rapi", "tepat"
]
NEG = [
    "buruk", "jelek", "lambat", "lemot", "mahal", "kecewa", "mengecewakan", "error", "rusak",
    "bohong", "tidak sesuai", "susah", "parah", "tolol", "bangsat", "goblok", "sampah", "payah",
    "ngehang", "hang", "lag", "crash", "bug", "refund", "komplain"
]

def deteksi_sentimen(teks: str) -> str:
    teks = teks.lower()
    sp, sn = 0, 0
    for kata in POS:
        if kata in teks:
            sp += 1
    for kata in NEG:
        if kata in teks:
            sn += 1
    if any(k in teks for k in ["tolol", "bangsat", "goblok", "sampah"]):
        return "Negatif"
    if sp > sn:
        return "Positif"
    elif sn > sp:
        return "Negatif"
    return "Netral"

# ============================================================
# 4) Antarmuka Streamlit
# ============================================================

# --- STATE: menu aktif (default = Input Teks)
if "active_menu" not in st.session_state:
    st.session_state["active_menu"] = "Input Teks"


def _go(menu: str) -> None:
    """Setter sederhana untuk berpindah menu (Streamlit akan rerun otomatis)."""
    st.session_state["active_menu"] = menu


# --- Sidebar: 3 tombol vertikal (Input / Upload / Tentang)
st.sidebar.subheader("Pilih Menu")
st.sidebar.button("📄 Input Teks", use_container_width=True, on_click=_go, args=("Input Teks",), key="btn_input")
st.sidebar.button("📂 Upload CSV", use_container_width=True, on_click=_go, args=("Upload File CSV",), key="btn_upload")
st.sidebar.button("ℹ️ Tentang",    use_container_width=True, on_click=_go, args=("Tentang",),     key="btn_about")

choice = st.session_state["active_menu"]


# ============================================================
# HALAMAN: Tentang
# ============================================================
if choice == "Tentang":
    # Header + subjudul
    st.markdown("""
        <h1 style='text-align:center; color:#ff4b4b;'>SENTINEX</h1>
        <h3 style='text-align:center; color:white;'>Sistem Deteksi Sentimen Aplikasi E-Commerce Indonesia</h3>
        <hr style='opacity:.2'>
    """, unsafe_allow_html=True)

    # CSS & foto kiri–kanan (avatar bulat)
    st.markdown("""
    <style>
      .about-photos{display:flex;justify-content:center;gap:60px;flex-wrap:wrap;margin-top:30px}
      .about-card{text-align:center}
      .about-cap{color:#ddd;font-size:14px;margin-top:8px;line-height:1.4}
    </style>
    """, unsafe_allow_html=True)

    # Render dua foto profil
    html_dosen = img_to_html_circle("assets/dosen.jpg", size=300, alt="Dosen Pembimbing")
    html_mhs   = img_to_html_circle("assets/mike.jpg", size=300, alt="Michael Manalu")

    st.markdown(f"""
    <div class="about-photos">
      <div class="about-card">
        {html_dosen}
        <div class="about-cap"><b>Dosen Pembimbing</b><br/>DARIUS ANDANA HARIS, S.KOM., M.T.I.</div>
      </div>
      <div class="about-card">
        {html_mhs}
        <div class="about-cap"><b>Mahasiswa</b><br/>MICHAEL PANGIHUTAN PARDOMUAN MANALU</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer kecil
    st.markdown("""
        <hr style='opacity:.2'>
        <div style='text-align:center; color:#aaa'>
            <b>Dikembangkan oleh:</b> Michael Manalu<br>
            Fakultas Teknologi Informasi, Universitas Tarumanagara
        </div>
    """, unsafe_allow_html=True)

    # Penting: hentikan render bagian lain agar halaman About benar-benar eksklusif
    st.stop()


# ============================================================
# HALAMAN: Utama (Input Teks / Upload CSV + hasil)
# ============================================================

# Judul utama halaman (muncul di semua menu selain "Tentang")
st.title("💬 Sistem Deteksi Sentimen & Pre-Processing Teks")


# --- ADD 2: kontrol scraping Google Play (di sidebar)
st.sidebar.markdown("---")
st.sidebar.subheader("Scraping Google Play")

pkg    = st.sidebar.text_input("Package ID", "blibli.mobile.commerce", key="pkg")
bahasa = st.sidebar.selectbox("Bahasa", ["id", "en"], index=0, key="bahasa")
negara = st.sidebar.selectbox("Negara", ["id", "us", "sg", "my"], index=0, key="negara")
jumlah = st.sidebar.slider("Jumlah ulasan", 100, 10000, 500, 50, key="jumlah")  # max 10k sesuai permintaan
urut   = st.sidebar.selectbox("Urutkan", ["NEWEST", "MOST_RELEVANT"], index=0, key="urut")
skor   = st.sidebar.selectbox("Filter skor", ["Semua", 1, 2, 3, 4, 5], index=0, key="skor")

if st.sidebar.button("Scrape sekarang"):
    filter_score = None if skor == "Semua" else int(skor)
    df_scraped = scrape_gplay_reviews(
        package_id=pkg,
        lang=bahasa,
        country=negara,
        count=jumlah,
        sort_key=urut,
        filter_score=filter_score
    )
    # simpan ke session agar bisa ditampilkan di bawah
    st.session_state["scraped_df"] = df_scraped


# ============================================================
# MODE 1: Input Teks Tunggal
# ============================================================
if choice == "Input Teks":
    teks = st.text_area("Masukkan ulasan:", "KURIR YG RETUR PENGGUNA YG NANGGUNG APLIKASI TOLOL")
    if st.button("Proses"):
        st.subheader("🔧 Tahapan Pre-processing")
        st.write("**Teks asli:**", teks)

        teks1 = cleaning(teks)
        st.write("1️⃣ Cleaning:", teks1)

        teks2 = teks1.lower()
        st.write("2️⃣ Lowercase:", teks2)

        teks3 = normalisasi(" " + teks2 + " ")
        st.write("3️⃣ Normalisasi:", teks3)

        teks4 = stopword(teks3)
        st.write("4️⃣ Stopword Removal:", teks4)

        teks5 = teks4.split()
        st.write("5️⃣ Tokenizing:", teks5)

        teks6 = stemming(teks5)
        st.write("6️⃣ Stemming:", teks6)

        hasil = deteksi_sentimen(teks6)
        st.success(f"Hasil Deteksi Sentimen: **{hasil}**")


# ============================================================
# MODE 2: Upload File CSV
# ============================================================
elif choice == "Upload File CSV":
    file = st.file_uploader("Unggah file CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.subheader("📄 Data Awal")
        st.write(df.head())

        kolom = st.text_input("Nama kolom teks (mis: review):", "review")
        if st.button("Proses CSV"):
            if kolom not in df.columns:
                st.error(f"Kolom '{kolom}' tidak ditemukan.")
            else:
                df["clean"]  = df[kolom].astype(str).apply(cleaning).str.lower()
                df["norm"]   = df["clean"].apply(lambda x: normalisasi(" " + x + " "))
                df["stop"]   = df["norm"].apply(stopword)
                df["token"]  = df["stop"].apply(lambda x: x.split())
                df["stem"]   = df["token"].apply(stemming)
                df["sentimen"] = df["stem"].apply(deteksi_sentimen)

                st.subheader("✅ Hasil Deteksi Sentimen")
                st.write(df[[kolom, "stem", "sentimen"]].head(15))

                st.download_button(
                    "💾 Download Hasil CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "hasil_sentimen.csv",
                    "text/csv"
                )


# ============================================================
# OUTPUT LAPORAN: TABEL EVALUASI → GRAFIK & TABEL DISTRIBUSI
# ============================================================
st.markdown("---")
st.header("📑 Hasil Akhir: Evaluasi Model & Distribusi Sentimen")

# ---------- A) Tabel Evaluasi ----------
st.subheader("A. Tabel Evaluasi (Accuracy, Precision, Recall, F1)")
eval_file = st.file_uploader(
    "Unggah file rekap evaluasi (opsional, default: /content/rekap_4_aplikasi.csv)",
    type=["csv"],
    key="eval_uploader"
)
eval_path_default = "/content/rekap_4_aplikasi.csv"
eval_df = None

try:
    if eval_file is not None:
        eval_df = pd.read_csv(eval_file)
    elif os.path.exists(eval_path_default):
        eval_df = pd.read_csv(eval_path_default)
    else:
        st.warning("Belum ada file rekap evaluasi.")
except Exception as e:
    st.error(f"Gagal membaca file rekap evaluasi: {e}")

if eval_df is not None:
    cols = ["App", "Accuracy", "Precision", "Recall", "F1"]
    eval_df = eval_df[[c for c in cols if c in eval_df.columns]]
    st.dataframe(
        eval_df.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
        use_container_width=True
    )
    st.download_button(
        "💾 Unduh Tabel Evaluasi (CSV)",
        eval_df.to_csv(index=False).encode("utf-8"),
        "tabel_evaluasi_4_aplikasi.csv",
        "text/csv"
    )
    st.markdown("""
**Penjelasan singkat:**
- Lazada memiliki performa terbaik secara keseluruhan dengan akurasi 0.6999, recall 0.5222, dan F1-score tertinggi sebesar 0.49, yang menunjukkan bahwa model FastText + Bi-GRU mampu mengenali sentimen dengan cukup baik dan seimbang.
- Tokopedia berada pada posisi berikutnya dengan akurasi 0.5633, recall 0.5062, dan F1-score 0.4305, menandakan kemampuan model dalam mendeteksi sentimen berada pada tingkat sedang.
- Blibli menempati posisi selanjutnya dengan akurasi 0.6165, recall 0.4675, dan F1-score 0.4281, yang menunjukkan performa model relatif lebih rendah dibandingkan Tokopedia.
- Shopee memiliki performa terendah dengan akurasi 0.6809, recall 0.3516, dan F1-score 0.2976, mengindikasikan bahwa model masih mengalami kesulitan dalam menyeimbangkan presisi dan recall pada data ulasan Shopee.
                
**Kesimpulan:** Model FastText + Bi-GRU bekerja paling baik pada data Lazada, diikuti oleh Tokopedia dan Blibli, sementara performa pada Shopee paling rendah.
""")


# ---------- Distribusi Sentimen (Grafik + Tabel digabung ----------
st.subheader("B. Tabel & Grafik Distribusi Sentimen per Aplikasi ")
st.write("Unggah **4 file CSV** (Blibli, Tokopedia, Lazada, Shopee).")

files = st.file_uploader(
    "Unggah 4 dataset (wajib):",
    type=["csv"],
    accept_multiple_files=True,
    key="dist_uploader_combined"
)


def _guess_app_name(filename: str) -> str | None:
    """Tebak nama app dari nama file (fallback jika pemetaan manual)."""
    n = filename.lower()
    patterns = {
        "Blibli": ["blibli", "bli-bli", "bli bli"],
        "Tokopedia": ["tokopedia", "toko pedia", "t0kopedia"],
        "Lazada": ["lazada", "lzd"],
        "Shopee": ["shopee", "shoppe", "shope", "sopee", "sope", "sopi", "sopy", "shoppee"],
    }
    for app, keys in patterns.items():
        if any(k in n for k in keys):
            return app
    return None


def _detect_label_column(df: pd.DataFrame) -> str:
    """Cari kolom label sentimen paling masuk akal."""
    candidates = ["label", "sentiment", "kelas", "target", "y"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    obj_cols = [c for c in df.columns if df[c].dtype == "O"]
    return obj_cols[-1] if obj_cols else df.columns[-1]


required_apps = ["Blibli", "Tokopedia", "Lazada", "Shopee"]
sent_df = None  # hasil akhir tabel distribusi

if files:
    # (1) Baca semua file
    raw_files = []
    for f in files:
        try:
            df_tmp = pd.read_csv(f)
        except Exception as e:
            st.error(f"Gagal membaca {f.name}: {e}")
            continue
        raw_files.append((f.name, df_tmp))

    # (2) Pemetaan otomatis (dari nama file) + manual jika perlu
    auto_map, unknown, used = {}, [], set()
    for fname, df_tmp in raw_files:
        app_guess = _guess_app_name(fname)
        if app_guess and app_guess not in used:
            auto_map[app_guess] = df_tmp
            used.add(app_guess)
        else:
            unknown.append((fname, df_tmp))

    remaining = [a for a in required_apps if a not in auto_map]
    if unknown and remaining:
        st.info("Beberapa file belum terpetakan otomatis. Silakan pilih aplikasinya secara manual.")
        manual_map = {}
        for idx, (fname, df_tmp) in enumerate(unknown):
            if not remaining:
                break
            ch = st.selectbox(
                f"Pilih aplikasi untuk file: {fname}",
                ["(pilih)"] + remaining,
                key=f"map_combined_{idx}"
            )
            if ch != "(pilih)":
                manual_map[ch] = df_tmp
                remaining.remove(ch)
        auto_map.update(manual_map)

    missing = [a for a in required_apps if a not in auto_map]
    if missing:
        st.warning(f"Belum lengkap: {', '.join(missing)}")

    # (3) Hitung distribusi bila 4 aplikasi lengkap
    if auto_map and not missing:
        rows = []
        for app in required_apps:
            df_ = auto_map[app].copy()
            label_col = _detect_label_column(df_)
            labels = df_[label_col].astype(str).str.strip().str.lower()
            pos = int((labels == "positif").sum())
            neg = int((labels == "negatif").sum())
            neu = int((labels == "netral").sum())
            tot = int(len(df_))
            rows.append({"App": app, "Positif": pos, "Negatif": neg, "Netral": neu, "Total": tot})

        if rows:
            sent_df = pd.DataFrame(rows).sort_values("App").reset_index(drop=True)

            # (4) Grafik batang
            df_plot = sent_df.set_index("App")[["Positif", "Negatif", "Netral"]]
            fig, ax = plt.subplots(figsize=(10, 6))
            df_plot.plot(kind="bar", ax=ax)
            ax.set_title("Distribusi Sentimen Tiap Aplikasi E-Commerce", fontsize=14, weight="bold")
            ax.set_xlabel("Aplikasi", fontsize=12)
            ax.set_ylabel("Jumlah Ulasan", fontsize=12)
            ax.set_ylim(0, 500)  # batas y default (0–500)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.legend(title="Kategori Sentimen")

            # Label angka di atas bar
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="bottom", fontsize=9
                )
            st.pyplot(fig)

            # (5) Tabel + tombol unduh
            st.subheader("Tabel Distribusi Sentimen per Aplikasi")
            st.dataframe(sent_df[["App", "Positif", "Negatif", "Netral", "Total"]], use_container_width=True)
            st.download_button(
                "💾 Unduh Tabel Distribusi (CSV)",
                sent_df.to_csv(index=False).encode("utf-8"),
                "tabel_distribusi_sentimen_4_aplikasi.csv",
                "text/csv"
            )

            # (6) Catatan ringkas
    st.markdown("""
**Penjelasan singkat:**
- Shopee memiliki ulasan positif tertinggi (341), dengan negatif (99) dan netral (60), menunjukkan mayoritas pengguna merasa puas dengan aplikasi ini.
- Lazada mencatat positif (279), negatif (161), dan netral (60), yang berarti pengguna umumnya puas namun masih ada keluhan.
- Blibli memperoleh positif (272), negatif (132), dan netral (96), menandakan pengguna cenderung positif dengan sebagian ulasan netral.
- Tokopedia memiliki positif paling sedikit (174), namun negatif tertinggi (196) dan netral tertinggi (127), menggambarkan pendapat pengguna paling beragam.
    
    Kesimpulan:
    - Positif tertinggi terdapat pada Shopee.
    - Negatif tertinggi terdapat pada Tokopedia.
    - Netral tertinggi juga terdapat pada Tokopedia.
""")
else:
    st.info("Unggah 4 dataset untuk menampilkan tabel distribusi dan penjelasannya.")


# ============================================================
# ADD 3: Tampilkan hasil scraping (jika ada)
# ============================================================
if "scraped_df" in st.session_state:
    st.markdown("---")
    st.header("📥 Hasil Scraping Google Play")
    df_scraped = st.session_state["scraped_df"]

    if df_scraped.empty:
        st.warning("Tidak ada data yang didapat. Coba ganti package/bahasa/negara/jumlah.")
    else:
        st.dataframe(df_scraped, use_container_width=True, height=380)
        st.download_button(
            "💾 Download Hasil Scraping (CSV)",
            df_scraped.to_csv(index=False).encode("utf-8"),
            "hasil_scraping_google_play.csv",
            "text/csv"
        )
