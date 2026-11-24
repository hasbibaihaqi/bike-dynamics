# BIKE-DYNAMICS
**Dynamic System Simulation for Bike Sharing Management**

> **Tugas Ujian Tengah Semester (UTS) — Pemodelan dan Simulasi**  
> **Judul Penelitian:** *"Pemodelan dan Simulasi Sistem Dinamik untuk Optimalisasi Ketersediaan Armada Sepeda pada Layanan Bike Sharing Berdasarkan Fluktuasi Permintaan dan Kondisi Cuaca"*

---

## 1. Deskripsi Proyek

`BIKE-DYNAMICS` adalah aplikasi berbasis web yang menggabungkan pendekatan **Machine Learning** dan **System Dynamics (Stock and Flow)** untuk mengelola dan mengoptimalkan ketersediaan armada sepeda pada layanan *bike sharing*.

Sistem ini dirancang untuk menjawab tantangan operasional utama:
1. **Fluktuasi Permintaan yang Dinamis:** Permintaan sepeda sangat dipengaruhi oleh waktu (*rush hour* jam kerja vs santai akhir pekan), suhu, kelembapan, kecepatan angin, dan kondisi cuaca ekstrem.
2. **Keterbatasan Kapasitas Armada:** Ketika stok sepeda tersedia (*Available Bikes*) di stasiun habis, sistem mengalami *stockout* yang menimbulkan permintaan tidak terpenuhi (*unmet demand*).
3. **Sirkulasi Kerusakan & Perawatan:** Sepeda yang selesai disewa dapat mengalami keausan/kerusakan yang meningkat pada kondisi cuaca buruk, sehingga harus masuk antrean perawatan (*Maintenance Bikes*) sebelum dapat digunakan kembali.

Aplikasi ini menyajikan analisis data historis, inferensi prediksi permintaan per jam berbasis *Gradient Boosting Regressor*, simulasi sirkulasi armada diskrit per jam, pengujian 5 skenario kebijakan manajerial, analisis sensitivitas parameter, serta ekspor laporan ke SQLite dan CSV.

---

## 2. Sumber Dataset

Dataset yang digunakan adalah **Kaggle Bike Sharing Demand Dataset** (`data/raw/train.csv`).

- **Jumlah Baris:** 10.886 observasi (data per jam tanggal 1 s.d. 19 setiap bulan periode 2011–2012).
- **Jumlah Kolom Asli:** 12 kolom:
  1. `datetime`: Stempel waktu per jam (`YYYY-MM-DD HH:MM:SS`).
  2. `season`: Musim (1: Semi, 2: Panas, 3: Gugur, 4: Dingin).
  3. `holiday`: Indikator hari libur resmi (0: Bukan, 1: Libur).
  4. `workingday`: Indikator hari kerja (1: Hari kerja, 0: Akhir pekan/libur).
  5. `weather`: Kondisi cuaca (1: Cerah/Sedikit berawan, 2: Berkabut/Mendung, 3: Hujan/Salju ringan, 4: Hujan lebat/Ekstrem).
  6. `temp`: Suhu udara aktual dalam derajat Celcius.
  7. `atemp`: Suhu yang dirasakan (*feels-like temperature*) dalam derajat Celcius.
  8. `humidity`: Kelembapan relatif udara (%).
  9. `windspeed`: Kecepatan angin (km/jam).
  10. `casual`: Jumlah sewa pengguna non-registrasi (*dibuang saat preprocessing untuk mencegah data leakage*).
  11. `registered`: Jumlah sewa pengguna terdaftar (*dibuang saat preprocessing untuk mencegah data leakage*).
  12. `count`: Total penyewaan sepeda aktual (**Target Prediksi ML**).

---

## 3. Penjelasan Singkat Metode

Sistem mengadopsi arsitektur terintegrasi dua lapis (*Hybrid ML-SD Framework*):

```
Dataset Historis (Kaggle)
        │
        ▼
[Pipeline Machine Learning] ──► Rekayasa Fitur & Model Gradient Boosting (R² = 0.91)
        │                                 │
        ▼                                 ▼
Prediksi Permintaan per Jam D(t)  ───────► [Mesin Simulasi System Dynamics]
                                                  │
                                                  ▼
                        Konservasi Stok: B_total = B_a(t) + B_r(t) + B_m(t)
                        Evaluasi 5 Skenario Kebijakan & Analisis Sensitivitas
```

### A. Machine Learning Pipeline (Prediksi Permintaan Potensial)
1. **Pencegahan Kebocoran Target (*Data Leakage Prevention*):** Kolom `casual` dan `registered` merupakan komponen aditif pembentuk target ($count = casual + registered$). Kedua kolom ini dibuang sejak awal agar model tidak mengeksploitasi data yang tidak tersedia pada saat waktu operasional masa depan.
2. **Pemisahan Data Kronologis (*Chronological Train-Test Split*):** Menjaga struktur deret waktu tanpa pengacakan (*no random shuffle*), 80% data awal (8.708 baris) untuk pelatihan model dan 20% data akhir (2.178 baris) sebagai data validasi uji murni.
3. **Rekayasa Fitur (*Feature Engineering*):** Transformasi siklikal waktu (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`), penanda jam sibuk (`is_rush_hour`), akhir pekan (`is_weekend`), serta perlakuan khusus anomali sensor kecepatan angin (`is_windspeed_zero`).
4. **Perbandingan 4 Model Regresi:**
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - **Gradient Boosting Regressor (Model Terbaik: $R^2 = 0{,}9089$, MAE = $44{,}25$, RMSE = $65{,}68$, RMSLE = $0{,}3812$)**
5. **Ekspor Artefak:** Model terbaik diekspor ke format `joblib` (`artifacts/demand_model.joblib`) sehingga aplikasi web Flask dapat melakukan inferensi langsung tanpa melakukan *retraining*.

### B. Pemodelan Sistem Dinamik (Stock and Flow Engine)
Sirkulasi armada sepeda dimodelkan menggunakan persamaan diferensi diskrit dengan resolusi langkah waktu per jam ($\Delta t = 1$ jam):

- **Variabel Stok (*Stock*):**
  - $B_a(t)$: *Available Bikes* (Sepeda siap sewa di stasiun).
  - $B_r(t)$: *Rented Bikes* (Sepeda yang sedang aktif digunakan pengguna).
  - $B_m(t)$: *Maintenance Bikes* (Sepeda dalam antrean atau proses perawatan di bengkel).

- **Variabel Aliran (*Flow*):**
  - $R(t)$: *Rental Rate* (Laju penyewaan aktual) $= \min(B_a(t), D(t))$.
  - $D_{\text{unmet}}(t)$: *Unmet Demand* (Permintaan tidak terpenuhi) $= \max(0, D(t) - R(t))$.
  - $U(t)$: *Return Rate* (Laju pengembalian sepeda) $= B_r(t) \cdot \min\left(1, \frac{\Delta t}{\tau_r}\right)$, dengan $\tau_r$ adalah rata-rata durasi sewa.
  - $M_i(t)$: *Maintenance Inflow* (Laju sepeda rusak masuk perawatan) $= U(t) \cdot \alpha_d^{\text{eff}}(t)$, dengan $\alpha_d^{\text{eff}}(t) = \min(0{,}95, \alpha_d \cdot \phi_w(W(t)))$ yang memperhitungkan faktor risiko cuaca.
  - $M_o(t)$: *Maintenance Outflow* (Laju sepeda selesai perbaikan) $= B_m(t) \cdot \min\left(1, \frac{\Delta t}{\tau_m}\right)$, dengan $\tau_m$ adalah durasi perbaikan.

- **Persamaan Pembaruan Stok (*Difference Equations*):**
  $$\begin{aligned}
  B_a(t+1) &= B_a(t) - R(t) + U(t) - M_i(t) + M_o(t) \\
  B_r(t+1) &= B_r(t) + R(t) - U(t) \\
  B_m(t+1) &= B_m(t) + M_i(t) - M_o(t)
  \end{aligned}$$

- **Hukum Konservasi Total Armada:**
  $$B_{\text{total}} = B_a(t) + B_r(t) + B_m(t) = \text{Konstan}$$
  Terbukti konsisten secara matematis dan komputasional sepanjang waktu simulasi tanpa ada stok bernilai negatif.

### C. Eksperimen 5 Skenario Kebijakan
1. **Skenario 1 (Normal Condition):** Kondisi operasional standar ($B_{\text{total}} = 600$, durasi perbaikan 6 jam, pengali permintaan 1.0).
2. **Skenario 2 (High Demand +20%):** Lonjakan permintaan 20% pada kapasitas armada eksisting.
3. **Skenario 3 (Fleet Expansion +20%):** Penambahan kapasitas armada menjadi 720 unit pada permintaan normal.
4. **Skenario 4 (Maintenance Optimization):** Percepatan perbaikan bengkel sebesar 50% ($\tau_m = 3$ jam).
5. **Skenario 5 (High Demand + Fleet Expansion):** Kombinasi lonjakan permintaan 20% yang diimbangi penambahan armada 20% untuk menguji daya serap ekspansi.

Evaluasi kelayakan skenario dirangking secara objektif menggunakan **Composite Policy Effectiveness Score**:
$$\text{Score} = 0{,}50 \times \text{Service Level} + 0{,}25 \times \text{Kesehatan Armada} + 0{,}25 \times \text{Produktivitas Aset}$$

---

## 4. Struktur Folder Proyek

```text
bike-dynamics/
├── app.py                              # Entry point inisialisasi aplikasi Flask
├── config.py                           # Konfigurasi path, database SQLite, dan parameter default
├── requirements.txt                    # Daftar pustaka dependensi Python
├── README.md                           # Panduan dan dokumentasi resmi proyek
├── .gitignore                          # Konfigurasi pengecualian cache & database lokal
├── data/
│   ├── raw/train.csv                   # Dataset mentah Kaggle Bike Sharing Demand
│   └── processed/cleaned_data.csv      # Dataset bersih hasil preprocessing
├── notebooks/
│   ├── 01_eda_dataset.ipynb            # Analisis eksplorasi data (EDA) & statistik
│   ├── 02_preprocessing.ipynb          # Pembersihan data & rekayasa fitur
│   ├── 03_train_demand_model.ipynb     # Pelatihan 4 model regresi ML
│   ├── 04_model_evaluation.ipynb       # Evaluasi komparatif & seleksi model terbaik
│   ├── 05_system_dynamics.ipynb        # Pemodelan CLD, SFD, persamaan matematis & sensitivitas
│   └── 06_scenario_experiments.ipynb   # Simulasi 5 skenario kebijakan & perhitungan skor
├── artifacts/
│   ├── demand_model.joblib             # Model terbaik terlatih (Gradient Boosting)
│   ├── preprocessing_pipeline.joblib   # Pipeline transformasi data ColumnTransformer
│   ├── candidate_models.joblib         # Model kandidat pembanding
│   ├── feature_config.json             # Metadata urutan fitur inferensi
│   ├── model_metrics.json              # Rekapitulasi metrik evaluasi model ML
│   ├── eda_summary.json                # Ringkasan statistik EDA untuk halaman web
│   ├── preprocessing_summary.json      # Catatan pembersihan data & anti-leakage
│   ├── sd_assumptions.json             # Parameter asumsi operasional System Dynamics
│   ├── scenario_experiments.json       # Hasil metrik & perankingan 5 skenario
│   ├── simulation_horizon_input.csv    # Deret input 168 jam untuk simulasi
│   ├── test_predictions.csv            # Hasil prediksi vs data aktual periode uji
│   ├── scenario_comparison_summary.csv # Tabel ringkasan perbandingan skenario
│   ├── scenario_timeseries.csv         # Deret waktu simulasi 168 jam kelima skenario
│   └── [subfolder gambar visualisasi]  # Grafik EDA, evaluasi ML, CLD, dan skenario
├── models/
│   ├── __init__.py                     # Interface modul models
│   ├── demand_predictor.py             # DemandPredictor (inferensi artefak ML)
│   ├── equations.py                    # Formulasi aliran matematis Stock and Flow
│   ├── scenarios.py                    # ScenarioRunner (eksekutor 5 skenario kebijakan)
│   ├── system_dynamics.py              # SystemDynamicsSimulator (mesin simulasi diskrit)
│   └── validation.py                   # Validasi ketat batas input form pengguna
├── services/
│   ├── __init__.py                     # Interface modul services
│   ├── data_service.py                 # Penyedia data overview dataset & pola permintaan
│   ├── prediction_service.py           # Penghubung inferensi ML dan horizon simulasi
│   ├── simulation_service.py           # Pengelola eksekusi simulasi kustom & SQLite DB
│   └── analysis_service.py             # Layanan perbandingan skenario, sensitivitas & CSV
├── routes/
│   ├── __init__.py                     # Helper rendering fallback Jinja2/JSON
│   ├── dashboard.py                    # Blueprint rute Dashboard utama (KPI & grafik)
│   ├── dataset.py                      # Blueprint rute Dataset Management
│   ├── analysis.py                     # Blueprint rute Demand Analysis & Model Performance
│   ├── simulation.py                   # Blueprint rute System Dynamics, Simulation Center, dsb.
│   └── results.py                      # Blueprint rute Results and Reports & Ekspor CSV
├── templates/
│   ├── base.html                       # Layout dasar sidebar responsif & topbar status
│   ├── dashboard.html                  # Halaman 1: Ringkasan KPI dan grafik stok
│   ├── dataset.html                    # Halaman 2: Pratinjau dataset & statistik deskriptif
│   ├── demand_analysis.html            # Halaman 3: Pola permintaan & kalkulator prediksi
│   ├── model_performance.html          # Halaman 4: Metrik perbandingan algoritma ML
│   ├── system_dynamics.html            # Halaman 5: Visualisasi CLD, SFD, dan formulasi matematis
│   ├── simulation_center.html          # Halaman 6: Pusat simulasi kustom interaktif
│   ├── scenario_analysis.html          # Halaman 7: Perbandingan 5 skenario kebijakan
│   ├── sensitivity_analysis.html       # Halaman 8: Uji sensitivitas 5 parameter
│   └── results.html                    # Halaman 9: Riwayat simulasi SQLite & laporan cetak
├── static/
│   ├── css/style.css                   # Gaya CSS modern, responsif, dan print-friendly
│   └── js/charts.js                    # Wrapper helper konfigurasi Chart.js
├── instance/
│   └── bike_dynamics.db                # Database SQLite lokal riwayat simulasi pengguna
└── tests/
    ├── __init__.py                     # Modul tests
    ├── conftest.py                     # Fixture client pengujian Flask
    ├── test_flask_routes.py            # Uji kelayakan 9 rute menu, API, dan ekspor CSV
    ├── test_ml_predictor.py            # Uji kesiapan model ML dan inferensi input
    └── test_system_dynamics.py         # Uji matematis konservasi armada & 5 skenario
```

---

## 5. Kebutuhan Instalasi

Pastikan lingkungan sistem Anda memenuhi spesifikasi berikut:
- **Sistem Operasi:** Windows 10/11, macOS, atau Linux.
- **Python:** Versi **Python 3.10** atau **Python 3.11** (disarankan 3.11).
- **Peramban Web (*Browser*):** Google Chrome, Mozilla Firefox, Microsoft Edge, atau Safari (mendukung JavaScript ES6+ untuk rendering Chart.js).
- **Manajer Paket:** `pip` versi terbaru.

---

## 6. Cara Memasang Dependensi

### Langkah 1: Buka Terminal / PowerShell
Navigasikan ke direktori utama proyek:
```bash
cd "c:\Users\HS\Desktop\UTS MOSI (TEORI DAN PRAK) - PEMODELAN DAN SIMULASI\bike-dynamics"
```

### Langkah 2: Buat Lingkungan Virtual (*Virtual Environment*)
```bash
# Windows
python -m venv .venv

# Linux / macOS
python3 -m venv .venv
```

### Langkah 3: Aktifkan Virtual Environment
```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt / CMD)
.\.venv\Scripts\activate.bat

# Linux / macOS (Bash / Zsh)
source .venv/bin/activate
```

### Langkah 4: Pasang Dependensi dari `requirements.txt`
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Daftar dependensi yang terpasang meliputi:
- `Flask>=3.0.0` (Web framework)
- `pandas>=2.1.0` & `numpy>=1.26.0` (Manipulasi data & numerik)
- `scikit-learn>=1.4.0` (Algoritma Machine Learning & transformer)
- `joblib>=1.3.2` (Serialisasi model terlatih)
- `matplotlib>=3.8.0` & `seaborn>=0.13.0` (Visualisasi data statis)
- `jupyter`, `nbconvert`, `nbformat` (Pengelolaan & eksekusi notebook)
- `pytest>=8.0.0` (Kerangka pengujian otomatis)

---

## 7. Cara Menjalankan Notebook

Seluruh alur riset dan eksperimen terdokumentasi dalam 6 notebook di folder `notebooks/`. Anda dapat menjalankannya dengan dua metode:

### Metode A: Eksekusi Otomatis Berurutan via Command Line (Disarankan)
Pastikan virtual environment aktif, lalu jalankan perintah berikut secara berurutan:
```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda_dataset.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/03_train_demand_model.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/04_model_evaluation.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/05_system_dynamics.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/06_scenario_experiments.ipynb --inplace
```

### Metode B: Menjalankan Secara Interaktif via Jupyter Lab / Notebook
Buka server Jupyter:
```bash
jupyter notebook
# atau
jupyter lab
```
Buka browser dan jalankan notebook sesuai urutan kronologis:
1. `01_eda_dataset.ipynb`: Memeriksa dataset, missing values, dan menghasilkan `artifacts/eda_summary.json`.
2. `02_preprocessing.ipynb`: Menghapus kolom kebocoran target (`casual`, `registered`), rekayasa fitur waktu, dan menyimpan `data/processed/cleaned_data.csv`.
3. `03_train_demand_model.ipynb`: Melatih 4 algoritma regresi pada pembagian kronologis 80:20.
4. `04_model_evaluation.ipynb`: Membandingkan metrik evaluasi pada data uji dan mengekspor `artifacts/demand_model.joblib`.
5. `05_system_dynamics.ipynb`: Memodelkan Stock and Flow, memverifikasi konservasi stok, dan melakukan sweep analisis sensitivitas.
6. `06_scenario_experiments.ipynb`: Menjalankan simulasi 5 skenario kebijakan dan mengekspor hasilnya ke artefak JSON/CSV.

---

## 8. Cara Menjalankan Pengujian Otomatis (*Pytest*)

Untuk memverifikasi integritas seluruh fungsi sistem (rute Flask, API JSON, pemuatan model tanpa training ulang, validasi batas input, serta hukum konservasi stok sistem dinamik):
```bash
pytest -v
```

Hasil pengujian yang diharapkan:
```text
============================= 20 passed in 4.25s ==============================
```

---

## 9. Cara Menjalankan Aplikasi Web Flask

### Langkah 1: Jalankan Server Aplikasi
Setelah dependensi terpasang dan virtual environment aktif:
```bash
python app.py
```

Output pada terminal akan menampilkan:
```text
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Langkah 2: Akses Aplikasi di Browser
Buka browser favorit Anda dan akses alamat:
```
http://127.0.0.1:5000
```

### Langkah 3: Navigasi 9 Menu Utama Aplikasi
1. **1. Dashboard (`/`):** Ringkasan kartu KPI operasional armada (*Total Fleet*, *Available Bikes*, *Rented Bikes*, *Maintenance Bikes*, *Service Level*, *Unmet Demand*, *Utilization Rate*) dan grafik interaktif sirkulasi sepeda per jam.
2. **2. Dataset Management (`/dataset`):** Status ketersediaan dataset raw dan preprocessed, rentang waktu observasi, pratinjau tabel data bersih, dan tabel statistik deskriptif.
3. **3. Demand Analysis (`/demand-analysis`):** Pola historis permintaan sepeda berdasarkan jam komuter vs akhir pekan, musim, cuaca, serta **Form Kalkulator Prediksi Permintaan Interaktif** menggunakan model Machine Learning terlatih.
4. **4. Model Performance (`/model-performance`):** Tabel komparatif kinerja 4 algoritma regresi (MAE, RMSE, $R^2$, RMSLE, waktu pelatihan), justifikasi seleksi model terbaik, dan grafik visualisasi *Aktual vs Prediksi*.
5. **5. System Dynamics (`/system-dynamics`):** Penjelasan konseptual Causal Loop Diagram (CLD), Stock and Flow Diagram (SFD), variabel stok-aliran, formulasi matematis diferensi diskrit, dan tabel dokumentasi asumsi parameter.
6. **6. Simulation Center (`/simulation-center`):** Laboratorium simulasi kustom pengguna. Pengguna dapat mengubah kapasitas armada, durasi sewa, tingkat kerusakan, durasi bengkel, durasi simulasi (hingga 720 jam), serta menyimpannya otomatis ke riwayat database SQLite.
7. **7. Scenario Analysis (`/scenario-analysis`):** Halaman perbandingan komparatif **5 skenario kebijakan manajerial** dilengkapi grafik perbandingan permintaan vs unmet demand, grafik dinamika stok tersedia, serta penentuan rekomendasi skenario terbaik berdasarkan skor komposit.
8. **8. Sensitivity Analysis (`/sensitivity-analysis`):** Eksperimen uji ketahanan sistem dengan menyapu (*sweeping*) nilai parameter (misal tingkat kerusakan, durasi bengkel, kapasitas armada) terhadap tingkat layanan (*service level*) dan *unmet demand*.
9. **9. Results and Reports (`/results`):** Ringkasan hasil simulasi aktif, riwayat simulasi tersimpan dari database SQLite, tombol cetak laporan ramah printer (*Print Report*), dan tombol **Unduh Data CSV** (`/results/export-csv`).

### Langkah 4: Endpoint Pemeriksaan Kesehatan & API
- **Health Check:** `GET http://127.0.0.1:5000/health` (Memeriksa kesiapan aplikasi dan ketersediaan model ML).
- **API Prediksi Permintaan:** `POST http://127.0.0.1:5000/api/predict-demand` (Menerima parameter JSON kondisi cuaca/waktu dan mengembalikan angka estimasi permintaan sepeda).
- **API Simulasi Sistem Dinamik:** `POST http://127.0.0.1:5000/api/simulate` (Menerima konfigurasi parameter armada dan mengembalikan hasil deret waktu simulasi).

---

## 10. Catatan Akademik & Pertanggungjawaban Model

- **Distingsi Data Empiris vs Asumsi:** Data historis Kaggle hanya menyediakan observasi permintaan penyewaan dan variabel iklim. Parameter internal sistem dinamik (kapasitas total armada 600 unit, rata-rata durasi sewa 1,5 jam, tingkat kerusakan dasar 4%, durasi perbaikan 6 jam) merupakan **asumsi peneliti yang dirumuskan secara rasional** untuk mengkaji sistem tertutup dan dapat diuji elastisitasnya melalui menu *Sensitivity Analysis*.
- **Tanpa Retraining di Web Server:** Aplikasi Flask tidak pernah melakukan pelatihan model berulang saat runtime; model dipersiapkan secara ilmiah melalui alur notebook dan dimuat secara efisien melalui format serialisasi `joblib`.
- **Integritas Konservasi:** Penjumlahan seluruh unit sepeda pada sistem terkonservasi penuh ($B_a + B_r + B_m = B_{\text{total}}$), memastikan tidak ada sepeda yang tercipta atau lenyap tanpa melalui aliran yang terdefinisi.
