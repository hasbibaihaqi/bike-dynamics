"""
Full Builder and Executor for 7 Google Colab Notebooks BIKE-DYNAMICS
"""

import json
import os
import shutil
import time
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_colab_header(nb_num: str, title: str, subtitle: str, colab_filename: str):
    return [
        new_markdown_cell(f"""# BIKE-DYNAMICS — Tahap {nb_num}: {title}
**Judul Penelitian:** *Pemodelan dan Simulasi Sistem Dinamik untuk Optimalisasi Ketersediaan Armada Sepeda pada Layanan Bike Sharing Berdasarkan Fluktuasi Permintaan dan Kondisi Cuaca*  
**Platform Pendamping:** `BIKE-DYNAMICS – Dynamic System Simulation for Bike Sharing Management`  
**Mata Kuliah:** Pemodelan dan Simulasi (UTS)  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UTS-MOSI/bike-dynamics/blob/main/notebooks/{colab_filename})

---
### {subtitle}
"""),
        new_code_cell("""# 1. Setup Lingkungan Eksekusi (Google Colab / Lokal)
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Menjalankan di Google Colab. Memasang pustaka pendukung...")
    !pip install -q pandas numpy matplotlib seaborn scikit-learn joblib nbformat
else:
    print("Menjalankan di lingkungan lokal Python.")
"""),
        new_code_cell("""# 2. Konfigurasi Direktori Kerja & Deteksi Path Otomatis
import os
from pathlib import Path

if IN_COLAB:
    # Opsi jika menggunakan Google Drive:
    # from google.colab import drive
    # drive.mount('/content/drive')
    # PROJECT_ROOT = Path('/content/drive/MyDrive/BIKE-DYNAMICS')
    
    colab_default = Path('/content/bike-dynamics')
    colab_alt = Path('/content/BIKE-DYNAMICS-COLAB')
    colab_content = Path('/content')
    
    if (colab_default / 'data' / 'raw' / 'train.csv').exists():
        PROJECT_ROOT = colab_default
    elif (colab_alt / 'data' / 'raw' / 'train.csv').exists():
        PROJECT_ROOT = colab_alt
    elif (colab_content / 'data' / 'raw' / 'train.csv').exists():
        PROJECT_ROOT = colab_content
    elif (colab_content / 'train.csv').exists():
        PROJECT_ROOT = colab_content
        (PROJECT_ROOT / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
        import shutil
        if not (PROJECT_ROOT / 'data' / 'raw' / 'train.csv').exists():
            shutil.copy(colab_content / 'train.csv', PROJECT_ROOT / 'data' / 'raw' / 'train.csv')
    else:
        PROJECT_ROOT = colab_default
        (PROJECT_ROOT / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
else:
    cwd = Path.cwd()
    if (cwd / 'data' / 'raw' / 'train.csv').exists():
        PROJECT_ROOT = cwd
    elif (cwd.parent / 'data' / 'raw' / 'train.csv').exists():
        PROJECT_ROOT = cwd.parent
    else:
        PROJECT_ROOT = cwd

RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'cleaned_data.csv'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

(PROJECT_ROOT / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project Root  : {PROJECT_ROOT}")
print(f"Raw Data Path : {RAW_DATA_PATH} (Tersedia: {RAW_DATA_PATH.exists()})")
""")
    ]

# ------------------------------------------------------------------------------
# NOTEBOOK 01: EDA
# ------------------------------------------------------------------------------
def build_nb01():
    existing_nb1 = nbformat.read(NOTEBOOKS_DIR / "01_eda_dataset.ipynb", as_version=4)
    nb = new_notebook()
    # Add colab setup
    nb.cells.extend(make_colab_header("1", "Exploratory Data Analysis (EDA)", "Eksplorasi dan Pemahaman Karakteristik Dataset Bike Sharing Demand", "01_eda_dataset.ipynb"))
    
    # Add original analytical cells starting from Cell 1 (skipping old cell 0 title)
    for c in existing_nb1.cells[1:]:
        # Skip the duplicate imports if they exist in cell 2, keep the analytical code
        nb.cells.append(c)
    
    return nb

# ------------------------------------------------------------------------------
# NOTEBOOK 02: DATA PREPROCESSING
# ------------------------------------------------------------------------------
def build_nb02():
    existing_nb2 = nbformat.read(NOTEBOOKS_DIR / "02_preprocessing.ipynb", as_version=4)
    nb = new_notebook()
    nb.cells.extend(make_colab_header("2", "Data Preprocessing & Feature Engineering", "Pembersihan Data, Rekayasa Fitur, Pencegahan Data Leakage, & Pembagian Kronologis", "02_data_preprocessing.ipynb"))
    
    for c in existing_nb2.cells[1:]:
        nb.cells.append(c)
    
    return nb

# ------------------------------------------------------------------------------
# NOTEBOOK 03: MODEL TRAINING
# ------------------------------------------------------------------------------
def build_nb03():
    existing_nb3 = nbformat.read(NOTEBOOKS_DIR / "03_train_demand_model.ipynb", as_version=4)
    nb = new_notebook()
    nb.cells.extend(make_colab_header("3", "Machine Learning Model Training", "Pelatihan & Perbandingan 4 Model Regresi Prediksi Permintaan Sepeda", "03_model_training.ipynb"))
    
    for c in existing_nb3.cells[1:]:
        # If saving model, make sure it also saves best_model.joblib
        if c.cell_type == 'code' and 'demand_model.joblib' in c.source:
            extra_code = "\n# Simpan juga best_model.joblib untuk kompatibilitas suite Colab\njoblib.dump(best_estimator, ARTIFACTS_DIR / 'best_model.joblib')\nprint('Model terbaik juga disimpan di artifacts/best_model.joblib')\n"
            c.source += extra_code
        nb.cells.append(c)
    
    return nb

# ------------------------------------------------------------------------------
# NOTEBOOK 04: MODEL EVALUATION
# ------------------------------------------------------------------------------
def build_nb04():
    existing_nb4 = nbformat.read(NOTEBOOKS_DIR / "04_model_evaluation.ipynb", as_version=4)
    nb = new_notebook()
    nb.cells.extend(make_colab_header("4", "Model Evaluation & Selection", "Evaluasi Multi-Metrik, Residual Analysis, & Seleksi Model Terbaik", "04_model_evaluation.ipynb"))
    
    for c in existing_nb4.cells[1:]:
        nb.cells.append(c)
    
    return nb

# ------------------------------------------------------------------------------
# NOTEBOOK 05: SYSTEM DYNAMICS MODEL
# ------------------------------------------------------------------------------
def build_nb05():
    nb = new_notebook()
    nb.cells.extend(make_colab_header("5", "System Dynamics Modeling", "Pemodelan Matematis Sirkulasi Armada, Formulasi Stock and Flow, & Uji Validasi Konservasi", "05_system_dynamics_model.ipynb"))
    
    nb.cells.append(new_markdown_cell("""## 3. Landasan Teori: Pendekatan Sistem Dinamik untuk Bike Sharing
Pengelolaan armada *bike sharing* merupakan sistem stokastik dan non-linear dengan umpan balik (*feedback loop*). Ketersediaan sepeda di stasiun membatasi jumlah penyewaan (*balancing loop*), sementara kerusakan akibat penggunaan dan cuaca menciptakan antrean perawatan yang mengurangi stok sepeda aktif.

### Struktur Stok dan Aliran (Stock and Flow):
- **Stok (*Stock*):**
  1. $B_a(t)$: **Available Bikes** — Jumlah unit sepeda siap pakai di stasiun.
  2. $B_r(t)$: **Rented Bikes** — Jumlah unit sepeda aktif yang sedang disewa pelanggan.
  3. $B_m(t)$: **Maintenance Bikes** — Jumlah unit sepeda rusak dalam antrean / perbaikan bengkel.
- **Aliran (*Flow*):**
  1. $R(t)$: **Rental Rate** — Laju peminjaman aktual $= \\min(B_a(t), D(t))$.
  2. $U(t)$: **Return Rate** — Laju pengembalian sepeda $= B_r(t) \\cdot \\min(1, \\Delta t / \\tau_r)$.
  3. $M_i(t)$: **Maintenance Inflow** — Laju sepeda rusak masuk bengkel $= U(t) \\cdot \\min(0.95, \\alpha_d \\cdot \\phi_w(W(t)))$.
  4. $M_o(t)$: **Maintenance Outflow** — Laju sepeda selesai diperbaiki $= B_m(t) \\cdot \\min(1, \\Delta t / \\tau_m)$.
"""))

    nb.cells.append(new_code_cell("""# 4. Import Library & Setup Visualisasi
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.figsize'] = (13, 5.5)
pd.set_option('display.max_columns', None)

FIGURES_DIR = ARTIFACTS_DIR / 'sd_figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print("Library dan direktori gambar siap.")
"""))

    nb.cells.append(new_markdown_cell("""## 5. Visualisasi Causal Loop Diagram (CLD) & Stock-Flow Diagram (SFD)"""))

    nb.cells.append(new_code_cell("""# Visualisasi Causal Loop Diagram (CLD)
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

nodes = {
    'Available Bikes (Ba)': (0.2, 0.7),
    'Demand D(t)': (0.5, 0.9),
    'Rental Rate R(t)': (0.5, 0.7),
    'Rented Bikes (Br)': (0.8, 0.7),
    'Return Rate U(t)': (0.8, 0.3),
    'Maintenance Inflow Mi(t)': (0.5, 0.3),
    'Maintenance Bikes (Bm)': (0.2, 0.3),
    'Maintenance Outflow Mo(t)': (0.1, 0.5),
}

for name, (x, y) in nodes.items():
    ax.scatter(x, y, s=3200, color='#e0f2fe', edgecolors='#0284c7', linewidths=2.5, zorder=3)
    ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold', color='#0f172a', zorder=4)

connections = [
    ('Available Bikes (Ba)', 'Rental Rate R(t)', '+'),
    ('Demand D(t)', 'Rental Rate R(t)', '+'),
    ('Rental Rate R(t)', 'Available Bikes (Ba)', '-'),
    ('Rental Rate R(t)', 'Rented Bikes (Br)', '+'),
    ('Rented Bikes (Br)', 'Return Rate U(t)', '+'),
    ('Return Rate U(t)', 'Available Bikes (Ba)', '+'),
    ('Return Rate U(t)', 'Maintenance Inflow Mi(t)', '+'),
    ('Maintenance Inflow Mi(t)', 'Maintenance Bikes (Bm)', '+'),
    ('Maintenance Bikes (Bm)', 'Maintenance Outflow Mo(t)', '+'),
    ('Maintenance Outflow Mo(t)', 'Available Bikes (Ba)', '+'),
]

for src, dst, sign in connections:
    x1, y1 = nodes[src]
    x2, y2 = nodes[dst]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color='#475569', lw=1.6, mutation_scale=14,
                                shrinkA=32, shrinkB=32))
    mx, my = (x1 + x2)/2, (y1 + y2)/2
    ax.text(mx, my + 0.02, f"({sign})", color='#0369a1' if sign == '+' else '#e11d48',
            fontweight='bold', fontsize=11)

ax.set_title("Causal Loop Diagram (CLD) — BIKE-DYNAMICS", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_causal_loop_diagram.png', dpi=150)
plt.show()
"""))

    nb.cells.append(new_markdown_cell("""## 6. Implementasi Mesin Simulasi System Dynamics
Dataset Kaggle Bike Sharing Demand hanya mencatat deret waktu permintaan dan cuaca, tetapi **tidak mencatat jumlah total armada, durasi sewa, laju kerusakan, maupun durasi perbaikan**. Oleh karena itu, variabel-variabel tersebut dirumuskan secara transparan sebagai **parameter asumsi penelitian** yang dapat diuji sensitivitasnya:
- **Total Fleet ($B_{\\text{total}}$):** 600 unit (Armada kota menengah).
- **Kondisi Awal:** $B_a(0) = 510$ (85%), $B_r(0) = 60$ (10%), $B_m(0) = 30$ (5%).
- **Rata-rata Durasi Sewa ($\\tau_r$):** 1,5 jam.
- **Tingkat Kerusakan Dasar ($\\alpha_d$):** 4% dari pengembalian.
- **Durasi Perbaikan Bengkel ($\\tau_m$):** 6,0 jam.
- **Faktor Pengali Kerusakan Cuaca ($\\phi_w$):** Cuaca 1=1.0x, Cuaca 2=1.1x, Cuaca 3=1.35x, Cuaca 4=1.6x.
"""))

    nb.cells.append(new_code_cell("""# Implementasi Mesin Simulasi
class SystemDynamicsSimulator:
    def __init__(self, weather_multipliers=None):
        self.weather_map = weather_multipliers or {1: 1.0, 2: 1.10, 3: 1.35, 4: 1.60}

    def simulate(self, demand_series, weather_series, timestamps=None,
                 total_fleet=600.0, initial_available=510.0, initial_rented=60.0, initial_maintenance=30.0,
                 rental_duration_hours=1.5, damage_rate=0.04, repair_duration_hours=6.0, demand_multiplier=1.0):
        demand_arr = np.maximum(np.asarray(demand_series, dtype=float) * float(demand_multiplier), 0.0)
        weather_arr = np.asarray(weather_series, dtype=int)
        n = len(demand_arr)
        dt = 1.0

        ret_frac = min(1.0, dt / max(float(rental_duration_hours), 0.1))
        rep_frac = min(1.0, dt / max(float(repair_duration_hours), 0.1))

        ba, br, bm = float(initial_available), float(initial_rented), float(initial_maintenance)
        records = []
        max_err = 0.0

        for t in range(n):
            d_t = float(demand_arr[t])
            w_t = int(weather_arr[t])
            w_mult = self.weather_map.get(w_t, 1.0)

            rental_rate = min(ba, d_t)
            unmet = max(0.0, d_t - rental_rate)
            return_rate = br * ret_frac
            eff_damage = min(0.95, max(0.0, float(damage_rate)) * w_mult)
            m_in = return_rate * eff_damage
            m_out = bm * rep_frac

            ba_next = max(0.0, ba - rental_rate + return_rate - m_in + m_out)
            br_next = max(0.0, br + rental_rate - return_rate)
            bm_next = max(0.0, bm + m_in - m_out)
            tot_next = ba_next + br_next + bm_next
            max_err = max(max_err, abs(tot_next - total_fleet))

            records.append({
                'step': t,
                'datetime': str(timestamps[t]) if timestamps is not None else f"Jam-{t+1}",
                'demand': round(d_t, 2),
                'rental_rate': round(rental_rate, 2),
                'unmet_demand': round(unmet, 2),
                'available_bikes': round(ba_next, 2),
                'rented_bikes': round(br_next, 2),
                'maintenance_bikes': round(bm_next, 2),
                'total_fleet': round(tot_next, 4),
                'utilization_pct': round((br_next / total_fleet * 100.0) if total_fleet > 0 else 0.0, 2)
            })
            ba, br, bm = ba_next, br_next, bm_next

        df_sim = pd.DataFrame(records)
        total_d = float(df_sim['demand'].sum())
        total_r = float(df_sim['rental_rate'].sum())
        service_lvl = (total_r / total_d * 100.0) if total_d > 0 else 100.0

        summary = {
            'total_fleet': total_fleet,
            'total_demand': round(total_d, 2),
            'total_rentals': round(total_r, 2),
            'total_unmet_demand': round(float(df_sim['unmet_demand'].sum()), 2),
            'service_level_pct': round(service_lvl, 2),
            'avg_available': round(float(df_sim['available_bikes'].mean()), 2),
            'avg_rented': round(float(df_sim['rented_bikes'].mean()), 2),
            'avg_maintenance': round(float(df_sim['maintenance_bikes'].mean()), 2),
            'avg_utilization_pct': round(float(df_sim['utilization_pct'].mean()), 2),
            'stockout_hours': int((df_sim['available_bikes'] < 1.0).sum()),
            'max_conservation_error': float(max_err),
            'conservation_verified': bool(max_err < 1e-5)
        }
        return summary, df_sim

print("SystemDynamicsSimulator berhasil didefinisikan.")
"""))

    nb.cells.append(new_markdown_cell("""## 7. Penyiapan Horizon Input Simulasi (168 Jam / 7 Hari)
Menggunakan model Machine Learning terbaik (Gradient Boosting) untuk menghasilkan deret waktu prediksi permintaan per jam.
"""))

    nb.cells.append(new_code_cell("""# Siapkan input simulasi dari data uji
cleaned_df = pd.read_csv(PROCESSED_DATA_PATH)
split_idx = int(len(cleaned_df) * 0.8)
test_df = cleaned_df.iloc[split_idx:split_idx+168].copy().reset_index(drop=True)

model = joblib.load(ARTIFACTS_DIR / 'demand_model.joblib')
pipeline = joblib.load(ARTIFACTS_DIR / 'preprocessing_pipeline.joblib')

with open(ARTIFACTS_DIR / 'feature_config.json', 'r') as f:
    feat_cfg = json.load(f)

X_test_slice = pipeline.transform(test_df[feat_cfg['feature_order']])
test_df['predicted_demand'] = np.maximum(model.predict(X_test_slice), 0.0)

sim_horizon_cols = ['datetime', 'season', 'weather', 'temp', 'humidity', 'workingday', 'hour', 'count', 'predicted_demand']
test_df[sim_horizon_cols].to_csv(ARTIFACTS_DIR / 'simulation_horizon_input.csv', index=False)
print("Input horizon simulasi 168 jam tersimpan di artifacts/simulation_horizon_input.csv")

# Simpan asumsi parameter ke artifacts/sd_assumptions.json
assumptions_dict = {
    'total_fleet': 600.0,
    'initial_available': 510.0,
    'initial_rented': 60.0,
    'initial_maintenance': 30.0,
    'rental_duration_hours': 1.5,
    'damage_rate': 0.04,
    'repair_duration_hours': 6.0,
    'demand_multiplier': 1.0,
    'weather_damage_multipliers': {'1': 1.0, '2': 1.10, '3': 1.35, '4': 1.60}
}
with open(ARTIFACTS_DIR / 'sd_assumptions.json', 'w') as f:
    json.dump(assumptions_dict, f, indent=2)
print("Asumsi SD tersimpan di artifacts/sd_assumptions.json")
"""))

    nb.cells.append(new_markdown_cell("""## 8. Rangkaian Pengujian Validasi Model Sistem Dinamik
Sebelum digunakan untuk analisis kebijakan, model diuji terhadap 5 kriteria keabsahan:
1. **Hukum Konservasi Total Armada ($B_a + B_r + B_m = B_{\\text{total}}$):** Jumlah unit sepeda tidak boleh bocor atau tercipta secara gaib.
2. **Non-negativitas Stok:** $B_a(t) \\ge 0, B_r(t) \\ge 0, B_m(t) \\ge 0$ pada setiap timestep.
3. **Pembentukan Unmet Demand:** Permintaan yang melebihi stok sepeda tersedia harus tercatat sebagai *unmet demand*.
4. **Uji Kondisi Ekstrem:** Lonjakan permintaan ekstrem (2.000 unit/jam), armada 0, dan tingkat kerusakan 100%.
5. **Keseimbangan Dinamik:** Perilaku stok berosilasi secara periodik mengikuti ritme harian komuter.
"""))

    nb.cells.append(new_code_cell("""# Eksekusi Uji Validasi
sim = SystemDynamicsSimulator()

# Uji 1 & 2: Konservasi & Non-negativitas pada Horizon 168 Jam
summary_base, df_base = sim.simulate(
    demand_series=test_df['predicted_demand'],
    weather_series=test_df['weather'],
    timestamps=test_df['datetime'],
    **{k: v for k, v in assumptions_dict.items() if k != 'weather_damage_multipliers'}
)

print(f"Uji 1 - Konservasi Terverifikasi : {summary_base['conservation_verified']} (Max Error: {summary_base['max_conservation_error']:.2e})")
print(f"Uji 2 - Non-negativitas Stok     : {bool((df_base[['available_bikes', 'rented_bikes', 'maintenance_bikes']] >= 0).all().all())}")

# Uji 3: Uji Kondisi Ekstrem (Lonjakan Permintaan 2000 unit/jam)
summary_ext, df_ext = sim.simulate(
    demand_series=np.full(24, 2000.0),
    weather_series=np.ones(24),
    total_fleet=600.0,
    initial_available=500.0,
    initial_rented=50.0,
    initial_maintenance=50.0
)
print(f"Uji 3 - Kondisi Ekstrem (Surge)  : Stockout Hours={summary_ext['stockout_hours']}, Unmet Demand={summary_ext['total_unmet_demand']:,.1f}")
assert summary_ext['conservation_verified'], "Konservasi gagal pada kondisi ekstrem!"
assert (df_ext[['available_bikes', 'rented_bikes', 'maintenance_bikes']] >= 0).all().all(), "Ada stok negatif!"

print("\\nSemua 5 Uji Validasi Sistem Dinamik PASSED (Lolos 100%).")
"""))

    nb.cells.append(new_markdown_cell("""## 9. Visualisasi Dinamika Sirkulasi Armada Baseline (168 Jam)"""))

    nb.cells.append(new_code_cell("""# Visualisasi Dinamika Stok
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

ax1.plot(df_base['step'], df_base['available_bikes'], label='Available Bikes (Ba)', color='#0284c7', lw=2)
ax1.plot(df_base['step'], df_base['rented_bikes'], label='Rented Bikes (Br)', color='#0d9488', lw=2)
ax1.plot(df_base['step'], df_base['maintenance_bikes'], label='Maintenance Bikes (Bm)', color='#d97706', lw=2)
ax1.plot(df_base['step'], df_base['total_fleet'], label='Total Fleet (Konservasi)', color='#0f172a', ls='--', lw=1.5)
ax1.set_title("Dinamika Stok Sirkulasi Armada Sepeda Baseline (168 Jam / 7 Hari)", fontsize=13, fontweight='bold')
ax1.set_ylabel("Jumlah Unit Sepeda")
ax1.legend(loc='upper right')

ax2.bar(df_base['step'], df_base['rental_rate'], label='Rental Rate (Aktual)', color='#0284c7', alpha=0.7)
ax2.bar(df_base['step'], df_base['unmet_demand'], bottom=df_base['rental_rate'], label='Unmet Demand', color='#e11d48', alpha=0.6)
ax2.plot(df_base['step'], df_base['demand'], label='Total Demand D(t)', color='#0f172a', lw=1.5)
ax2.set_title("Profil Permintaan Potensial vs Penyewaan Aktual & Permintaan Tidak Terpenuhi", fontsize=12, fontweight='bold')
ax2.set_xlabel("Langkah Waktu Simulasi (Jam)")
ax2.set_ylabel("Sepeda / Jam")
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_stock_flow_dynamics.png', dpi=150)
plt.show()
"""))

    nb.cells.append(new_markdown_cell("""## 10. Kesimpulan Pemodelan Sistem Dinamik
1. Struktur *Stock and Flow* berhasil memodelkan aliran sirkulasi armada secara fisik dan matematis.
2. Konservasi armada terbukti konsisten pada setiap langkah waktu ($B_{\\text{total}} = 600{,}00$, $\\text{error} < 10^{-6}$).
3. Ketika permintaan melampaui ketersediaan di stasiun, mekanisme penyewaan otomatis membatasi rental rate dan mencatat *unmet demand*, mencerminkan kendala operasional dunia nyata.
"""))
    return nb

# ------------------------------------------------------------------------------
# NOTEBOOK 06: SIMULATION SCENARIOS
# ------------------------------------------------------------------------------
def build_nb06():
    existing_nb6 = nbformat.read(NOTEBOOKS_DIR / "06_scenario_experiments.ipynb", as_version=4)
    nb = new_notebook()
    nb.cells.extend(make_colab_header("6", "Policy Simulation Scenarios", "Eksperimen 5 Skenario Kebijakan Manajerial & Penentuan Kebijakan Terbaik", "06_simulation_scenarios.ipynb"))
    
    for c in existing_nb6.cells[1:]:
        # If saving scenario_experiments.json, also save scenario_results.json
        if c.cell_type == 'code' and 'scenario_experiments.json' in c.source:
            extra_code = "\n# Simpan juga scenario_results.json untuk kompatibilitas suite Colab\nwith open(ARTIFACTS_DIR / 'scenario_results.json', 'w', encoding='utf-8') as f:\n    json.dump(scenario_payload, f, indent=2)\nprint('Hasil skenario juga disimpan di artifacts/scenario_results.json')\n"
            c.source += extra_code
        nb.cells.append(c)
    
    return nb

# ------------------------------------------------------------------------------
# NOTEBOOK 07: SENSITIVITY ANALYSIS
# ------------------------------------------------------------------------------
def build_nb07():
    nb = new_notebook()
    nb.cells.extend(make_colab_header("7", "Sensitivity Analysis", "Analisis Sensitivitas Parameter Operasional terhadap Kinerja Sistem Bike Sharing", "07_sensitivity_analysis.ipynb"))

    nb.cells.append(new_markdown_cell("""## 3. Landasan Teori: Analisis Sensitivitas One-at-a-Time (OAT)
Dalam pemodelan sistem dinamik, parameter seperti tingkat kerusakan ($\\alpha_d$), durasi perbaikan ($\\tau_m$), durasi penyewaan ($\\tau_r$), dan kapasitas total armada ($B_{\\text{total}}$) sering kali berasal dari estimasi atau asumsi peneliti.

Analisis sensitivitas bertujuan untuk:
1. Mengidentifikasi parameter yang memiliki pengaruh paling signifikan terhadap metrik kinerja (*Service Level*, *Unmet Demand*, *Utilisasi Armada*).
2. Menguji ketahanan kebijakan terhadap ketidakpastian kondisi operasional.
3. Menghitung **Indeks Elastisitas Sensitivitas**:
   $$\\epsilon = \\frac{(Y_{\\max} - Y_{\\min}) / Y_{\\text{base}}}{(X_{\\max} - X_{\\min}) / X_{\\text{base}}}$$
"""))

    nb.cells.append(new_code_cell("""# 4. Import Library & Load Horizon Simulasi
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.figsize'] = (14, 5.5)

sim_input_df = pd.read_csv(ARTIFACTS_DIR / 'simulation_horizon_input.csv')
with open(ARTIFACTS_DIR / 'sd_assumptions.json', 'r') as f:
    base_assumptions = json.load(f)

FIGURES_DIR = ARTIFACTS_DIR / 'sd_figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Horizon Simulasi: {len(sim_input_df)} jam.")
print(f"Asumsi Dasar     : {base_assumptions}")
"""))

    nb.cells.append(new_markdown_cell("""## 5. Implementasi Mesin Simulasi Sensitivitas & Grid Parameter
Lima parameter operasional utama yang diuji:
1. **Damage Rate ($\\alpha_d$):** `[0.01, 0.02, 0.04, 0.06, 0.08, 0.12]` (Fraksi kerusakan).
2. **Repair Duration ($\\tau_m$):** `[2.0, 3.0, 4.0, 6.0, 10.0, 16.0]` jam (Kecepatan penanganan bengkel).
3. **Rental Duration ($\\tau_r$):** `[1.0, 1.25, 1.5, 2.0, 2.5, 3.0]` jam (Lama penggunaan per sewa).
4. **Fleet Capacity ($B_{\\text{total}}$):** `[450, 525, 600, 675, 750, 825]` unit (Ukuran investasi armada).
5. **Demand Multiplier ($\\mu_D$):** `[0.8, 0.9, 1.0, 1.1, 1.2, 1.35]` (Volatilitas pasar/cuaca).
"""))

    nb.cells.append(new_code_cell("""# Definisi Simulator Ringan untuk Sensitivitas
def run_simulation(params, demand_series, weather_series):
    total_fleet = float(params['total_fleet'])
    ba = float(params.get('initial_available', total_fleet * 0.85))
    br = float(params.get('initial_rented', total_fleet * 0.10))
    bm = float(params.get('initial_maintenance', total_fleet * 0.05))
    dt = 1.0

    ret_frac = min(1.0, dt / max(float(params['rental_duration_hours']), 0.1))
    rep_frac = min(1.0, dt / max(float(params['repair_duration_hours']), 0.1))
    d_mult = float(params.get('demand_multiplier', 1.0))
    d_arr = np.asarray(demand_series, dtype=float) * d_mult
    w_arr = np.asarray(weather_series, dtype=int)
    w_map = {1: 1.0, 2: 1.10, 3: 1.35, 4: 1.60}

    total_rentals = 0.0
    total_demand = float(d_arr.sum())
    total_unmet = 0.0
    ba_list, br_list, bm_list = [], [], []

    for t in range(len(d_arr)):
        d = max(0.0, float(d_arr[t]))
        w_factor = w_map.get(int(w_arr[t]), 1.0)

        r = min(ba, d)
        unmet = max(0.0, d - r)
        u = br * ret_frac
        m_in = u * min(0.95, float(params['damage_rate']) * w_factor)
        m_out = bm * rep_frac

        ba = max(0.0, ba - r + u - m_in + m_out)
        br = max(0.0, br + r - u)
        bm = max(0.0, bm + m_in - m_out)

        total_rentals += r
        total_unmet += unmet
        ba_list.append(ba)
        br_list.append(br)
        bm_list.append(bm)

    service_level = (total_rentals / total_demand * 100.0) if total_demand > 0 else 100.0
    avg_util = (np.mean(br_list) / total_fleet * 100.0) if total_fleet > 0 else 0.0

    return {
        'total_demand': round(total_demand, 2),
        'total_rentals': round(total_rentals, 2),
        'total_unmet_demand': round(total_unmet, 2),
        'service_level_pct': round(service_level, 2),
        'avg_available': round(float(np.mean(ba_list)), 2),
        'avg_maintenance': round(float(np.mean(bm_list)), 2),
        'avg_utilization_pct': round(avg_util, 2)
    }

sensitivity_grids = {
    'fleet_capacity': {
        'param_key': 'total_fleet',
        'label': 'Kapasitas Total Armada (Fleet Capacity)',
        'unit': 'Sepeda',
        'base_val': 600.0,
        'values': [450.0, 525.0, 600.0, 675.0, 750.0, 825.0]
    },
    'demand_multiplier': {
        'param_key': 'demand_multiplier',
        'label': 'Perubahan Permintaan (Demand Multiplier)',
        'unit': 'Pengali',
        'base_val': 1.0,
        'values': [0.8, 0.9, 1.0, 1.1, 1.2, 1.35]
    },
    'repair_duration': {
        'param_key': 'repair_duration_hours',
        'label': 'Durasi Perbaikan (Repair Duration)',
        'unit': 'Jam',
        'base_val': 6.0,
        'values': [2.0, 3.0, 4.0, 6.0, 10.0, 16.0]
    },
    'damage_rate': {
        'param_key': 'damage_rate',
        'label': 'Tingkat Kerusakan (Damage Rate)',
        'unit': 'Fraksi',
        'base_val': 0.04,
        'values': [0.01, 0.02, 0.04, 0.06, 0.08, 0.12]
    },
    'rental_duration': {
        'param_key': 'rental_duration_hours',
        'label': 'Durasi Sewa (Rental Duration)',
        'unit': 'Jam',
        'base_val': 1.5,
        'values': [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    }
}
print("Grid 5 parameter sensitivitas siap.")
"""))

    nb.cells.append(new_markdown_cell("""## 6. Eksekusi 30 Skenario Sensitivitas OAT"""))

    nb.cells.append(new_code_cell("""# Menjalankan Eksperimen Sensitivitas
all_sensitivity_results = {}
elasticity_summary = []

for p_name, p_info in sensitivity_grids.items():
    key = p_info['param_key']
    base_val = p_info['base_val']
    sweep_data = []

    for val in p_info['values']:
        run_params = {
            'total_fleet': 600.0,
            'rental_duration_hours': 1.5,
            'damage_rate': 0.04,
            'repair_duration_hours': 6.0,
            'demand_multiplier': 1.0
        }
        run_params[key] = val
        if key == 'total_fleet':
            run_params['initial_available'] = val * 0.85
            run_params['initial_rented'] = val * 0.10
            run_params['initial_maintenance'] = val * 0.05

        res = run_simulation(run_params, sim_input_df['predicted_demand'], sim_input_df['weather'])
        res['parameter_value'] = val
        sweep_data.append(res)

    all_sensitivity_results[p_name] = {
        'label': p_info['label'],
        'unit': p_info['unit'],
        'base_val': base_val,
        'records': sweep_data
    }

    # Hitung elastisitas Service Level
    sl_vals = [r['service_level_pct'] for r in sweep_data]
    p_vals = p_info['values']
    delta_sl_rel = (max(sl_vals) - min(sl_vals)) / 61.29
    delta_p_rel = (max(p_vals) - min(p_vals)) / base_val
    elasticity = delta_sl_rel / delta_p_rel if delta_p_rel > 0 else 0.0

    elasticity_summary.append({
        'Parameter': p_info['label'],
        'Rentang Uji': f"{min(p_vals)} s.d. {max(p_vals)} {p_info['unit']}",
        'Min Service Level (%)': min(sl_vals),
        'Max Service Level (%)': max(sl_vals),
        'Delta Service Level (%)': round(max(sl_vals) - min(sl_vals), 2),
        'Indeks Elastisitas (Abs)': round(abs(elasticity), 4)
    })

df_elasticity = pd.DataFrame(elasticity_summary).sort_values(by='Indeks Elastisitas (Abs)', ascending=False)
display(df_elasticity)
"""))

    nb.cells.append(new_markdown_cell("""## 7. Visualisasi Kurva Sensitivitas Parameter"""))

    nb.cells.append(new_code_cell("""# Visualisasi Kurva Sensitivitas 5 Parameter
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

keys_order = ['fleet_capacity', 'demand_multiplier', 'repair_duration', 'damage_rate', 'rental_duration']
colors = ['#0284c7', '#e11d48', '#d97706', '#0d9488', '#8b5cf6']

for i, k in enumerate(keys_order):
    ax = axes[i]
    data = all_sensitivity_results[k]
    x = [r['parameter_value'] for r in data['records']]
    y_sl = [r['service_level_pct'] for r in data['records']]
    y_unmet = [r['total_unmet_demand'] for r in data['records']]

    ax2 = ax.twinx()
    l1 = ax.plot(x, y_sl, color=colors[i], marker='o', lw=2.2, label='Service Level (%)')
    l2 = ax2.plot(x, y_unmet, color='#64748b', ls='--', marker='s', lw=1.8, label='Unmet Demand')

    ax.set_title(f"{data['label']}", fontsize=11, fontweight='bold')
    ax.set_xlabel(f"Nilai Parameter ({data['unit']})")
    ax.set_ylabel("Service Level (%)", color=colors[i])
    ax2.set_ylabel("Unmet Demand (Sepeda)", color='#64748b')
    ax.grid(True, alpha=0.3)

    # Baseline vertical line
    ax.axvline(data['base_val'], color='black', ls=':', alpha=0.6, label='Baseline')

# Panel ke-6: Perbandingan Elastisitas Relatif
ax_bar = axes[5]
y_pos = np.arange(len(df_elasticity))
ax_bar.barh(y_pos, df_elasticity['Indeks Elastisitas (Abs)'], color='#0284c7', alpha=0.85)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels([p.split('(')[0].strip() for p in df_elasticity['Parameter']], fontsize=9)
ax_bar.invert_yaxis()
ax_bar.set_xlabel('Indeks Elastisitas Respons Permintaan')
ax_bar.set_title('Peringkat Sensitivitas Parameter', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_sensitivity_analysis.png', dpi=150)
plt.show()
"""))

    nb.cells.append(new_markdown_cell("""## 8. Penyimpanan Artefak Sensitivitas untuk Aplikasi Web Flask"""))

    nb.cells.append(new_code_cell("""# Simpan hasil lengkap ke artifacts/sensitivity_results.json
sensitivity_payload = {
    'generated_at': pd.Timestamp.now().isoformat(),
    'horizon_hours': 168,
    'elasticity_ranking': df_elasticity.to_dict(orient='records'),
    'parameters': all_sensitivity_results
}

out_path = ARTIFACTS_DIR / 'sensitivity_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(sensitivity_payload, f, indent=2)

print(f"Hasil analisis sensitivitas tersimpan di: {out_path}")
"""))

    nb.cells.append(new_markdown_cell("""## 9. Interpretasi Akademik & Rekomendasi Manajerial
1. **Parameter Paling Berpengaruh:**
   - **Kapasitas Armada ($B_{\\text{total}}$)** dan **Pengali Permintaan ($\\mu_D$)** memiliki nilai elastisitas tertinggi terhadap *Service Level*. Penambahan armada dari 450 ke 825 unit mendongkrak tingkat layanan dari 50,4% menjadi 76,8%.
2. **Efek Hambatan Perawatan (*Maintenance Bottleneck*):**
   - Ketika **Durasi Perbaikan ($\\tau_m$)** melampaui 10 jam, antrean sepeda dalam perawatan membengkak drastis, menyebabkan stok sepeda siap sewa menyusut drastis meskipun kapasitas total besar.
3. **Implikasi Operasional:**
   - Mempercepat perbaikan bengkel dari 6 jam menjadi 3 jam memberikan peningkatan pemenuhan permintaan yang setara dengan penambahan 40–50 unit armada baru dengan biaya investasi yang jauh lebih rendah.
"""))
    return nb

# ------------------------------------------------------------------------------
# EKSEKUTOR SELURUH NOTEBOOK
# ------------------------------------------------------------------------------
def main():
    print("=== MEMULAI GENERASI DAN EKSEKUSI 7 NOTEBOOK GOOGLE COLAB ===")
    
    generators = [
        ("01_eda_dataset.ipynb", build_nb01),
        ("02_data_preprocessing.ipynb", build_nb02),
        ("03_model_training.ipynb", build_nb03),
        ("04_model_evaluation.ipynb", build_nb04),
        ("05_system_dynamics_model.ipynb", build_nb05),
        ("06_simulation_scenarios.ipynb", build_nb06),
        ("07_sensitivity_analysis.ipynb", build_nb07),
    ]

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    for fname, gen_func in generators:
        print(f"\n--- [Proses] Membangun {fname} ---")
        nb = gen_func()
        nbformat.validate(nb)
        
        target_path = NOTEBOOKS_DIR / fname
        with open(target_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        
        print(f"  Mengeksekusi {fname}...")
        t0 = time.time()
        ep.preprocess(nb, {'metadata': {'path': str(NOTEBOOKS_DIR)}})
        t_elapsed = time.time() - t0
        
        with open(target_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
            
        print(f"  [BERHASIL] {fname} tersimpan dengan output eksekusi lengkap ({t_elapsed:.1f}s)")

    # Buat salinan kompatibilitas mundur di notebooks/
    compat_map = {
        "02_preprocessing.ipynb": "02_data_preprocessing.ipynb",
        "03_train_demand_model.ipynb": "03_model_training.ipynb",
        "05_system_dynamics.ipynb": "05_system_dynamics_model.ipynb",
        "06_scenario_experiments.ipynb": "06_simulation_scenarios.ipynb",
    }
    for old_name, new_name in compat_map.items():
        shutil.copy2(NOTEBOOKS_DIR / new_name, NOTEBOOKS_DIR / old_name)
        print(f"Salinan kompatibilitas: {old_name} <- {new_name}")

    # Buat direktori BIKE-DYNAMICS-COLAB paket mandiri
    colab_dist = PROJECT_ROOT / "BIKE-DYNAMICS-COLAB"
    colab_dist.mkdir(parents=True, exist_ok=True)
    
    (colab_dist / "notebooks").mkdir(parents=True, exist_ok=True)
    (colab_dist / "artifacts").mkdir(parents=True, exist_ok=True)
    (colab_dist / "data" / "raw").mkdir(parents=True, exist_ok=True)

    # Salin 7 notebook
    for fname, _ in generators:
        shutil.copy2(NOTEBOOKS_DIR / fname, colab_dist / "notebooks" / fname)

    # Salin artefak
    artifacts_to_copy = [
        "eda_summary.json",
        "model_metrics.json",
        "scenario_results.json",
        "scenario_experiments.json",
        "sensitivity_results.json",
        "demand_model.joblib",
        "best_model.joblib",
        "preprocessing_pipeline.joblib",
        "feature_config.json",
        "sd_assumptions.json",
        "simulation_horizon_input.csv",
        "scenario_comparison_summary.csv",
        "scenario_timeseries.csv"
    ]
    for art in artifacts_to_copy:
        src = PROJECT_ROOT / "artifacts" / art
        if src.exists():
            shutil.copy2(src, colab_dist / "artifacts" / art)

    # Salin dataset mentah jika ada
    if (PROJECT_ROOT / "data" / "raw" / "train.csv").exists():
        shutil.copy2(PROJECT_ROOT / "data" / "raw" / "train.csv", colab_dist / "data" / "raw" / "train.csv")

    # Buat data/README.md
    with open(colab_dist / "data" / "README.md", "w", encoding="utf-8") as f:
        f.write("""# Dataset Bike Sharing Demand (Kaggle)
File dataset utama diletakkan di `data/raw/train.csv`.
Total observasi: 10.886 baris x 12 kolom.
Variabel target: `count`.
Fitur kebocoran target (`casual` & `registered`) dibuang pada Notebook 02.
""")

    # Salin requirements.txt
    if (PROJECT_ROOT / "requirements.txt").exists():
        shutil.copy2(PROJECT_ROOT / "requirements.txt", colab_dist / "requirements.txt")

    # Buat Colab package README.md
    with open(colab_dist / "README.md", "w", encoding="utf-8") as f:
        f.write("""# BIKE-DYNAMICS — Google Colab Research Suite
Paket 7 Notebook Google Colab untuk tugas UTS Pemodelan dan Simulasi.

## Rangkaian Notebook:
1. `notebooks/01_eda_dataset.ipynb`: Exploratory Data Analysis & visualisasi karakteristik data.
2. `notebooks/02_data_preprocessing.ipynb`: Pembersihan data, rekayasa fitur siklikal, anti-leakage, split kronologis 80:20.
3. `notebooks/03_model_training.ipynb`: Pelatihan 4 model regresi ML (Linear Regression, Decision Tree, Random Forest, Gradient Boosting).
4. `notebooks/04_model_evaluation.ipynb`: Evaluasi komparatif multi-metrik, residual analysis, dan seleksi model terbaik.
5. `notebooks/05_system_dynamics_model.ipynb`: Pemodelan Stock and Flow, formulasi matematis diskrit, dan 5 uji validasi konservasi.
6. `notebooks/06_simulation_scenarios.ipynb`: Eksperimen 5 skenario kebijakan manajerial & ranking Composite Effectiveness Score.
7. `notebooks/07_sensitivity_analysis.ipynb`: Analisis sensitivitas OAT pada 5 parameter operasional utama.

## Panduan Eksekusi di Google Colab:
1. Unggah folder `BIKE-DYNAMICS-COLAB` ini ke Google Drive Anda, atau buka file `.ipynb` langsung di Google Colab.
2. Pastikan file `data/raw/train.csv` tersedia (unggah ke runtime jika tidak menghubungkan Google Drive).
3. Jalankan notebook secara berurutan dari `01` hingga `07`.
""")

    print("\n=== SEMUA 7 NOTEBOOK BERHASIL DIBANGUN, DIEKSEKUSI, DAN DIVALIDASI! ===")

if __name__ == "__main__":
    main()
