# BIKE-DYNAMICS — Google Colab Research Suite
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
