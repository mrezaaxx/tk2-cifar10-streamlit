# Klasifikasi Gambar CIFAR-10 (Tugas Kelompok 2)

Aplikasi web Streamlit yang menerima gambar (JPG/PNG) dan menampilkan prediksi
model CNN untuk 10 kelas CIFAR-10: airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, truck.

Fitur:
- Unggah gambar, tampilkan gambar asli dan versi 32×32 yang dilihat model.
- Prediksi top-1 + confidence, top-3 dalam bar chart.
- Peringatan bila confidence < 50%.
- Penanganan galat: format salah, file kosong/rusak, file > 5 MB, model gagal dimuat.

## Struktur

```
app.py                  # antarmuka Streamlit
inference.py            # preprocessing + prediksi (dipakai app & skrip evaluasi)
config.json             # kontrak model: kelas, ukuran input, normalisasi
model/model.keras       # model final tim (CNN hasil tuning)
model/metrics.json      # hyperparameter + akurasi model final
train_final.py          # pelatihan ulang model final tim (Trial #17)
train_temp.py           # pelatihan model sementara (lama, tidak dipakai)
evaluate_new_images.py  # uji gambar baru -> results.csv
test_images/<kelas>/    # gambar uji di luar dataset
requirements.txt
```

## Menjalankan secara lokal

Butuh **Python 3.11** (TensorFlow 2.18 belum mendukung Python 3.13+).

```bash
git clone <url-repo> cifar10-app
cd cifar10-app
python -m venv .venv            # pastikan python = 3.11
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501.

## Model final tim

Model yang dipakai adalah konfigurasi terbaik dari notebook *Hyperparameter
Tuning CNN pada CIFAR-10* (Random Search Keras Tuner, 20 trial, Trial #17):

| Hyperparameter | Nilai |
|---|---|
| Arsitektur | Conv 32 → Pool → Conv 64 → Pool → Conv 128 → Dense 128 → Dropout → Dense 10 |
| Kernel | 3×3 |
| Optimizer | SGD (momentum 0,9, Nesterov) |
| Learning rate | 0,00376 |
| Batch size | 32 |
| Epoch | maks 40, EarlyStopping val_accuracy (patience 5) |
| Dropout | 0,7 |
| Augmentasi | tidak |

Hasil di Colab: val accuracy 0,7372, test accuracy 0,7259, macro F1 0,7260.

`model/model.keras` dilatih ulang secara lokal dengan `python train_final.py`
(seed 42, split 90/10 stratified), sehingga angkanya bisa sedikit berbeda dari
Colab; lihat `model/metrics.json`. Jika file `cnn_best_tuned.keras` dari ZIP
Colab tersedia, cukup salin ke `model/model.keras`.

## Mengganti model

1. Simpan model sebagai `model/model.keras` (atau `.h5`, lalu ubah `model_path`).
2. Sesuaikan `config.json`:
   - `class_names` — urutan **harus sama** dengan urutan output model.
   - `input_size` — ukuran input model, mis. `[32, 32]`.
   - `normalization` — saat ini `rescale_255` (piksel ÷ 255). Jika model memakai
     normalisasi lain (mis. preprocess_input untuk transfer learning), tambahkan di
     `inference.preprocess`.
3. Pastikan versi TensorFlow sama dengan yang dipakai saat training.

Normalisasi atau urutan kelas yang berbeda tidak memunculkan error, tetapi
membuat prediksi salah secara diam-diam.

## Uji gambar baru

Taruh gambar di `test_images/<nama_kelas>/`, contoh `test_images/cat/kucing1.jpg`.
Gambar di luar 10 kelas boleh ditaruh di `test_images/unknown/`.

```bash
python evaluate_new_images.py
```

Hasil per gambar ditulis ke `results.csv` untuk bab evaluasi laporan.

## Keterbatasan

- Model hanya mengenal 10 kelas; gambar di luar kelas tetap dipaksa ke salah satunya.
- Input diperkecil ke 32×32, sehingga detail gambar beresolusi tinggi hilang.
- Confidence softmax tinggi bukan jaminan prediksi benar.
