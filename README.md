# Klasifikasi Gambar CIFAR-10 (Tugas Kelompok 2)

Aplikasi web Streamlit yang menerima gambar (JPG/PNG) dan menampilkan prediksi
model MobileNetV2 (transfer learning) untuk 10 kelas CIFAR-10: airplane,
automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Fitur:
- Unggah gambar, tampilkan gambar asli dan versi 32×32 yang dilihat model.
- Prediksi top-1 + confidence, top-3 dalam bar chart.
- Peringatan bila confidence < 50%.
- Penanganan galat: format salah, file kosong/rusak, file > 5 MB, model gagal dimuat.

## Struktur

```
app.py                  # antarmuka Streamlit
inference.py            # preprocessing + prediksi (dipakai app & skrip evaluasi)
config.json             # kontrak model: kelas, ukuran input, resize, normalisasi
model/model.keras       # model final tim (MobileNetV2, eksperimen D)
model/metadata.json     # metadata, konfigurasi training + metrik model final
train_final.py          # pelatihan CNN hasil tuning (Trial #17, lama, tidak dipakai)
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

Model yang dipakai adalah hasil optimasi tim (`tk2_hasil_optimasi_model`),
**eksperimen D**: transfer learning MobileNetV2 (bobot ImageNet) dengan
augmentasi. Gambar masuk ke model sebagai 32×32 RGB (piksel ÷ 255); augmentasi
dan resize ke 96×96 untuk MobileNetV2 sudah ada di dalam model.

| Konfigurasi | Nilai |
|---|---|
| Backbone | MobileNetV2 (ImageNet), input internal 96×96 |
| Training head | 3 epoch, lr 0,001, dropout 0,2 |
| Fine-tuning | dari layer 54, 12 epoch, lr 0,0001, EarlyStopping patience 4 |
| Batch size | 128 |
| Augmentasi | ya |
| Parameter | 2.270.794 (9,7 MB) |

Dipilih karena val accuracy tertinggi dibanding seluruh eksperimen
(CNN terbaik C4: 0,8424; CNN Random Search: 0,72–0,73).

| Metrik | Validasi | Test |
|---|---|---|
| Accuracy | 0,9286 | 0,9217 |
| Macro F1 | 0,9285 | 0,9213 |

Test accuracy 0,9217 juga terverifikasi ulang secara lokal dengan
TensorFlow 2.18.1 + Keras 3.15.1. Detail lengkap ada di `model/metadata.json`.
Model dilatih di Colab (TF 2.21 / Keras 3.13); file `.keras` termuat tanpa
`custom_objects` di versi yang dipasang `requirements.txt`.

## Mengganti model

1. Simpan model sebagai `model/model.keras` (atau `.h5`, lalu ubah `model_path`).
2. Sesuaikan `config.json`:
   - `class_names` — urutan **harus sama** dengan urutan output model.
   - `input_size` — ukuran input model, mis. `[32, 32]`.
   - `resample` — metode resize PIL, samakan dengan training (mis. `bilinear`).
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
- Input diperkecil ke 32×32 (bilinear), sehingga detail gambar beresolusi tinggi hilang.
- Inferensi di CPU lebih lambat dari CNN kecil (±165 ms/gambar di Colab).
- Confidence softmax tinggi bukan jaminan prediksi benar.
