# Vision Transformer Image Captioning on MSCOCO

## 1. Task Description

**Task:** Image Captioning  
**Deskripsi:**  
Menghasilkan deskripsi teks (caption) otomatis untuk gambar menggunakan model Vision Transformer (ViT) dengan decoder Transformer.

## 2. Dataset

- **Nama:** MSCOCO (HuggingFaceM4/COCO)
- **Sumber:** [HuggingFace Datasets - HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO)
- **Total gambar:** ~21.000 (train), ~1.500 (val) (subset digunakan)
- **Keterangan:**  
  - Hanya gambar RGB yang digunakan.
  - Setiap gambar memiliki satu caption (caption utama).
  - Gambar diresize ke 224x224 pixel.
  - Caption di-tokenisasi dengan custom tokenizer (vocab size ~2027, min_freq=5).
  - Split train/val mengikuti dataset asli.

**Contoh kode akses data:**
```python
from datasets import load_dataset
dataset = load_dataset("HuggingFaceM4/COCO", trust_remote_code=True)
```

## 3. Preprocessing

- Filter hanya gambar RGB.
- Resize gambar ke 224x224 dan normalisasi (mean/std ImageNet).
- Tokenisasi caption dengan custom tokenizer (lowercase, hapus tanda baca, min_freq=5).
- Padding dan truncation caption ke panjang 50 token.

## 4. Model

- **Arsitektur:** Vision Transformer (ViT) encoder + Transformer decoder.
- **Eksperimen:**  
  - Baseline (embed_dim=256, patch_size=16, 8 heads)
  - Small Embedding (embed_dim=128)
  - Small Patch (patch_size=8)
  - More Heads (num_heads=16)
  - RoPE (Rotary Positional Embedding)
- **Tokenisasi:** Custom tokenizer (bukan HuggingFace tokenizer).

## 5. Training

- **Optimizer:** AdamW
- **Learning Rate:** 1e-4
- **Epochs:** 10
- **Batch Size:** 32
- **Loss:** CrossEntropyLoss (label smoothing 0.1, ignore padding)
- **Evaluasi:** CIDEr metric (HuggingFace evaluate)

## 6. Evaluasi & Visualisasi

- **Metode evaluasi:**  
  - Loss (train/val)
  - CIDEr score (val)
- **Visualisasi:**  
  - Kurva loss dan CIDEr per epoch untuk semua model.
  - Bar chart best CIDEr tiap model.
  - Contoh caption hasil generate dari semua model pada gambar yang sama.

## 7. Cara Menjalankan

1. Install dependensi:
    - `transformers`
    - `datasets`
    - `torch`
    - `matplotlib`
    - `evaluate`
    - `tqdm`
2. Jalankan notebook `visual-transformers-deep-learning.ipynb` untuk seluruh pipeline (preprocessing, training, evaluasi, visualisasi).
3. Hasil model dan metrik disimpan otomatis (checkpoint dan JSON).

---

**Catatan:**  
Seluruh pipeline (preprocessing, training, evaluasi, visualisasi, perbandingan model) diimplementasikan di notebook utama (`visual-transformers-deep-learning.ipynb`).  
Silakan buka notebook untuk detail kode, eksperimen, dan hasil.
