# Studi Eksperimen Vision Transformer pada Klasifikasi Kucing vs Anjing

## 1. Nama dan Deskripsi Task

**Task:** Klasifikasi gambar kucing vs anjing menggunakan arsitektur Vision Transformer (ViT) dengan berbagai konfigurasi (jumlah attention heads, dimensi embedding, ukuran patch, dan perbandingan RoPE).

**Deskripsi:**  
Studi ini membandingkan performa beberapa varian Vision Transformer pada tugas klasifikasi dua kelas (kucing dan anjing). Eksperimen dilakukan untuk memahami pengaruh komponen arsitektur utama seperti jumlah attention heads, dimensi embedding, ukuran patch, serta positional encoding (RoPE vs standar).

## 2. Nama Dataset, Link, Statistik, dan Contoh

**Nama Dataset:** Cats vs Dogs  
**Sumber:** [HuggingFace Datasets - cats_vs_dogs](https://huggingface.co/datasets/cats_vs_dogs)

**Statistik Dataset:**
- Total gambar: 23.262
- Jumlah kelas: 2 (kucing = 0, anjing = 1)
- Proporsi data:  
  - Train: ~90%  
  - Test: ~10%  
- Semua gambar yang digunakan memiliki 3 channel (RGB).

**Contoh Isi Dataset:**

![sample_images](attachment/sample_images.png)


**Contoh kode akses data:**
```python
from datasets import load_dataset
dataset = load_dataset("cats_vs_dogs")
sample = dataset['train'][0]
print(sample['image'])  # PIL Image
print(sample['labels']) # 0 (kucing) atau 1 (anjing)
```

**Distribusi kelas (setelah filter RGB):**
- Kucing: sekitar 11.682 gambar
- Anjing: sekitar 11.580 gambar

---
**Catatan:**  
Dataset diunduh otomatis dari HuggingFace Datasets. Seluruh preprocessing dan pembagian train/test dilakukan di notebook.
