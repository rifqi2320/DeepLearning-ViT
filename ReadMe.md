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

| image (PIL.Image) | labels |
|-------------------|--------|
| ![contoh1](https://datasets-server.huggingface.co/assets/microsoft/cats_vs_dogs/--/b5ae3589204019bc2cc97e99e4914a54589333ef/--/default/train/0/image/image.jpg?Expires=1748230130&Signature=SopT5kqiT973bfuZ~JbJyDdugjE3LtJa4qL59ZykdpjIrE6CYMO5otvOVvyKxf5g3ErvGE6w~pwaxQa~ao5c5VeBeuO3C1NoHXCmtZhg17oInBti0W0F9lB1mwJZBUMevMj8HSmmBpJ1xtE7BCYGF1qEgLpFXjtcTMP-BVnTKJhDr6ycWC6aA9S3zulo1elIdbr7LFJgqFKv8k2UqCdvB-A3hGs8tP6JeOmd6QakMBRjT9kuaMi1YcdXBkGzFUjpfnPboxy9HD7jhyWHGJnJh5-zQ2k5qKLRQGJKaIQQrLASee8UG-io9R0Sh202soQ-AcQvGDbiLwrgiaUjRdSt1A__&Key-Pair-Id=K3EI6M078Z3AC3) | 0 (kucing) |
| ![contoh2](https://datasets-server.huggingface.co/cached-assets/microsoft/cats_vs_dogs/--/b5ae3589204019bc2cc97e99e4914a54589333ef/--/default/train/11744/image/image.jpg?Expires=1748230218&Signature=sMAFreyXrcHOdXWHZHTp4oFBsGLG-xQEYG8T6jNvRy~WkkGY11oDm3W~MPfSuhTBr~MT2jUVhu~lH-fJSi7lr96sFd7UrOjRY4Zpuj0QOwOzwAKfBPerysatiq7HGQFTmr5rSrpeigXxtfBmNa4f7PILAVOGHwd0PrL8Orad0NMdF85ff3qMOeQggooABjQk0dHNr7yrLKRI89QHBXw8IqgQnMtEV0thxDOYzW8YKMvYUYujQLaawbcD-wa34k5qff6RPCVTv8~NBMW5ixTgi8oCDB-Z2elX5s~BMq0UmOfop9iMNIGzIcZt0ZIceF5h2J2efTL0wL-dsIWVdUIIDw__&Key-Pair-Id=K3EI6M078Z3AC3) | 1 (anjing) |

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
