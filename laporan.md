# Analisis dan Pembahasan Model Deep Learning untuk Klasifikasi Chest X-Ray

## 1. Analisis Perubahan Kode

### 1.1 Perubahan pada model.py

**Sebelum perubahan:**  
Model CNN yang digunakan sebelumnya relatif sederhana, terdiri dari beberapa layer convolution, batch normalization, dan pooling yang disusun secara berurutan. Model ini cukup baik untuk baseline, namun memiliki keterbatasan dalam menangkap fitur kompleks dan sering mengalami masalah vanishing gradient pada jaringan yang lebih dalam.

**Setelah perubahan:**  
Model diubah menjadi arsitektur **DenseNet** yang lebih modern dan efisien.  
Perubahan utama yang diterapkan:
- **Dense Connections:** Setiap layer dalam satu block menerima input dari semua layer sebelumnya, sehingga memperbaiki aliran gradien, meningkatkan feature reuse, dan mempercepat konvergensi.
- **Growth Rate:** Penambahan parameter growth rate untuk mengatur jumlah feature map baru pada setiap layer, sehingga arsitektur dapat diatur lebar dan kedalamannya.
- **Transition Layer:** Penambahan layer transisi untuk mengurangi dimensi feature map, menjaga efisiensi komputasi meskipun model lebih dalam.
- **Adaptive Pooling & Dropout:** Penambahan adaptive pooling agar model fleksibel terhadap ukuran input, serta dropout untuk regularisasi dan mencegah overfitting.
- **Batch Normalization:** Digunakan secara konsisten untuk mempercepat training dan menjaga stabilitas distribusi aktivasi.

### 1.2 Perubahan pada train.py

**Sebelum perubahan:**  
Proses training dilakukan dengan loop sederhana, tanpa banyak fitur tambahan seperti early stopping, checkpointing, atau learning rate scheduling yang adaptif. Monitoring training juga terbatas.

**Setelah perubahan:**  
Proses training diubah menjadi lebih modular, robust, dan mengikuti best practice deep learning modern:
- **Object-Oriented:** Training dibungkus dalam kelas `ChestXrayTrainer` untuk modularitas, kemudahan pengembangan, dan pengelolaan state training.
- **Optimizer & Scheduler:** Menggunakan AdamW (varian Adam dengan weight decay) dan OneCycleLR untuk optimisasi yang lebih baik dan adaptif.
- **Gradient Clipping:** Menambahkan gradient clipping untuk mencegah exploding gradients, menjaga stabilitas training.
- **Early Stopping & Checkpointing:** Implementasi early stopping untuk menghentikan training jika tidak ada peningkatan pada validation loss, serta penyimpanan model terbaik secara otomatis.
- **Class Imbalance Handling:** Penambahan `pos_weight` pada loss function untuk mengatasi data imbalance, sehingga model tidak bias ke kelas mayoritas.
- **Monitoring:** Tracking loss, akurasi, dan learning rate setiap epoch serta visualisasi hasil prediksi pada data validasi.

---

## 2. Analisis Hasil Perubahan pada Gambar dan Training

### 2.1 Dampak pada Hasil Training

- **Kualitas Prediksi Meningkat:**  
  Model DenseNet mampu menangkap fitur yang lebih kompleks pada citra X-ray, sehingga prediksi pada gambar validasi menjadi lebih akurat dan robust, terutama pada kasus minoritas seperti Pneumothorax.
- **Loss dan Akurasi Lebih Stabil:**  
  Dengan scheduler dan gradient clipping, kurva loss dan akurasi pada training dan validasi menjadi lebih stabil, mengurangi risiko overfitting dan underfitting.
- **Convergence Lebih Cepat:**  
  Dengan OneCycleLR, model mencapai performa optimal dalam jumlah epoch yang lebih sedikit, sehingga training lebih efisien.
- **Visualisasi Prediksi Lebih Baik:**  
  Hasil visualisasi prediksi pada gambar validasi menunjukkan model lebih jarang melakukan kesalahan klasifikasi, dan prediksi lebih sesuai dengan label ground truth.

### 2.2 Contoh Perubahan pada Plot

- **Sebelum:**  
  Kurva loss cenderung fluktuatif, akurasi validasi stagnan, dan prediksi pada gambar validasi sering salah, terutama pada kelas minoritas.
- **Sesudah:**  
  Kurva loss menurun konsisten, akurasi validasi meningkat, dan prediksi pada gambar validasi lebih akurat serta seimbang antar kelas.

---

## 3. Monitoring dan Visualisasi

### 3.1 Metrics Tracking
- Training loss dan accuracy
- Validation loss dan accuracy
- Learning rate history

### 3.2 Visualisasi
- Plot training history (loss dan akurasi)
- Visualisasi random predictions pada validation set untuk melihat performa model secara kualitatif

---

## 4. Best Practices yang Diimplementasikan

- **Device Management:**  
  Otomatis memilih GPU jika tersedia, dan memastikan semua tensor berada di device yang sama.
- **Error Handling:**  
  Try-except untuk menangani interupsi training, serta pembersihan resource yang baik.
- **Code Organization:**  
  Menggunakan object-oriented design, modularisasi kode, dan pemisahan concern antara model, data, dan training.

---

## 5. Potensi Pengembangan

- **Arsitektur:**  
  Eksperimen dengan growth rate yang berbeda, menambah/mengurangi jumlah dense blocks, atau mencoba varian DenseNet lain.
- **Training:**  
  Implementasi learning rate finder, cross-validation, dan data augmentation yang lebih advanced.
- **Monitoring:**  
  Penambahan metrics seperti F1-score, ROC-AUC, integrasi TensorBoard, dan visualisasi confusion matrix.

---

## 6. Kesimpulan

Perubahan arsitektur ke DenseNet dan proses training yang lebih modern memberikan dampak signifikan pada performa model:
- Model lebih dalam dan efisien, mampu belajar fitur lebih baik.
- Proses training lebih stabil dan adaptif terhadap data imbalance.
- Hasil prediksi pada gambar validasi lebih akurat dan dapat diandalkan.
- Framework lebih siap untuk pengembangan dan eksperimen lanjutan.

Implementasi ini menunjukkan pendekatan yang solid untuk klasifikasi medical imaging dengan deep learning, menggunakan best practices modern dan teknik-teknik yang proven untuk meningkatkan performa dan stabilitas training. Arsitektur DenseNet yang digunakan bersama dengan strategi training yang optimal memberikan framework yang kuat untuk pengembangan model klasifikasi chest x-ray yang dapat diandalkan.