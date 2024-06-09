Emotion Classification using Convolutional Neural Network (CNN)

Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan sebuah sistem yang mampu mengklasifikasikan emosi manusia berdasarkan gambar wajah menggunakan Convolutional Neural Network (CNN). Proyek ini mencakup pembuatan dataset, pelatihan model CNN, serta penggunaan model untuk mengklasifikasikan emosi pada gambar wajah.

Struktur Direktori
1. images/: Direktori yang berisi dataset gambar wajah untuk pelatihan dan pengujian model.
2. classification.ipynb: Notebook Jupyter yang berisi kode untuk pelatihan model CNN dan evaluasi performanya.
3. main.py: Berkas Python yang berfungsi sebagai skrip utama untuk menggunakan model yang telah dilatih untuk mengklasifikasikan emosi pada gambar wajah.
4. model.h5: Berkas yang berisi model CNN yang telah dilatih dan disimpan dalam format h5.
5. haarcascade.xml: Berkas XML yang berisi konfigurasi Haar Cascade untuk deteksi wajah pada gambar.
Cara Penggunaan
Persiapan Dataset: Pastikan dataset gambar wajah tersedia dalam direktori images/. Dataset dapat berisi gambar-gambar dengan label emosi yang sesuai.
Pelatihan Model: Jalankan notebook classification.ipynb untuk melatih model CNN menggunakan dataset yang telah disiapkan. Pastikan semua dependensi telah diinstall sebelum menjalankan notebook ini.
Evaluasi Performa Model: Setelah pelatihan selesai, Anda dapat mengevaluasi performa model menggunakan metrik yang sesuai dalam notebook tersebut.
Penggunaan Model: Anda dapat menggunakan model yang telah dilatih untuk mengklasifikasikan emosi pada gambar wajah. Gunakan main.py untuk menjalankan skrip utama.
Dependensi
Pastikan telah menginstall dependensi berikut sebelum menjalankan notebook atau skrip utama:

Python 3.x
TensorFlow
Keras
OpenCV
