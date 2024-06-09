from tensorflow.keras.models import load_model # Mengimpor fungsi load_model dari modul models dalam library Keras yang terintegrasi dengan TensorFlow
from time import sleep # Untuk memberikan jeda antara iterasi loop
from tensorflow.keras.preprocessing.image import img_to_array # Mengonversi objek gambar menjadi array (NumPy)
import cv2 # Digunakan untuk mengimpor modul cv2 (OpenCV) dalam Python
import numpy as np # Untuk komputasi numerik dan array multidimensi



# Memanggil cascade classifier yang tersimpan pada file .xml
face_classifier = cv2.CascadeClassifier(r'D:\semester 4\pengertian AI\UAS\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')

# Memuat model yang telah dilatih sebelumnya untuk klasifikasi emosi
classifier = load_model(r'D:\semester 4\pengertian AI\UAS\Emotion_Detection_CNN-main\model.h5')

# Mendefinisikan klasifikasi output emosi yang mungkin akan terjadi
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Mengambil video menggunakan kamera default laptop
cap = cv2.VideoCapture(0)

while True: # Saat kamera dinyalakan maka loop dimulai
    _, frame = cap.read() # Membaca frame dari video
    labels = [] # Menyimpan label emosi yang terdeteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Mengubah frame menjadi citra grayscale
    faces = face_classifier.detectMultiScale(gray) # Mendeteksi wajah dalam citra grayscale
    
    for (x, y, w, h) in faces: # Mulai loop untuk setiap wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2) # Membuat persegi menggunakan OpenCV untuk menggambar kotak pada frame
        roi_gray = gray[y:y+h, x:x+w] # Region of interest: Wajah yang akan diproses selanjutnya
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA) # Proses resize dari OpenCV
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0 # Konversi area wajah menjadi tipe data float dan melakukan normalisasi
            roi = img_to_array(roi) # Mengonversi area wajah ke dalam bentuk array
            roi = np.expand_dims(roi, axis=0) # Menambahkan dimensi tambahan pada array
            
            prediction = classifier.predict(roi)[0] # Melakukan prediksi emosi pada area wajah yang telah diproses menggunakan model CNN
            label = emotion_labels[prediction.argmax()] # Memilih label emosi dengan nilai probabilitas tertinggi dari prediksi
            label_position = (x, y) # Menentukan posisi label pada frame berdasarkan koordinat wajah yang terdeteksi
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Menampilkan teks label emosi pada frame
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detector', frame) # Menampilkan frame gambar dalam sebuah jendela tampilan dengan judul "Emotion Detector"
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Menghentikan loop utama dalam program jika tombol 'q' pada keyboard ditekan
        break

cap.release() # Menghentikan pengambilan frame
cv2.destroyAllWindows() # Membersihkan jendela tampilan setelah aplikasi selesai dieksekusi
