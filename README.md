# Smart Greeting System
Sistem greeting otomatis berbasis deteksi wajah dengan teknologi AI untuk menyambut pengunjung menggunakan bahasa Jawa sesuai konteks (gender, usia, dan waktu).

## Fitur Utama

- **Deteksi Wajah Real-time**: Menggunakan YuNet Face Detector (OpenCV) untuk akurasi tinggi
- **Prediksi Gender & Usia**: Deep learning model berbasis Caffe
- **Boundary Line Detection**: Sistem deteksi crossing dengan support rotasi garis arbitrary (0-360 derajat)
- **Greeting Bahasa Jawa Otomatis**: Sapaan kontekstual berdasarkan:
  - Gender (Mas/Mbak/Pakdhe/Budhe)
  - Kategori usia (Bocah/Nom/Dewasa/Sepuh)
  - Waktu (Enjing/Siyang/Sonten/Ndalu)
  - Jumlah pengunjung (Single/Grup/Rombongan)
- **Text-to-Speech**: Audio greeting menggunakan gTTS
- **Face Tracking**: Sistem tracking per individu untuk konsistensi hasil
- **Anti-Flickering**: Voting system dengan buffer 10 frame untuk stabilitas prediksi
- **Smart Cooldown**: Mencegah greeting berulang yang tidak perlu
- **Visitor Counter**: Pencatatan otomatis data pengunjung harian (gender, usia) ke CSV

## Teknologi

- **Face Detection**: YuNet (ONNX model dari OpenCV Zoo)
- **Age & Gender Classification**: Caffe models
- **TTS Engine**: Google Text-to-Speech (gTTS)
- **Audio Playback**: Pygame mixer
- **Computer Vision**: OpenCV (cv2)

## Prerequisites

- Python 3.8 atau lebih tinggi
- Webcam/Camera
- OS: Windows/Linux/MacOS

## Instalasi

1. Clone repository:
```bash
git clone https://github.com/shluf/greeting-fr
cd greeting-fr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pastikan model files tersedia di direktori project:
   - `face_detection_yunet_2023mar.onnx` (auto-download jika belum ada)
   - `age_deploy.prototxt`
   - `age_net.caffemodel`
   - `gender_deploy.prototxt`
   - `gender_net.caffemodel`

## Cara Penggunaan

1. Jalankan aplikasi:
```bash
python main.py
```

2. Konfigurasi (edit file `config.py`):
   - `BOUNDARY_LINE_POSITION`: Posisi garis batas (0.0-1.0, default 0.4)
   - `BOUNDARY_ROTATION_ANGLE`: Sudut rotasi garis (0-360 derajat, default 30)
   - `GREETING_COOLDOWN_FRAMES`: Jeda antar greeting (default 1 frame)
   - `CAMERA_ID`: ID kamera yang digunakan (default 2)
   - Dan banyak parameter lainnya untuk fine-tuning

3. Tekan 'q' untuk keluar

## Struktur Project

```
greeting/
├── main.py                              # Program utama
├── config.py                            # File konfigurasi
├── requirements.txt                     # Python dependencies
├── README.md                           # Dokumentasi
├── models/                             # Folder model AI
│   ├── face_detection_yunet_2023mar.onnx
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
├── sounds/                             # Folder audio greeting (auto-generated)
│   └── greeting_*.mp3
└── data/                               # Folder data pengunjung
    └── visitor_log.csv                 # Log pengunjung harian
```

## Cara Kerja Sistem

1. **Face Detection**: Deteksi wajah menggunakan YuNet dengan confidence threshold 80%
2. **Face Tracking**: Setiap wajah diberi ID unik dan di-track posisinya
3. **Feature Extraction**: Crop wajah dengan padding 15% untuk konteks lebih baik
4. **Classification**: Prediksi gender dan usia menggunakan deep learning
5. **Stabilization**: Voting system dari 10 frame terakhir untuk hasil yang stabil
6. **Boundary Crossing**: Deteksi saat wajah melewati garis batas yang telah dikonfigurasi
7. **Greeting Generation**: Generate sapaan bahasa Jawa berdasarkan konteks
8. **Audio Playback**: Convert text ke speech dan play audio greeting
9. **Data Logging**: Catat data pengunjung (tanggal, waktu, gender, usia) ke CSV

## Visitor Counter System

Sistem secara otomatis mencatat setiap pengunjung yang melewati boundary line dengan informasi:
- Tanggal dan waktu kunjungan
- Gender (Male/Female)
- Kategori usia (bayi, bocah, nom, mudha, dewasa, sepuh)
- Rentang usia prediksi
- Jumlah orang dalam grup

Data disimpan di `data/visitor_log.csv` dan dapat dianalisis untuk statistik pengunjung harian, mingguan, atau bulanan.

Format CSV:
```
Tanggal,Waktu,Gender,Kategori_Umur,Rentang_Umur,Jumlah_Grup
2026-02-10,14:30:15,Male,mudha,(25-32),1
2026-02-10,14:35:22,Female,nom,(15-20),2
```

Statistik hari ini ditampilkan real-time di layar:
- Total pengunjung hari ini
- Jumlah laki-laki dan perempuan

## Algoritma Boundary Crossing

Sistem menggunakan algoritma geometric untuk deteksi crossing pada garis dengan rotasi arbitrary:

- Menghitung sisi titik (face center) relatif terhadap garis menggunakan dot product
- Deteksi crossing saat titik berpindah dari sisi negatif ke positif
- Support rotasi 0-360 derajat dengan perhitungan vektor normal

## Kategori Usia

- (0-2): Bayi
- (4-6): Bocah
- (8-12): Bocah
- (15-20): Nom
- (25-32): Mudha
- (38-43): Dewasa
- (48-53): Sepuh
- (60-100): Sepuh

## Contoh Greeting

- **Single visitor (pria muda, pagi)**: "Sugeng enjing mas, sugeng rawuh wonten BMKG stasiun klimatologi Yogyakarta"
- **Dua pengunjung (wanita)**: "Sugeng siyang mbak-mbak, sugeng rawuh wonten BMKG stasiun klimatologi Yogyakarta"
- **Rombongan**: "Sugeng sonten sedanten, sugeng rawuh wonten BMKG stasiun klimatologi Yogyakarta"

## Troubleshooting

### Kamera tidak terdeteksi
Ubah parameter di `cv2.VideoCapture(X)` dengan nilai 0, 1, atau 2 sesuai device index kamera.

### Model tidak ditemukan
Pastikan semua file model (.prototxt dan .caffemodel) berada di direktori yang sama dengan main.py.

### Audio tidak keluar
Pastikan pygame mixer ter-install dengan benar dan sistem audio berfungsi normal.

### Deteksi tidak akurat
- Pastikan pencahayaan cukup
- Posisi wajah menghadap kamera
- Jarak optimal: 0.5-2 meter dari kamera

## Lisensi Model

- **YuNet**: OpenCV Zoo (Apache License 2.0)
- **Age & Gender Models**: Caffe Model Zoo

## Kontribusi

Project ini dikembangkan untuk BMKG Stasiun Klimatologi Yogyakarta. Untuk kontribusi atau pertanyaan, silakan hubungi tim pengembang.

## Author

Developed for BMKG Stasiun Klimatologi Yogyakarta

## Version

1.0.0 (February 2026)
