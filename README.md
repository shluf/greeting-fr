# Face Recognition Greeting System 👋

Sistem pengenalan wajah otomatis dengan sapaan text-to-speech dalam Bahasa Indonesia menggunakan Piper TTS.

## Quick Install

```bash
git clone https://github.com/shluf/greeting-fr.git
cd greeting-fr
chmod +x install.sh
./install.sh
```

## Requirements

- Python 3.7+
- Webcam
- Linux (Ubuntu/Debian recommended)
- 2GB RAM minimum

## Dependencies

- face_recognition
- opencv-python
- piper-tts
- numpy
- requests

## Getting Started

### 1. Install Dependencies

Run the installation script:
```bash
./install.sh
```

Or install manually:
```bash
pip3 install -r requirements.txt
```

### 2. Download Piper TTS Model

```bash
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx.json
```

### 3. Add Face Images

Tambahkan foto wajah ke folder `images/` dengan format nama:
```
images/
├── Budi_Santoso.jpg
├── Ahmad.png
├── Siti_Nurhaliza.jpg
```

**Note**: Underscore `_` akan otomatis diganti menjadi spasi saat ditampilkan.

### 4. Build Face Dataset

```bash
python3 add_face.py
```

### 5. Run the Application

```bash
python3 main.py
```

Tekan **'q'** untuk keluar.

## Usage

1. Jalankan `python3 main.py`
2. Aplikasi akan membuka webcam
3. Tunjukkan wajah ke kamera
4. Sistem akan:
   - Mendeteksi dan mengenali wajah
   - Menampilkan nama di layar
   - Mengucapkan sapaan "Selamat datang [Nama]"
   - Menyimpan foto dengan timestamp

## Configuration

### Ubah Cooldown Duration

Edit di [main.py](main.py#L108):
```python
    cooldown_duration = 30  # ubah 30 ke durasi yang diinginkan (detik)
```

### Enable Push Notifications

Edit di [main.py](main.py#L32-L33):
```python
"token": "YOUR_PUSHOVER_TOKEN",
"user": "YOUR_PUSHOVER_USER_KEY",
```

### Ubah Resolusi Kamera

Edit di [main.py](main.py#L43):
```python
video_capture = cv2.VideoCapture(0)  # 0 = default camera, ubah ke 1, 2, dst untuk camera lain
```

## Troubleshooting

### Camera tidak terdeteksi
```bash
# Check camera
ls /dev/video*

# Test camera
ffplay /dev/video0
```

### Audio tidak keluar
```bash
# Install audio player
sudo apt-get install alsa-utils

# Test audio
aplay welcome.wav
```

### Face recognition error
```bash
# Install dependencies
sudo apt-get install cmake build-essential libopenblas-dev liblapack-dev
pip3 install --upgrade face_recognition
```

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Credits

- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [Piper TTS](https://github.com/rhasspy/piper) by Rhasspy
- [OpenCV](https://opencv.org/)

## Contact

For questions or support, please open an issue on GitHub.

---

