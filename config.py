"""
Konfigurasi untuk Smart Greeting System
"""

# ==============================================================================
# KONFIGURASI MODEL
# ==============================================================================

# Path ke folder model
MODEL_PREFIX = "./models/"

# Nilai rata-rata warna untuk normalisasi (Standard Model Caffe)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# List Klasifikasi (Model membagi umur dalam rentang)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# ==============================================================================
# KONFIGURASI FACE DETECTION
# ==============================================================================

# Confidence threshold untuk deteksi wajah (0.0 - 1.0)
# Nilai lebih tinggi = lebih selektif, lebih sedikit false positive
FACE_CONFIDENCE_THRESHOLD = 0.8

# YuNet detector parameters
YUNET_SCORE_THRESHOLD = 0.8
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000

# Ukuran minimum wajah yang dideteksi (dalam pixel)
MIN_FACE_WIDTH = 30
MIN_FACE_HEIGHT = 30

# Padding ratio untuk crop wajah (0.0 - 1.0)
# Nilai lebih besar = margin lebih luas di sekitar wajah
FACE_PADDING_RATIO = 0.15

# ==============================================================================
# KONFIGURASI FACE TRACKING
# ==============================================================================

# Ukuran buffer untuk voting system (stabilizer)
TRACKING_BUFFER_SIZE = 10

# Jarak maksimum untuk mencocokkan wajah di frame berbeda (dalam pixel)
MAX_TRACKING_DISTANCE = 100

# Jumlah frame sebelum wajah dianggap hilang
FRAMES_BEFORE_REMOVAL = 30

# Minimum frame untuk data stabil sebelum greeting
MIN_STABLE_FRAMES = 5

# ==============================================================================
# KONFIGURASI BOUNDARY LINE
# ==============================================================================

# Posisi garis batas (0.0 - 1.0, dari atas layar)
# 0.0 = paling atas, 0.5 = tengah, 1.0 = paling bawah
BOUNDARY_LINE_POSITION = 0.6

# Sudut rotasi garis batas (dalam derajat)
# 0 = horizontal, 45 = diagonal, 90 = vertikal
BOUNDARY_ROTATION_ANGLE = -30

# Ketebalan garis batas (dalam pixel)
BOUNDARY_THICKNESS = 3

# Warna garis batas (BGR format)
# (0, 0, 255) = Merah, (0, 255, 0) = Hijau, (255, 0, 0) = Biru
BOUNDARY_COLOR = (0, 0, 255)

# ==============================================================================
# KONFIGURASI GREETING
# ==============================================================================

# Cooldown frames antara greeting (1 frame jika FPS=30 maka ~0.033 detik)
# Nilai lebih besar = jeda lebih lama antara greeting
GREETING_COOLDOWN_FRAMES = 1

# Path ke folder audio greeting
AUDIO_PREFIX = "./sounds/"

# ==============================================================================
# KONFIGURASI KAMERA
# ==============================================================================

# ID kamera (0 = default, 1 = kamera kedua, dst)
CAMERA_ID = 2

# ==============================================================================
# KONFIGURASI TAMPILAN
# ==============================================================================

# Warna untuk kotak wajah (BGR)
FACE_BOX_COLOR = (0, 255, 0)  # Hijau
FACE_BOX_THICKNESS = 2

# Warna untuk label teks (BGR)
TEXT_COLOR = (0, 255, 255)  # Kuning
TEXT_THICKNESS = 2
TEXT_FONT_SCALE = 0.8

# Tampilkan label boundary line di layar
SHOW_BOUNDARY_LABEL = False
