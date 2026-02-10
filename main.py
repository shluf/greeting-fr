import cv2
import math
import time
from collections import deque
import statistics
from gtts import gTTS
import pygame
import os
import hashlib
import csv
from datetime import datetime
from config import *

def highlightFace(detector, frame, conf_threshold=FACE_CONFIDENCE_THRESHOLD):
    # YuNet menggunakan API yang berbeda (FaceDetectorYN)
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    # Update ukuran input YuNet sesuai ukuran frame
    detector.setInputSize((frameWidth, frameHeight))
    
    # Deteksi wajah menggunakan YuNet
    _, faces = detector.detect(frame)
    faceBoxes = []
    
    # YuNet mengembalikan None jika tidak ada wajah
    if faces is not None:
        for face in faces:
            # YuNet format: [x, y, w, h, ...landmarks, confidence]
            x, y, w, h = face[0:4].astype(int)
            confidence = face[-1]
            
            if confidence > conf_threshold:
                # Konversi ke format [x1, y1, x2, y2]
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                # PENINGKATAN: Validasi ukuran wajah (filter yang terlalu kecil)
                if w >= MIN_FACE_WIDTH and h >= MIN_FACE_HEIGHT:
                    faceBoxes.append([x1, y1, x2, y2])
                    
                    # Gambar kotak wajah dengan info confidence
                    cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), FACE_BOX_COLOR, FACE_BOX_THICKNESS, cv2.LINE_AA)
                    # Tampilkan confidence score
                    cv2.putText(frameOpencvDnn, f'{confidence*100:.1f}%', (x1, y1-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, FACE_BOX_COLOR, 1, cv2.LINE_AA)
    
    return frameOpencvDnn, faceBoxes

# --- 1. Load Model Files (Pastikan file ada di folder yang sama) ---
# File untuk Umur
ageProto = MODEL_PREFIX + "age_deploy.prototxt"
ageModel = MODEL_PREFIX + "age_net.caffemodel"
# File untuk Gender
genderProto = MODEL_PREFIX + "gender_deploy.prototxt"
genderModel = MODEL_PREFIX + "gender_net.caffemodel"

# --- 2. Inisialisasi Network ---
print("Sedang memuat model...")

try:
    # YuNet - Face Detector modern dari OpenCV
    yunet_model_path = MODEL_PREFIX + "face_detection_yunet_2023mar.onnx"
    
    # Coba load model lokal, jika tidak ada download otomatis
    try:
        faceNet = cv2.FaceDetectorYN.create(
            yunet_model_path,
            "",  # config (kosong untuk ONNX)
            (320, 320),  # input size (akan diupdate saat runtime)
            score_threshold=YUNET_SCORE_THRESHOLD,
            nms_threshold=YUNET_NMS_THRESHOLD,
            top_k=YUNET_TOP_K
        )
        print("✓ YuNet model loaded dari file lokal")
    except:
        # Download dari OpenCV Zoo jika belum ada
        print("Downloading YuNet model dari OpenCV Zoo...")
        import urllib.request
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        urllib.request.urlretrieve(url, yunet_model_path)
        print("✓ Download selesai")
        
        faceNet = cv2.FaceDetectorYN.create(
            yunet_model_path,
            "",
            (320, 320),
            score_threshold=YUNET_SCORE_THRESHOLD,
            nms_threshold=YUNET_NMS_THRESHOLD,
            top_k=YUNET_TOP_K
        )
    
    # Load model Age & Gender (tetap menggunakan Caffe)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    print("✓ Age & Gender models loaded")
    
except Exception as e:
    print(f"Error: Gagal memuat model!\nDetail: {e}")
    exit()

# Buka Webcam (0 adalah ID kamera default)
video = cv2.VideoCapture(CAMERA_ID)

# --- PENINGKATAN AKURASI ---
# Dictionary untuk tracking setiap wajah
face_trackers = {}
next_face_id = 0

def get_rotated_line_points(frame_width, frame_height, position_ratio, angle):
    """Menghitung titik awal dan akhir garis dengan rotasi"""
    # Titik tengah garis (pusat rotasi)
    center_x = frame_width // 2
    center_y = int(frame_height * position_ratio)
    
    # Panjang garis (diagonal frame untuk memastikan garis menutupi frame)
    line_length = int(math.sqrt(frame_width**2 + frame_height**2))
    half_length = line_length // 2
    
    # Konversi sudut ke radian
    angle_rad = math.radians(angle)
    
    # Hitung titik awal dan akhir dengan rotasi
    dx = half_length * math.cos(angle_rad)
    dy = half_length * math.sin(angle_rad)
    
    x1 = int(center_x - dx)
    y1 = int(center_y - dy)
    x2 = int(center_x + dx)
    y2 = int(center_y + dy)
    
    return (x1, y1), (x2, y2)

def check_boundary_crossing(face_center_x, face_center_y, frame_width, frame_height, face_id, angle):
    """Cek apakah wajah melewati garis batas yang dirotasi"""
    # Hitung posisi garis batas
    boundary_center_x = frame_width // 2
    boundary_center_y = int(frame_height * BOUNDARY_LINE_POSITION)
    
    # Simpan posisi sebelumnya jika belum ada
    if face_id not in face_trackers:
        return False
    
    tracker = face_trackers[face_id]
    
    # Cek apakah sudah pernah crossing
    if 'has_crossed' not in tracker:
        tracker['has_crossed'] = False
        # Hitung posisi awal relatif terhadap garis
        tracker['last_side'] = calculate_side_of_line(
            face_center_x, face_center_y, 
            boundary_center_x, boundary_center_y, 
            angle
        )
    
    # Tentukan sisi saat ini
    current_side = calculate_side_of_line(
        face_center_x, face_center_y,
        boundary_center_x, boundary_center_y,
        angle
    )
    
    # Deteksi crossing (dari sisi negatif ke positif)
    if (tracker['last_side'] < 0 and current_side > 0 
        and not tracker['has_crossed']):
        tracker['has_crossed'] = True
        tracker['last_side'] = current_side
        return True
    
    # Update posisi
    tracker['last_side'] = current_side
    
    # Reset jika wajah kembali ke sisi negatif (untuk visit berikutnya)
    if current_side < 0 and tracker['has_crossed']:
        tracker['has_crossed'] = False
    
    return False

def calculate_side_of_line(px, py, line_cx, line_cy, angle):
    """
    Menghitung sisi mana titik berada relatif terhadap garis yang dirotasi
    Return: negatif jika di satu sisi, positif jika di sisi lain
    """
    # Konversi sudut ke radian
    angle_rad = math.radians(angle)
    
    # Vektor normal garis (tegak lurus terhadap garis)
    nx = -math.sin(angle_rad)
    ny = math.cos(angle_rad)
    
    # Vektor dari titik tengah garis ke titik yang dicek
    dx = px - line_cx
    dy = py - line_cy
    
    # Dot product untuk menentukan sisi
    return dx * nx + dy * ny

# --- SISTEM GREETING BAHASA JAWA ---
# Inisialisasi pygame mixer untuk audio
pygame.mixer.init()

# Variabel untuk tracking greeting
last_greeting_hash = None
greeting_cooldown = 0

# --- SISTEM VISITOR COUNTER ---
def init_visitor_log():
    """Inisialisasi file CSV untuk logging pengunjung"""
    if not ENABLE_VISITOR_LOGGING:
        return
    
    # Buat direktori data jika belum ada
    os.makedirs(os.path.dirname(VISITOR_LOG_FILE), exist_ok=True)
    
    # Cek apakah file sudah ada
    file_exists = os.path.isfile(VISITOR_LOG_FILE)
    
    # Jika file belum ada, buat dengan header
    if not file_exists:
        with open(VISITOR_LOG_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Tanggal', 'Waktu', 'Gender', 'Kategori_Umur', 'Rentang_Umur', 'Jumlah_Grup'])

def log_visitor(genders, ages, count):
    """Simpan data pengunjung ke CSV"""
    if not ENABLE_VISITOR_LOGGING:
        return
    
    try:
        now = datetime.now()
        tanggal = now.strftime('%Y-%m-%d')
        waktu = now.strftime('%H:%M:%S')
        
        # Jika grup, catat semua anggota
        with open(VISITOR_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            for i in range(len(genders)):
                gender = genders[i]
                age = ages[i]
                age_category = get_age_category(age)
                
                writer.writerow([
                    tanggal,
                    waktu,
                    gender,
                    age_category,
                    age,
                    count
                ])
        
        print(f"[LOG] Data pengunjung disimpan: {count} orang pada {waktu}")
    except Exception as e:
        print(f"Error logging visitor: {e}")

def get_visitor_stats_today():
    """Mendapatkan statistik pengunjung hari ini"""
    if not ENABLE_VISITOR_LOGGING or not os.path.isfile(VISITOR_LOG_FILE):
        return {'total': 0, 'male': 0, 'female': 0, 'age_categories': {}}
    
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        total = 0
        male_count = 0
        female_count = 0
        age_categories = {}
        
        with open(VISITOR_LOG_FILE, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Tanggal'] == today:
                    total += 1
                    if row['Gender'] == 'Male':
                        male_count += 1
                    else:
                        female_count += 1
                    
                    age_cat = row['Kategori_Umur']
                    age_categories[age_cat] = age_categories.get(age_cat, 0) + 1
        
        return {
            'total': total,
            'male': male_count,
            'female': female_count,
            'age_categories': age_categories
        }
    except Exception as e:
        print(f"Error getting visitor stats: {e}")
        return {'total': 0, 'male': 0, 'female': 0, 'age_categories': {}}

# Inisialisasi visitor log
init_visitor_log()

def get_age_category(age_string):
    """Mengkategorikan umur ke dalam kelompok"""
    age_map = {
        '(0-2)': 'bayi',
        '(4-6)': 'bocah',
        '(8-12)': 'bocah',
        '(15-20)': 'nom',
        '(25-32)': 'mudha',
        '(38-43)': 'dewasa',
        '(48-53)': 'sepuh',
        '(60-100)': 'sepuh'
    }
    return age_map.get(age_string, 'mudha')

def generate_javanese_greeting(genders, ages, count):
    """Generate sapaan bahasa Jawa berdasarkan komposisi pengunjung"""
    
    # Hitung komposisi gender
    male_count = genders.count('Male')
    female_count = genders.count('Female')
    
    # Kategorikan umur
    age_categories = [get_age_category(age) for age in ages]
    dominant_age = max(set(age_categories), key=age_categories.count)
    
    # Tentukan sapaan berdasarkan waktu
    hour = time.localtime().tm_hour
    if 3 <= hour < 11:
        waktu = "enjing"
    elif 11 <= hour < 15:
        waktu = "siyang"
    elif 15 <= hour < 18:
        waktu = "sonten"
    else:
        waktu = "ndalu"
    
    # Logika sapaan berdasarkan situasi
    if count == 1:
        # Satu orang
        if genders[0] == 'Male':
            if dominant_age in ['bayi', 'bocah']:
                sapaan = f"Sugeng {waktu} dhik, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
            elif dominant_age == 'nom':
                sapaan = f"Sugeng {waktu} mas, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
            elif dominant_age in ['sepuh']:
                sapaan = f"Sugeng {waktu} pakdhe, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
            else:
                sapaan = f"Sugeng {waktu} mas, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
        else:
            if dominant_age in ['bayi', 'bocah']:
                sapaan = f"Sugeng {waktu} dhik, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
            elif dominant_age == 'nom':
                sapaan = f"Sugeng {waktu} mbak, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
            elif dominant_age in ['sepuh']:
                sapaan = f"Sugeng {waktu} budhe, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
            else:
                sapaan = f"Sugeng {waktu} mbak, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
    
    elif count == 2:
        # Dua orang
        if male_count == 2:
            if dominant_age in ['bayi', 'bocah']:
                sapaan = f"Sugeng {waktu} dhik-dhik, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
            else:
                sapaan = f"Sugeng {waktu} mas-mas, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
        elif female_count == 2:
            if dominant_age in ['bayi', 'bocah']:
                sapaan = f"Sugeng {waktu} dhik-dhik, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
            else:
                sapaan = f"Sugeng {waktu} mbak-mbak, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
        else:
            # Campuran gender
            if dominant_age in ['bayi', 'bocah']:
                sapaan = f"Sugeng {waktu} dhik-dhik, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
            else:
                sapaan = f"Sugeng {waktu} mas mbak, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
    
    else:
        # Lebih dari 2 orang (rombongan)
        if dominant_age in ['bayi', 'bocah']:
            sapaan = f"Sugeng {waktu} adhik-adhik, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta. Sampun {count} bocah ingkang rawuh"
        elif male_count > female_count:
            sapaan = f"Sugeng {waktu} mas-mas, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
        elif female_count > male_count:
            sapaan = f"Sugeng {waktu} mbak-mbak, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
        else:
            sapaan = f"Sugeng {waktu} sedanten, sugeng rawuh wonten B M K G stasiun klimatologi Yogyakarta"
    
    return sapaan

def play_greeting(text):
    """Generate dan play audio greeting"""
    try:
        # Buat hash dari text untuk nama file unik
        text_hash = hashlib.md5(text.encode()).hexdigest()
        audio_file = f"{AUDIO_PREFIX}greeting_{text_hash}.mp3"
        
        # Cek apakah audio sudah pernah di-generate
        if not os.path.exists(audio_file):
            print(f"Generating audio: {text}")
            tts = gTTS(text=text, lang='id', slow=False)
            tts.save(audio_file)
        
        # Play audio
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        return text_hash
    except Exception as e:
        print(f"Error playing greeting: {e}")
        return None

def get_face_center(faceBox):
    """Menghitung titik tengah wajah untuk tracking"""
    return ((faceBox[0] + faceBox[2]) // 2, (faceBox[1] + faceBox[3]) // 2)

def find_closest_face(center, face_trackers, max_distance=MAX_TRACKING_DISTANCE):
    """Mencari face ID terdekat berdasarkan posisi"""
    min_dist = max_distance
    closest_id = None
    for face_id, data in face_trackers.items():
        if 'last_center' in data:
            dist = math.sqrt((center[0] - data['last_center'][0])**2 + 
                           (center[1] - data['last_center'][1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_id = face_id
    return closest_id

print("Tekan 'q' pada keyboard untuk keluar.")

# Variabel untuk tracking visitor yang sudah di-log
logged_visitors = set()

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    # 1. Deteksi Wajah
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    
    # Gambar garis batas dengan rotasi
    frame_height = resultImg.shape[0]
    frame_width = resultImg.shape[1]
    
    # Dapatkan titik awal dan akhir garis dengan rotasi
    pt1, pt2 = get_rotated_line_points(frame_width, frame_height, 
                                       BOUNDARY_LINE_POSITION, 
                                       BOUNDARY_ROTATION_ANGLE)
    
    # Gambar garis
    cv2.line(resultImg, pt1, pt2, BOUNDARY_COLOR, BOUNDARY_THICKNESS)
    
    # Tambahkan label garis batas dengan info sudut (opsional)
    if SHOW_BOUNDARY_LABEL:
        label_text = f'BOUNDARY LINE ({BOUNDARY_ROTATION_ANGLE})'
        cv2.putText(resultImg, label_text, (frame_width - 250, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOUNDARY_COLOR, 2, cv2.LINE_AA)
    
    # Hitung jumlah wajah
    jumlah_wajah = len(faceBoxes)
    
    # Tampilkan jumlah wajah di pojok kiri atas
    cv2.putText(resultImg, f'Wajah: {jumlah_wajah}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

    # if not faceBoxes:
        # print("Wajah tidak terdeteksi")
    
    # 2. Analisis Gender & Umur untuk setiap wajah
    current_face_ids = []
    
    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox
        w = x2 - x1
        h = y2 - y1
        
        # PENINGKATAN 1: Hitung koordinat baru dengan padding ratio (margin)
        nx1 = max(0, x1 - int(w * FACE_PADDING_RATIO))
        ny1 = max(0, y1 - int(h * FACE_PADDING_RATIO))
        nx2 = min(frame.shape[1], x2 + int(w * FACE_PADDING_RATIO))
        ny2 = min(frame.shape[0], y2 + int(h * FACE_PADDING_RATIO))
        
        # Potong gambar dengan margin baru
        face = frame[ny1:ny2, nx1:nx2]

        if face.size == 0: # Skip jika crop error
            continue
        
        # Cari atau buat face tracker untuk wajah ini
        center = get_face_center(faceBox)
        face_id = find_closest_face(center, face_trackers)
        
        if face_id is None:
            # Wajah baru
            face_id = next_face_id
            next_face_id += 1
            face_trackers[face_id] = {
                'gender_buffer': deque(maxlen=TRACKING_BUFFER_SIZE),
                'age_buffer': deque(maxlen=TRACKING_BUFFER_SIZE),
                'last_center': center,
                'frames_lost': 0
            }
        else:
            # Update posisi wajah yang sudah ada
            face_trackers[face_id]['last_center'] = center
            face_trackers[face_id]['frames_lost'] = 0
        
        current_face_ids.append(face_id)

        # Proses Blob untuk input ke AI
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # -- Prediksi Gender --
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = GENDER_LIST[genderPreds[0].argmax()]
        
        # -- Prediksi Umur --
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = AGE_LIST[agePreds[0].argmax()]
        
        # PENINGKATAN 2: Tambahkan ke buffer untuk stabilizer
        face_trackers[face_id]['gender_buffer'].append(gender)
        face_trackers[face_id]['age_buffer'].append(age)
        
        # Ambil hasil dari voting system (modus dari 10 frame terakhir)
        try:
            final_gender = statistics.mode(face_trackers[face_id]['gender_buffer'])
            final_age = statistics.mode(face_trackers[face_id]['age_buffer'])
        except:
            # Jika buffer belum penuh atau ada hasil seri, gunakan hasil terbaru
            final_gender = gender
            final_age = age

        # Buat Label dengan hasil yang sudah di-stabilkan
        label = f"{final_gender}, {final_age}"
        
        # Tulis Label di atas kotak wajah
        cv2.putText(resultImg, label, (faceBox[0], faceBox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
    
    # Bersihkan face tracker yang sudah tidak terdeteksi lagi
    faces_to_remove = []
    for face_id in face_trackers:
        if face_id not in current_face_ids:
            face_trackers[face_id]['frames_lost'] += 1
            # Hapus jika sudah tidak terdeteksi untuk beberapa frame
            if face_trackers[face_id]['frames_lost'] > FRAMES_BEFORE_REMOVAL:
                faces_to_remove.append(face_id)
    
    for face_id in faces_to_remove:
        del face_trackers[face_id]
    
    # --- SISTEM GREETING OTOMATIS (HANYA JIKA MELEWATI GARIS BATAS) ---
    greeting_cooldown -= 1
    
    # Track crossing events
    crossing_detected = False
    
    # Cek apakah ada wajah yang melewati garis batas
    for face_id in current_face_ids:
        if face_id in face_trackers and 'last_center' in face_trackers[face_id]:
            face_center_x, face_center_y = face_trackers[face_id]['last_center']
            if check_boundary_crossing(face_center_x, face_center_y, 
                                      frame_width, frame_height, 
                                      face_id, BOUNDARY_ROTATION_ANGLE):
                crossing_detected = True
                print(f"✓ Wajah {face_id} melewati garis batas!")
                break
    
    # Hanya greeting jika ada crossing dan cooldown sudah habis
    if crossing_detected and len(faceBoxes) > 0 and greeting_cooldown <= 0:
        # Kumpulkan data dari face trackers yang stabil (minimal 5 frame)
        stable_genders = []
        stable_ages = []
        crossing_face_ids = []
        
        for face_id in current_face_ids:
            tracker = face_trackers[face_id]
            # Pastikan buffer sudah cukup data
            if len(tracker['gender_buffer']) >= MIN_STABLE_FRAMES:
                try:
                    gender = statistics.mode(tracker['gender_buffer'])
                    age = statistics.mode(tracker['age_buffer'])
                    stable_genders.append(gender)
                    stable_ages.append(age)
                    crossing_face_ids.append(face_id)
                except:
                    pass
        
        # Jika ada data yang stabil, generate greeting
        if len(stable_genders) > 0:
            greeting_text = generate_javanese_greeting(stable_genders, stable_ages, len(stable_genders))
            greeting_hash = hashlib.md5(greeting_text.encode()).hexdigest()
            
            # Hanya play jika greeting berbeda dari yang terakhir
            if greeting_hash != last_greeting_hash:
                play_greeting(greeting_text)
                last_greeting_hash = greeting_hash
                greeting_cooldown = GREETING_COOLDOWN_FRAMES
                
                # Log visitor data (hanya untuk wajah yang belum di-log)
                visitors_to_log = []
                genders_to_log = []
                ages_to_log = []
                
                for i, face_id in enumerate(crossing_face_ids):
                    if face_id not in logged_visitors:
                        visitors_to_log.append(face_id)
                        genders_to_log.append(stable_genders[i])
                        ages_to_log.append(stable_ages[i])
                        logged_visitors.add(face_id)
                
                if len(visitors_to_log) > 0:
                    log_visitor(genders_to_log, ages_to_log, len(visitors_to_log))
                
                # Tampilkan greeting di layar
                cv2.putText(resultImg, "Greeting: Playing...", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Tampilkan status monitoring
    if greeting_cooldown > 0:
        cv2.putText(resultImg, f"Cooldown: {greeting_cooldown//30}s", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Tampilkan statistik pengunjung hari ini
    if ENABLE_VISITOR_LOGGING:
        stats = get_visitor_stats_today()
        cv2.putText(resultImg, f"Hari ini: {stats['total']} pengunjung", (20, frame_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f"L: {stats['male']} | P: {stats['female']}", (20, frame_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
    # Tampilkan window
    cv2.imshow("Real-time Face/Age/Gender Detection", resultImg)

# Bersihkan memori saat selesai
video.release()
cv2.destroyAllWindows()