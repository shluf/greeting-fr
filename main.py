import face_recognition
import cv2
import numpy as np
import os
import time
import pickle
import numpy as np
import datetime
import requests
import subprocess




def get_greeting_time():
    """Generate dynamic greeting based on current time"""
    current_hour = datetime.datetime.now().hour
    
    if 3 <= current_hour < 11:
        return "Selamat pagi"
    elif 11 <= current_hour < 15:
        return "Selamat siang"
    elif 15 <= current_hour < 18:
        return "Selamat sore"
    else:
        return "Selamat malam"


def Send_Push_Notifications(Name , Image_File):
    Image_File = 'Webcam_'+str(Image_File)+'.jpg'
    r = requests.post("https://api.pushover.net/1/messages.json", data = {
      "token": "YOUR TOKEN HERE",
      "user": "YOUR USER KEY HERE",
      "message": "\""+"Found : "+str(Name)+" With your Laptop.\n"+"Time :"+ Image_File +"\""
    },
    files = {
      "attachment": ("image.jpg", open(Image_File, "rb"), "image/jpeg")
    })
    print(r.text)



video_capture = cv2.VideoCapture(4)

print("Loading Application..")

with open('./models/face/dataset_faces.dat', 'rb') as f:
	all_face_encodings = pickle.load(f)

# Create arrays of known face encodings and their names
known_face_encodings =  np.array(list(all_face_encodings.values()))
known_face_names = list(all_face_encodings.keys())


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

print("Loading App Done.")

print("Press 'q' to quit")

time.sleep(2)

last_greeted_names = []
greeting_cooldown = {}

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read(1)
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        # Process text-to-speech only for new faces or after cooldown
        current_time = time.time()
        new_faces = []
        cooldown_duration = 30  # durasi cooldown dalam detik
        
        for name in face_names:
            if name == "Unknown":
                continue
                
            if name not in greeting_cooldown:
                # Wajah pertama kali terdeteksi
                new_faces.append(name)
                greeting_cooldown[name] = current_time
                print(f"✓ {name}: Pertama kali terdeteksi - akan diputar sapaan")
            else:
                time_since_last_greeting = current_time - greeting_cooldown[name]
                if time_since_last_greeting > cooldown_duration:
                    # Cooldown selesai, bisa sapaan lagi
                    new_faces.append(name)
                    greeting_cooldown[name] = current_time
                    print(f"✓ {name}: Cooldown selesai ({int(time_since_last_greeting)}s) - akan diputar sapaan")
                else:
                    # Masih dalam cooldown
                    remaining_time = int(cooldown_duration - time_since_last_greeting)
                    print(f"⏳ {name}: Dalam cooldown - sisa {remaining_time} detik")

        if new_faces:
            print(f"🎤 Memutar sapaan untuk: {new_faces}")
            face_names_string = " ".join(new_faces)
            face_names_string = face_names_string.replace("_", " ")

            # Generate dynamic greeting based on time
            greeting = get_greeting_time()
            mytext = f'{greeting}, "{face_names_string}". Selamat datang di BMKG Stasiun Klimatologi Yogyakarta.'
            
            # Generate speech using Piper TTS (lightweight, fast, offline)
            print(f"🔊 Generating TTS: \"{mytext}\"")
            piper_cmd = f'echo "{mytext}" | piper --model ./models/tts/id_ID-news_tts-medium.onnx --output-file welcome.wav'
            subprocess.run(piper_cmd, shell=True, check=True, capture_output=True)
            
            # Play the audio file
            print("▶️  Playing audio...")
            os.system("aplay welcome.wav 2>/dev/null")
            print("✓ Audio selesai diputar")

            x = datetime.datetime.now()
            cv2.imwrite('Webcam_'+str(x)+'.jpg', frame)
            # Send_Push_Notifications(str(face_names_string), x)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Tampilkan gambar dengan nama
    cv2.imshow('Face Recognition', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()






