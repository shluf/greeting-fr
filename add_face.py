import face_recognition
import pickle
import os
from tqdm import tqdm

all_face_encodings = {}


print("Loading known face encodings... \nPlease wait..It may take some momemts....")


images = os.listdir('images')

for image in tqdm(images,ncols=100):
    # load the image
    known_image = face_recognition.load_image_file("images/" + image)

    # Get the face encodings for the known images
    face_encodings = face_recognition.face_encodings(known_image)
    
    if len(face_encodings) > 0:
        all_face_encodings[str(image[:-4])] = face_encodings[0]
    else:
        print(f"\nWarning: No face detected in {image}. Skipping...")

with open('./models/face/dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)