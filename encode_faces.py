from mtcnn import MTCNN
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import os
import pickle

# FaceNet এর পরিবর্তে আমরা ResNet50 ব্যবহার করব embeddings তৈরি করার জন্য
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

detector = MTCNN()

dataset_dir = "database/me"  # main folder
encodings = []
names = []

for img_name in os.listdir(dataset_dir):
    img_path = os.path.join(dataset_dir, img_name)
    if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    img_array_rgb = img_array[:, :, ::-1]  # BGR->RGB if needed

    # Face detect
    faces = detector.detect_faces(img_array_rgb)
    if len(faces) != 1:
        print(f"Warning: {img_path} has no face or multiple faces detected.")
        continue

    x, y, w, h = faces[0]['box']
    face_img = img_array_rgb[y:y+h, x:x+w]
    face_img = image.smart_resize(face_img, (224, 224))
    face_img = np.expand_dims(face_img, axis=0)
    face_img = preprocess_input(face_img)

    embedding = model.predict(face_img)[0]
    encodings.append(embedding)
    names.append("me")

# Save embeddings
data = {"encodings": encodings, "names": names}
with open("encodings_mtcnn.pkl", "wb") as f:
    pickle.dump(data, f)

print("Face embeddings saved successfully using MTCNN + ResNet50!")
