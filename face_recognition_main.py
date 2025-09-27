import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from telegram_notify import send_telegram_message
import time

# ===============================
# Load Encodings
# ===============================
data = pickle.load(open("encodings_mtcnn.pkl", "rb"))
known_encodings = data["encodings"]
known_names = data["names"]

# ===============================
# Load Models
# ===============================
detector = MTCNN()
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# ===============================
# Webcam Setup
# ===============================
video = cv2.VideoCapture(0)
last_alert_time = 0
ALERT_COOLDOWN = 30  # seconds
DISTANCE_THRESHOLD = 0.75  # adjustable threshold

def get_embedding(face_img):
    face_img = image.smart_resize(face_img, (224, 224))
    face_img = np.expand_dims(face_img, axis=0)
    face_img = preprocess_input(face_img)
    embedding = model.predict(face_img)[0]
    return embedding

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]  # BGR -> RGB
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']
        # Correct negative coordinates
        x, y = max(0, x), max(0, y)
        face_img = rgb_frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue  # skip if face crop failed

        embedding = get_embedding(face_img)

        # Compare with known embeddings
        distances = [np.linalg.norm(embedding - enc) for enc in known_encodings]
        if len(distances) > 0 and min(distances) < DISTANCE_THRESHOLD:
            idx = distances.index(min(distances))
            name = known_names[idx]
        else:
            name = "Unknown"

        # Draw rectangle & name
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Send Telegram notification for unknown
        if name == "Unknown":
            current_time = time.time()
            if current_time - last_alert_time > ALERT_COOLDOWN:
                send_telegram_message("Unknown Person Detected! Be Aware Rakin!!!")
                last_alert_time = current_time

    # Show frame
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
