import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cnn_model.h5")

model = load_model(MODEL_PATH)

IMG_SIZE = 128

# 🔧 TUNED VALUES (IMPORTANT)
THRESHOLD = 0.85          # higher → fewer false positives
FREQ_THRESHOLD = 6.5      # ignores normal photo edits

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- FREQUENCY ANALYSIS ----------------
def frequency_artifact_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return np.mean(magnitude)

# ---------------- IMAGE PREDICTION ----------------
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid Image"

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        return "REAL IMAGE ✅"

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    # ---------- CNN FEATURE ----------
    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face_norm = face_resized / 255.0
    face_norm = np.reshape(face_norm, (1, IMG_SIZE, IMG_SIZE, 3))

    cnn_score = model.predict(face_norm, verbose=0)[0][0]

    # ---------- FACE BOUNDARY CHECK ----------
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges)

    # ---------- STRICT DECISION (VERY IMPORTANT) ----------
    if cnn_score > 0.95 and edge_density < 8:
        return f"DEEPFAKE IMAGE ❌ (confidence: {cnn_score*100:.1f}%)"
    else:
        return f"REAL IMAGE ✅ (confidence: {(1-cnn_score)*100:.1f}%)"

# ---------------- VIDEO PREDICTION ----------------
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 3))

        pred = model.predict(face, verbose=0)[0][0]
        preds.append(pred)

    cap.release()

    if len(preds) < 10:
        return "REAL VIDEO ✅"

    mean_pred = np.mean(preds)
    std_pred = np.std(preds)

    # ---------- TEMPORAL CONSISTENCY RULE ----------
    if mean_pred > 0.90 and std_pred > 0.08:
        return f"DEEPFAKE VIDEO ❌ (confidence: {mean_pred*100:.1f}%)"
    else:
        return f"REAL VIDEO ✅ (confidence: {(1-mean_pred)*100:.1f}%)"

