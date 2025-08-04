import cv2
import numpy as np
import uuid
import os
from insightface.app import FaceAnalysis
from Silent_Face_Anti_Spoofing.test import test

# Setup
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

model_dir = 'D:/Python project/Face recognization/face-attendance-system/Silent_Face_Anti_Spoofing/resources/anti_spoof_models'
temp_dir = "./temp_faces"
os.makedirs(temp_dir, exist_ok=True)

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Avoid out-of-bound errors
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = frame[y1:y2, x1:x2]
        temp_img_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
        cv2.imwrite(temp_img_path, face_crop)

        # Run anti-spoofing prediction
        score = test(image=temp_img_path, model_dir=model_dir, device_id=0)
        os.remove(temp_img_path)

        label = "Real" if score > 0.5 else "Fake"
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Anti-Spoofing", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
