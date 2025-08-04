import cv2
import numpy as np
import os
import time
from insightface.app import FaceAnalysis

from Silent_Face_Anti_Spoofing.src.anti_spoof_predict import AntiSpoofPredict
from Silent_Face_Anti_Spoofing.src.generate_patches import CropImage
from Silent_Face_Anti_Spoofing.src.utility import parse_model_name

# Anti-spoofing model path and config
MODEL_DIR = "D:/Python project/Face recognization/face-attendance-system/Silent_Face_Anti_Spoofing/resources/anti_spoof_models"
DEVICE_ID = 0


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True
# Anti-spoof function
def test_antispoof(image, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    image = cv2.resize(image, (int(image.shape[0] * 3 / 4), image.shape[0]))
    result = check_image(image)
    if result is False:
        return -1
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    label = np.argmax(prediction)
    return label  # 1 = Real, 0 = Fake

# Initialize face detector (InsightFace)
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame. Exiting ...")
        break

    faces = face_app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box

        # Ensure coordinates are within the frame
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        label = test_antispoof(face_crop, MODEL_DIR, DEVICE_ID)

        if label == 1:
            color = (0, 255, 0)
            text = "Real"
        elif label == 0:
            color = (0, 0, 255)
            text = "Fake"
        else:
            color = (0, 255, 255)
            text = "Invalid"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Detection + Anti-Spoofing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
