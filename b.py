import cv2
from insightface.app import FaceAnalysis

# Initialize face analysis
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use CPU
app.prepare(ctx_id=0, det_size=(640, 640))

# Start webcam
cap = cv2.VideoCapture(0)  # Try 1 or 2 if this doesn't work

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame. Exiting ...")
        break

    faces = app.get(frame)

    # Draw detected faces
    for face in faces:
        box = face.bbox.astype(int)  # [x1, y1, x2, y2]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow('Webcam Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
