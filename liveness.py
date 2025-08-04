import cv2
from Silent_Face_Anti_Spoofing.test import test


def run_spoof_detection_webcam(model_dir, device_id=0):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = test(image=frame, model_dir=model_dir, device_id=device_id)

        # Display result
        if label == 1:
            cv2.putText(frame, 'Real Face', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Fake Face', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Spoof Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_spoof_detection_webcam(
        model_dir='D:/Python project/Face recognization/face-attendance-system/Silent_Face_Anti_Spoofing/resources/anti_spoof_models',
        device_id=0
    )
