import cv2
import time
import winsound

# Load Haar Cascade Classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Webcam
cap = cv2.VideoCapture(0)

# Settings
CLOSED_TIME_LIMIT = 1.0  # Alarm after 1 second

eye_closed_start = None

print("Driver Drowsiness Detection Started")
print("Press 'Q' to Exit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    eyes_detected = False

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=8
        )

        if len(eyes) >= 2:
            eyes_detected = True

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    roi_color,
                    (ex, ey),
                    (ex+ew, ey+eh),
                    (255, 0, 0),
                    2
                )

    # Drowsiness Logic
    if not eyes_detected:

        if eye_closed_start is None:
            eye_closed_start = time.time()

        elapsed_time = time.time() - eye_closed_start

        cv2.putText(
            frame,
            f"Eyes Closed: {elapsed_time:.1f}s",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if elapsed_time >= CLOSED_TIME_LIMIT:

            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            winsound.Beep(2500, 300)

    else:

        eye_closed_start = None

        cv2.putText(
            frame,
            "Driver Active",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()