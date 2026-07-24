import cv2
import mediapipe as mp
import math
import time

# ==========================================
# CALIBRATION VALUES
# ==========================================
FOCAL_LENGTH = 507          # Your calibrated focal length
REAL_FACE_WIDTH = 15        # Average human face width (cm)

# ==========================================
# MEDIAPIPE FACE DETECTION
# ==========================================
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# ==========================================
# WEBCAM
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

previous_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    h, w, _ = frame.shape

    # Dashboard Panel
    cv2.rectangle(frame, (5, 5), (330, 250), (40, 40, 40), -1)

    # Center Guide Line
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 2)

    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            if width <= 0:
                continue

            face_width = width

            face_center_x = x + width // 2
            face_center_y = y + height // 2

            # Distance
            distance = (FOCAL_LENGTH * REAL_FACE_WIDTH) / face_width

            # Horizontal Angle
            angle = math.degrees(
                math.atan((face_center_x - (w // 2)) / FOCAL_LENGTH)
            )

            # Direction
            if angle < -5:
                direction = "LEFT"
            elif angle > 5:
                direction = "RIGHT"
            else:
                direction = "CENTER"

            # Distance Status
            if distance < 30:
                status = "Too Close"
                status_color = (0, 0, 255)
            elif distance > 100:
                status = "Too Far"
                status_color = (0, 0, 255)
            else:
                status = "Optimal"
                status_color = (0, 255, 0)

            # Draw Face Box
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            # Face Center
            cv2.circle(
                frame,
                (face_center_x, face_center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Face Label
            cv2.putText(
                frame,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Dashboard Text
            cv2.putText(frame, "2D FACE ESTIMATOR",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

            cv2.putText(frame,
                        f"Width      : {face_width} px",
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2)

            cv2.putText(frame,
                        f"Distance : {distance:.1f} cm",
                        (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

            cv2.putText(frame,
                        f"Angle      : {angle:.1f} deg",
                        (15, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2)

            cv2.putText(frame,
                        f"Direction : {direction}",
                        (15, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

            cv2.putText(frame,
                        f"Status     : {status}",
                        (15, 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        status_color,
                        2)

    else:

        cv2.putText(frame,
                    "No Face Detected",
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(frame,
                f"FPS : {int(fps)}",
                (15, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2)

    cv2.imshow("2D Face Distance & Angle Estimator", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if cv2.getWindowProperty(
        "2D Face Distance & Angle Estimator",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break

cap.release()
cv2.destroyAllWindows()