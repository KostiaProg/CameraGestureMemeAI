import cv2
import mediapipe as mp

import numpy as np
import time

def camera() -> cv2.VideoCapture:
    hand_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    webcam = cv2.VideoCapture(0) # 0 default

    if not webcam.isOpened():
        print("Couldn't find a camera")
        return None

    WAIT_TIME = 1.5

    time = None
    while True:
        ret, frame = webcam.read()
        if ret: # true if frame captured correctly
            time = camera_logic(frame, hand_detector, time, WAIT_TIME)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Couldn't capture a frame")
            break

    webcam.release()
    cv2.destroyAllWindows()

def camera_logic(frame: cv2.typing.MatLike, hand_detector, start: float = None, wait_time: float = 1.0) -> float:
    w, h = 600, 600
    RED = (0, 0, 255)

    response_image = np.ones((h, w, 3), dtype=np.uint8)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (w, h))

    if start is None or time.perf_counter() - start >= wait_time:
        cv2.putText(response_image, "Ready!", (int(w/3), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 5)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = hand_detector.process(frame_rgb)

        hand_imgs = []
        if hands.multi_hand_landmarks:
            for hand_landmarks in hands.multi_hand_landmarks:
                x1, y1, x2, y2 = w, h, 0, 0

                for point in hand_landmarks.landmark:
                    x = point.x * w
                    y = point.y * h

                    x1 = min(x1, max(0, x-30))
                    y1 = min(y1, max(0, y-30))
                    x2 = max(x2, min(w, x+30))
                    y2 = max(y2, min(h, y+30))

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 5)

                hand_imgs.append(frame[x1:x2, y1:y2])
                start = time.perf_counter()
    else:
        cv2.putText(response_image, "Wait!", (int(w/3), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 5)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Info", response_image)
    
    return start

camera()