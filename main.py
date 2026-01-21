import cv2
import mediapipe as mp

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

    ready = True
    while True:
        ret, frame = webcam.read()
        if ret: # true if frame captured correctly
            camera_logic(ready, frame, hand_detector)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Couldn't capture a frame")
            break

    webcam.release()
    cv2.destroyAllWindows()

def camera_logic(ready: bool, frame: cv2.typing.MatLike, hand_detector):
    w, h = 600, 600
    RED = (0, 0, 255)

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (w, h))
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

            hand_imgs.append(frame[x1:x2, y1:y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 5)
            

    cv2.imshow("webcam", frame)

camera()