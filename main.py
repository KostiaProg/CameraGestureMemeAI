import cv2
import mediapipe as mp

import numpy as np
import time
import operator

from model import get_saved_model, get_fingers

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
    is_num2 = False
    num1 = 0
    num2 = 0
    chosen = 4
    operations = [operator.add, operator.sub, operator.mul, operator.floordiv]

    finger_model = get_saved_model()
    memory = []

    time = None
    add_info = None
    while True:
        while chosen < 0 or chosen > 3:
            chosen = int(input("What operation do you want to do (0 - '+', 1 - '-', 2 - '*', 3 - '/'): "))

        ret, frame = webcam.read()
        if ret: # true if frame captured correctly
            if not is_num2:
                time, memory, is_num2, add_info = camera_logic(num1, is_num2, frame, hand_detector, finger_model, memory, time, WAIT_TIME, add_info)
                if is_num2:
                    num1 = int("".join(map(str, memory)))
                    add_info = str(num1)
                    memory.clear()
            else:
                time, memory, is_num2, add_info = camera_logic(num2, is_num2, frame, hand_detector, finger_model, memory, time, WAIT_TIME, add_info)
                if not is_num2:
                    num2 = int("".join(map(str, memory)))
                    if num2 != 0:
                        ans = operations[chosen](num1, num2)
                        add_info = str(ans)
                    else:
                        add_info = "Can't divide by 0!"

                    num1 = 0
                    num2 = 0
                    chosen = 4
                    

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Couldn't capture a frame")
            break
    
    webcam.release()
    cv2.destroyAllWindows()

def camera_logic(num: int, is_num2: bool, frame: cv2.typing.MatLike, hand_detector, finger_model, memory: list, start: float = None, wait_time: float = 1.0, add_info: str = None) -> float:
    w, h = 600, 600
    RED = (0, 0, 255)

    # some basic settings for frame
    response_image = np.ones((h, w, 3), dtype=np.uint8)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (w, h))

    # wait some time before getting next finger count
    if start is None or time.perf_counter() - start >= wait_time:
        cv2.putText(response_image, "Ready!", (int(w/3), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 5, cv2.LINE_AA)
        add_info = None # restart shown num

        # hands need another color scheme
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = hand_detector.process(frame_rgb)

        # find all hands on the picture
        fingers_on_hands = []
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

                # draw hands position
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 5)

                # save the amount of fingers on each hand
                fingers_on_hands.append(get_fingers(finger_model, cv2.cvtColor(frame[x1:x2, y1:y2], cv2.COLOR_RGB2GRAY)))
                start = time.perf_counter()


        if len(fingers_on_hands) == 2 and fingers_on_hands[0] == 0 and fingers_on_hands[1] == 0:
            # go the next number
            is_num2 = not is_num2
        elif len(fingers_on_hands) != 2 or fingers_on_hands[0] != 5 or fingers_on_hands[1] != 5: # can't have 10
            # add digit to number
            memory.append(sum(fingers_on_hands))

    else:
        if not add_info:
            cv2.putText(response_image, "Wait!", (int(w/3), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 5, cv2.LINE_AA)
        else:
            cv2.putText(response_image, add_info, (int(w/3), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 5, cv2.LINE_AA)


    cv2.imshow("Webcam", frame)
    cv2.imshow("Info", response_image)
    
    return start, memory, is_num2, add_info

camera()