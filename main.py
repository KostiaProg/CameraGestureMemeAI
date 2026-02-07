import cv2
import mediapipe as mp

import numpy as np
import operator
from time import perf_counter

from model import get_saved_model, get_fingers

def camera() -> cv2.VideoCapture:
    # setup
    hand_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.75
    )
    webcam = cv2.VideoCapture(0) # 0 default

    if not webcam.isOpened():
        print("Couldn't find a camera")
        return None

    is_num2 = False # are we taking num2 already
    num1 = 0
    num2 = 0
    chosen = -1 # operation index
    operations = [operator.add, operator.sub, operator.mul, operator.floordiv]

    finger_model = get_saved_model()
    memory = []

    time = perf_counter() # wait time
    add_info = None # info to print out
    while True:
        ret, frame = webcam.read()
        if ret: # true if frame captured correctly
            if not is_num2:
                chosen, time, memory, is_num2, add_info = camera_logic(chosen, is_num2, frame, hand_detector, finger_model, memory, time, add_info)
                if is_num2: # when change happens calculate
                    num1 = int("".join(map(str, memory)))
                    add_info = str(num1)
                    memory.clear()
            else:
                chosen, time, memory, is_num2, add_info = camera_logic(chosen, is_num2, frame, hand_detector, finger_model, memory, time, add_info)
                if not is_num2: # when change happens calculate
                    num2 = int("".join(map(str, memory)))
                    if chosen != 3 or num2 != 0:
                        ans = operations[chosen](num1, num2)
                        add_info = str(ans)
                    else:
                        add_info = "Can't divide by 0!"

                    # restart calculations and nums
                    memory.clear()
                    num1 = 0
                    num2 = 0
                    chosen = -1
                    

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Couldn't capture a frame")
            break
    
    webcam.release()
    cv2.destroyAllWindows()

def add_text(text: str, img):
    RED = (0, 0, 255)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1.5
    THICKNESS = 3

    ready_size, _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    ready_center = (img.shape[1] // 2 - ready_size[0], img.shape[0] // 2 + ready_size[1])
    cv2.putText(img, text, ready_center, FONT, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

def get_hand_and_fingers(img, hand_landmarks, fingers_on_hands, prev, wait_time, finger_model):
    TOO_CLOSE = 40
    skin_mask_ycrcb_min = np.array((0, 2, 2))
    skin_mask_ycrcb_max = np.array((40, 240, 255))
    w, h = img.shape[1], img.shape[0]

    x1, y1, x2, y2 = w, h, 0, 0
    for point in hand_landmarks.landmark:
        x = point.x * w
        y = point.y * h

        x1 = min(x1, max(0, x-55))
        y1 = min(y1, max(0, y-55))
        x2 = max(x2, min(w, x+55))
        y2 = max(y2, min(h, y+55))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # if hands are over each other - incorrect input, break
    if prev[0] != -1 and x2-TOO_CLOSE >= prev[0] and prev[2]-TOO_CLOSE >= x1 and y2-TOO_CLOSE >= prev[1] and prev[3]-TOO_CLOSE >= y1:
        fingers_on_hands.clear()
        return fingers_on_hands, perf_counter()-(wait_time*0.8), prev
    
    # save this hand position to future check if hands are over each other
    hand_frame = cv2.subtract(img[y1:y2, x1:x2], np.ones(img[y1:y2, x1:x2].shape, dtype="uint8") * 50) # decrease brigthness
    prev = [x1, y1, x2, y2]

    # remove bg
    hands_hls = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2HLS_FULL)
    skin_mask = cv2.GaussianBlur(cv2.inRange(hands_hls, skin_mask_ycrcb_min, skin_mask_ycrcb_max), (9, 9), 3)
    dilated_skin_mask = cv2.cvtColor(cv2.dilate(skin_mask, np.ones((2, 2), np.uint8), iterations=3), cv2.COLOR_GRAY2RGB) # to black-white image with 3 color channels
    # masked_hands = cv2.cvtColor(cv2.bitwise_and(hand_frame, hand_frame, mask=dilated_skin_mask), cv2.COLOR_HLS2RGB_FULL) # uncomment if not a black/white dataset

    # save the amount of fingers on each hand
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5) # draw a rectagle for visualisation
    fingers_on_hands.append(get_fingers(finger_model, dilated_skin_mask))

    return fingers_on_hands, perf_counter(), prev # set timer

def camera_logic(chosen: int, is_num2: bool, frame: cv2.typing.MatLike, hand_detector, finger_model, memory: list, start: float = None, add_info: str = None) -> float:
    w, h = 600, 600
    WAIT_TIME = 3

    # some basic settings for frame
    response_image = np.ones((h, w, 3), dtype=np.uint8)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (w, h))
    # wait some time before getting next finger count
    if start is None or perf_counter() - start >= WAIT_TIME:
        add_text("Ready", response_image)
        add_info = None

        # hands need another color scheme
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = hand_detector.process(frame_rgb)

        # find all hands on the picture
        fingers_on_hands = []
        prev = [-1, -1, -1, -1] # x1, y1, x2, y2
        if hands.multi_hand_landmarks and len(hands.multi_hand_landmarks) <= 2:
            for hand_landmarks in hands.multi_hand_landmarks:
                fingers_on_hands, start, prev = get_hand_and_fingers(frame, hand_landmarks, fingers_on_hands, prev, WAIT_TIME, finger_model)

            if chosen == -1:
                if len(fingers_on_hands) == 1 and fingers_on_hands[0] >= 0 and fingers_on_hands[0] <= 3:
                    chosen = fingers_on_hands[0]
                    match chosen:
                        case 0:
                            add_info = "+"
                        case 1:
                            add_info = "-"
                        case 2:
                            add_info = "+"
                        case 3:
                            add_info = "/"
            else:
                if len(fingers_on_hands) == 0:
                    pass
                elif len(memory) > 0 and len(fingers_on_hands) == 2 and fingers_on_hands[0] == 0 and fingers_on_hands[1] == 0:
                    # go the next number
                    is_num2 = not is_num2
                elif len(fingers_on_hands) != 2 or fingers_on_hands[0] != 5 or fingers_on_hands[1] != 5: # can't have 10
                    # add digit to number
                    memory.append(sum(fingers_on_hands))
            
            print(memory)

    else:
        if add_info:
            add_text(add_info, response_image)
        else:
            add_text("Wait!", response_image)


    cv2.imshow("Webcam", frame)
    cv2.imshow("Info", response_image)
    
    return chosen, start, memory, is_num2, add_info

camera()