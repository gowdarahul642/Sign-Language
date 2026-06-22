'''import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('sign_language_model (2).h5')
import json

with open("class_names.json", "r") as f:
    class_names = json.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64,64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    label = class_names[class_id]

    cv2.putText(frame, label, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

count = len(oss.listdir("C:\\Users\\Rahul Gowda\\OneDrive\\Desktop\\Sign_lan1\\Sign-Language-To-Text-and-Speech-Conversion-master\\AtoZ_3.1"))
c_dir = 'A'

offset = 15
step = 1
flag = False
suv = 0

white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("C:\\Users\\Rahul Gowda\\OneDrive\\Desktop\\Sign_lan1\\Sign-Language-To-Text-and-Speech-Conversion-master\\white.jpg", white)

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands_data = hd.findHands(frame, draw=False, flipType=True)
        hands = hands_data[0] if isinstance(hands_data, tuple) else hands_data
        white = cv2.imread("C:\\Users\\Rahul Gowda\\OneDrive\\Desktop\\Sign_lan1\\Sign-Language-To-Text-and-Speech-Conversion-master\\white.jpg")

        if hands and isinstance(hands, list) and len(hands) > 0:
            hand = hands[0]  # First detected hand
            if isinstance(hand, dict) and 'bbox' in hand:
                x, y, w, h = hand['bbox']
                print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")
            else:
                print("Unexpected hand structure:", hand)
                continue

            image = np.array(frame[y - offset:y + h + offset, x - offset:x + w + offset])
            handz_data = hd2.findHands(image, draw=True, flipType=True)
            handz = handz_data[0] if isinstance(handz_data, tuple) else handz_data

            if handz and isinstance(handz, list) and len(handz) > 0:
                hand = handz[0]
                pts = hand['lmList']
                os_x = ((400 - w) // 2) - 15
                os_y = ((400 - h) // 2) - 15
                
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t+1][0] + os_x, pts[t+1][1] + os_y), (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t+1][0] + os_x, pts[t+1][1] + os_y), (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t+1][0] + os_x, pts[t+1][1] + os_y), (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t+1][0] + os_x, pts[t+1][1] + os_y), (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t+1][0] + os_x, pts[t+1][1] + os_y), (0, 255, 0), 3)
                
                cv2.imshow("Hand Skeleton", white)

        frame = cv2.putText(frame, "dir=" + str(c_dir) + "  count=" + str(count), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            break

    except Exception as e:
        print("Error:", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import tensorflow as tf

import tensorflow as tf

model = tf.keras.models.load_model(
    'sign_language_model (2).h5',
    compile=False
)

print("Model loaded successfully")

class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64,64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    label = class_names[class_id]

    cv2.putText(frame, label, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import os
import traceback

# ================== LOAD MODEL ==================
model = tf.keras.models.load_model("sign_language_model (2).h5",   # <-- SavedModel folder (NOT .h5)
    compile=False
)
print("Model loaded successfully")

# ================== CLASS NAMES ==================
class_names = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

# ================== IMAGE PREPROCESS ==================
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ================== CAMERA & HAND DETECTOR ==================
cap = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 15

# White background
white = np.ones((400, 400, 3), np.uint8) * 255

label = "None"
confidence = 0.0

# ================== MAIN LOOP ==================
while True:
    try:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        hands, _ = hd.findHands(frame, draw=False)

        white[:] = 255

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            img_crop = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            hands2, _ = hd2.findHands(img_crop, draw=False)

            if hands2:
                pts = hands2[0]['lmList']

                os_x = (400 - w) // 2
                os_y = (400 - h) // 2

                fingers = [
                    (0,1,2,3,4),
                    (5,6,7,8),
                    (9,10,11,12),
                    (13,14,15,16),
                    (17,18,19,20)
                ]

                for finger in fingers:
                    for i in range(len(finger)-1):
                        cv2.line(
                            white,
                            (pts[finger[i]][0]+os_x, pts[finger[i]][1]+os_y),
                            (pts[finger[i+1]][0]+os_x, pts[finger[i+1]][1]+os_y),
                            (0,255,0),
                            3
                        )

                # ===== MODEL PREDICTION =====
                white_rgb = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
                input_img = preprocess_image(white_rgb)

                pred = model.predict(input_img, verbose=0)
                class_id = np.argmax(pred)
                confidence = float(np.max(pred))

                if class_id < len(class_names):
                    label = class_names[class_id]
                else:
                    label = "Unknown"

                cv2.imshow("Hand Skeleton", white)

        # ===== DISPLAY RESULT =====
        cv2.putText(
            frame,
            f"Prediction: {label} ({confidence:.2f})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        print(traceback.format_exc())

cap.release()
cv2.destroyAllWindows()







 
