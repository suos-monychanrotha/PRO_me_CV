import pickle

import cv2
import mediapipe as mp
import numpy as np

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf"
khmer_font = ImageFont.truetype(FONT_PATH, 60)

def put_khmer_text(frame, text, pos, font, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)  # color is RGB here
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)  # built-in camera
if not cap.isOpened():
    raise SystemExit('Could not open camera. Check camera permission for Terminal.')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,   # video stream
    max_num_hands=2,
    min_detection_confidence=0.3,
)

FEATURES_PER_HAND = 42
MAX_HANDS = 2

labels_dict = { 0: 'ក',
                1: 'ខ', 
                2: 'គ',
                3: 'ឃ',
                4: 'ង', 
                5: 'ច', 
                6: 'ឆ', 
                7: 'ជ', 
                8: 'ឈ', 
                9: 'ញ', 
                10: 'ដ', 
                11: 'ឋ', 
                12: 'ឌ', 
                13: 'ឍ', 
                14: 'ណ', 
                15: 'ត', 
                16: 'ថ', 
                17: 'ទ', 
                18: 'ធ', 
                19: 'ន', 
                20: 'ប', 
                21: 'ផ', 
                22: 'ព', 
                23: 'ភ', 
                24: 'ម', 
                25: 'យ',
                26: 'រ',
                27: 'ល',
                28: 'វ',
                29: 'ស',
                30: 'ហ',
                31: 'ឡ',
                32: 'អ',
                33: '០'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Same feature building as create_2.py
        hand_list = sorted(
            results.multi_hand_landmarks[:MAX_HANDS],
            key=lambda h: h.landmark[0].x,
        )
        x_ = [lm.x for h in hand_list for lm in h.landmark]
        y_ = [lm.y for h in hand_list for lm in h.landmark]

        data_aux = []
        for h in hand_list:
            for lm in h.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
        data_aux += [0.0] * (FEATURES_PER_HAND * MAX_HANDS - len(data_aux))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        proba = model.predict_proba([np.asarray(data_aux)])[0]
        print(dict(zip(model.classes_, proba.round(2))))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        frame = put_khmer_text(frame, predicted_character, (x1, max(y1 - 70, 0)), khmer_font)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()