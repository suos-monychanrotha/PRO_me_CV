import pickle
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

model = pickle.load(open('./model.p', 'rb'))['model']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit('Could not open camera.')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.3)

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
                33: '\u17D2',
                34: 'ៅ'}   # add new letters here
                

# --- spelling settings ---
HOLD_FRAMES = 15    # frames the same letter must be held (~0.5 s)
MIN_CONF = 0.6      # minimum model confidence to count a frame

FONT_PATH = "/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf"
big_font = ImageFont.truetype(FONT_PATH, 60)
word_font = ImageFont.truetype(FONT_PATH, 70)


def put_khmer_text(frame, text, pos, font, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img_pil).text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


recent = deque(maxlen=HOLD_FRAMES)  # last predictions
word = ""
last_added = None   # letter just added
locked = False      # blocks re-adding until hand leaves or sign changes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    progress = 0.0

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Same feature building as create_2.py
        hand_list = sorted(results.multi_hand_landmarks[:MAX_HANDS],
                           key=lambda h: h.landmark[0].x)
        x_ = [lm.x for h in hand_list for lm in h.landmark]
        y_ = [lm.y for h in hand_list for lm in h.landmark]
        data_aux = []
        for h in hand_list:
            for lm in h.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
        data_aux += [0.0] * (FEATURES_PER_HAND * MAX_HANDS - len(data_aux))

        proba = model.predict_proba([np.asarray(data_aux)])[0]
        best = int(np.argmax(proba))
        conf = proba[best]
        char = labels_dict[int(model.classes_[best])] if conf >= MIN_CONF else None
        recent.append(char)

        # unlock if the sign changed
        if locked and char is not None and char != last_added:
            locked = False

        # count how many recent frames agree with the newest prediction
        if char is not None:
            agree = sum(1 for c in recent if c == char)
            progress = agree / HOLD_FRAMES
            if agree >= HOLD_FRAMES and not locked:
                word += char
                last_added = char
                locked = True
                recent.clear()

        # box + current guess
        x1 = max(int(min(x_) * W) - 10, 0)
        y1 = max(int(min(y_) * H) - 10, 0)
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        if char:
            frame = put_khmer_text(frame, char, (x1, max(y1 - 75, 0)), big_font)
    else:
        # hand left the frame: reset, so the same letter can be added again
        recent.clear()
        locked = False
        last_added = None

    # hold progress bar
    cv2.rectangle(frame, (10, 10), (210, 30), (255, 255, 255), 2)
    cv2.rectangle(frame, (10, 10), (10 + int(200 * progress), 30), (0, 200, 0), -1)

    # word at the bottom
    cv2.rectangle(frame, (0, H - 115), (W, H), (0, 0, 0), -1)
    frame = put_khmer_text(frame, word, (20, H - 135), word_font, (255, 255, 255))

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):      # backspace
        word = word[:-1]
    elif key == ord('c'):      # clear all
        word = ""
    elif key == ord('1'):      # add coeng
        word += '\u17D2'

cap.release()
cv2.destroyAllWindows()