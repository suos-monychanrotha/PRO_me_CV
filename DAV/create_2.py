import os
import pickle

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3,
)

DATA_DIR = './data2'
FEATURES_PER_HAND = 42   # 21 landmarks * (x, y)
MAX_HANDS = 2

data2 = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # skip .DS_Store etc.

    for img_path in os.listdir(dir_path):
        if img_path.startswith('.'):
            continue

        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            continue

        # Sort hands left-to-right (by wrist x) so the order is consistent
        hand_list = sorted(
            results.multi_hand_landmarks[:MAX_HANDS],
            key=lambda h: h.landmark[0].x,
        )

        # Normalize using the min x/y across all detected hands,
        # so the position of one hand relative to the other is kept
        x_ = [lm.x for h in hand_list for lm in h.landmark]
        y_ = [lm.y for h in hand_list for lm in h.landmark]

        data_aux = []
        for h in hand_list:
            for lm in h.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        # Pad with zeros so every sample has exactly 84 features
        data_aux += [0.0] * (FEATURES_PER_HAND * MAX_HANDS - len(data_aux))

        data2.append(data_aux)
        labels.append(dir_)

with open('data2.pickle', 'wb') as f:
    pickle.dump({'data2': data2, 'labels': labels}, f)

print('Saved', len(data2), 'samples')
