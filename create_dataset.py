import os
import pickle
import cv2
import mediapipe as mp

DATA_DIR = "./data"

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

data = []
labels = []
skipped_images = 0


for letter in sorted(os.listdir(DATA_DIR)):
    letter_folder = os.path.join(DATA_DIR, letter)
    if not os.path.isdir(letter_folder):
        continue
    print(f"We're doing letter {letter} rn")
    for image_name in os.listdir(letter_folder):
        image_path = os.path.join(letter_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            skipped_images += 1
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            skipped_images += 1
            continue
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coordinates = []
        y_coordinates = []
        for landmark in hand_landmarks.landmark:
            x_coordinates.append(landmark.x)
            y_coordinates.append(landmark.y)
        data_aux = []
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_coordinates))
            data_aux.append(landmark.y - min(y_coordinates))
        if len(data_aux) != 42:
            print(f"Skipping {image_path}: ")
            print(f"expected 42 values, got {len(data_aux)}")
            skipped_images += 1
            continue
        data.append(data_aux)
        labels.append(letter)


hands.close()

with open("data.pickle", "wb") as file:
    pickle.dump({"data": data, "labels": labels}, file)
print()
print(f"Saved {len(data)} valid examples.")
print(f"Skipped {skipped_images} images.")
print("Every saved example contains 42 values.")