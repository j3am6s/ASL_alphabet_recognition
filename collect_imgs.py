import os
import cv2


DATA_DIR = "./data"
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DATASET_SIZE = 200
cap = cv2.VideoCapture(0)

if not cap.isOpened(): raise RuntimeError("The camera not opened.")

os.makedirs(DATA_DIR, exist_ok=True)


for letter in LETTERS:
    letter_folder = os.path.join(DATA_DIR, letter)
    os.makedirs(letter_folder, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("The camera not return image.")
        cv2.putText(frame, f"Letter {letter} - press Q", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Collecting", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        cv2.putText(frame, f"{letter}: {counter + 1}/{DATASET_SIZE}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Collecting", frame)
        image_path = os.path.join(letter_folder, f"{counter}.jpg")
        cv2.imwrite(image_path, frame)
        cv2.waitKey(25)
        counter +=1

cap.release()
cv2.destroyAllWindows()