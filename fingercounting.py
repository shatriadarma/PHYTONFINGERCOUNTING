import cv2
import time
import mediapipe as mp
import numpy as np

# Inisialisasi Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Fungsi untuk menghitung jumlah jari yang terangkat
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 5
    fingers = []

    # Cek jari selain ibu jari
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)  # Jari terangkat
        else:
            fingers.append(0)  # Jari tidak terangkat

    # Cek ibu jari
    if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_tip - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers.count(1)

# Buka kamera
cap = cv2.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Konversi ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Ambil tinggi & lebar frame
    h, w, c = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Hitung jumlah jari
            finger_count = count_fingers(hand_landmarks)

            # Tampilkan gambar sesuai jumlah jari
            try:
                finger_image = cv2.imread(f"{finger_count}.jpg")
                if finger_image is not None:
                    finger_image = cv2.resize(finger_image, (150, 150))
                    frame[10:160, 10:160] = finger_image
            except Exception as e:
                print(f"Error loading image: {e}")

            # Tampilkan jumlah jari
            cv2.putText(frame, f"Fingers: {finger_count}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Tampilkan FPS di kanan atas
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Finger Counting", frame)

    # Keluar jika tekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
