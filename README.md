import cv2
import mediapipe as mp

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Kamera
cap = cv2.VideoCapture(0)

# Gunakan model MediaPipe
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    prev_x = None
    wave_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip agar seperti mirror
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # Ambil koordinat pergelangan tangan (wrist index = 0)
                h, w, _ = frame.shape
                cx = int(handLms.landmark[0].x * w)

                # Cek pergerakan kanan-kiri
                if prev_x is not None:
                    if abs(cx - prev_x) > 40:  # jika geser jauh
                        wave_count += 1

                prev_x = cx

                # Jika melambai 3x → anggap salam perkenalan
                if wave_count >= 3:
                    cv2.putText(frame, "Halo, nama saya ChatGPT", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    wave_count = 0  # reset supaya bisa ulang

        # Tampilkan hasil
        cv2.imshow("Hand Gesture Perkenalan", frame)

        # Tekan ESC untuk keluar
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()