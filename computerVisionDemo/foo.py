import cv2
import numpy as np

capture = cv2.VideoCapture('venv/Resources/video.mp4')
success, img = capture.read()

if not capture.isOpened():
    print("HATA:Video akışı açılamadı.")

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        cv2.imshow("Ekran", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
