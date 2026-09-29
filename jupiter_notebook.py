"""Webcam emotion recognition for Jupyter notebooks (frames shown inline).

Run the cells of this file in a notebook; interrupt the kernel to stop.
For a regular window, use: python emotion_detection.py
"""
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.models import load_model

from emotion_detection import annotate, ensure_model, predict_faces

model = load_model(ensure_model(), compile=False)  # extracts the bundled model on first run

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame")
                break
            frame = annotate(frame, predict_faces(model, frame))
            clear_output(wait=True)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Emotion Detection")
            plt.axis("off")
            plt.show()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        cap.release()
