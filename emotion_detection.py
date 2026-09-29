"""Real-time facial emotion recognition (OpenCV Haar cascade + Keras CNN trained on FER-2013).

Usage:
    python emotion_detection.py                 # webcam, press q to quit
    python emotion_detection.py --image face.jpg --output annotated.jpg
"""
import argparse
import os
import zipfile

import cv2
import numpy as np
from tensorflow.keras.models import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ZIP = os.path.join(HERE, "model for recognition.zip")
MODEL_PATH = os.path.join(HERE, "model", "model.h5")

# The 7 FER-2013 emotion classes, in the model's output order.
CLASS_NAMES = ["Angry", "Disgusted", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def ensure_model(path=MODEL_PATH):
    """Extract the bundled model from the zip on first run."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with zipfile.ZipFile(MODEL_ZIP) as z:
            z.extract("model.h5", os.path.dirname(path))
    return path


def predict_faces(model, frame):
    """Detect faces and return [(x, y, w, h, label, confidence), ...]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    results = []
    for (x, y, w, h) in faces:
        # The model was trained on raw 0-255 grayscale 48x48 crops (no rescaling).
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48)).astype("float32")[None, :, :, None]
        probs = model.predict(face, verbose=0)[0]
        i = int(np.argmax(probs))
        results.append((x, y, w, h, CLASS_NAMES[i], float(probs[i])))
    return results


def annotate(frame, results):
    for (x, y, w, h, label, conf) in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"{label} {conf:.0%}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


def run_webcam(model, camera=0):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try --camera 1, or use --image.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = annotate(frame, predict_faces(model, frame))
            cv2.imshow("Emotion Detection (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="run on an image file instead of the webcam")
    ap.add_argument("--output", default="annotated.jpg", help="where to save the annotated image")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--model", default=MODEL_PATH, help="path to the Keras .h5 model")
    args = ap.parse_args()

    model = load_model(ensure_model(args.model) if args.model == MODEL_PATH else args.model, compile=False)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        results = predict_faces(model, frame)
        for (_, _, _, _, label, conf) in results:
            print(f"{label}: {conf:.1%}")
        cv2.imwrite(args.output, annotate(frame, results))
        print(f"{len(results)} face(s) found; saved {args.output}")
    else:
        run_webcam(model, args.camera)


if __name__ == "__main__":
    main()
