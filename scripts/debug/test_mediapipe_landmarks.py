import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
for _candidate in (_THIS_FILE.parent, *_THIS_FILE.parents):
    if (_candidate / "data_utils").exists():
        _PROJECT_ROOT = _candidate
        break
else:
    _PROJECT_ROOT = _THIS_FILE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import mediapipe as mp
import os

test_dir = os.path.join(os.path.dirname(__file__), '..', 'test')
image_path = os.path.join(test_dir, '1.png')

# New mediapipe tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download model if needed
model_path = str(_PROJECT_ROOT / 'assets' / 'pretrained' / 'face_landmarker.task')
if not os.path.exists(model_path):
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print(f"Downloading face_landmarker model...")
    urllib.request.urlretrieve(url, model_path)
    print("Done.")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1)

with FaceLandmarker.create_from_options(options) as landmarker:
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    print(f"Image: {image_path} ({w}x{h})")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print("No face detected!")
    else:
        landmarks = result.face_landmarks[0]
        print(f"Detected {len(landmarks)} landmarks")

        # Draw landmarks on image
        annotated = image.copy()
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (x, y), 1, (0, 255, 0), -1)

        # Draw connections for face oval
        FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        for i in range(len(FACE_OVAL) - 1):
            p1 = landmarks[FACE_OVAL[i]]
            p2 = landmarks[FACE_OVAL[i + 1]]
            cv2.line(annotated, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 200, 0), 1)

        out_path = os.path.join(test_dir, '1_mediapipe_landmarks.png')
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")

        # Print key landmark positions
        key_indices = {1: "nose_tip", 33: "left_eye_inner", 263: "right_eye_inner", 13: "upper_lip", 14: "lower_lip"}
        for idx, name in key_indices.items():
            lm = landmarks[idx]
            print(f"  {name:20s}: x={lm.x:.4f} y={lm.y:.4f} z={lm.z:.4f} -> pixel ({int(lm.x*w)}, {int(lm.y*h)})")
