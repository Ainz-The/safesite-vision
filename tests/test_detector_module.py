import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import cv2
from detector import PPEDetector

IMAGE_PATH = "image (6).png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")

detector = PPEDetector()
detections = detector.detect(image)

print("Total detections:", len(detections))

for detection in detections:
    print(
        detection["label"],
        f'{detection["conf"]:.3f}',
        detection["box"],
    )
