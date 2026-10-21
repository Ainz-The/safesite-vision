
import cv2
from detector import PPEDetector

IMAGE_PATH = "workers_test2.png"
OUTPUT_PATH = "ppe_visualization.jpg"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}")

detector = PPEDetector()
detections = detector.detect(image)

annotated = image.copy()

# Draw every PPE prediction, including Worker boxes.
for detection in detections:
    label = detection["label"]
    confidence = detection["conf"]
    x1, y1, x2, y2 = detection["box"]

    if label == "Worker":
        color = (255, 150, 0)       # Blue: worker
    elif label.startswith("No-"):
        color = (0, 0, 255)         # Red: possible missing PPE
    else:
        color = (0, 180, 0)         # Green: PPE detected

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    text = f"{label} {confidence:.2f}"
    text_y = max(20, y1 - 6)

    # Solid background makes labels easier to read.
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        2,
    )

    cv2.rectangle(
        annotated,
        (x1, max(0, text_y - text_height - baseline)),
        (min(image.shape[1] - 1, x1 + text_width), text_y + baseline),
        color,
        -1,
    )

    cv2.putText(
        annotated,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

if not cv2.imwrite(OUTPUT_PATH, annotated):
    raise RuntimeError(f"Could not save {OUTPUT_PATH}")

print("Image size:", image.shape[1], "x", image.shape[0])
print("Total detections:", len(detections))

print("\nDetection counts:")
counts = {}
for detection in detections:
    label = detection["label"]
    counts[label] = counts.get(label, 0) + 1

for label, count in sorted(counts.items()):
    print(f"{label}: {count}")

print("\nSaved visualization:", OUTPUT_PATH)
