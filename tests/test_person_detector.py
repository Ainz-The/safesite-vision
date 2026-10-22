
from ultralytics import YOLO
import cv2

IMAGE_PATH = "workers_test2.png"
MODEL_PATH = "yolo11n.pt"
OUTPUT_PATH = "person_detection_comparison.jpg"

# Load a pretrained general-purpose object detector.
model = YOLO(MODEL_PATH)

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# COCO class 0 is "person".
results = model.predict(
    source=image,
    classes=[0],
    conf=0.25,
    verbose=False,
)

result = results[0]
annotated = image.copy()

person_count = 0

for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    confidence = float(box.conf[0])

    person_count += 1

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated,
        f"Person {confidence:.2f}",
        (x1, max(20, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

if not cv2.imwrite(OUTPUT_PATH, annotated):
    raise RuntimeError("Could not save the output image.")

print("Image:", IMAGE_PATH)
print("Persons detected:", person_count)
print("Saved visualization:", OUTPUT_PATH)
