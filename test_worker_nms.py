
import cv2
import numpy as np
from detector import PPEDetector, letterbox, INPUT_SIZE

IMAGE_PATH = "workers_test2.png"
CONF_THRESHOLD = 0.10
NMS_THRESHOLD = 0.45
WORKER_CLASS_ID = 13

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}")

detector = PPEDetector()
padded, scale, pad_x, pad_y = letterbox(image, INPUT_SIZE)

blob = cv2.dnn.blobFromImage(
    padded,
    scalefactor=1 / 255.0,
    size=(INPUT_SIZE, INPUT_SIZE),
    swapRB=True,
    crop=False,
)

detector.net.setInput(blob)
predictions = detector.net.forward()[0].T

height, width = image.shape[:2]
boxes, scores = [], []

for prediction in predictions:
    confidence = float(prediction[4 + WORKER_CLASS_ID])
    if confidence < CONF_THRESHOLD:
        continue

    cx, cy, w, h = prediction[:4]

    x1 = int(round((cx - w / 2 - pad_x) / scale))
    y1 = int(round((cy - h / 2 - pad_y) / scale))
    x2 = int(round((cx + w / 2 - pad_x) / scale))
    y2 = int(round((cy + h / 2 - pad_y) / scale))

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        continue

    boxes.append([x1, y1, x2 - x1, y2 - y1])
    scores.append(confidence)

kept = cv2.dnn.NMSBoxes(
    boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD
)

kept = np.array(kept).flatten() if len(kept) else []
# Experimental filter: reject extremely narrow worker boxes.
filtered = []

for index in kept:
    x, y, w, h = boxes[int(index)]
    aspect_ratio = w / h

    if aspect_ratio < 0.12:
        print(
            f"Rejected narrow box: "
            f"confidence={scores[int(index)]:.3f}, "
            f"box={[x, y, x + w, y + h]}, "
            f"aspect_ratio={aspect_ratio:.3f}"
        )
        continue

    filtered.append(int(index))

kept = filtered

print("Worker candidates before NMS:", len(boxes))
print("Worker boxes after NMS:", len(kept))

annotated = image.copy()

for number, index in enumerate(kept, start=1):
    x, y, w, h = boxes[int(index)]
    confidence = scores[int(index)]

    print(
        f"W{number}: confidence={confidence:.3f}, "
        f"box={[x, y, x + w, y + h]}, "
        f"width={w}, height={h}, "
        f"aspect_ratio={w / h:.3f}"
    )

    cv2.rectangle(
        annotated, (x, y), (x + w, y + h), (255, 150, 0), 2
    )
    cv2.putText(
        annotated,
        f"W{number} {confidence:.2f}",
        (x, max(20, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 150, 0),
        2,
    )

output_path = "worker_only_nms.jpg"
if not cv2.imwrite(output_path, annotated):
    raise RuntimeError("Could not save output image")

print("Saved:", output_path)
