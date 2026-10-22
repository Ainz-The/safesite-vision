
import cv2
import numpy as np

MODEL_PATH = (
    "models/"
    "2e4afbc243a7efb5180de7ccd8aec0b27b955da71a663b9d24071afb72bf7e54.onnx"
)
IMAGE_PATH = "image (6).png"

CLASSES = [
    "Boots", "Ear-Protection", "Glass", "Glove", "Hard_hat",
    "Mask", "No-Boots", "No-Ear-Protection", "No-Glass",
    "No-Glove", "No-Helmet", "No-Mask", "No-Vest", "Worker",
    "Vest", "Circular_Saw", "Fire_Extinguisher",
    "Fire_prevention_Net", "Welding_Equipment",
]

INPUT_SIZE = 640
CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45


def letterbox(image, size=640):
    """Resize without stretching, then pad to a square."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)

    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = cv2.resize(image, (new_width, new_height))

    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2

    left = round(pad_x - 0.1)
    right = size - new_width - left
    top = round(pad_y - 0.1)
    bottom = size - new_height - top

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    return padded, scale, left, top


image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

height, width = image.shape[:2]

padded, scale, pad_x, pad_y = letterbox(image, INPUT_SIZE)

net = cv2.dnn.readNetFromONNX(MODEL_PATH)

blob = cv2.dnn.blobFromImage(
    padded,
    scalefactor=1.0 / 255.0,
    size=(INPUT_SIZE, INPUT_SIZE),
    swapRB=True,
    crop=False,
)

net.setInput(blob)
output = net.forward()

# Model output: (1, 23, 8400)
predictions = output[0].T

boxes = []
scores = []
class_ids = []

for prediction in predictions:
    cx, cy, box_w, box_h = prediction[:4]
    class_scores = prediction[4:]

    class_id = int(np.argmax(class_scores))
    confidence = float(class_scores[class_id])

    if confidence < CONF_THRESHOLD:
        continue

    # Convert from model coordinates back to original-image coordinates.
    x1 = (cx - box_w / 2 - pad_x) / scale
    y1 = (cy - box_h / 2 - pad_y) / scale
    x2 = (cx + box_w / 2 - pad_x) / scale
    y2 = (cy + box_h / 2 - pad_y) / scale

    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))

    if x2 <= x1 or y2 <= y1:
        continue

    boxes.append([x1, y1, x2 - x1, y2 - y1])
    scores.append(confidence)
    class_ids.append(class_id)

# Apply NMS independently to each class.
kept_indices = []

for class_id in sorted(set(class_ids)):
    class_indices = [
        i for i, detected_class in enumerate(class_ids)
        if detected_class == class_id
    ]

    class_boxes = [boxes[i] for i in class_indices]
    class_scores = [scores[i] for i in class_indices]

    kept = cv2.dnn.NMSBoxes(
        class_boxes,
        class_scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD,
    )

    if len(kept) > 0:
        for local_index in np.array(kept).flatten():
            kept_indices.append(class_indices[int(local_index)])

annotated = image.copy()

# Different colors help distinguish different detected classes.
CLASS_COLORS = {
    "Worker": (255, 150, 0),
    "Vest": (0, 200, 0),
    "Hard_hat": (0, 220, 255),
    "No-Glove": (255, 0, 200),
    "No-Mask": (0, 0, 255),
    "No-Ear-Protection": (255, 100, 0),
    "Glass": (200, 100, 255),
}

print("Detections after class-aware NMS:", len(kept_indices))

# Draw boxes without placing text at every box corner.
for index in kept_indices:
    x, y, w, h = boxes[index]
    label = CLASSES[class_ids[index]]
    color = CLASS_COLORS.get(label, (0, 200, 0))

    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        color,
        2,
    )

# Put all labels in a separate panel to avoid overlapping text.
panel_width = 235
panel_height = min(
    height,
    12 + 25 * len(kept_indices)
)

panel = annotated.copy()
cv2.rectangle(
    panel,
    (0, 0),
    (panel_width, panel_height),
    (25, 25, 25),
    -1,
)
annotated = cv2.addWeighted(panel, 0.82, annotated, 0.18, 0)

for row, index in enumerate(kept_indices):
    label = CLASSES[class_ids[index]]
    confidence = scores[index]
    color = CLASS_COLORS.get(label, (0, 200, 0))

    text = f"{label}: {confidence:.2f}"
    y = 22 + row * 25

    cv2.putText(
        annotated,
        text,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )

output_path = "detector_test_result_v3.jpg"

if not cv2.imwrite(output_path, annotated):
    raise RuntimeError("Could not save annotated image.")

print("Saved annotated image to:", output_path)
