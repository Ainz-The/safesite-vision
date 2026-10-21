
import cv2
from detector import PPEDetector
from agent import associate_ppe_with_workers

image = cv2.imread("workers_test2.png")

if image is None:
    raise FileNotFoundError("Could not read multi_workers.jpg")

detector = PPEDetector()
detections = detector.detect(image)

association = associate_ppe_with_workers(detections)

print("Workers found:", len(association["workers"]))

for worker in association["workers"]:
    print(f"\nWorker {worker['worker_id']}:")
    print("Box:", worker["worker_box"])

    for item in worker["ppe"]:
        print(
            f"  {item['label']}: "
            f"{item['conf']:.3f}, box={item['box']}"
        )

print("\nUnassigned detections:", len(association["unassigned"]))

for item in association["unassigned"]:
    print(item["label"], item["box"])


# Create an annotated copy of the image.
annotated = image.copy()


annotated = image.copy()

# Different colours help distinguish individual workers.
colors = [
    (255, 100, 0),
    (0, 200, 255),
    (255, 0, 180),
    (0, 220, 100),
    (180, 100, 255),
    (0, 140, 255),
    (255, 220, 0),
    (100, 255, 100),
    (200, 150, 50),
    (255, 100, 100),
    (150, 220, 255),
]

for worker in association["workers"]:
    worker_id = worker["worker_id"]
    x1, y1, x2, y2 = worker["worker_box"]
    color = colors[(worker_id - 1) % len(colors)]

    # Draw the worker bounding box.
    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    # Stagger label positions to reduce overlapping text.
    label = f"W{worker_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )

    label_y = min(
        max(y1 + text_h + 8, text_h + 8 + (worker_id - 1) * 22),
        max(text_h + 8, y2 - 5),
    )

    label_x = min(x1, image.shape[1] - text_w - 6)
    label_x = max(0, label_x)

    cv2.rectangle(
        annotated,
        (label_x, label_y - text_h - 6),
        (label_x + text_w + 6, label_y + baseline),
        color,
        -1,
    )

    cv2.putText(
        annotated,
        label,
        (label_x + 3, label_y - 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

# Show unassigned detections in red.
for detection in association["unassigned"]:
    x1, y1, x2, y2 = detection["box"]
    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2,
    )

output_path = "multi_worker_association.jpg"

if not cv2.imwrite(output_path, annotated):
    raise RuntimeError("Failed to save visualization")

print("Saved visualization:", output_path)
