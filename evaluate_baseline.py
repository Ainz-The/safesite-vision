
import cv2
import csv
from pathlib import Path
from collections import Counter, defaultdict
from agent import inspect

# Test images
test_images = [
    "workers_test2.png",
    "multi_workers.jpg",
    "image (6).png",
    "enhanced_test.png",
    "hidden_worker_raw_debug.jpg",
]

output_dir = Path("evaluation_results")
output_dir.mkdir(exist_ok=True)

predictions_path = output_dir / "predictions.csv"
ground_truth_path = output_dir / "ground_truth.csv"

prediction_columns = [
    "image", "label", "confidence",
    "x1", "y1", "x2", "y2",
]

ground_truth_columns = [
    "image", "label", "x1", "y1", "x2", "y2",
]


def normalize_box(box):
    """Convert a detection box into four integer coordinates."""
    return [int(round(float(value))) for value in box]


all_predictions = []

for filename in test_images:
    image = cv2.imread(filename)

    if image is None:
        print(f"\nCould not load: {filename}")
        continue

    result = inspect(image)
    detections = result["detections"]
    counts = Counter(d["label"] for d in detections)

    print(f"\n{'=' * 45}")
    print(f"Image: {filename}")
    print(f"Resolution: {image.shape[1]} x {image.shape[0]}")
    print(f"Total detections: {len(detections)}")
    print(f"Label counts: {dict(counts)}")

    for detection in detections:
        x1, y1, x2, y2 = normalize_box(detection["box"])

        all_predictions.append({
            "image": filename,
            "label": detection["label"],
            "confidence": round(float(detection["conf"]), 5),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    output_path = output_dir / f"{Path(filename).stem}_result.jpg"

    if not cv2.imwrite(
        str(output_path), result["annotated_image"]
    ):
        print(f"Warning: could not save {output_path}")
    else:
        print(f"Saved annotated image: {output_path}")

# Export predictions
with open(predictions_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=prediction_columns)
    writer.writeheader()
    writer.writerows(all_predictions)

print(f"\nExported {len(all_predictions)} predictions to:")
print(predictions_path)

# Create a ground-truth template without overwriting existing annotations.
if not ground_truth_path.exists():
    with open(
        ground_truth_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=ground_truth_columns
        )
        writer.writeheader()

    print(f"\nCreated annotation template: {ground_truth_path}")
    print("Add manually verified ground-truth boxes to this CSV.")
    print("No accuracy metrics are available until it is annotated.")

else:
    # Evaluate predictions against manually verified ground truth.
    with open(
        ground_truth_path, newline="", encoding="utf-8"
    ) as file:
        ground_truth = list(csv.DictReader(file))

    ground_truth = [
        row for row in ground_truth
        if row.get("image", "").strip()
        and row.get("label", "").strip()
    ]

    if not ground_truth:
        print("\nGround-truth CSV is empty; metrics cannot be calculated.")
    else:
        def box_iou(box_a, box_b):
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - intersection

            return intersection / union if union else 0.0

        # Read prediction rows and normalize types.
        with open(
            predictions_path, newline="", encoding="utf-8"
        ) as file:
            predictions = list(csv.DictReader(file))

        def parse_row(row, is_prediction=False):
            return {
                "image": row["image"].strip(),
                "label": row["label"].strip(),
                "box": [
                    float(row["x1"]), float(row["y1"]),
                    float(row["x2"]), float(row["y2"]),
                ],
                "confidence": (
                    float(row["confidence"])
                    if is_prediction else None
                ),
            }

        try:
            predictions = [
                parse_row(row, True) for row in predictions
            ]
            ground_truth = [
                parse_row(row) for row in ground_truth
            ]
        except (KeyError, ValueError) as error:
            raise ValueError(
                "Check CSV column names and coordinate values."
            ) from error

        # Match same-class boxes in the same image at IoU >= 0.50.
        groups = defaultdict(lambda: {"pred": [], "gt": []})

        for item in predictions:
            groups[(item["image"], item["label"])]["pred"].append(item)

        for item in ground_truth:
            groups[(item["image"], item["label"])]["gt"].append(item)

        class_counts = defaultdict(
            lambda: {"TP": 0, "FP": 0, "FN": 0}
        )

        for (_, label), group in groups.items():
            preds = sorted(
                group["pred"],
                key=lambda item: item["confidence"],
                reverse=True,
            )
            truths = group["gt"]
            used_truths = set()

            for pred in preds:
                candidates = [
                    (box_iou(pred["box"], truth["box"]), index)
                    for index, truth in enumerate(truths)
                    if index not in used_truths
                ]

                best_iou, best_index = max(
                    candidates, default=(0.0, None)
                )

                if best_index is not None and best_iou >= 0.50:
                    class_counts[label]["TP"] += 1
                    used_truths.add(best_index)
                else:
                    class_counts[label]["FP"] += 1

            class_counts[label]["FN"] += len(truths) - len(used_truths)

        total_tp = sum(c["TP"] for c in class_counts.values())
        total_fp = sum(c["FP"] for c in class_counts.values())
        total_fn = sum(c["FN"] for c in class_counts.values())

        precision = (
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp else 0.0
        )
        recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        print("\n--- Evaluation metrics (IoU >= 0.50) ---")
        print(f"True positives:  {total_tp}")
        print(f"False positives: {total_fp}")
        print(f"False negatives: {total_fn}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 score:  {f1:.3f}")

        print("\nPer-class results:")
        for label, counts in sorted(class_counts.items()):
            tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            print(
                f"{label}: TP={tp}, FP={fp}, FN={fn}, "
                f"precision={p:.3f}, recall={r:.3f}"
            )

print("\nBaseline evaluation export complete.")
