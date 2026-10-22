
import cv2
from ultralytics import YOLO
from detector import PPEDetector

# Test image and output
IMAGE_PATH = "multi_workers.jpg"
PERSON_MODEL_PATH = "yolo11n.pt"
OUTPUT_PATH = "combined_multi_workers.jpg"

MIN_PPE_OVERLAP = 0.60
AMBIGUITY_MARGIN = 0.08

# Load models and image
person_model = YOLO(PERSON_MODEL_PATH)
ppe_detector = PPEDetector()

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# Detect people. A lower NMS IoU threshold can suppress
# some duplicate overlapping boxes.
person_result = person_model.predict(
    source=image,
    classes=[0],
    conf=0.25,
    iou=0.50,
    verbose=False,
)[0]

people = []

for box in person_result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    people.append({
        "box": [x1, y1, x2, y2],
        "conf": float(box.conf[0]),
        "ppe": [],
    })

# Detect PPE; use the dedicated person model for people.
detections = [
    d for d in ppe_detector.detect(image)
    if d["label"] != "Worker"
]


def body_region(label):
    label = label.lower()

    if any(word in label for word in [
        "hard_hat", "helmet", "glass", "mask", "ear-protection"
    ]):
        return "head"
    if "vest" in label:
        return "torso"
    if "glove" in label:
        return "hands"
    if "boots" in label:
        return "feet"
    return "other"


def overlap_fraction(ppe_box, person_box):
    x1, y1, x2, y2 = map(float, ppe_box)
    px1, py1, px2, py2 = map(float, person_box)

    ix1 = max(x1, px1)
    iy1 = max(y1, py1)
    ix2 = min(x2, px2)
    iy2 = min(y2, py2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (x2 - x1) * (y2 - y1))

    return intersection / area


def score_candidate(detection, person):
    x1, y1, x2, y2 = map(float, detection["box"])
    px1, py1, px2, py2 = map(float, person["box"])

    height = py2 - py1
    width = px2 - px1

    if height <= 0 or width <= 0:
        return {
            "score": 0.0,
            "overlap": 0.0,
            "region_ok": False,
        }

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    relative_y = (cy - py1) / height

    region = body_region(detection["label"])

    limits = {
        "head": (0.00, 0.35, 0.15),
        "torso": (0.15, 0.75, 0.45),
        "hands": (0.20, 0.90, 0.60),
        "feet": (0.65, 1.05, 0.90),
        "other": (0.00, 1.00, 0.50),
    }

    low, high, expected_y = limits[region]
    region_ok = low <= relative_y <= high
    overlap = overlap_fraction(detection["box"], person["box"])

    vertical_fit = 1.0 - min(
        1.0, abs(relative_y - expected_y) / 0.5
    )

    center_inside = (
        px1 <= cx <= px2 and py1 <= cy <= py2
    )
    center_factor = 1.0 if center_inside else 0.25
    region_factor = 1.0 if region_ok else 0.35

    score = (
        0.60 * overlap
        + 0.25 * vertical_fit
        + 0.15 * center_factor
    ) * region_factor

    return {
        "score": score,
        "overlap": overlap,
        "region_ok": region_ok,
    }


# Associate each PPE detection with a person, or request review.
unassigned = []
diagnostics = []

for detection in detections:
    candidates = []

    for index, person in enumerate(people):
        metrics = score_candidate(detection, person)
        candidates.append({
            "person_index": index,
            **metrics,
        })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    label = detection["label"]
    confidence = detection["conf"]

    if not candidates:
        unassigned.append(detection)
        diagnostics.append({
            "label": label,
            "conf": confidence,
            "result": "NO PEOPLE DETECTED",
        })
        continue

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    second_score = second["score"] if second else None
    margin = (
        best["score"] - second_score
        if second else 1.0
    )

    accepted = (
        best["overlap"] >= MIN_PPE_OVERLAP
        and best["region_ok"]
        and margin >= AMBIGUITY_MARGIN
    )

    if accepted:
        people[best["person_index"]]["ppe"].append(detection)
        status = f"Person {best['person_index'] + 1}"
    else:
        unassigned.append(detection)
        status = "REVIEW"

    diagnostics.append({
        "label": label,
        "conf": confidence,
        "result": status,
        "best_person": best["person_index"] + 1,
        "best_score": best["score"],
        "best_overlap": best["overlap"],
        "best_region_ok": best["region_ok"],
        "second_person": (
            second["person_index"] + 1 if second else None
        ),
        "second_score": second_score,
        "margin": margin,
    })


# Count accepted matches before consolidating duplicate labels.
accepted_before_dedup = sum(
    item["result"].startswith("Person ")
    for item in diagnostics
)

# Keep the highest-confidence detection for each label per person.
for person in people:
    best_by_label = {}

    for detection in person["ppe"]:
        label = detection["label"]
        previous = best_by_label.get(label)

        if (
            previous is None
            or detection["conf"] > previous["conf"]
        ):
            best_by_label[label] = detection

    person["ppe"] = list(best_by_label.values())

unique_findings_after_dedup = sum(
    len(person["ppe"]) for person in people
)


# Print association diagnostics.
print("\n--- Association diagnostics ---")

for item in diagnostics:
    if item["result"] == "NO PEOPLE DETECTED":
        print(
            f"{item['label']} conf={item['conf']:.2f}: "
            "no people detected"
        )
        continue

    second_text = (
        f"{item['second_score']:.3f}"
        if item["second_score"] is not None
        else "N/A"
    )

    print(
        f"{item['label']} conf={item['conf']:.2f}"
        f" | result={item['result']}"
        f" | best person={item['best_person']}"
        f" score={item['best_score']:.3f}"
        f" overlap={item['best_overlap']:.2f}"
        f" region_ok={item['best_region_ok']}"
        f" | second person={item['second_person']}"
        f" score={second_text}"
        f" | margin={item['margin']:.3f}"
    )


# Draw people and accepted PPE associations.
annotated = image.copy()

for index, person in enumerate(people, start=1):
    x1, y1, x2, y2 = person["box"]

    cv2.rectangle(
        annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
    )
    cv2.putText(
        annotated,
        f"Person {index} {person['conf']:.2f}",
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    for detection in person["ppe"]:
        bx1, by1, bx2, by2 = map(int, detection["box"])
        label = detection["label"]

        color = (
            (0, 0, 255)
            if label.startswith("No-")
            else (255, 180, 0)
        )

        cv2.rectangle(
            annotated, (bx1, by1), (bx2, by2), color, 2
        )
        cv2.putText(
            annotated,
            f"{label} {detection['conf']:.2f}",
            (bx1, max(18, by1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

if not cv2.imwrite(OUTPUT_PATH, annotated):
    raise RuntimeError(f"Could not save {OUTPUT_PATH}")


# Print summary.
print("\n--- Inspection summary ---")
print(f"Image: {IMAGE_PATH}")
print(f"People detected: {len(people)}")
print(f"PPE detections evaluated: {len(detections)}")
print(
    "Accepted associations before deduplication: "
    f"{accepted_before_dedup}"
)
print(
    "Unique person-label findings after deduplication: "
    f"{unique_findings_after_dedup}"
)
print(f"Unassigned / review detections: {len(unassigned)}")

for index, person in enumerate(people, start=1):
    print(f"\nPerson {index}:")

    if not person["ppe"]:
        print("  No PPE detections accepted")
    else:
        for detection in person["ppe"]:
            print(
                f"  {detection['label']} "
                f"(confidence: {detection['conf']:.2f})"
            )

print(f"\nSaved visualization: {OUTPUT_PATH}")
print("Geometric matches are not verified safety violations.")
