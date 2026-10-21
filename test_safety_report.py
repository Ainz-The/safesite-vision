
import cv2
from detector import PPEDetector
from agent import associate_ppe_with_workers

IMAGE_PATH = "workers_test2.png"

# PPE categories the current model can assess.
PPE_REQUIREMENTS = {
    "Helmet": ("Hard_hat", "No-Helmet"),
    "Vest": ("Vest", "No-Vest"),
    "Boots": ("Boots", "No-Boots"),
    "Glasses": ("Glass", "No-Glass"),
    "Gloves": ("Glove", "No-Glove"),
    "Mask": ("Mask", "No-Mask"),
}


def assess_worker(ppe_detections):
    """Create a provisional report from the model's PPE predictions."""
    report = {}

    for requirement, (present_label, missing_label) in PPE_REQUIREMENTS.items():
        present = [
            d for d in ppe_detections
            if d["label"] == present_label
        ]
        missing = [
            d for d in ppe_detections
            if d["label"] == missing_label
        ]

        if present and missing:
            status = "REVIEW: conflicting predictions"
        elif missing:
            status = "REVIEW: possible missing " + requirement
        elif present:
            status = "Detected: " + requirement
        else:
            status = "UNKNOWN: no reliable evidence"

        report[requirement] = status

    return report


image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}")

detector = PPEDetector()
detections = detector.detect(image)
association = associate_ppe_with_workers(detections)

print("PROVISIONAL PPE SAFETY REPORT")
print("=" * 50)
print("Workers detected:", len(association["workers"]))

for worker in association["workers"]:
    report = assess_worker(worker["ppe"])

    print(f"\nWorker {worker['worker_id']}")
    print("Box:", worker["worker_box"])

    for requirement, status in report.items():
        print(f"  {requirement}: {status}")

    if not worker["ppe"]:
        print("  Note: No PPE detections assigned to this worker.")

print("\nUnassigned PPE detections:", len(association["unassigned"]))
for detection in association["unassigned"]:
    print(
        f"  {detection['label']} "
        f"(score={detection['conf']:.2f}, box={detection['box']})"
    )

print("\nIMPORTANT: This is a provisional model-based report.")
print("It is not a verified safety inspection.")
