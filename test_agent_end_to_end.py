
import cv2
import json
from pathlib import Path

from agent import inspect

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "evaluation_results" / "agent_end_to_end"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMAGES = [
    "workers_test2.png",
    "multi_workers.jpg",
    "image (6).png",
    "enhanced_test.png",
    "hidden_worker_raw_debug.jpg",
]

report = {
    "test_type": "end_to_end_pipeline_smoke_test",
    "warning": (
        "Successful execution does not establish PPE detection accuracy "
        "or confirm that a potential violation is real."
    ),
    "images": [],
}

passed = 0
failed = 0

for filename in TEST_IMAGES:
    image_path = ROOT / filename

    if not image_path.exists():
        print(f"SKIP: {filename} — file not found")
        continue

    image = cv2.imread(str(image_path))

    try:
        if image is None:
            raise ValueError("OpenCV could not read this image.")

        result = inspect(image)
        annotated = result["annotated_image"]

        if annotated is None or annotated.size == 0:
            raise ValueError("Agent returned an empty annotated image.")

        output_image = OUTPUT_DIR / f"{image_path.stem}_annotated.jpg"

        if not cv2.imwrite(str(output_image), annotated):
            raise OSError(f"Could not save {output_image.name}")

        summary = result["summary"]

        record = {
            "image": filename,
            "status": result["status"],
            "action": result["action"],
            "summary": summary,
            "detections": result["detections"],
            "potential_violations": result["potential_violations"],
            "uncertain_detections": result["uncertain_detections"],
            "rechecks": result["rechecks"],
            "enhanced_comparison": result["enhanced_comparison"],
            "trace": result["trace"],
            "annotated_image": str(output_image.relative_to(ROOT)),
        }

        # Convert NumPy scalar types if any appear in the result.
        def json_safe(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            if hasattr(value, "item"):
                return value.item()
            raise TypeError(
                f"Object of type {type(value).__name__} "
                "cannot be converted to JSON"
            )

        json.dumps(record, default=json_safe)

        report["images"].append(record)

        print(
            f"PASS: {filename} | "
            f"status={result['status']} | "
            f"saved={output_image.name}"
        )
        passed += 1

    except Exception as error:
        print(f"FAIL: {filename} | {type(error).__name__}: {error}")
        failed += 1

report["test_summary"] = {
    "passed": passed,
    "failed": failed,
    "images_processed": len(report["images"]),
}

report_path = OUTPUT_DIR / "inspection_report.json"

with report_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2, default=json_safe)

print("\n========== END-TO-END SUMMARY ==========")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Report: {report_path}")
print(f"Annotated images: {OUTPUT_DIR}")
print("========================================")
