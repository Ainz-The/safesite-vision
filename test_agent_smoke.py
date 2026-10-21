
import cv2
from pathlib import Path
from agent import inspect

test_images = [
    "workers_test2.png",
    "multi_workers.jpg",
    "image (6).png",
    "enhanced_test.png",
    "hidden_worker_raw_debug.jpg",
]

passed = 0
failed = 0

for filename in test_images:
    path = Path(filename)

    if not path.exists():
        print(f"SKIP: {filename} — image not found")
        continue

    image = cv2.imread(str(path))

    try:
        result = inspect(image)

        assert result["action"] == "human_approval"
        assert "status" in result
        assert "summary" in result
        assert "detections" in result
        assert "potential_violations" in result
        assert "uncertain_detections" in result
        assert "trace" in result
        assert result["annotated_image"] is not None

        print(
            f"PASS: {filename} | "
            f"Status: {result['status']} | "
            f"Detections: {len(result['detections'])} | "
            f"Potential issues: "
            f"{len(result['potential_violations'])} | "
            f"Uncertain: "
            f"{len(result['uncertain_detections'])}"
        )
        passed += 1

    except Exception as error:
        print(f"FAIL: {filename} | {type(error).__name__}: {error}")
        failed += 1

try:
    inspect(None)
    print("FAIL: Empty input was not rejected")
    failed += 1
except ValueError:
    print("PASS: Empty input rejected")
    passed += 1

print("\n========== TEST SUMMARY ==========")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print("==================================")
