
import cv2
from agent import inspect

IMAGE_PATH = "image (6).png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")

result = inspect(image)

print("Action:", result["action"])
print("Detections:", len(result["detections"]))
print("Trace:")

for step in result["trace"]:
    print(step)

output_path = "agent_test_result.jpg"

if not cv2.imwrite(output_path, result["annotated_image"]):
    raise RuntimeError("Could not save annotated image.")

print("Saved:", output_path)
