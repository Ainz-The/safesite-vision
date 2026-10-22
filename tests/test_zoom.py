import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import cv2
from vision import zoom_crop

image = cv2.imread("image (6).png")

if image is None:
    raise FileNotFoundError("Could not load image (6).png")

height, width = image.shape[:2]

# Use the center of the image as a temporary test region.
box = [
    width // 4,
    height // 4,
    3 * width // 4,
    3 * height // 4
]

zoomed = zoom_crop(image, box)

cv2.imwrite("zoomed_test.png", zoomed)

print("Original dimensions:", image.shape)
print("Zoomed dimensions:", zoomed.shape)
print("Saved zoomed image as zoomed_test.png")
