
import cv2
from vision import check_quality, enhance

image = cv2.imread("image (6).png")

if image is None:
    raise FileNotFoundError("Could not load image (6).png")

enhanced = enhance(image)

print("Original quality:")
print(check_quality(image))

print("\nEnhanced quality:")
print(check_quality(enhanced))

cv2.imwrite("enhanced_test.png", enhanced)
print("\nSaved enhanced image as enhanced_test.png")
