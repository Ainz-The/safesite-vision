
import cv2
from vision import check_quality

image = cv2.imread("image (6).png")

if image is None:
    raise FileNotFoundError("Could not load image (6).png")

result = check_quality(image)

print("Blur score:", result["blur_score"])
print("Brightness:", result["brightness"])
print("Blurry:", result["blurry"])
print("Too dark:", result["too_dark"])
print("Too bright:", result["too_bright"])
print("Acceptable:", result["acceptable"])
