import cv2


def check_quality(image):
    # Convert the image from BGR to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the blur score
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # Calculate average brightness
    brightness = float(gray.mean())

    # Initial thresholds
    blurry = blur_score < 100.0
    too_dark = brightness < 40.0
    too_bright = brightness > 215.0

    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "blurry": blurry,
        "too_dark": too_dark,
        "too_bright": too_bright,
        "acceptable": not (blurry or too_dark or too_bright)
    }

def enhance(image):
    # Convert the image from BGR to LAB color space.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the image into lightness and color channels.
    lightness, channel_a, channel_b = cv2.split(lab)

    # Improve contrast in the lightness channel.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced_lightness = clahe.apply(lightness)

    # Combine the improved lightness with the original colors.
    enhanced_lab = cv2.merge(
        (enhanced_lightness, channel_a, channel_b)
    )

    # Convert back to the original BGR format.
    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced

def zoom_crop(image, box, padding=20, scale=2):
    # Read the image dimensions.
    height, width = image.shape[:2]

    # Get the bounding-box coordinates.
    x1, y1, x2, y2 = box

    # Add padding around the detected object.
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    # Crop the selected region.
    crop = image[y1:y2, x1:x2]

    # Reject an invalid or empty crop.
    if crop.size == 0:
        raise ValueError("The bounding box produced an empty crop")

    # Enlarge the crop.
    enlarged = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    return enlarged
