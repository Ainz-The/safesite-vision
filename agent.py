
import cv2
from collections import Counter

from vision import check_quality, enhance, zoom_crop
from detector import PPEDetector


detector = PPEDetector()

MIN_PPE_CONFIDENCE = 0.50
RECHECK_CONFIDENCE = 0.35
CROP_PADDING = 20
CROP_SCALE = 2
MATCH_IOU_THRESHOLD = 0.30

VIOLATION_LABELS = {
    "No-Helmet",
    "No-Vest",
    "No-Glove",
    "No-Mask",
    "No-Ear-Protection",
    "No-Boots",
    "No-Glass",
}


def count_labels(detections):
    """Count predictions by label."""
    return dict(Counter(d["label"] for d in detections))


def box_iou(a, b):
    """Calculate intersection over union for two bounding boxes."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def draw_detections(image, detections):
    """
    Draw all prediction boxes.
    Only label predictions with confidence >= MIN_PPE_CONFIDENCE
    to reduce text clutter.
    """
    annotated = image.copy()
    image_height, image_width = image.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box"])
        label = detection["label"]
        confidence = float(detection["conf"])

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        color = (
            (0, 0, 255)
            if label in VIOLATION_LABELS
            else (0, 200, 0)
        )

        thickness = 2 if confidence >= MIN_PPE_CONFIDENCE else 1

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        # Keep lower-confidence boxes visible, but omit their text.
        if confidence < MIN_PPE_CONFIDENCE:
            continue

        text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        text_thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            text_thickness,
        )

        text_x = max(
            0,
            min(x1, image_width - text_width - 4),
        )
        text_y = max(text_height + 5, y1 - 5)

        # Solid background improves readability on busy images.
        background_top = max(0, text_y - text_height - 4)
        background_bottom = min(
            image_height - 1,
            text_y + baseline,
        )

        cv2.rectangle(
            annotated,
            (text_x, background_top),
            (
                min(image_width - 1, text_x + text_width + 4),
                background_bottom,
            ),
            color,
            -1,
        )

        cv2.putText(
            annotated,
            text,
            (text_x + 2, text_y - 1),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return annotated


def inspect(image):
    """
    Inspect an image for potential PPE issues.

    Model predictions are candidates for human review.
    This function does not certify workplace safety.
    """
    if image is None or not hasattr(image, "size") or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    trace = []
    rechecks = []

    # 1. Check image quality.
    quality = check_quality(image)

    trace.append({
        "step": 0,
        "check": "quality",
        "result": quality,
    })

    if not quality.get("acceptable", False):
        reason = (
            "Image quality is insufficient for reliable inspection. "
            "Capture a clearer image before reviewing PPE."
        )

        trace.append({
            "step": 1,
            "action": "human_approval",
            "reason": reason,
        })

        return {
            "action": "human_approval",
            "status": "IMAGE_QUALITY_INSUFFICIENT",
            "summary": {
                "message": reason,
                "total_detections": 0,
                "label_counts": {},
                "potential_violation_count": 0,
                "uncertain_detection_count": 0,
                "human_review_required": True,
            },
            "detections": [],
            "potential_violations": [],
            "uncertain_detections": [],
            "rechecks": [],
            "enhanced_comparison": None,
            "trace": trace,
            "annotated_image": image.copy(),
        }

    # 2. Run the detector on the original image.
    original = detector.detect(image)

    trace.append({
        "step": 1,
        "check": "initial_detection",
        "result": {
            "detection_count": len(original),
            "labels": count_labels(original),
        },
    })

    # 3. Identify lower-confidence predictions for diagnostic rechecks.
    uncertain = [
        d for d in original
        if RECHECK_CONFIDENCE <= float(d["conf"])
        < MIN_PPE_CONFIDENCE
    ]

    height, width = image.shape[:2]

    for detection in uncertain:
        x1, y1, x2, y2 = map(int, detection["box"])

        cx1 = max(0, x1 - CROP_PADDING)
        cy1 = max(0, y1 - CROP_PADDING)
        cx2 = min(width, x2 + CROP_PADDING)
        cy2 = min(height, y2 + CROP_PADDING)

        record = {
            "label": detection["label"],
            "original_confidence": float(detection["conf"]),
            "outcome": "RECHECK_ERROR",
        }

        try:
            crop = zoom_crop(
                image,
                detection["box"],
                padding=CROP_PADDING,
                scale=CROP_SCALE,
            )

            if crop is None or crop.size == 0:
                raise ValueError("Crop operation returned an empty image.")

            crop_detections = detector.detect(crop)
            mapped = []

            # Map crop predictions back into original-image coordinates.
            # This assumes zoom_crop enlarges the crop by CROP_SCALE
            # without additional geometric transformations.
            for crop_detection in crop_detections:
                bx1, by1, bx2, by2 = map(
                    float,
                    crop_detection["box"],
                )

                mapped_box = [
                    cx1 + bx1 / CROP_SCALE,
                    cy1 + by1 / CROP_SCALE,
                    cx1 + bx2 / CROP_SCALE,
                    cy1 + by2 / CROP_SCALE,
                ]

                mapped.append({
                    **crop_detection,
                    "box": mapped_box,
                })

            same_label = [
                item for item in mapped
                if item["label"] == detection["label"]
            ]

            if not same_label:
                record.update({
                    "outcome": "NOT_FOUND_IN_CROP",
                    "crop_detection_count": len(crop_detections),
                })
            else:
                best = max(
                    same_label,
                    key=lambda item: box_iou(
                        detection["box"],
                        item["box"],
                    ),
                )

                overlap = box_iou(
                    detection["box"],
                    best["box"],
                )

                record.update({
                    "crop_confidence": float(best["conf"]),
                    "box_iou": round(overlap, 3),
                    "outcome": (
                        "MATCHED"
                        if overlap >= MATCH_IOU_THRESHOLD
                        else "LABEL_FOUND_BOX_MISMATCH"
                    ),
                    "note": (
                        "A crop match is diagnostic only; it does not "
                        "verify that a PPE violation is real."
                    ),
                })

        except (ValueError, cv2.error) as error:
            record.update({
                "outcome": "RECHECK_ERROR",
                "error": str(error),
            })

        rechecks.append(record)

    # 4. Compare original and enhanced-image predictions.
    # The enhanced pass never replaces the original detections.
    enhanced_comparison = None

    if uncertain:
        try:
            enhanced_image = enhance(image)

            if enhanced_image is None or enhanced_image.size == 0:
                raise ValueError(
                    "Enhancement returned an empty image."
                )

            enhanced_detections = detector.detect(enhanced_image)

            original_counts = count_labels(original)
            enhanced_counts = count_labels(enhanced_detections)

            labels = sorted(
                set(original_counts) | set(enhanced_counts)
            )

            enhanced_comparison = {
                "original_detection_count": len(original),
                "enhanced_detection_count": len(enhanced_detections),
                "label_count_comparison": {
                    label: {
                        "original": original_counts.get(label, 0),
                        "enhanced": enhanced_counts.get(label, 0),
                    }
                    for label in labels
                },
                "note": (
                    "Differences show sensitivity to enhancement, "
                    "not which prediction is correct."
                ),
            }

        except (ValueError, cv2.error) as error:
            enhanced_comparison = {
                "error": str(error),
                "note": "Enhanced-image comparison could not be completed.",
            }

    # 5. Separate high-confidence potential violations from uncertain
    # predictions. Neither category is a verified safety finding.
    potential_violations = [
        d for d in original
        if d["label"] in VIOLATION_LABELS
        and float(d["conf"]) >= MIN_PPE_CONFIDENCE
    ]

    uncertain_detections = [
        {**d, "uncertain": True}
        for d in uncertain
    ]

    if potential_violations:
        status = "POTENTIAL_PPE_ISSUES"
        message = (
            "The model flagged possible missing-PPE issues. "
            "A reviewer must verify the findings and affected workers."
        )
    elif uncertain:
        status = "UNCERTAIN_FINDINGS"
        message = (
            "Some detections have lower confidence. "
            "Review the image and findings manually."
        )
    else:
        status = "NO_HIGH_CONFIDENCE_VIOLATION_LABELS_DETECTED"
        message = (
            "No high-confidence missing-PPE labels were detected. "
            "This does not establish that workers or the site are safe."
        )

    trace.append({
        "step": 2,
        "check": "targeted_crop_comparison",
        "result": {
            "uncertain_count": len(uncertain),
            "rechecks": rechecks,
            "note": "Crop comparisons are diagnostic only.",
        },
    })

    trace.append({
        "step": 3,
        "check": "enhanced_full_image_comparison",
        "result": enhanced_comparison,
    })

    trace.append({
        "step": 4,
        "action": "human_approval",
        "status": status,
        "reason": message,
        "potential_violation_count": len(potential_violations),
    })

    # Preserve all original predictions in the returned results.
    annotated_detections = [
        {
            **d,
            "potential_violation": d["label"] in VIOLATION_LABELS,
        }
        for d in original
    ]

    return {
        "action": "human_approval",
        "status": status,
        "summary": {
            "message": message,
            "total_detections": len(original),
            "label_counts": count_labels(original),
            "potential_violation_count": len(potential_violations),
            "uncertain_detection_count": len(uncertain),
            "human_review_required": True,
        },
        "detections": annotated_detections,
        "potential_violations": potential_violations,
        "uncertain_detections": uncertain_detections,
        "rechecks": rechecks,
        "enhanced_comparison": enhanced_comparison,
        "trace": trace,
        "annotated_image": draw_detections(image, original),
    }
