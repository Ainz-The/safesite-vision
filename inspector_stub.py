import cv2

def inspect(image):
    height, width = image.shape[:2]
    annotated = image.copy()
    box = [width // 4, height // 4, 3 * width // 4, 3 * height // 4]
    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
    return {
        "action": "alert",
        "detections": [
            {"label": "person_no_helmet", "conf": 0.91, "box": box, "violation": True}
        ],
        "trace": [
            {"step": 0, "check": "quality", "result": "ok (placeholder)"},
            {"step": 1, "action": "alert"},
        ],
        "annotated_image": annotated,
    }