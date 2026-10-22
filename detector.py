
import cv2
import numpy as np


MODEL_PATH = (
    "models/"
    "2e4afbc243a7efb5180de7ccd8aec0b27b955da71a663b9d24071afb72bf7e54.onnx"
)

CLASSES = [
    "Boots",
    "Ear-Protection",
    "Glass",
    "Glove",
    "Hard_hat",
    "Mask",
    "No-Boots",
    "No-Ear-Protection",
    "No-Glass",
    "No-Glove",
    "No-Helmet",
    "No-Mask",
    "No-Vest",
    "Worker",
    "Vest",
    "Circular_Saw",
    "Fire_Extinguisher",
    "Fire_prevention_Net",
    "Welding_Equipment",
]

INPUT_SIZE = 640
CONF_THRESHOLD = 0.35
WORKER_CONF_THRESHOLD = 0.10
NMS_THRESHOLD = 0.45

WORKER_CLASS_ID = CLASSES.index("Worker")


def letterbox(image, size=640):
    """Resize an image while preserving aspect ratio, then pad it."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)

    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = cv2.resize(image, (new_width, new_height))

    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2

    left = round(pad_x - 0.1)
    right = size - new_width - left
    top = round(pad_y - 0.1)
    bottom = size - new_height - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded, scale, left, top


class PPEDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def detect(self, image):
        """
        Detect workers and PPE objects.

        Returns detections in original-image coordinates:
        {"label": str, "conf": float, "box": [x1, y1, x2, y2]}
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        height, width = image.shape[:2]

        padded, scale, pad_x, pad_y = letterbox(
            image, INPUT_SIZE
        )

        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1 / 255.0,
            size=(INPUT_SIZE, INPUT_SIZE),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        output = self.net.forward()

        predictions = output[0].T

        # Collect candidates separately by class.
        candidates = {class_id: [] for class_id in range(len(CLASSES))}

        for prediction in predictions:
            cx, cy, box_w, box_h = map(float, prediction[:4])
            class_scores = prediction[4:]

            if len(class_scores) != len(CLASSES):
                raise ValueError(
                    f"Model returned {len(class_scores)} class scores, "
                    f"but {len(CLASSES)} classes are configured."
                )

            # Process Worker independently so a different class winning
            # the argmax does not automatically discard its Worker score.
            worker_conf = float(class_scores[WORKER_CLASS_ID])

            if worker_conf >= WORKER_CONF_THRESHOLD:
                self._add_candidate(
                    candidates,
                    WORKER_CLASS_ID,
                    worker_conf,
                    cx, cy, box_w, box_h,
                    scale, pad_x, pad_y,
                    width, height,
                )

            # Select the strongest non-Worker class for this prediction.
            non_worker_scores = class_scores.copy()
            non_worker_scores[WORKER_CLASS_ID] = -1.0

            class_id = int(np.argmax(non_worker_scores))
            confidence = float(non_worker_scores[class_id])

            if confidence >= CONF_THRESHOLD:
                self._add_candidate(
                    candidates,
                    class_id,
                    confidence,
                    cx, cy, box_w, box_h,
                    scale, pad_x, pad_y,
                    width, height,
                )

        # Apply NMS independently for each class.
        detections = []

        for class_id, class_candidates in candidates.items():
            if not class_candidates:
                continue

            boxes = [item["nms_box"] for item in class_candidates]
            scores = [item["conf"] for item in class_candidates]

            threshold = (
                WORKER_CONF_THRESHOLD
                if class_id == WORKER_CLASS_ID
                else CONF_THRESHOLD
            )

            kept = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                threshold,
                NMS_THRESHOLD,
            )

            if len(kept) == 0:
                continue

            for local_index in np.array(kept).flatten():
                item = class_candidates[int(local_index)]

                detections.append({
                    "label": CLASSES[class_id],
                    "conf": item["conf"],
                    "box": item["box"],
                })

        return detections

    @staticmethod
    def _add_candidate(
        candidates,
        class_id,
        confidence,
        cx,
        cy,
        box_w,
        box_h,
        scale,
        pad_x,
        pad_y,
        image_width,
        image_height,
    ):
        """Convert one model box to image coordinates and store it."""
        x1 = (cx - box_w / 2 - pad_x) / scale
        y1 = (cy - box_h / 2 - pad_y) / scale
        x2 = (cx + box_w / 2 - pad_x) / scale
        y2 = (cy + box_h / 2 - pad_y) / scale

        x1 = max(0, min(int(round(x1)), image_width - 1))
        y1 = max(0, min(int(round(y1)), image_height - 1))
        x2 = max(0, min(int(round(x2)), image_width - 1))
        y2 = max(0, min(int(round(y2)), image_height - 1))

        if x2 <= x1 or y2 <= y1:
            return

        candidates[class_id].append({
            "conf": confidence,
            "box": [x1, y1, x2, y2],
            "nms_box": [x1, y1, x2 - x1, y2 - y1],
        })
