# apps/ai_service/services.py
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_PATH = "my_model.pt"

try:
    logger.info(f"Loading YOLO model from {MODEL_PATH}")
    yolo_model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    yolo_model = None


def detect_from_image(frame):
    """
    Run YOLO on a given image (numpy array) and return detections + annotated image
    """
    if yolo_model is None:
        return {"error": "YOLO model is not loaded"}

    try:
        results = yolo_model(frame)
        logger.debug(f"YOLO Raw Results: {results}")
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                xyxy = [float(x) for x in box.xyxy[0].tolist()]

                detections.append({
                    "class": str(yolo_model.names[cls_id]),
                    "confidence": conf,
                    "bbox": xyxy
                })

                # Draw boxes
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{yolo_model.names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode frame to base64
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        return {"detections": detections, "image": jpg_as_text}

    except Exception as e:
        import traceback
        logger.error(f"Error in detect_from_image: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}
