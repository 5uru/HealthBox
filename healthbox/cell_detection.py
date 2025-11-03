from ultralytics import YOLO
import tempfile

Model_path = 'models/cell_detection.onnx'
class_names = {"0": "RBC", "1": "WBC", "2": "Platelet"}
def make_prediction(image, confidence=0.25):
    """Make prediction on the input image using the YOLO model."""
    model = YOLO(Model_path, task="detect")
    results = model(
            source=image,  # path to test images
            conf=confidence,        # confidence threshold
    )

    result = results[0]

    # Extract bounding boxes, confidence scores, and class names
    boxes = result.boxes
    detections = []
    if boxes is not None:
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get confidence score
            conf = float(box.conf[0])
            # Get class ID and name
            cls_id = str(box.cls[0].int().item())
            class_name = class_names[cls_id]

            detections.append({
                    "Class": class_name,
                    "Confidence": f"{conf:.2f}",
                    "Coordinates": f"({x1}, {y1}) - ({x2}, {y2})"
            })

    return result, detections
