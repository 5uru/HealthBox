from ultralytics import YOLO
from healthbox.blood_stain.base import Model

Model_path = 'models/bone_fracture.onnx'
class_names = {"0": "fractured", "1": "nonfractured"}

class BoneFractureDetectionModel(Model):
    def __init__(self, model_path: str = Model_path, confidence: float = 0.25):
        self.model_path = model_path
        self.confidence = confidence
        self.model = YOLO(self.model_path, task="detect")

    def predict(self, image) -> dict:
        """Make prediction on the input image using the YOLO model."""
        results = self.model(
                source=image,  # path to test images
                conf=self.confidence,        # confidence threshold
                rect=True,                # use rectangular inference
                augment=True,              # augmented inference
                verbose=False,             # disable verbose output
                show_conf=True               # disable showing confidence scores on output image
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

                annotated_bgr = result.plot()
                annotated_rgb = annotated_bgr[..., ::-1]

                if "fractured" in detections:
                    fractured_status = ":red[Bone Fractured]"
                else:
                    fractured_status = ":blue[No Fracture]"

        return {
                "result": annotated_rgb,
                "detections": fractured_status
        }