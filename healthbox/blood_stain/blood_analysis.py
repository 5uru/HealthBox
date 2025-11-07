from PIL import Image

from healthbox.blood_stain.registry import MODEL_REGISTRY, CLASSIFIER_REGISTRY
from utils import crop_cells


def run_blood_analysis(image: Image.Image, conf_threshold: float = 0.25, enabled_tasks=None):
    # Update detection model threshold
    detector = MODEL_REGISTRY["detection"]
    detector.confidence = conf_threshold

    # Stage 1: Detection
    detection = detector.predict(image)

    results = {
            "detection": {"result": detection["result"], "detections": detection["detections"]},
            "classification": {}
    }

    # Stage 2: Classification
    classifier_registry_keys = list(CLASSIFIER_REGISTRY.keys())
    for key in classifier_registry_keys:
        model_info = CLASSIFIER_REGISTRY[key]
        if  not enabled_tasks.get(key, True):
            continue
        if [d for d in detection["detections"] if d["Class"] == model_info["input_class"]]:
            cropped_cells = crop_cells(
                    image,
                    detection["result"],
                    model_info["input_class"],
                    padding=model_info["input_padding"],
                    output_size=model_info["input_size"]
            )

            model = model_info["model"]
            model_results = [model.predict(crop) for crop in cropped_cells]
            print(model_results)
            result = {
                    "name": model_info["name"],
                    "info": model_info["info"],
                    "results": model_results
            }
            results["classification"][key] = result



    return results