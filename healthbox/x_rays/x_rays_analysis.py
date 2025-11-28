from PIL import Image

from healthbox.x_rays.registry import DETECTION_REGISTRY, CLASSIFIER_REGISTRY



def run_x_rays_analysis(image: Image.Image, conf_threshold: float = 0.25, enabled_tasks=None):

    results = {
            "detection": {},
            "classification": {}
    }

    # Stage 1: Detection
    for key in DETECTION_REGISTRY:
        model_info = DETECTION_REGISTRY[key]
        model = model_info["model"]
        model.confidence = conf_threshold
        detection = model.predict(image)
        results["detection"][key] = {
                "name": model_info["name"],
                "info": model_info["info"],
                "result": detection["result"],
                "detections": detection["detections"]
        }



    # Stage 2: Classification
    for key in CLASSIFIER_REGISTRY:
        model_info = CLASSIFIER_REGISTRY[key]
        model = model_info["model"]
        model_results = model.predict(image)
        result = {
                "name": model_info["name"],
                "info": model_info["info"],
                "results": model_results
        }
        results["classification"][key] = result

    return results