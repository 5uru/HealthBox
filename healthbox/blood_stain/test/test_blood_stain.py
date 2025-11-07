from healthbox.analyse_model import analyze_model
from healthbox.blood_stain.registry import CLASSIFIER_REGISTRY, DETECTION_REGISTRY

for  model_name, model_class in CLASSIFIER_REGISTRY.items():
    analyze_model(model_class["model"])

for model_name, model_class in DETECTION_REGISTRY.items():
    analyze_model(model_class["model"])
