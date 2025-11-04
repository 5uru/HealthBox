from healthbox.blood_stain.detection.cell_detection import CellDetectionModel
from healthbox.blood_stain.classification.wbc_classifier import WBCClassifier
from healthbox.blood_stain.classification.malaria_classifier import MalariaClassifier

# Model registry: maps task → model instance
MODEL_REGISTRY = {
        "detection": CellDetectionModel(confidence=0.25),
        "wbc_classification": WBCClassifier(),
        "malaria_classification": MalariaClassifier(),
}

CLASSIFIER_REGISTRY = {
        "wbc_classification":{
            "name": "WBC Classification",
            "info": "Classifies White Blood Cells (WBCs) into subtypes such as Neutrophils, Lymphocytes, Monocytes, Eosinophils, and Basophils.",
            "model": WBCClassifier(),
            "input_size": 224,
            "input_class": "WBC",
            "input_padding": 20,

        },
        "malaria_classification": {
            "name": "Malaria Detection in RBCs",
            "info": "Screens Red Blood Cells (RBCs) for malaria infection.",
            "model": MalariaClassifier(),
            "input_size": 224,
            "input_class": "RBC",
            "input_padding": 0,
        }
}

OPTIONAL_TASKS = {
    key: CLASSIFIER_REGISTRY[key]["name"] for key in CLASSIFIER_REGISTRY
}
# Class mapping (shared across modules)
CLASS_NAMES = ["RBC", "WBC", "Platelet"]
CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}