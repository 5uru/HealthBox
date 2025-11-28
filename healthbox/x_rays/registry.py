from healthbox.x_rays.bone.bone_fracture import BoneFractureDetectionModel
from healthbox.x_rays.chest.x_ray_vision import XRayVision
from healthbox.x_rays.chest.pneumonia import PneumoniaDetection
from healthbox.x_rays.chest.pneumothorax import PneumothoraxDetection
from healthbox.x_rays.chest.segmentation import ChestSegmentation



# Model registry: maps task → model instance


DETECTION_REGISTRY = {
        "ChestSegmentation": {
                "name": "Chest Segmentation",
                "info": "Chest Segmentation using a pretrained model.",
                "model": ChestSegmentation(),
        },
        "PneumoniaDetection": {
                "name": "Pneumonia Detection",
                "info": "Pneumonia Detection using a pretrained model.",
                "model": PneumoniaDetection(),
        },
        "PneumothoraxDetection": {
                "name": "Pneumothorax Detection",
                "info": "Pneumothorax Detection using a pretrained model.",
                "model": PneumothoraxDetection(),
        },
        "fracture_detection": {
                "name": "Bone Fracture Detection",
                "info": "Detects bone fractures in X-ray images.",
                "model": BoneFractureDetectionModel(confidence=0.25),
        },
}

CLASSIFIER_REGISTRY = {
        "x_ray_vision": {
                "name": "XRayVision Chest Diagnostics",
                "info": "14 Chest diagnostics",
                "model": XRayVision(),
        }

}

OPTIONAL_TASKS = {
        key: CLASSIFIER_REGISTRY[key]["name"] for key in CLASSIFIER_REGISTRY
}
