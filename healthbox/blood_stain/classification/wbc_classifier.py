import torch
import torch.nn as nn
import timm
from PIL import Image
from typing import Dict, Any, Union
import os
from healthbox.blood_stain.base import Model

class WBCClassifier(Model):
    """
    White Blood Cell (WBC) classifier using EfficientNet backbone.
    Supports inference on 8 WBC subtypes commonly found in peripheral blood smears.
    """

    MODEL_PATH = 'models/wbc_classifier.pth'
    MODEL_NAME = 'tf_efficientnet_b0'
    NUM_CLASSES = 8
    CLASS_NAMES = [
            'basophil', 'eosinophil', 'erythroblast', 'immature granulocytes',
            'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
    ]

    def __init__(self):
        """Initialize model, preprocessing pipeline, and execution device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        self.transforms = self._get_transforms()
        self._load_weights()

    def _create_model(self) -> nn.Module:
        """
        Build the model by loading an EfficientNet backbone (without classifier)
        and attaching a custom, well-initialized classification head.
        """
        # Load backbone without its original classifier (num_classes=0 removes it)
        model = timm.create_model(
                self.MODEL_NAME,
                pretrained=False,   # We'll load fine-tuned weights separately
                num_classes=0       # Strips the default classifier head
        )
        num_in_features = model.num_features

        # Custom lightweight classifier head
        model.classifier = nn.Sequential(
                nn.BatchNorm1d(num_in_features),
                nn.Linear(num_in_features, 512),
                nn.ReLU(inplace=True),    # In-place saves memory during inference
                nn.Dropout(0.3),
                nn.Linear(512, self.NUM_CLASSES)
        )

        # Initialize new layers using best practices
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                # Kaiming init suits ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        return model.to(self.device)

    def _get_transforms(self):
        """
        Use TIMM's built-in config resolver to get the correct preprocessing
        transforms that match the model's training setup.
        """
        data_config = timm.data.resolve_data_config(
                {},
                model=self.model,
                use_test_size=True  # Ensures correct input resolution for inference
        )
        return timm.data.create_transform(**data_config, is_training=False)

    def _load_weights(self):
        """Load trained model weights with robust error handling."""
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(
                    f"Model weights file not found at: {self.MODEL_PATH}. "
                    "Ensure the model is exported to the expected path."
            )
        try:
            state_dict = torch.load(self.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Critical: disable dropout & batch norm updates
        except Exception as e:
            raise RuntimeError(
                    f"Failed to load or apply model weights from '{self.MODEL_PATH}': {e}"
            ) from e

    @torch.inference_mode()  # More efficient than torch.no_grad() for pure inference
    def predict(self, image: Union[Image.Image, str]) -> Dict[str, Any]:
        """
        Run inference on a single WBC image.

        Args:
            image: Either a PIL.Image.Image (RGB) or a valid image file path (str).

        Returns:
            dict with keys:
                - 'class': predicted class name (str)
                - 'confidence': top-1 confidence (%) rounded to 2 decimals (float)
                - 'probabilities': {class_name: confidence%} for all 8 classes (dict)
        """
        # Load image from path if needed
        if isinstance(image, str):
            if not os.path.isfile(image):
                raise FileNotFoundError(f"Image file does not exist: {image}")
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(
                    "Input must be a PIL.Image.Image object or a valid image file path (str)."
            )

        # Preprocess, add batch dimension, and move to device efficiently
        tensor = self.transforms(image).unsqueeze(0).to(self.device, non_blocking=True)

        # Forward pass
        logits = self.model(tensor)  # Shape: [1, 8]
        probabilities = torch.softmax(logits[0], dim=0)  # Remove batch dim, apply softmax
        top_prob, top_idx = torch.max(probabilities, dim=0)

        # Format output as human-readable dictionary
        return {
                "class": self.CLASS_NAMES[top_idx.item()],
                "confidence": round(top_prob.item() * 100, 2),
                "probabilities": {
                        name: round(prob.item() * 100, 2)
                        for name, prob in zip(self.CLASS_NAMES, probabilities)
                }
        }