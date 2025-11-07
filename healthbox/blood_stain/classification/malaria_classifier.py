import torch
import torch.nn as nn
import timm
from PIL import Image
from typing import Dict, Any, Union
import os
from healthbox.blood_stain.base import Model

class MalariaClassifier(Model):
    """EfficientNet-based malaria cell image classifier with optimized inference pipeline."""

    MODEL_PATH = 'models/malaria_classifier.pth'
    MODEL_NAME = 'tf_efficientnet_b0'
    NUM_CLASSES = 2
    CLASS_NAMES = ['Parasitized', 'Uninfected']

    def __init__(self):
        """Initialize model, preprocessing transforms, and device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        self.transforms = self._get_transforms()
        self._load_weights()

    def _create_model(self) -> nn.Module:
        """
        Create the backbone (EfficientNet) without its default classifier head,
        then attach a custom lightweight classifier with proper initialization.
        """
        # Create backbone without classifier (num_classes=0 disables it in timm)
        model = timm.create_model(
                self.MODEL_NAME,
                pretrained=False,  # We'll load our own fine-tuned weights
                num_classes=0      # Removes the original classifier head
        )
        num_in_features = model.num_features  # Get feature dimension from backbone

        # Custom classifier head: lightweight, batch-normalized, and regularized
        model.classifier = nn.Sequential(
                nn.BatchNorm1d(num_in_features),
                nn.Linear(num_in_features, 512),
                nn.ReLU(inplace=True),       # In-place to reduce memory footprint
                nn.Dropout(0.3),
                nn.Linear(512, self.NUM_CLASSES)
        )

        # Proper weight initialization for new layers
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        return model.to(self.device)

    def _get_transforms(self):
        """Retrieve validation-time preprocessing transforms from TIMM config."""
        data_config = timm.data.resolve_data_config(
                {},
                model=self.model,
                use_test_size=True  # Ensures correct input resolution for inference
        )
        return timm.data.create_transform(**data_config, is_training=False)

    def _load_weights(self):
        """Load trained weights with robust error handling."""
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"Model weights not found at: {self.MODEL_PATH}")

        try:
            state_dict = torch.load(self.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)
        except Exception as e:
            raise RuntimeError(f"Failed to load or apply model weights from {self.MODEL_PATH}: {e}") from e

    @torch.inference_mode()  # Preferred over no_grad() for pure inference (disables autograd + optimizes)
    def predict(self, image: Union[Image.Image, str]) -> Dict[str, Any]:
        """
        Perform inference on a single malaria cell image.

        Args:
            image: Either a PIL Image (RGB) or a valid path to an image file.

        Returns:
            A dictionary containing:
                - 'class': predicted class name ('Parasitized' or 'Uninfected')
                - 'confidence': top class confidence (%) rounded to 2 decimals
                - 'probabilities': per-class probabilities (%) as a dict
        """
        # Load image from path if needed
        if isinstance(image, str):
            if not os.path.isfile(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image or a valid image file path.")

        # Preprocess, add batch dimension, and move to device efficiently
        tensor = self.transforms(image).unsqueeze(0).to(self.device, non_blocking=True)

        # Forward pass
        logits = self.model(tensor)  # Shape: [1, 2]
        probabilities = torch.softmax(logits[0], dim=0)  # Remove batch dim, apply softmax
        top_prob, top_idx = torch.max(probabilities, dim=0)

        # Format results
        return {
                "class": self.CLASS_NAMES[top_idx.item()],
                "confidence": round(top_prob.item() * 100, 2),
                "probabilities": {
                        name: round(prob.item() * 100, 2)
                        for name, prob in zip(self.CLASS_NAMES, probabilities)
                }
        }
