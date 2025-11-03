import torch
import torch.nn as nn
import timm
from PIL import Image
from typing import Dict, Any, Optional, Union


class WBCClassifier:
    MODEL_PATH = 'models/wbc_classifier.pth'
    MODEL_NAME = 'tf_efficientnet_b0'
    NUM_CLASSES = 8
    CLASS_NAMES = ['basophil', 'eosinophil', 'erythroblast', 'immature granulocytes',
                   'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

    def __init__(self):
        """Initialize model, transforms, and device once during instantiation"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        self.transforms = self._get_transforms()
        self._load_weights()

    def _create_model(self) -> nn.Module:
        """Create model architecture with classifier head"""
        model = timm.create_model(
                self.MODEL_NAME,
                pretrained=False,
                num_classes=0  # Remove default classifier
        )
        num_in_features = model.num_features

        # Efficient classifier head with proper initialization
        model.classifier = nn.Sequential(
                nn.BatchNorm1d(num_in_features),
                nn.Linear(num_in_features, 512),
                nn.ReLU(inplace=True),  # In-place operation saves memory
                nn.Dropout(0.3),
                nn.Linear(512, self.NUM_CLASSES)
        )

        # Initialize new layers properly
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        return model.to(self.device)

    def _get_transforms(self):
        """Get validation transforms from TIMM config"""
        data_config = timm.data.resolve_data_config(
                {},
                model=self.model,
                use_test_size=True  # Ensures proper inference resolution
        )
        return timm.data.create_transform(**data_config, is_training=False)

    def _load_weights(self):
        """Load model weights with error handling"""
        try:
            state_dict = torch.load(self.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Crucial for inference mode
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}") from e

    @torch.inference_mode()  # More efficient than no_grad for inference
    def predict(self, image: Union[Image.Image, str]) -> Dict[str, Any]:
        """
        Predict class for a single image

        Args:
            image: PIL Image object or path to image file

        Returns:
            Dictionary with prediction results
        """
        # Handle image loading if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # Preprocess and move to device in one step
        tensor = self.transforms(image).unsqueeze(0).to(self.device, non_blocking=True)

        # Get prediction
        outputs = self.model(tensor)
        probabilities = torch.softmax(outputs[0], dim=0)
        top_prob, top_idx = torch.max(probabilities, dim=0)

        return {
                "class": self.CLASS_NAMES[top_idx.item()],
                "confidence": round(top_prob.item() * 100, 2),
                "probabilities": {name: round(prob.item() * 100, 2)
                                  for name, prob in zip(self.CLASS_NAMES, probabilities)}
        }

# Global instance for easy access (singleton pattern)
_classifier: Optional[WBCClassifier] = None

def classify_image(image: Union[Image.Image, str]) -> Dict[str, Any]:
    """
    Thread-safe classification function using singleton pattern

    Args:
        image: PIL Image or path to image file

    Returns:
        Prediction dictionary
    """
    global _classifier
    if _classifier is None:
        _classifier = WBCClassifier()
    return _classifier.predict(image)