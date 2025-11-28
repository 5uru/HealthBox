import cv2
import torch
from transformers import AutoModel
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from healthbox.blood_stain.base import Model


def make_overlay(
        img: np.ndarray, mask: np.ndarray, alpha: float = 0.7
) -> np.ndarray[np.uint8]:
    overlay = alpha * img + (1 - alpha) * mask
    return overlay.astype(np.uint8)



class PneumoniaDetection(Model):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained("ianpan/pneumonia-cxr",
                                          trust_remote_code=True, local_files_only=True)
        self.model = model.eval().to(self.device)

    def predict(self, image) -> dict:
        image = np.array(image)
        if image.ndim == 3:
            rg = image.copy()
        else:
            rg = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Le modèle attend une image 2D (H, W). Si l'image est en couleur, on la convertit.
        if image.ndim == 3:
            # Si format (H, W, 3) standard OpenCV/image
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Si format (3, H, W)
            elif image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x = self.model.preprocess(image)
        x = torch.from_numpy(x).float().to(self.device)
        x = rearrange(x, "h w -> 1 1 h w")

        with torch.inference_mode():
            out = self.model(x.to(self.device))

        ptx_mask = out["mask"]
        h, w = rg.shape[:2]
        ptx_mask = F.interpolate(ptx_mask, size=(h, w), mode="bilinear")
        ptx_mask = (ptx_mask.cpu().numpy()[0, 0] * 255).astype(np.uint8)
        ptx_mask = cv2.applyColorMap(ptx_mask, cv2.COLORMAP_JET)
        ptx_overlay = make_overlay(rg, ptx_mask[..., ::-1])

        ptx_score = round(out["cls"].item(), 2)

        if ptx_score > 0.5:
            info_string = f":red[Pneumonia: {ptx_score}]"
        else:
            info_string = f":blue[Pneumonia: {ptx_score}]"


        return {
                "result": ptx_overlay,
                "detections": info_string
        }

