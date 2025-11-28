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

def calculate_ctr(mask: np.ndarray) -> float:
    # mask.ndim = 2, (height, width)
    lungs = np.zeros_like(mask)
    lungs[mask == 1] = 1
    lungs[mask == 2] = 1
    heart = (mask == 3).astype("int")
    y, x = np.stack(np.where(lungs == 1))
    lung_min = x.min()
    lung_max = x.max()
    y, x = np.stack(np.where(heart == 1))
    heart_min = x.min()
    heart_max = x.max()
    lung_range = lung_max - lung_min
    heart_range = heart_max - heart_min
    return heart_range / lung_range


class ChestSegmentation(Model):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained("ianpan/chest-x-ray-basic",
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
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        x = x.float()

        with torch.inference_mode():
            out = self.model(x.to(self.device))

        info_mask = out["mask"]
        h, w = rg.shape[:2]
        info_mask = F.interpolate(info_mask, size=(h, w), mode="bilinear")
        info_mask = info_mask.argmax(1)[0]
        info_mask_3ch = F.one_hot(info_mask, num_classes=4)[..., 1:]
        info_mask_3ch = (info_mask_3ch.cpu().numpy() * 255).astype(np.uint8)
        info_overlay = make_overlay(rg, info_mask_3ch[..., ::-1])

        view = out["view"].argmax(1).item()
        info_string = ""
        if view in {0, 1}:
            info_string += "This is a frontal chest radiograph "
            if view == 0:
                info_string += "**(AP projection)**."
            elif view == 1:
                info_string += "**(PA projection)**."
        elif view == 2:
            info_string += "This is a lateral chest radiograph."

        age = out["age"].item()
        info_string += f"\n\nThe patient's predicted age is **{round(age)}** years."
        sex = out["female"].item()
        if sex < 0.5:
            sex = "male"
        else:
            sex = "female"
        info_string += f"\n\nThe patient's predicted sex is **{sex}**."

        if view in {0, 1}:
            ctr = calculate_ctr(info_mask.cpu().numpy())
            info_string += f"\n\nThe estimated cardiothoracic ratio (CTR) is **{ctr:0.2f}**."
            if view == 0:
                info_string += (
                        "\n\nNote that the cardiac silhuoette is magnified in the AP projection."
                )

        if view == 2:
            info_string += (
                    "\n\nNOTE: The below outputs are NOT VALID for lateral radiographs."
            )


        return {
                "result": info_overlay,
                "detections": info_string
        }

