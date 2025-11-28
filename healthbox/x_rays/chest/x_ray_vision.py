import torchxrayvision as xrv
import skimage, torch, torchvision
import numpy as np
from PIL import Image
from healthbox.blood_stain.base import Model






class XRayVision(Model):
    def __init__(self):

        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")

    def predict(self, image):

        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = skimage.io.imread(image)
        # Prepare the image:
        img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
        img = img.mean(2)[None, ...] # Make single color channel

        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

        img = transform(img)
        img = torch.from_numpy(img)

        results = self.model(img[None,...])
        results = dict(zip(self.model.pathologies,results[0].detach().numpy()))

        outputs = []
        for k,v in results.items():
            if k in ['Pneumothorax', 'Pneumonia', 'Fracture']:
                continue
            result = {
                    "name": k,
                    "value": round(v.item(), 2)
            }
            outputs.append(result)
        # sort by value
        outputs.sort(key=lambda x: x['value'], reverse=True)

        result = ""
        for i in outputs:
            if  i['value'] > 0.5:
                result += f":red[{i['name']}: {i['value']:.2f}] \n\n "
            else:
                result += f":blue[{i['name']}: {i['value']:.2f}] \n\n "
        return result

