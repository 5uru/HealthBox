from tinygrad import Tensor, nn
import numpy as np
from tinygrad import dtypes

Tensor.default_dtype = dtypes.float32
class VGG16:
    def __init__(self, num_classes=2):
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)

        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)

        self.conv_4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)

        self.conv_5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn_5 = nn.BatchNorm2d(256)

        self.conv_6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_6 = nn.BatchNorm2d(256)

        self.conv_7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_7 = nn.BatchNorm2d(256)

        self.conv_8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn_8 = nn.BatchNorm2d(512)

        self.conv_9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn_9 = nn.BatchNorm2d(512)

        self.conv_10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn_10 = nn.BatchNorm2d(512)

        self.conv_11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn_11 = nn.BatchNorm2d(512)

        self.conv_12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn_12 = nn.BatchNorm2d(512)

        self.conv_13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn_13 = nn.BatchNorm2d(512)

        self.fc_1 = nn.Linear(7*7*512, 4096)
        self.fc_2 = nn.Linear(4096, 4096)
        self.fc_3 = nn.Linear(4096, num_classes)


    def __call__(self, x:Tensor) -> Tensor:
        out = self.conv_1(x)
        out = self.bn_1(out).relu()

        out = self.conv_2(out)
        out = self.bn_2(out).relu().max_pool2d(kernel_size = (2,2), stride = 2)

        out = self.conv_3(out)
        out = self.bn_3(out).relu()

        out = self.conv_4(out)
        out = self.bn_4(out).relu().max_pool2d(kernel_size = (2,2), stride = 2)

        out = self.conv_5(out)
        out = self.bn_5(out).relu()

        out = self.conv_6(out)
        out = self.bn_6(out).relu()

        out = self.conv_7(out)
        out = self.bn_7(out).relu().max_pool2d(kernel_size = (2,2), stride = 2)

        out = self.conv_8(out)
        out = self.bn_8(out).relu()

        out = self.conv_9(out)
        out = self.bn_9(out).relu()

        out = self.conv_10(out)
        out = self.bn_10(out).relu().max_pool2d(kernel_size = (2,2), stride = 2)

        out = self.conv_11(out)
        out = self.bn_11(out).relu()

        out = self.conv_12(out)
        out = self.bn_12(out).relu()

        out = self.conv_13(out)
        out = self.bn_13(out).relu().max_pool2d(kernel_size = (2,2), stride = 2)

        out = out.reshape(out.size(0), -1)

        out = out.dropout(0.5)
        out = self.fc_1(out)
        out = out.relu()

        out = out.dropout(0.5)
        out = self.fc_2(out).relu()

        out = self.fc_3(out)


        return out

if __name__ == "__main__":

    model = VGG16()
    x = Tensor(np.random.randn(1, 3, 224, 224), dtype=dtypes.float32)
    y = model(x)
    print(y.numpy())  # Should be (1, num_classes)
    print(y.shape)  # Should be (1, num_classes)