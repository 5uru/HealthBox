from tinygrad import Tensor, nn
import numpy as np
from tinygrad import dtypes


class CNN:
    def __init__(self):
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Fully connected layers
        self.fc1 = nn.Linear(100352, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)


    def __call__(self, x:Tensor) -> Tensor:
        # Feature extraction
        x = (self.bn1(self.conv1(x))).relu()
        x = x.max_pool2d()

        x = (self.bn2(self.conv2(x))).relu()
        x = x.max_pool2d()

        x = (self.bn3(self.conv3(x))).relu()
        x = x.max_pool2d()

        x = (self.bn4(self.conv4(x))).relu()
        x = x.avg_pool2d()

        # Classification
        x = x.flatten(1)
        x = x.dropout(0.5)
        x = (self.fc1(x)).relu()
        x = x.dropout(0.3)
        x = self.fc2(x)
        x = x.dropout(0.3)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    # Example usage
    model = CNN()
    input_tensor = Tensor(np.random.rand(1, 3, 224, 224).astype(np.float32))
    output = model(input_tensor)
    print(output.shape)  # Should print (1, 2)
    print(output.numpy())  # Should print the output values