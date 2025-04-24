from tinygrad import Tensor, nn
import numpy as np
from tinygrad import dtypes


class MLP:
    def __init__(self, num_classes=2):
        self.fc_1 = nn.Linear(224*224*3, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, 64)
        self.fc_4 = nn.Linear(64, num_classes)

    def __call__(self, x:Tensor) -> Tensor:
        out = x.reshape(x.shape[0], -1)
        out = self.fc_1(out).relu()
        out = self.fc_2(out).relu()
        out = self.fc_3(out).relu()
        out = self.fc_4(out).sigmoid()
        return out



if __name__ == "__main__":
    # Example usage
    model = MLP(num_classes=2)
    input_tensor = Tensor(np.random.rand(1, 3, 224, 224).astype(np.float32))
    output = model(input_tensor)
    print(output.shape)  # Should print (1, 2)
    print(output.numpy())  # Should print the output values