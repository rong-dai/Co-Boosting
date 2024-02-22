from torch import nn
import torch.nn.functional as F

class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        feature = x.view(x.size(0), -1)
        x = self.fc1(x)
        if return_features:
            return x, feature
        return x

    def feat_forward(self, x):
        return self.fc1(x)


def cnn(num_classes=10):
    return CNNCifar(num_classes)