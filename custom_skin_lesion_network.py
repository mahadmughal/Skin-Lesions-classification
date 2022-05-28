from torch import nn
import torch.nn.functional as F
import torch

custom_model = nn.Sequential(
    nn.Conv2d(3,8,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(8),
    nn.Conv2d(8,16,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(128),

    nn.MaxPool2d(2,2),
    nn.Flatten(),
    nn.Linear(256*4*4,1024),
    nn.ReLU(),
    nn.Linear(1024,512),
    nn.ReLU(),
    nn.Linear(512,7)
  )


# class CustomSkinLesionNetwork(nn.Module):
#   def __init__(self):
#     super(CustomSkinLesionNetwork, self).__init__()
#     self.conv1 = nn.Conv2d(3, 64, 224, 224)
#     self.conv2 = nn.Conv2d(8, 16, 3, 1)
#     self.conv3 = nn.Conv2d(16, 32, 3, 1)
#     self.conv4 = nn.Conv2d(32, 64, 3, 1)
#     self.conv5 = nn.Conv2d(64, 128, 3, 1)

#   def forward(self, x):
#     x = self.conv1(x)
#     x = F.relu(x)
#     x = F.BatchNorm2d(x)
#     x = self.conv2(x)
#     x = F.relu(x)
#     x = F.BatchNorm2d(x)
#     x = self.conv3(x)
#     x = F.relu(x)
#     x = F.BatchNorm2d(x)
#     x = self.conv4(x)
#     x = F.relu(x)
#     x = F.BatchNorm2d(x)
#     x = self.conv5(x)
#     x = F.relu(x)
#     x = F.BatchNorm2d(x)
#     x = F.max_pool2d(x, 2)
#     x = torch.flatten(x, 1)

#     output = F.log_softmax(x, dim=1)

#     return output

# #     model = NeuralNetwork().to(device)
# #     print(model)

class CustomSkinLesionNetwork(nn.Module):

  def __init__(self, num_classes=7):
    super(CustomSkinLesionNetwork, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, num_classes)
    )

  def forward(self, x):
      x = self.features(x)
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
      return x
