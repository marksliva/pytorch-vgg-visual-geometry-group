import torch
from torch.nn import Module, Conv2d, MaxPool2d, Linear, Softmax, ReLU
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from functools import reduce

class Vgg16(Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        # consider wrapping in a sequential
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.maxpool1 = MaxPool2d(2, stride=2)
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = ReLU()
        self.conv4 = Conv2d(128, 128, kernel_size=3, stride=1)
        self.relu4 = ReLU()
        self.conv5 = Conv2d(128, 256, kernel_size=3, stride=1)
        self.relu5 = ReLU()
        self.maxpool2 = MaxPool2d(2, stride=2)
        self.conv6 = Conv2d(256, 256, kernel_size=3, stride=1)
        self.relu6 = ReLU()
        self.conv7 = Conv2d(256, 256, kernel_size=3, stride=1)
        self.relu7 = ReLU()
        self.conv8 = Conv2d(256, 512, kernel_size=3, stride=1)
        self.relu8 = ReLU()
        self.maxpool3 = MaxPool2d(2, stride=2)
        self.conv9 = Conv2d(512, 512, kernel_size=3, stride=1)
        self.relu9 = ReLU()
        self.conv10 = Conv2d(512, 512, kernel_size=3, stride=1)
        self.relu10 = ReLU()
        self.conv11 = Conv2d(512, 512, kernel_size=3, stride=1)
        self.relu11 = ReLU()
        self.maxpool4 = MaxPool2d(2, stride=2)
        self.conv12 = Conv2d(512, 512, kernel_size=3, stride=1)
        self.relu12 = ReLU()
        self.conv13 = Conv2d(512, 512, kernel_size=3, stride=1)
        self.relu13 = ReLU()
        self.conv14 = Conv2d(512, 512, kernel_size=3, stride=1)
        self.relu14 = ReLU()
        self.maxpool5 = MaxPool2d(2, stride=2)
        self.fc1 = Linear(1, 4096)
        self.fc2 = Linear(4096, 4096)
        self.fc3 = Linear(4096, 10)
        self.softmax = Softmax()

    def forward(self, batch):
        batch = self.conv1(batch)
        batch = self.relu1(batch)
        batch = self.conv2(batch)
        batch = self.relu2(batch)
        batch = self.maxpool1(batch)
        batch = self.conv3(batch)
        batch = self.relu3(batch)
        batch = self.conv4(batch)
        batch = self.relu4(batch)
        batch = self.conv5(batch)
        batch = self.relu5(batch)
        batch = self.maxpool2(batch)
        batch = self.conv6(batch)
        batch = self.relu6(batch)
        batch = self.conv7(batch)
        batch = self.relu7(batch)
        batch = self.conv8(batch)
        batch = self.relu8(batch)
        batch = self.maxpool3(batch)
        batch = self.conv9(batch)
        batch = self.relu9(batch)
        batch = self.conv10(batch)
        batch = self.relu10(batch)
        batch = self.conv11(batch)
        batch = self.relu11(batch)
        batch = self.maxpool4(batch)
        batch = self.conv12(batch)
        batch = self.relu12(batch)
        batch = self.conv13(batch)
        batch = self.relu13(batch)
        batch = self.conv14(batch)
        batch = self.relu14(batch)
        batch = self.maxpool5(batch)
        batch = self.fc1(batch)
        batch = self.fc2(batch)
        batch = self.fc3(batch)
        return self.softmax(batch)


# need to add the preprocessing step of subtracting the mean RGB value from each pixel
image_folder = ImageFolder("data/train", transform=ToTensor())
dataloader = DataLoader(image_folder, batch_size=1, shuffle=True)
print(torch.cuda.is_available())
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
		
model = Vgg16().to(device)

print(model)

total = 0
for param in model.parameters():
    print(param.shape)
    total += reduce((lambda d1, d2: d1 * d2), param.shape)

print("total trainable params: ", total)
for inputs, labels in dataloader:
    model.forward(inputs.to(device))
