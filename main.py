import torch
from torch.nn import Module, Conv2d, MaxPool2d, Linear, Softmax, ReLU, CrossEntropyLoss
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from functools import reduce

class Vgg16(Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        # consider wrapping in a sequential
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU(True)
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU(True)
        self.maxpool1 = MaxPool2d(2, stride=2)
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = ReLU(True)
        self.conv4 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = ReLU(True)
        self.conv5 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu5 = ReLU(True)
        self.maxpool2 = MaxPool2d(2, stride=2)
        self.conv6 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu6 = ReLU(True)
        self.conv7 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu7 = ReLU(True)
        self.conv8 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu8 = ReLU(True)
        self.maxpool3 = MaxPool2d(2, stride=2)
        self.conv9 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu9 = ReLU(True)
        self.conv10 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu10 = ReLU(True)
        self.conv11 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu11 = ReLU(True)
        self.maxpool4 = MaxPool2d(2, stride=2)
        self.conv12 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu12 = ReLU(True)
        self.conv13 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu13 = ReLU(True)
        self.conv14 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu14 = ReLU(True)
        self.maxpool5 = MaxPool2d(2, stride=2)
        self.fc1 = Linear(in_features=512 * 7 * 7, out_features=4096, bias=True)
        self.fc2 = Linear(in_features=4096, out_features=4096, bias=True)
        self.fc3 = Linear(in_features=4096, out_features=10, bias=True)
        self.softmax = Softmax(dim=0)

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
        batch = batch.reshape(batch.size(0), -1)
        batch = self.fc1(batch)
        batch = self.fc2(batch)
        batch = self.fc3(batch)
        return self.softmax(batch)


# need to add the preprocessing step of subtracting the mean RGB value from each pixel
image_folder = ImageFolder("data/train", transform=ToTensor())
dataloader = DataLoader(image_folder, batch_size=2, shuffle=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Vgg16().to(device)


# print the number of parameters
# print(model)
# total = 0
# for param in model.parameters():
#     print(param.shape)
#     total += reduce((lambda d1, d2: d1 * d2), param.shape)
# print("total trainable params: ", total)


helpers = dict(
    loss_function = CrossEntropyLoss(),
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
)

torch.random.manual_seed(1)

for inputs, labels in dataloader:
    print("inputs shape ", inputs.shape)
    print("labels", labels)
    helpers['optimizer'].zero_grad
    outputs = model.forward(inputs.to(device))
    loss = helpers['loss_function'](outputs, labels)
    loss.backward()
    helpers['optimizer'].step()
