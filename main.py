import torch
from torch.nn import Module, Conv2d, MaxPool2d, Linear, Softmax, ReLU, CrossEntropyLoss, Dropout, Sequential
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from functools import reduce
from torchvision.transforms import CenterCrop, Compose
import numpy as np
from torch.nn.init import xavier_uniform_

epochs = 1

class Vgg16(Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        self.convolutions = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            MaxPool2d(2, stride=2),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            MaxPool2d(2, stride=2),
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            MaxPool2d(2, stride=2),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            MaxPool2d(2, stride=2),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(True),
            MaxPool2d(2, stride=2)
        )

        self.connected = Sequential(
            Linear(in_features=512 * 7 * 7, out_features=4096, bias=True),
            ReLU(True),
            Dropout(p=0.5),
            Linear(in_features=4096, out_features=4096, bias=True),
            ReLU(True),
            Dropout(p=0.5),
            Linear(in_features=4096, out_features=2, bias=True),
            Softmax(dim=1)
        )

        self.convolutions.apply(self._init_weights)
        self.connected.apply(self._init_weights)

    def forward(self, batch):
        batch = self.convolutions(batch)
        batch = batch.view(-1, 512 * 7 * 7)
        batch = self.connected(batch)
        print("after running connected layers ", batch)
        return batch


    def _init_weights(self, m):
        if type(m) == Linear or type(m) == Conv2d:
            xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


# need to add the preprocessing step of subtracting the mean RGB value from each pixel
transforms = Compose([CenterCrop(224), ToTensor()])
train_image_folder = ImageFolder("data/train", transform=transforms)
test_image_folder = ImageFolder("data/train", transform=transforms)
train_dataloader = DataLoader(train_image_folder, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_image_folder, batch_size=1, shuffle=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Vgg16().to(device)

# print the number of parameters
# print(model)
# total = 0
# for param in model.parameters():
#     print(param.shape)
#     total += reduce((lambda d1, d2: d1 * d2), param.shape)
# print("total trainable params: ", total)

loss_function = CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005)

torch.random.manual_seed(2)

model.train()

for i in range(epochs):
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        print("loss ", loss)
        _, preds = torch.max(outputs, 1)
        print("train prediction: ", preds, "matched actual: ", preds.equal(labels))
        print("\n\n")


print("**test**")

model.eval()

for test_inputs, expected_labels in test_dataloader:
    inputs = test_inputs.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    print("test prediction: ", preds, "matched actual: ", preds.equal(expected_labels))
    print("\n\n")
