'''
import os


def name_converter():
    os.chdir(os.getcwd() + "\\source\\pet\\SinChangSeop\\cat")
    name_list = os.listdir()

    for idx, name in enumerate(name_list):
        os.rename(name, "cat." + str(idx + 1 + 1000) + ".jpg")
'''
import torch
#

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

#
import numpy as np
from matplotlib import pyplot as plt

#
from torchvision import models

#
from torch import nn, optim

#
from device_converter import device

# set data

hyperparams = {
    "batch_size": 4,
    "learning_rate": 0.0001,
    "epochs": 5,
    "transform": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48235, 0.45882, 0.40784],
                std=[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
            )
        ]
    )
}

# 클래스명은 폴더명으로 적용됨
train_dataset = ImageFolder('./source/pet/train', transform=hyperparams["transform"])
test_dataset = ImageFolder('./source/pet/SinChangSeop', transform=hyperparams["transform"])

train_dataloader = DataLoader(
    train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True
)


# data visualization

def showimages():
    mean = [0.48235, 0.45882, 0.40784]
    std = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]

    images, labels = next(iter(train_dataloader))
    for image, label in zip(images, labels):
        image = image.numpy().transpose((1, 2, 0))  # Tensor는 HWC가 아닌 CHW순서다
        image = ((std * image + mean) * 255).astype(np.uint)  # 픽셀범위 조정

        plt.imshow(image)
        plt.title(train_dataset.classes[int(label)])
        plt.show()


# get VGG-16 models & weights
model = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
# print(model)

# fine tuning, 2개의 클래스로 output 크기를 변경함
# 사전 학습된걸 쓸 수 없으니 새로운 가중치를 학습해야함!
model.classifier[6] = nn.Linear(4096, len(train_dataset.classes))

# 모델/손실함수/최적화 기법 선정
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=hyperparams["learning_rate"])

#####
# model.load_state_dict(torch.load("models/VGG16.pt"))

# 학습
for epoch in range(hyperparams["epochs"]):
    cost = 0.0

    for images, classes in train_dataloader:
        images = images.to(device)
        classes = classes.to(device)

        output = model(images)
        loss = criterion(output, classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(train_dataloader)
    print(f"Epoch : {epoch + 1:4d}, Cost : {cost:.3f}")

with torch.no_grad():
    model.eval()

    accuracy = 0.0

    for images, classes in test_dataloader:
        images = images.to(device)
        classes = classes.to(device)

        outputs = model(images)
        probs = nn.functional.softmax(outputs, dim=-1)
        outputs_classes = torch.argmax(probs, dim=-1)

        print(torch.eq(classes, outputs_classes))

        accuracy += int(torch.eq(classes, outputs_classes).sum())

    # 책의 방식과는 라이브러리가 약간 달라진듯
    # print(f"acc@1 : {accuracy / (len(test_dataset) * hyperparams['batch_size']) * 100:.2f}%")
    print(f"acc@1 : {accuracy / len(test_dataset) * 100:.2f}%")

torch.save(model.state_dict(), "models/VGG16.pt")
print("Saved the model weights")
