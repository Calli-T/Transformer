from torch import nn
from torchvision import models

from PIL import Image
from torchvision import transforms

import torch
from torch.nn import functional as F

import matplotlib.pyplot as plt

model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1").eval()
features = nn.Sequential(*list(model.children())[:-2])  # avgpool과 fc층은 분류기이므로 제외

# 전처리
transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

image = Image.open("./source/airplane.jpg")
target = transforms(image).unsqueeze(0)

# 모델이 판단한 클래스 idx 추출
output = model(target)
class_idx = int(output.argmax())

# FC에서 idx에 해당하는 가중치를 추출, W-c_k를 계산
# Resnet-18에서 FC는 입력 512차원 출력 1024차원
weights = model.fc.weight[class_idx].reshape(-1, 1, 1)  # [512] -> [512, 1, 1] 차원 확장
features_output = features(target).squeeze()  # [1, 512, 7, 7] -> [512, 7, 7] 연산을 위해 차원변경

print(weights.shape)
print(features_output.shape)

#

cam = features_output * weights
cam = torch.sum(cam, dim=0)  # [512, 7, 7] -> [7, 7]
cam = F.interpolate( # 이미지 7등분에서 어떤 영역에서 가장 많은 영향을 미쳤는지 알려줌 -> 보간하여 이미지 크기와 동일하게 변경
    input=cam.unsqueeze(0).unsqueeze(0), # [7, 7] -> [1, 1, 7, 7] 보간함수는 4차원배열을 입력
    size=(image.size[1], image.size[0]), # 이미지 크기와 같이!
    mode="bilinear", # 보간법은 이중 선형 방식을 통해
).squeeze().detach().numpy() # [이미지 너비, 높이로 변경] -> 넘파이 배열로

# 보여주기
plt.imshow(image)
plt.imshow(cam, cmap="jet", alpha=0.5)
plt.axis("off")
plt.show()