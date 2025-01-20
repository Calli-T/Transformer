from torchvision import models
from torchinfo import summary
from device_converter import device

import torch
from PIL import Image
from torchvision import models, transforms

# 모델 불러오기
model = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1").eval().to(device)
# summary(model, (1, 3, 224, 244), device="cpu")

# 클래스 불러오기
with open("./source/imagenet_classes.txt", "r") as file:
    classes = file.read().splitlines()

# print(f"클래스 개수 : {len(classes)}")
# print(f"첫 번째 클래스 테이블 : {classes[0]}")

# 이미지 전처리
transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # imagenet 데이터셋의 평균과 표준 편차, 각각 R채널/G채널/B채널의 평균/표준편차이다
    ]
)

tensors = []
files = ["./source/airplane.jpg", "./source/bus.jpg"]
for file in files:
    image = Image.open(file)
    tensors.append(transforms(image))

tensors = torch.stack(tensors)
print(f"입력 텐서의 크기 : {tensors.shape}")

# 추론

import numpy as np
from torch.nn import functional as F

# 기울기 계산 엔진 비활성화, 임의의 값으로 모델을 확인
with torch.no_grad():
    outputs = model(tensors.to(device))
    probs = F.softmax(outputs, dim=-1)
    top_probs, top_idxs = probs.topk(5)
    # 가장 높은 5개를 뽑아서 확인

# 5개 가져온거 값이랑 인덱스 있는데 인덱스 가지고 클래스 리스트에서 인덱싱해서 가져옴
top_probs = top_probs.detach().cpu().numpy()
top_idxs = top_idxs.detach().cpu().numpy()
top_classes = np.array(classes)[top_idxs]

for idx, (cls, prob) in enumerate(zip(top_classes, top_probs)):
    print(f"{files[idx]} 추론 결과")
    for c, p in zip(cls, prob):
        print(f" - {c:<30} : {p * 100:>5.2f}%")
