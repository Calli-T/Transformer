from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    # 각각 밝기 대비 채도 색상/이미지는 거리나 조명 등에 의해 색상이 크게 달라질 수 있다고 한다
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # 정규화 클래스는 PIL.image말고 Tensor식으로 받고 그 방식은 (input[channel] - mean[channel]) / std[channel]
    transforms.ToPILImage()  # permute가 필요없어진다?
])

image = Image.open('./Dog.jpeg')
transformed_image = transform(image)
plt.imshow(transformed_image)  # .permute(1, 2, 0))
plt.show()
