from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
    # 각도, 이동, 척도, 전단을 이용한 이미지 변형
    transforms.ToTensor()
])

image = Image.open('Dog.jpeg')
transformed_image = transform(image)
plt.imshow(transformed_image.permute(1, 2, 0))
plt.show()
