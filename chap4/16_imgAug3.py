from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.RandomCrop(size=(400, 420)),  # 자르기, 위치는 랜덤인듯
    transforms.Pad(padding=30, fill=(127, 127, 255), padding_mode='constant'),
    # 패딩 넣기, 패딩 색, 패딩모드/reflect나 symmetric으로 주면 RGB값은 무시
    transforms.ToTensor()
])

image = Image.open('./Dog.jpeg')
transformed_image = transform(image)
plt.imshow(transformed_image.permute(1, 2, 0))  # 아마 순서 바꾸기 인듯
plt.show()
