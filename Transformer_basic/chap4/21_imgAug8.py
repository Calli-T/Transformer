from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=1.0, value=0), # value 0이면 컷아웃
    transforms.RandomErasing(p=1.0, value='random'), # random이면 무작위 지우기, 둘 다 Tensor에만 가능
    transforms.ToPILImage()
])

image = Image.open('Dog.jpeg')
transformed_image = transform(image)
plt.imshow(transformed_image)
plt.show()
