from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize(size=(300, 300)),
    transforms.ToTensor()
])

image = Image.open('./Dog.jpeg')
transformed_image = transform(image)
plt.imshow(transformed_image.permute(1, 2, 0))
plt.show()
