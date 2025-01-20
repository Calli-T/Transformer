from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class Mixup:
    def __init__(self, target, scale, alpha=0.5, beta=0.5):
        self.target = target
        self.scale = scale
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image):
        image = np.array(image)
        target = self.target.resize(self.scale)

        # 버전이 좀 안맞아서 손보았다
        target = np.array(target) / 255
        image = image / 255

        mix_image = image * self.alpha + target * self.beta
        return Image.fromarray((mix_image * 255).astype(np.uint8))


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    Mixup(
        target=Image.open('Cat.png').convert('RGB'),
        scale=(512, 512),
        alpha=0.5,
        beta=0.5,
    )
])

image1 = Image.open('Dog.jpeg')
transformed_image = transform(image1)
plt.imshow(transformed_image)
plt.show()
