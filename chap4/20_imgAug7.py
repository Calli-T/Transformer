from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa

class IaaTransforms:
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.SaltAndPepper(p=(0.03, 0.07)),
            iaa.Rain(speed=(0.3, 0.7))
        ])

    def __call__(self, images):
        images = np.array(images)
        augmented = self.seq.augment_image(images)
        return Image.fromarray(augmented)


transform = transforms.Compose([
    IaaTransforms()
])

image = Image.open('./Dog.jpeg')
transformed_image = transform(image)
plt.imshow(transformed_image)
plt.show()

# 패키지 버전 충돌
