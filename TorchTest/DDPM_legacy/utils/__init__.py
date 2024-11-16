from matplotlib import pyplot as plt
import cv2


def show_images(images, h, w):
    plt.figure(figsize=(w, h))
    for idx, image in enumerate(images):
        plt.subplot(h, w, idx + 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    plt.show()
