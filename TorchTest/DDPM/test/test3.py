import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../datasets/sunflower/1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, interpolation='bicubic')
plt.show()