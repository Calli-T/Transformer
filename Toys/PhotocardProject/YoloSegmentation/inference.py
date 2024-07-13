from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

model = YOLO('./yolov8x-seg.pt')

'''
results = model.predict(
    "./mu.png",
    save=True,
    imgsz=640,
    conf=0.5,
    device="cuda"
)
'''
results = model("./mu.png", project="./runs", imgsz=640)  # , save=True, imgsz=736)

'''
for obj in results[0]:
    print(obj.masks)
    print(obj.masks.shape)
    break

'''

'''
for field_method in dir(results[0]):
    print(field_method)
'''

# print(type(results[0]))
# print(type(results[0].plot(show=False)))
# print(results[0].plot(show=False).shape)
# print(results[0].masks[0].xy[0].shape)
# print(results[0].masks[0].xyn[0].shape)
# print(len(results[0].masks.xy))

'''
img = plt.imread("./mu.png")
# plt.imshow(img)
for point in results[0].masks[0].xy[0]:
    plt.scatter(point[0], point[1], s=1)
plt.show()
'''

# for obj in results[0].masks:
#     for point in obj.xy[0]:
#         plt.scatter(point[0], (640 - point[1]), s=1)
# plt.show()

# print(type(results[0].masks[0].xy[0]))

origin = cv2.imread("./mu.png")
# mask[0]부터 각각의 객체 정보의 xy[0]는 하나의 큰 numpy 배열, 그걸로 다각형을 만들면 그게 누끼,
for obj in results[0].masks:
    points = obj.xy[0].astype(np.int32)
    # for point in points:
    #     point[1] = point[1]
    cv2.polylines(origin, [points], True, (255, 0, 0), 4)
cv2.imshow("img", origin)
cv2.waitKey(0)

# 클래스 정보를 찾아보자