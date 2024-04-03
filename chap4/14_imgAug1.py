from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(size=(512, 512)),  # 512x512 크기로 증강
        transforms.ToTensor()  # [0.0, 1.0] 사이 값으로 최소 최대 정규화를 때린다/(높이, 너비, 채널) -> (채널, 높이, 너비)
    ]
)

image = Image.open("./Dog.jpeg")
transformed_image = transform(image)

print(transformed_image.shape)
# image.show()
