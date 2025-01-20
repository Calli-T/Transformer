from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees=30, expand=False, center=None), # expand=True의 경우 회전과정에서 여백생성
        transforms.RandomRotation(degrees=[-180, 180], expand=True, center=None), # 임의 범위 설정은 리스트로
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ]
)

image = Image.open("Dog.jpeg")
transformed_image = transform(image)

print(transformed_image.shape)
