import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, root, train, _transform=None):
        super().__init__()
        directory = "train" if train else "val"
        annotations = root + "/annotations" + f"/{directory}_annotations.json"

        self.coco = COCO(annotations)
        self.image_path = os.path.join(root, directory)
        self._transform = _transform

        self.categories = self._get_categories()
        self.data = self._load_data()

    def _get_categories(self):  # 카테고리 가져옴
        categories = {0: "background"}  # 배경을 의미
        for category in self.coco.cats.values():  # 나머지 카테고리 이름을 가져옴
            categories[category["id"]] = category["name"]

        return categories

    def _load_data(self):  # 이미지들에 박스랑 클래스 정보 가져와서 처리함
        data = []
        for _id in self.coco.imgs:
            file_name = self.coco.loadImgs(_id)[0]["file_name"]
            image_path = os.path.join(self.image_path, file_name)
            image = Image.open(image_path).convert("RGB")

            boxes = []
            labels = []

            anns = self.coco.loadAnns(self.coco.getAnnIds(_id))  # imageId2annId
            for ann in anns:
                x, y, w, h = ann["bbox"]  # boundary box
                # 경계상자랑 클래스
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

                target = {
                    "image_id": torch.LongTensor([_id]),
                    "boxes": torch.FloatTensor(boxes),
                    "labels": torch.LongTensor(labels)
                }
                data.append([image, target])
        return data

    # idx2transformedImage&target
    def __getitem__(self, index):
        image, target = self.data[index]
        if self._transform:
            image = self._transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


# ---------------------------------------

from torchvision import transforms
from torch.utils.data import DataLoader


# batch로 묶을 때 필요한 함수를 정의
def collator(batch):
    return tuple(zip(*batch))


transform = transforms.Compose(
    [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float)
    ]
)

# 경로수정할것
train_dataset = COCODataset("./source", train=True, _transform=transform)
test_dataset = COCODataset("./source", train=False, _transform=transform)

train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=collator
)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=collator
)

# 1