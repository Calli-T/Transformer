from itertools import chain
from collections import defaultdict
from torch.utils.data import Subset
from torchvision import datasets

# -

import torch
from torchvision import transforms
from transformers import AutoImageProcessor

# -

from torch.utils.data import DataLoader


# --- subset ---


# 잘라 담아주는 함수인가?
def subset_sampler(dataset, classes, max_len):
    target_idx = defaultdict(list)
    for idx, label in enumerate(dataset.train_labels):
        target_idx[int(label)].append(idx)

    indices = list(
        chain.from_iterable(
            [target_idx[idx][:max_len] for idx in range(len(classes))]
        )
    )

    return Subset(dataset, indices)


train_dataset = datasets.FashionMNIST(root="./source", download=True, train=True)
test_dataset = datasets.FashionMNIST(root="./source", download=True, train=False)

classes = train_dataset.classes
class_to_idx = train_dataset.class_to_idx

subset_train_dataset = subset_sampler(dataset=train_dataset, classes=train_dataset.classes, max_len=1000)
subset_test_dataset = subset_sampler(dataset=test_dataset, classes=test_dataset.classes, max_len=100)

'''print(train_dataset)
print(test_dataset)
print(classes)
print(class_to_idx)
print(len(subset_train_dataset))
print(len(subset_test_dataset))
print(train_dataset[0])'''

# --- image preprocessing ---

# 허깅 페이스의 이미지 처리기 프로세스를 전처리에 사용, crop이나 resize 등 사용가능
'''image_processor = AutoImageProcessor.from_pretrained(
    pretrained_model_name_or_path="google/vit-base-patch16-224-in21k"  # 224 size, 16 image patch
)'''
image_processor = AutoImageProcessor.from_pretrained(
    pretrained_model_name_or_path="microsoft/swin-tiny-patch4-window7-224"  # 224 size, 16 image patch
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            size=(
                image_processor.size["height"],
                image_processor.size["width"]
            )
        ),
        transforms.Lambda(
            lambda x: torch.cat([x, x, x], 0)  # 단일채널복제, 다중 채널로 변환
        ),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        )
    ]
)

'''print(f"size: {image_processor.size}")
print(f"mean: {image_processor.image_mean}")
print(f"std: {image_processor.image_std}")'''


# --- dataloader ---
def collator(data, transform):
    images, labels = zip(*data)
    pixel_values = torch.stack([transform(image) for image in images])
    labels = torch.tensor([label for label in labels])

    return {"pixel_values": pixel_values, "labels": labels}  # 입력형식이 픽셀벨류와 라벨


train_dataloader = DataLoader(
    subset_train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda x: collator(x, transform),
    drop_last=True
)
valid_dataloader = DataLoader(
    subset_test_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda x: collator(x, transform),
    drop_last=True
)

'''
batch = next(iter(train_dataloader))
for key, value in batch.items():
    print(f"{key}: {value.shape}")
'''
