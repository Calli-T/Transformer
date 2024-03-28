from device_converter import device
from CocoDataLoader import *
from model import *

from torch import optim

params = [p for p in model.parameters() if p.requires_grad]  # 필요 패러미터만 가져오는 묘수인가?
optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)  # 학습율/모멘텀/가중치감쇠
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                         gamma=0.1)  # 5 에퐄마다 학습률 0.1씩 줄어듬, 스케줄링 # 스케줄러도 optimizer처럼 step으로 학습률 갱신가능, 일반적으로 한 epoch마다

for epoch in range(5):
    cost = 0.0
    for idx, (images, targets) in enumerate(train_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()  # 이거 loss_dick로 하는게 맞지 않나?
        optimizer.step()

        cost += losses

    lr_scheduler.step()
    cost = cost / len(train_dataloader)
    print(f"Epoch : {epoch + 1:4d}, Cost : {cost:.3f}")

torch.save(model.state_dict(), "./models")
