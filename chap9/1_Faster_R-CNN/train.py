from CocoDataLoader import *
from model import *

from torch import optim
from time import time

params = [p for p in model.parameters() if p.requires_grad]  # 필요 패러미터만 가져오는 묘수인가?
optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)  # 학습율/모멘텀/가중치감쇠
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                         gamma=0.1)  # 5 에퐄마다 학습률 0.1씩 줄어듬, 스케줄링 # 스케줄러도 optimizer처럼 step으로 학습률 갱신가능, 일반적으로 한 epoch마다

for epoch in range(5):
    cost = 0.0

    # time check
    count = 0
    start = time()

    for idx, (images, targets) in enumerate(train_dataloader):

        # time check
        now = int(time() - start)
        count += 1
        if count % 100 == 0:
            print(f"count: {count}, {now//60}m {now%60}s")

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        cost += losses

    lr_scheduler.step()
    cost = cost / len(train_dataloader)
    print(f"Epoch : {epoch + 1:4d}, Cost : {cost:.3f}")

torch.save(model.state_dict(), "./models/faster_rcnn.pt")
