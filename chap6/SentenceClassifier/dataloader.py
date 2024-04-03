import torch
from torch.utils.data import TensorDataset, DataLoader
from integer_encoding_with_padding import getIds
from korpora_dataset import getDataset

train_ids, test_ids = getIds()
train, test = getDataset()

# 정수 인코딩된 토큰들
train_ids = torch.tensor(train_ids)
test_ids = torch.tensor(test_ids)

# 오리지널 데이터셋의 라벨, 긍정 1/ 부정 0으로 라벨링 되어있더라
train_lables = torch.tensor(train.label.values, dtype=torch.float32)
test_lables = torch.tensor(test.label.values, dtype=torch.float32)

# 데이터셋(텐서화)과 데이터로더, 파이토치의 클래스로
train_dataset = TensorDataset(train_ids, train_lables)
test_dataset = TensorDataset(test_ids, test_lables)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


def getDataLoader():
    return train_loader, test_loader
