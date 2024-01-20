from torch import nn
import torch_directml

device = torch_directml.device()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 2),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(2, 1)
        self._init_weights()

    def _init_weights(self):
        # Sequential의 층(들)?을 제이비어 초기화, 편향은 상수로
        nn.init.xavier_uniform_(self.layer[0].weight)
        self.layer[0].bias.data.fill_(0.01)

        # 한 층을 제이비어 초기화, 편향은 상수로
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)


model = Net().to(device)
