import math
import torch
from torch import nn
from matplotlib import pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # 0부터 max_len-1까지의 값이 담긴 텐서를 차원을 하나 추가해서 준다. 벡터->행렬
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        '''
        position은 말그대로 pos
        div_term에서 임베딩의 인덱스는 2i또는 2i+1로 표시,
        arange는 [0, 2, 4, 6, ... , 126]즉 2i이고 뒤는 -ln(10000)/d_model
        이를 exp함수에 넣으면 각각의 값 n에 대해 e^n이 나올텐데, e^(2i * (-ln(10000)/d_model)) = e^((2i/d_model) * (- log e 10000)) 
        = 1/e^((2i/d_model) * log e 10000)) 
        = 1/e^(ln(10000^(2i/d_model)))
        = 1/10000^(2i/d_model)
        
        position = div_term = pos/10000^(2i/d_model)
        이는 위치 인코딩에서 sin과 cos에 넣는 값과도 같다
        '''
        pe = torch.zeros(max_len, 1, d_model)  # [최대 시퀀스 길이, 1, 입력 임베딩의 차원]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # 매개변수 갱신 안하도록 설정

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


encoding = PositionalEncoding(d_model=128, max_len=50)

plt.pcolormesh(encoding.pe.numpy().squeeze(), cmap="RdBu")
plt.xlabel("Embedding Dimension")
plt.xlim((0, 128))
plt.ylabel("Position")
plt.colorbar()
plt.show()
