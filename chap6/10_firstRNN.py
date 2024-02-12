import torch
from torch import nn

input_size = 128
output_size = 256
num_layers = 3
bidirectional = True

model = nn.RNN(input_size=input_size, hidden_size=output_size, num_layers=num_layers, nonlinearity="tanh",
               batch_first=True, bidirectional=bidirectional)

batch_size = 4
sequence_len = 6

inputs = torch.randn(batch_size, sequence_len, input_size) # 입력값
h_0 = torch.rand(num_layers * (int(bidirectional) + 1), batch_size, output_size) # 초기 은닉 상태

outputs, hidden = model(inputs, h_0)
print(outputs.shape)
print(hidden.shape)

# 그래서 이게 뭐냐 ???