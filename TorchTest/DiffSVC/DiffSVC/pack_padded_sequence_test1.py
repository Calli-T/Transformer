# import torch
# from torch.nn.utils.rnn import pack_padded_sequence
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # 1MT -> B1MT 텐서로 제작 ㄱㄱ
# tensor_list = [torch.zeros([1, 128, 1287]),
#                torch.zeros([1, 128, 1300]),
#                torch.zeros([1, 128, 1557])]
#
import torch
from torch.nn.utils.rnn import pack_padded_sequence

# 입력 시퀀스 (패딩 포함)
input_tensor = torch.tensor([
    [1, 2, 3, 0],   # 시퀀스 1
    [4, 5, 0, 0],   # 시퀀스 2
    [6, 0, 0, 0]    # 시퀀스 3
], dtype=torch.float32)

# 길이 정보 (각 시퀀스의 실제 길이)
lengths = torch.tensor([3, 2, 1])

# 배치 크기 우선 (B, T)
packed = pack_padded_sequence(input_tensor, lengths, batch_first=True, enforce_sorted=False)

print(packed)