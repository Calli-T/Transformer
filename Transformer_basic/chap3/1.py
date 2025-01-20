import torch

# 47p
print(torch.tensor([1, 2, 3]))
print(torch.Tensor([[1, 2, 3], [4, 5, 6]]))
print(torch.LongTensor([1, 2, 3]))
print(torch.FloatTensor([1, 2, 3]))

# 토치는 넘파이에서 배열을 생성하는 방식과 동일함
# tensor 보다는 Tensor권장, 빈 구조로 생성하는게 불가능하므로 의도하지 않은 자료형으로 바뀔일이 없다.
# Long Float등등은 해당 자료형으로
