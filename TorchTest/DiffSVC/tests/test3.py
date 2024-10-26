import torch

t = torch.tensor([[1, 2], [3, 4]])

print(torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]])))
print(torch.gather(t, 1, torch.tensor([[0, 1], [1, 0]])))
print(torch.gather(t, 1, torch.tensor([[1, 1], [1, 0]])))
print(torch.gather(t, 0, torch.tensor([[0, 0], [1, 0]])))
print(torch.gather(t, 0, torch.tensor([[1, 1], [1, 0]])))

'''
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
'''

'''tensor_3d = torch.tensor(
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24],
                                                                                     [25, 26, 27]]])
'''