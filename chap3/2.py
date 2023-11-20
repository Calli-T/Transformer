import torch

# https://oculus.tistory.com/41
# 랜덤시드 고정방법에 대하여
# 아래 방식은 잘 안통했다

#import random
#random.seed(42)

#import numpy as np
#np.random.seed(42)

#  텐서의 속성은 형태 타입 장치가 있다. 하나라도 안맞으면 연산이 동작하지 않음
tensor = torch.rand(1, 2)
print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)