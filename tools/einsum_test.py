import torch

a = torch.rand(32, 192, 17)
b = torch.rand(3, 17, 17)

c = torch.einsum('ncv,kvw->ncw', (a, b)) # 'ncv,kvw->ncw'   32,192,17, 3,17,17-> 32,192,17

print('einsum--', c)

b = torch.sum(b, dim=0)
# a = a.reshape(3, 32, 192, 17)

a_b = torch.matmul(a, b)

print('ddd--', a_b)
