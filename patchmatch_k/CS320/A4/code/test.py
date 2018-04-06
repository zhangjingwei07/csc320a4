import numpy as np
from heapq import heappush, heappushpop, nlargest, heappop
a = np.random.rand(10, 10, 3)

# target_index = np.zeros((5, 10, 10, 2)).astype(int)
# target_y = np.clip(target_index[:, :, :, 0], 0, 10)
# target_x = np.clip(target_index[:, :, :, 1], 0, 10)
# b = a[target_y, target_x]
# print(target_x.shape)
# print(b.shape)
#print(len(a[:, :, 0]))

# h = []
# heappush(h, (5, 'write code'))
# heappush(h, (7, 'release product'))
# heappush(h, (1, 'write spec'))
# heappush(h, (3, 'create tests'))
# print(h)
a = [1, 2, 3]
target_idx = np.array([1, 2])
target_idx.astype(float)
print(target_idx)