import math

c = []
d = []

H = 3
W = 4

for j in range(H*W):
    j_ = j + 1
    c_ = H - (H * (W + 1)- j_) % H
    c.append(c_)
    d_ = math.ceil(j_ / 3)
    d.append(d_)
    
    print(f'M_x{j_} x_{c_}{d_}')