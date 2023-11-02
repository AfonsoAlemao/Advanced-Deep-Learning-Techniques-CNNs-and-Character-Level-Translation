import numpy as np
import math
x = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
#x = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=float)
#x = np.array([[20, 35, 35, 35, 35, 20],[29, 46, 44, 42, 42, 27],[16, 25, 21, 19, 19, 12],[66, 120, 116, 154, 114, 62],[74, 216, 174, 252, 172, 112],[70, 210, 170, 250, 170, 110]], dtype=float)

W_pesos = np.array([[1,2],[3,4]], dtype=float)
#W_pesos = np.array([[1,2],[3,4],[5,6]], dtype=float)
#W_pesos = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]], dtype=float)

#z_line_esperado = np.array([37,67,47,77], dtype=float)
#z_line_esperado = np.array([149,170,191], dtype=float)
#z_line_esperado = np.array([225,458,708,1000,258,566,981,1488,250, 552, 887, 1320, 209, 472, 802, 1224], dtype=float)

H = x.shape[0]
W = x.shape[1]

M = W_pesos.shape[0]
N = W_pesos.shape[1]

H_line = H - M + 1
W_line = W - N + 1

M_sol = np.zeros((H_line * W_line, H * W))

for i in range(H_line * W_line):
    i_ = i + 1
    k = math.floor((i_ - 1) / 2)
    for j in range(H * W):
        j_ = j + 1
        b = math.ceil(j_ / H) - math.floor((i_ - 1) /  H_line)
        a = j_ - H * math.floor((j_ - 1) /  H) - (i_ - 1) + H_line * math.floor((i_ - 1) /  H_line)
        
        if (a <= M and b <= N and a > 0 and b > 0):
            print(f'M_sol[{i_}][{j_}] = W[{a}][{b}]')
            M_sol[i][j] = W_pesos[a - 1][b - 1]

        
vec_x = x.reshape((-1, 1), order="F")
print(M_sol)
z_linha = np.array(np.matmul(M_sol, vec_x))
print(z_linha)