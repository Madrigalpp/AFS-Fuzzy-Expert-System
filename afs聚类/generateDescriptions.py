import numpy as np
from trian import TriangularFunction

def generate_weight_matrix(data, parameters):
    num_rows, num_cols = data.shape
    weight_matrix = np.zeros((num_rows, 3 * num_cols))

    flag = 0

    for j in range(num_cols):
        tf = [None] * 3

        for i in range(len(tf)):
            if i == 0:
                tf[i] = TriangularFunction(parameters[0, j], parameters[0, j], parameters[2, j])
            elif i == len(tf) - 1:
                tf[i] = TriangularFunction(parameters[0, j], parameters[2, j], parameters[2, j])
            else:
                tf[i] = TriangularFunction(parameters[0, j], parameters[1, j], parameters[2, j])

        for i in range(len(tf)):
            for m in range(num_rows):
                weight_matrix[m, flag] = tf[0].apply(data[m, j])
                weight_matrix[m, flag + 1] = tf[1].apply(data[m, j])
                weight_matrix[m, flag + 2] = tf[2].apply(data[m, j])

        flag += 3

    return weight_matrix

# 示例调用
# data_matrix = np.array([[1, 2, 3],
#                         [4, 5, 6],
#                         [7, 8, 9]])

data_matrix = np.array([[1],
                        [4],
                        [7]])

parameters_matrix = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]])
# parameters[0, j] 是左边界，parameters[1, j] 是顶点，parameters[2, j] 是右边界

# 函数将对每一列数据应用三个三角函数

result = generate_weight_matrix(data_matrix, parameters_matrix)
print(result)
