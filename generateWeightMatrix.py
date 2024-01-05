# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by Wankang_Zhai (zhai@dlmu.edu.cn)
# ------------------------------------------------------------------------------

from trian import TriangularFunction

import numpy as np

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


# def example():
#     # 示例数据和参数
#     data = np.array([[1] ])
# # 有一个特征 对应的有一列权重
#     parameters = np.array([[0],
#                            [1],
#                            [2]])
#
#     # 调用生成权重矩阵的函数
#     weight_matrix = generate_weight_matrix(data, parameters)
#
#     # 打印生成的权重矩阵
#     print(weight_matrix)
#
# # 调用示例函数
# example()