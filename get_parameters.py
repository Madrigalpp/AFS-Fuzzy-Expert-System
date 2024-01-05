# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by Wankang_Zhai (zhai@dlmu.edu.cn)
# ------------------------------------------------------------------------------
import numpy as np

def get_parameters(data):
    parameters = np.zeros((3, data.shape[1]))

    for j in range(data.shape[1]):
        max_val = float('-inf')
        min_val = float('inf')
        sum_val = 0

        for i in range(data.shape[0]):
            # 寻找最大值
            max_val = max(max_val, data[i, j])

            # 寻找最小值
            min_val = min(min_val, data[i, j])

            # 累加计算总和
            sum_val += data[i, j]

        # 设置统计参数矩阵的值
        # 最小值
        parameters[0, j] = min_val

        # 均值
        parameters[1, j] = sum_val / data.shape[0]

        # 最大值
        parameters[2, j] = max_val

    return parameters

import numpy as np

# 创建一个示例数据矩阵
data_matrix = np.array([[1, 2, 3],
                       [4, 50, 6],
                       [101, 8, 11]])

# 调用get_parameters函数
result = get_parameters(data_matrix)

# 打印结果
print("输入数据矩阵:")
print(data_matrix)
print("\n统计参数矩阵:")
print(result)

# 输入矩阵
# [[  1   2   3]
#  [  4  50   6]
#  [101   8  11]]

# 统计参数矩阵
# 统计参数矩阵:
# [[  1.           2.           3.        ]
#  [ 35.33333333  20.           6.66666667]
#  [101.          50.          11.        ]]