# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by Wankang_Zhai (zhai@dlmu.edu.cn)
# ------------------------------------------------------------------------------
import numpy as np

def get_membership_degree(weight_matrix, index_of_concept, index_of_instance):

    '''
    计算一个实例对于一个Fuzzy_Term的隶属度。
    :param weight_matrix: 一个包含权重值的矩阵，(indexOfInstance,indexOfConcept)
    :param index_of_concept: 表示概念的索引，即在权重矩阵中的列索引。
    :param index_of_instance: 表示实例的索引，即在权重矩阵中的行索引。
    函数通过循环遍历权重矩阵的每一行，检查每一行对应的 weight_matrix[i, index_of_concept]
    是否小于等于给定概念的权重值 weight_matrix[index_of_instance, index_of_concept]。如果满足这个条件，就增加计数器的值。
    计算隶属度，即满足条件的行数除以总行数
    :return: (int) 隶属度值
    '''

    count = sum(weight_matrix[i, index_of_concept] <= weight_matrix[index_of_instance, index_of_concept] for i in range(weight_matrix.shape[0]))
    return count / weight_matrix.shape[0]



# 示例权重矩阵
weight_matrix_data = np.array([[1.2, 0.8, 1.5],
                               [0.7, 1.0, 1.2],
                               [1.1, 0.9, 1.3]])



# 示例调用
membership_degree = get_membership_degree(weight_matrix_data, 1, 0)
print(f"Membership Degree: {membership_degree}")
