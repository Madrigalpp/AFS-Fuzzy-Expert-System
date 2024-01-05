# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by Wankang_Zhai (zhai@dlmu.edu.cn)
# ------------------------------------------------------------------------------
from get_membership_degree import get_membership_degree

def get_membership_complex_degree(weight_matrix, description, index_of_instance):
    '''
    这个函数计算了每个描述在给定样本索引上的隶属度，然后返回这些隶属度的平均值
    :param ：
    :return:  the same as get_membership_degree
    you should check the get_membership_degree.py
    '''
    degree = 0

    for value in description:
        degree += get_membership_degree(weight_matrix, value, index_of_instance)

    return degree / len(description)




