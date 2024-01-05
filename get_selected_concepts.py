import numpy  as np
from get_membership_degree import get_membership_degree

def get_selected_concepts(weight_matrix, index_of_instance, concept, epsilon):
    degrees = {}

    for index_of_concept in concept:
        degrees[index_of_concept] = get_membership_degree(weight_matrix, index_of_concept, index_of_instance)

    new_concept = []
    flag = 0
    degree_max = float('-inf')
    index_max = 0

    for c in concept:
        flag += 1
        if degrees[c] > degree_max:
            degree_max = degrees[c]
            index_max = c

        if flag == 3:
            new_concept.append(index_max)
            flag = 0
            degree_max = float('-inf')
            index_max = 0

    max_degree = float('-inf')

    for c in new_concept:
        if degrees[c] > max_degree:
            max_degree = degrees[c]

    result = [c for c in new_concept if degrees[c] > (max_degree - epsilon)]

    return result
# weight_matrix = [
#     [0.2, 0.5, 0.8],
#     [0.6, 0.3, 0.1],
#     [0.4, 0.7, 0.9]
# ]
# weight_matrix = np.array(weight_matrix)
#
# index_of_instance = 1
# concept = [0, 1, 2]  # 假设有9个concept
# epsilon = 0.1
# result = get_selected_concepts(weight_matrix, index_of_instance, concept, epsilon)
#
# print(result)