import numpy as np
from select_attributes_by_fnsn import select_attributes_by_fnsn
from get_selected_concepts import get_selected_concepts
from get_concepts import get_concepts

def generate_descriptions(instances, weight_matrix, neighbour, feature, epsilon):
    descriptions = []
    for i, ins in enumerate(instances):
        attributes = select_attributes_by_fnsn(instances, ins, neighbour, feature)
        concepts = get_concepts(attributes)

        description = get_selected_concepts(weight_matrix, i, concepts, epsilon)
        descriptions.append(description)
        print(f"第{i+1}个样本的描述生成完毕！")

    return descriptions

class Instance:
    def __init__(self, values):
        self.values = values
class Instances:
    def __init__(self, instances, class_index):
        self.instances = instances
        self.class_index = class_index

    def numAttributes(self):
        # 返回实例集合的属性数量
        if self.instances:
            return len(self.instances[0].values) + 1  # 加上类属性
        else:
            return 0

    def __iter__(self):
        return iter(self.instances)

# 创建实例对象
instance_x = Instance([1.0, 2.0, 3.0])
instance_1 = Instance([2.0, 3.0, 4.0])
instance_2 = Instance([3.0, 4.0, 5.0])
instance_3 = Instance([4.0, 5.0, 6.0])

# 创建实例集合对象
instances = Instances([instance_1, instance_2, instance_3], class_index=1)
weight_matrix = [
    [0.2],
    [0.6],
    [0.4]
]
weight_matrix = np.array(weight_matrix)
neighbour = 2
feature = 2
epsilon = 0.1
descriptions = generate_descriptions(instances, weight_matrix, neighbour, feature, epsilon)
