from getSalientValue import get_salient_value

def select_attributes_by_fnsn(instances, instance, neighbour, feature):
    salient_map = {}

    flag = 0
    for i in range(instances.numAttributes()):
        if i == instances.class_index:
            continue
        else:
            salient_value = get_salient_value(instances, instance, flag, neighbour)
            salient_map[i] = salient_value
            flag += 1

    # 对 salient_map 进行按 value 降序排列
    sorted_salient_map = sorted(salient_map.items(), key=lambda x: x[1], reverse=True)

    result = [entry[0] for entry in sorted_salient_map[:feature]]
    return result



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
            return len(self.instances[1].values) # 加上类属性
        else:
            return 0

    def __iter__(self):
        return iter(self.instances)

# 创建实例对象
instance_1 = Instance([1.0, 2.0, 3.0])
instance_2 = Instance([2.0, 3.0, 4.0])
instance_3 = Instance([3.0, 4.0, 5.0])
instance_x = Instance([1.0, 2.0, 3.0])

# 创建 Instances 对象
instances_list = [instance_1, instance_2, instance_3]
class_index = 2  # 替换为实际的类索引
instances = Instances(instances_list, class_index)
print(instances.numAttributes())


neighbour = 2
feature = 2
selected_attributes = select_attributes_by_fnsn(instances, instance_x, neighbour, feature)

# 打印选择的属性
print("Selected Attributes:", selected_attributes)
