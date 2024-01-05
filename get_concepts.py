import numpy as np
def get_concepts(selected_attributes):
    result = []

    for i in selected_attributes:
        result.extend([i * 3, i * 3 + 1, i * 3 + 2])

    return result

