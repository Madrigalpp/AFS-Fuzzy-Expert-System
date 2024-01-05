# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by Wankang_Zhai (zhai@dlmu.edu.cn)
# ------------------------------------------------------------------------------
class TriangularFunction:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def apply(self, x):
        # 确认模糊函数的类别
        if self.a != self.b and self.b != self.c and self.a != self.c:  # class 1: a! = b != c
            if x < self.a or x > self.c:
                return 0.0
            elif x == self.a or x == self.c:
                return 0.0
            elif x == self.b:
                return 1.0
            elif self.a < x < self.b:
                return (x - self.a) / (self.b - self.a)
            elif self.b < x < self.c:
                return (self.c - x) / (self.c - self.b)

        elif self.a == self.b and self.b != self.c:  # class 2: a = b != c
            if x == self.a:
                return 1.0
            elif x < self.a:
                return 1.0
            elif x == self.c:
                return 0.0
            elif x > self.c:
                return 0.0
            elif self.b < x < self.c:
                return (self.c - x) / (self.c - self.b)

        elif self.a != self.b and self.b == self.c:  # class 3: a != b = c
            if x < self.a:
                return 0.0
            elif x == self.a:
                return 0.0
            elif x == self.c:
                return 1.0
            elif x > self.b:
                return 1.0
            elif self.a < x < self.b:
                return (x - self.a) / (self.b - self.a)

        # 如果计算错误，返回double的最大值
        return float('inf')

