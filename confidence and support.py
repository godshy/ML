# 计算买了A产品再买B产品的概率，即支持度和置信度，再计算发生的概率。

import numpy
from collections import defaultdict

feature = ['bread', 'milk', 'cheese', 'apple', 'banana']
D_FILE = 'affinity_dataset.txt'
X = numpy.loadtxt(D_FILE)

valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occur = defaultdict(int)

for sample in X:
    for product in range(5):
        if sample[product] == 0:  # 如果第一个买的东西值为0，继续查找。
            continue
        else:
            num_occur[product] += 1
            for conclusion in range(5):
                if conclusion == product:
                    continue
                elif sample[conclusion] == 0:
                    invalid_rules[product, conclusion] += 1
                else:
                    valid_rules[product, conclusion] += 1

# 这个字典的valid格式为 key：（买了一个是哪个东西，买了第二个是什么东西）发生的次数
support = valid_rules
confidence = defaultdict(float)

for product, conclusion in valid_rules.keys():  # 计算置信度
    rule = (product, conclusion)
    confidence[rule] = valid_rules[rule] / num_occur[product]


def compute(x, y):
    x = int(x)
    y = int(y)
    product_a = feature[x]
    product_b = feature[y]
    print('people who bought {0} also bought {1}'.format(product_a, product_b))
    print('the support value is %d, and confidence value is %.3f' % (support[(x, y)], confidence[(x, y)]))

# 两种格式化方式，f表示浮点数，3表示小数点后3位


for a in range(0, 5):
    for b in range(0, 5):
        if a != b:
            print(compute(a, b))
        else:
            continue

list_a = []
for k, v in support.items():
    tu = (v, k)
    list_a.append(tu)

for k, v in sorted(list_a, reverse=True):
    print(v, k)

list_b = []
for k, v in confidence.items():
    tv = (v, k)
    list_b.append(tv)

for k, v in sorted(list_b, reverse=True):
    print(v, k)
