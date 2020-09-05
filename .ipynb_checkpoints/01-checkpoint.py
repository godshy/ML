import numpy
from collections import defaultdict
D_FILE = 'affinity_dataset.txt'
X = numpy.loadtxt(D_FILE)

valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occur = defaultdict(int)
num_valid = 0
num_invalid = 0
num_bread = 0
num_milk = 0
num_cheese = 0
num_apple = 0
num_banana = 0

for sample in X:
    if sample[3] == 1:
        num_apple += 1
        if sample[4] == 1:
            num_valid += 1
        else:
            num_invalid +=1


print('there are {} people who bought apples'.format(num_apple))
print('invalid is: {}'.format(num_invalid))
print('valid is: {}'.format(num_valid))
