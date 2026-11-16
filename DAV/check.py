# import pickle

# with open('data2.pickle', 'rb') as f:
#     d = pickle.load(f)

# print('samples:', len(d['data2']))
# print('classes:', sorted(set(d['labels'])))
# print('feature lengths:', set(len(x) for x in d['data2']))


import pickle
from collections import Counter

d = pickle.load(open('data2.pickle', 'rb'))
print('samples per class:', Counter(d['labels']))
print('feature lengths:', set(len(x) for x in d['data2']))