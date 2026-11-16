# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data2.pickle', 'rb'))

# data2 = np.asarray(data_dict['data2'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data2, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()

import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data2.pickle', 'rb'))

data2 = np.asarray(data_dict['data2'])
labels = np.asarray(data_dict['labels'])
print('data shape:', data2.shape)   # should be (300, 84)

x_train, x_test, y_train, y_test = train_test_split(
    data2, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print('accuracy: {}%'.format(accuracy_score(y_predict, y_test) * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)