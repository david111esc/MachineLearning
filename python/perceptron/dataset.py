import numpy as np
from matplotlib import pyplot as plt


class DataSet():
    def __init__(self,
                 data_train,
                 target_train,
                 data_val=None,
                 target_val=None):
        self.__data_train = data_train
        self.__target_train = target_train
        self.__data_val = data_val
        self.__target_val = target_val

    @property
    def data_train(self):
        return self.__data_train

    @property
    def target_train(self):
        return self.__target_train

    @property
    def data_val(self):
        return self.__data_val

    @property
    def target_val(self):
        return self.__target_val


x = np.array([[0, 1.2],   [0, 2.3], [0.6, 2], [0.4, 3.1],
              [0.9, 2.2], [1.1, 3.6], [1.2, 2.7], [1.6, 3.1],
              [2.1, 2.7], [1.9, 3.3], [2.1, 3.8], [3, 4.1],

              [1.6, 0.3], [1.3, 0.8], [3.7, 1.8], [0.9, 0.1],
              [2.3, 0.7], [2.7, 0.8],  [2.9, 1.8], [3.9, 2.8],
              [3.3, 1.4],  [2.8, 1.3], [3.2, 0.1], [0.3, 0]])
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
x_p = list(range(0, 12, 1))
x_n = list(range(12, 24, 1))

plt.scatter(x[x_p, 0], x[x_p, 1], marker='^', color='b')
plt.scatter(x[x_n, 0], x[x_n, 1], marker='o', color='r')
plt.show()
