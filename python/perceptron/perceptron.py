import logging
import numpy as np
from matplotlib import pyplot as plt
logging.basicConfig(level=logging.INFO)


class Perceptron():
    def __init__(self,
                 data_dimension):
        self.__update_times = 0
        self.__w = np.zeros(shape=data_dimension) + 0.01
        self.__b = np.zeros(shape=1)

    @property
    def update_times(self):
        return self.__update_times

    @property
    def weight(self):
        return self.__w

    @property
    def bias(self):
        return self.__b

    def train(self, dataset, target):
        eta = 0.1
        R = 2
        while self.__update_times < 20:
            last_epoch_update_times = self.update_times
            for i in list(range(dataset.shape[0])):
                if (dataset[i].dot(self.weight) + self.bias) * target[i] <= 0:
                    self.__w = self.__w + eta * target[i] * dataset[i]
                    self.__b = self.__b + eta * target[i] * R * R
                    self.__update_times += 1
            x_p = target > 0
            x_n = target < 0
            plt.scatter(dataset[x_p, 0], dataset[x_p, 1],
                        marker='^', color='b')
            plt.scatter(dataset[x_n, 0], dataset[x_n, 1],
                        marker='o', color='r')

            def f_out(x):
                return -(x * self.weight[0] + self.bias)/self.weight[1]
            dx = np.arange(-1, 4, 0.01)
            print(self.weight)
            print(self.bias)
            print(self.__update_times)
            plt.plot(dx, f_out(dx), color='y')
            plt.show()
            if last_epoch_update_times == self.update_times:
                break


#         # Initial State
# WEIGHT_SHAPE = [3]
# BIAS_SHAPE = [1]
# w = np.zeros(shape=WEIGHT_SHAPE)
# b = np.zeros(shape=BIAS_SHAPE)
# logging.info(w)
# logging.info(b)

# # Predict State
# x = np.array([1, 2, 3])
# logging.info(x)
# logging.info(w*x + b)


# def predict(w,
#             b,
#             x):
#     return w * x + b


# def decision(y_p,
#              y_r):
#     return y_p * y_r

# # Update State


# def w_update_direct(x,
#                     y,
#                     eta):
#     return eta * y * x


# def b_update_direct(r,
#                     y,
#                     eta):
#     return eta * y * r * r


# def perceptron_update(w,
#                       b,
#                       x,
#                       y,
#                       r,
#                       eta):
#     w = w + w_update_direct(x, y, eta)
#     b = b + b_update_direct(r, y, eta)
