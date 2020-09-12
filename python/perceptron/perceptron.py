import logging
import numpy as np
import matplotlib
logging.basicConfig(level=logging.INFO)


# class Perceptron():
#     def


# Initial State
WEIGHT_SHAPE = [3]
BIAS_SHAPE = [1]
w = np.zeros(shape=WEIGHT_SHAPE)
b = np.zeros(shape=BIAS_SHAPE)
logging.info(w)
logging.info(b)

# Predict State
x = np.array([1, 2, 3])
logging.info(x)
logging.info(w*x + b)


def predict(w,
            b,
            x):
    return w * x + b


def decision(y_p,
             y_r):
    return y_p * y_r

# Update State


def w_update_direct(x,
                    y,
                    eta):
    return eta * y * x


def b_update_direct(r,
                    y,
                    eta):
    return eta * y * r * r


def perceptron_update(w,
                      b,
                      x,
                      y,
                      r,
                      eta):
    w = w + w_update_direct(x, y, eta)
    b = b + b_update_direct(r, y, eta)
