# Author: Bingzhen Lyu
# Date: 2022/1/14
# Kalman
import numpy as np
import math
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Todo:set variable
unit_num_state = 6  # x，y，dx，dy,ddy,ddy
num_point = 1  # 8 Points of test Point
num_state = unit_num_state * num_point
num_dimension = 2  # 2 Dimension(x,y)
t = 1/30  # delta t =1/30
c = 0.028  # Change of acceleration
n_diff = 2  # Number of derivation

# TODO: set unit D
d = np.zeros((unit_num_state, unit_num_state))
d[0][2] = 1
d[1][3] = 1
d[2][4] = 1
d[3][5] = 1
# d[4][6] = 1
# d[5][7] = 1

def set_kalman_all():

    return kalmanfilter(num_state, num_point, num_dimension, unit_num_state, n_diff, d, t, c)

# ------set KalmanFilter--------


""" Kalman Filter
     Refer to http:/github.com/rlabbe/filterpy
     """


def kalmanfilter(num_state, num_point, num_dimension, unit_num_state, n_diff, d, t, c):
    # TODO: Set KalmanFilter

    kalman = KalmanFilter(num_state, num_dimension*num_point)

    # TODO: set Initial state x
    x = get_x(num_state, unit_num_state, num_dimension)
    kalman.x = np.array(x)
    # TODO: set transitionMatrix F
    #
    unif = unit_f(d, n_diff, t)
    f = get_f(unif, num_point)
    kalman.F = np.array(f)
    # TODO: set measurementMatrix H
    h = get_h(num_point, num_state, unit_num_state, num_dimension)
    kalman.H = np.array(h)

    # TODO: set covariance Matrix P
    p = np.eye(num_state)*(c**2)
    kalman.P = np.array(p)

    # TODO: set uncertainty R
    # r = get_r(c, num_point, num_dimension)
    kalman.R = np.array(np.eye(num_point * num_dimension) * (c**2))
    # TODO: Set processNoiseCov Q
    # q = get_q(unif, num_point, c)
    # kalman.Q = np.array(q)
    # kalman.Q = Q_discrete_white_noise(num_dimension, t, c, block_size=num_point)

    kalman.B = 0
    kalman.Q = Q_discrete_white_noise(3, t, c, 2)
    #
    # print("D:", d)
    print("Initial state x:\n", kalman.x)
    print("transitionMatrix F: \n", kalman.F)
    print("measurementMatrix H: \n", kalman.H)
    print("covariance Matrix P:\n", kalman.P)
    print("uncertainty R:\n", kalman.R)
    print("processNoiseCov Q:\n", kalman.Q)
    return kalman

# ------------calculation process-------------#


def matrixpow(matrix, n):
    if type(matrix) == list:
        matrix = np.array(matrix)
    if n == 1:
        return matrix
    else:
        return np.matmul(matrix, matrixpow(matrix, n - 1))


def unit_f(unit_d, n_diff, t):  # unit_d, Number of derivation , delta_t
    unif = 0
    for i in range(1, n_diff+1):
        unif = unif + (matrixpow((unit_d * t), i)) / math.factorial(i)
    unif = unif+np.eye(unit_d.shape[0])
    # print("f = ")
    # print(f)
    return unif


def get_f(unif, n_point):  # n number of point
    a = n_point*(unif.shape[0])
    b = n_point*(unif.shape[0])

    new_f = np.zeros((a, b))
    for i in range(0, new_f.shape[0], unif.shape[0]):
        for j in range(0, new_f.shape[1], unif.shape[1]):
            if i == j:
                for k in range(unif.shape[0]):
                    for g in range(unif.shape[1]):
                        new_f[i+k][j+g] = unif[k][g]
    return new_f


def get_r(unif):  # n=number of state

    uni_r = np.zeros((unif.shape[0], 1))  #
    for i in range(uni_r.shape[0]):
        uni_r[uni_r.shape[0]-1-i][0] = unif[unif.shape[1]-1-i][unif.shape[0]-1]
    return uni_r



def get_q(unif, n_point, c):
    unir = get_r(unif)
    q1 = unir*unir.T
    # print("q1\n", q1)
    # print("-------------")
    q = np.zeros((n_point*unif.shape[0], n_point*unif.shape[0]))

    for i in range(0, q.shape[0], q1.shape[0]):  #
        for j in range(0, q.shape[1], q1.shape[1]):
            if i == j:
                for k in range(q1.shape[0]):
                    for g in range(q1.shape[1]):
                        if k == g:
                            if q1[k][g] == 0:
                                q[i+k][j+g] = 1
                                continue
                        q[i+k][j+g] = q1[k][g]

    q = q * c**2
    return q


def get_h(num_point, num_state_total, num_state_each, num_dimension):
    h = np.zeros((num_point*num_dimension, num_state_total))

    i = 0
    if num_dimension == 1:
        for j in range(0, h.shape[1], num_state_each):
            h[i][j] = 1
            i += num_dimension

    if num_dimension == 2:
        for j in range(0, h.shape[1], num_state_each):
            h[i][j] = 1
            h[i+1][j+1] = 1
            i += num_dimension
    # print("h:\n", h)
    return h


def get_x(num_state, unit_num_state, num_dimension):
    x = np.zeros((num_state, 1))

    for i in range(0, x.shape[0], unit_num_state):
        for j in range(num_dimension):
            x[i+j][0] = 1
    return x
