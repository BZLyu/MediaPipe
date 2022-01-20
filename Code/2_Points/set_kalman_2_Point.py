# Author: Bingzhen Lyu 
# Date: 2022/1/14
# Kalman
import cv2
import numpy as np
import math


def set_kalman():
    # TODO: Set KalmanFilter
    kalman = cv2.KalmanFilter(8, 4)  # x1,y1,dx1,dy1,x2,y3,dx2,dy2
    # TODO: set D
    d = np.zeros((4, 4))
    d[0][2] = 1
    d[1][3] = 1
    # D[4][6]=1
    # D[5][7]=1
    #   print("D = ")
    # print(D)
    # TODO: Reset D:
    d = new_d(d, 2)  # 2 is number of point
    # TODO: set measurementMatrix H
    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]], np.float32)
    # kalman.measurementMatrix = np.array(get_h(4, 8), np.float32)  # get_H(num_measurement,num_state):

    # TODO: set transitionMatrix F
    # kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    f = get_f(d, 1, 1)  # (D,n,t)

    kalman.transitionMatrix = np.array(f, np.float32)
    # kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # TODO: Set processNoiseCov Q
    kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 1]], np.float32)  # * 0.03
    # q = get_q(d, f) + np.eye(4)
    # kalman.processNoiseCov = np.array(q, np.float32) * 0.03

    print("D:", d)
    print("measurementMatrix H:", kalman.measurementMatrix)
    print("transitionMatrix F: ", kalman.transitionMatrix)
    print("processNoiseCov Q:", kalman.processNoiseCov)
    return kalman

# D ^n:


def matrixpow(matrix, n):
    if type(matrix) == list:
        matrix = np.array(matrix)
    if n == 1:
        return matrix
    else:
        return np.matmul(matrix, matrixpow(matrix, n - 1))
# F


def get_f(d, n, t):  # D, Number of Measurement, delta_t
    f = 0
    for i in range(1, n+1):
        f = f + (matrixpow((d * t), i)) / math.factorial(i)
    f = f+np.eye(d.shape[0])
    # print("f = ")
    # print(f)
    return f

# H


def get_h(num_measurement, num_state):
    h = np.zeros((num_measurement, num_state))
    # print(h)
    for i in range(num_state):
        for j in range(num_measurement):
            if i == j:
                h[i][j] = 1
                break
    return h

# Q :


def get_q(d, f):
    r = np.zeros((f.shape[1], 1))
    # print(F[[F.shape[1]-4],[F.shape[1]-1]])
    # print(F)
    # print("r.shape",r.shape)
    copyline = d.shape[0]
    # print("copyline:",copyline)
    i = 0
    j = 0

    while i < r.shape[0]:
        if j < copyline:
            #   print("i",i,"j",j)
            #     print(F[[F.shape[1]-4+j],[F.shape[1]-1]])
            r[[i], [0]] = f[[f.shape[1] - 4 + j], [f.shape[1] - 1]]
            j = j+1
            i = i+1
        else:
            j = 0

    #      np.array(r)[0][i]=np.array(F)[i][F.shape[1]-1]
    # print("r = ")
    # print(r)
    #   print("r_T = ")
    #    print(r.T)
    q = r*r.T
    #    print("q =")
    #    print (q)

    return q
# 合并

# 33*4=132
# Z= np.zeros((132,132))


def new_d(d, n):  # n number of point
    #   print("D:")
    #   print(D)
    z = np.zeros((d.shape[0] * n, d.shape[0] * n))
    #    print("new D is ",D.shape[0]*n,"*",D.shape[0]*n,"Matrix")
    first_one = 0
    # second_one = 0
    i = 0
    b = 0
    while i < z.shape[0]:

        for j in range(z.shape[1]):
            if i == j:
                if first_one == 0:
                    b = j+2
                    z[i][b] = 1
                    first_one = 1

                    # print("i=",i)
                    #  print("j=",b)
                    # print("第一个 1：")
                    # print(z)

                    i = i+1
                    break

            if first_one == 1:
                b += 1
                z[i][b] = 1
                i = i+3
                first_one = 0

                # print("i=",i)
                # print("j=",b)
                # print("第二个 1：")
                # print(z)

                if i >= z.shape[0]-1 & b >= z.shape[1]-1:
                    # print("new D: ")
                    # print(z)
                    return z
                b = 0
                break
