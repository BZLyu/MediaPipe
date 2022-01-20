# Author: Bingzhen Lyu 
# Date: 2022/1/14
# Kalman
import cv2
import numpy as np


def set_kalman_angle():
    kalman = cv2.KalmanFilter(4, 1)  # (dynamparams,measureparams)
    delta_t = 0.3
    c = 1
    # measurementMatrix H
    kalman.measurementMatrix = np.array([[1, 0, 0, 0]], np.float32)

    # transitionMatrix F
    kalman.transitionMatrix = np.array([[1, delta_t, 0.5*delta_t**2, (1/6)*delta_t**3], [0, 1, delta_t, 0.5*delta_t**2],
                                        [0, 0, 1, delta_t], [0, 0, 0, 1]], np.float32)
    # processNoiseCov Q
    kalman.processNoiseCov = np.array([[(1/36)*delta_t**6, (1/12)*delta_t**5, (1/6)*delta_t**4, (1/6)*delta_t**3],
                                       [(1/12)*delta_t**5, (1/4)*delta_t**4, (1/2)*delta_t**3, (1/2)*delta_t**2],
                                       [(1/6)*delta_t**4, 0.5*delta_t**3, delta_t**2, delta_t],
                                       [(1/6)*delta_t, 0.5*delta_t**2, delta_t, 1]], np.float32) * c**2

    # print("measurementMatrix", kalman.measurementMatrix)
    # print("transitionMatrix", kalman.transitionMatrix)
    # print("processNoiseCov", kalman.processNoiseCov)

    return kalman
