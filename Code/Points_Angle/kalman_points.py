# Author: Bingzhen Lyu 
# Date: 2022/1/18

import numpy as np


def all_points(list_kalman, landmarks):

    current_prediction = np.ones((33, 2))

    for i in range(len(landmarks)):
        kalman_i = list_kalman[i][0]
        # print("kalman", kalman_i)
        current = np.array([[np.float32(landmarks[i].x)], [np.float32(landmarks[i].y)]])
        # if i == 11:
        #     print("current_SHOULDER:", current)
        kalman_i.correct(current)
        # print("new_current_measurement", current_measurement)
        prediction = list_kalman[i][0].predict()
        current_prediction[i] = prediction[0], prediction[1]

    return current_prediction
