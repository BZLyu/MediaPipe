# # Author: Bingzhen Lyu
# # Date: 2022/1/18

import numpy as np
import time
import set_kalman


def all_points(all_kalman, prevlandmarks, landmarks):
    # todo: set variable
    unit_num_state = 4  # x，y，dx，dy
    num_point = 8  # 8 test Points
    # num_state = unit_num_state * num_point
    num_dimension = 2  # 2 Dimension(x,y)

    current_measurement = np.ones((num_point*num_dimension, 1))
    last_measurement = np.ones((num_point*num_dimension, 1))

    current_prediction = np.ones((num_point, 2))

    # todo : Determine if Mediapipe is right
    right = 1
    j = 0
    point_index = [11, 12, 23, 24, 25, 26, 27, 28]  # 8testpoint
    for i in point_index:
        current_measurement[j] = np.float32(landmarks[i].x)
        current_measurement[j+1] = np.float32(landmarks[i].y)
        j += 2

    j = 0
    for i in point_index:
        last_measurement[j] = np.float32(prevlandmarks[i].x)
        last_measurement[j+1] = np.float32(prevlandmarks[i].y)
        j += 2
    # change Q

    for i in range(len(point_index)):

        a = all_kalman.x[2+i*4][0]
        b = all_kalman.x[3+i*4][0]

        if abs(a) > 0.03 or abs(b) > 0.1:  #
            right = 0
            # print(abs(a))
            # print(abs(b))
            # print(i)
            break
    # if right ==0:
        # print("!")
    set_kalman.resetq(all_kalman, right)
    all_kalman.predict()
    all_kalman.update(current_measurement)

    prediction = all_kalman.x

    j = 0

    for i in range(current_prediction.shape[0]):
        current_prediction[i] = prediction[j][0], prediction[j+1][0]
        j += unit_num_state
    # print("prediction", current_prediction)
    # print("-------------------------------")
    prevlandmarks = landmarks
    return current_prediction, prevlandmarks

def frist_x(all_kalman, landmarks):
    unit_num_state = 4  # x，y，dx，dy
    num_point = 8  # 8 test Points
    # num_state = unit_num_state * num_point
    num_dimension = 2  # 2 Dimension(x,y)

    j = 0
    point_index = [11, 12, 23, 24, 25, 26, 27, 28]  # 8testpoint
    for i in point_index:
        all_kalman.x[j] = np.float32(landmarks[i].x)
        all_kalman.x[j + 1] = np.float32(landmarks[i].y)
        j += 2
