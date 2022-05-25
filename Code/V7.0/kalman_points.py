# # Author: Bingzhen Lyu
# # Date: 2022/1/18

import numpy as np
import time
import set_kalman


def all_points(kalman1, kalman2, prevlandmarks, landmarks):
    # todo: set variable
    unit_num_state = 4  # x，y，dx，dy
    num_point = 4  # 4 Points for each Kalman
    # num_state = unit_num_state * num_point
    num_dimension = 2  # 2 Dimension(x,y)

    stabel_current_measurement = np.ones((num_point*num_dimension, 1))
    stabel_last_measurement = np.ones((num_point*num_dimension, 1))

    instabilityl_current_measurement = np.ones((num_point*num_dimension, 1))
    instability_last_measurement = np.ones((num_point*num_dimension, 1))

    current_prediction =  np.ones((num_point*2, 2))
    # todo : Determine if Mediapipe is right
    right = 1
    j = 0
    point_index = [11, 12, 23, 24, 25, 26, 27, 28]  # 8testpoint
    for i in point_index:
        if i < 25:
            stabel_current_measurement[j] = np.float32(landmarks[i].x)
            stabel_current_measurement[j+1] = np.float32(landmarks[i].y)
            stabel_last_measurement[j] = np.float32(prevlandmarks[i].x)
            stabel_last_measurement[j+1] = np.float32(prevlandmarks[i].y)
            j += 2
    j = 0
    for i in point_index:
        if i > 24:
            instabilityl_current_measurement[j] = np.float32(landmarks[i].x)
            instabilityl_current_measurement[j+1] = np.float32(landmarks[i].y)
            instability_last_measurement[j] = np.float32(prevlandmarks[i].x)
            instability_last_measurement[j+1] = np.float32(prevlandmarks[i].y)
            j += 2

    # change Q

    for i in range(4):

        a = kalman1.x[2+i*4][0]
        b = kalman1.x[3+i*4][0]

        if abs(a) > 0.03 or abs(b) > 0.1:  #
            right = 0
            break
    for i in range(4):
        a = kalman2.x[2+i*4][0]
        b = kalman2.x[3+i*4][0]

        if abs(a) > 0.03 or abs(b) > 0.1:  #
            right = 0
            # print(abs(a))
            # print(abs(b))
            # print(i)
            break
    # if right ==0:
        # print("!")
    set_kalman.resetq(kalman1, right)
    set_kalman.resetq(kalman2, right)
    kalman1.predict()
    kalman2.predict()
    kalman1.update(stabel_current_measurement)
    kalman2.update(instabilityl_current_measurement)

    stabe_prediction = kalman1.x
    instability_prediciton=kalman2.x

    j = 0

    for i in range(4):
        current_prediction[i] = stabe_prediction[j][0], stabe_prediction[j+1][0]
        j += unit_num_state
    j=0
    for i in range(4,8):
        current_prediction[i] = instability_prediciton[j][0], instability_prediciton[j+1][0]
        j += unit_num_state
    #             curret_prediction[i] = instability_prediciton[j][0], instability_prediciton[j + 1][0]
    # print("prediction", current_prediction)
    # print("-------------------------------")
    prevlandmarks = landmarks
    return current_prediction, prevlandmarks

def frist_x(kalman1,kalman2, landmarks):
    unit_num_state = 4  # x，y，dx，dy
    num_point = 4  # 4 Points for each Kalman
    # num_state = unit_num_state * num_point
    num_dimension = 2  # 2 Dimension(x,y)

    stabel_current_measurement = np.ones((num_point * num_dimension, 1))
    instabilityl_current_measurement = np.ones((num_point*num_dimension, 1))
    j = 0
    point_index = [11, 12, 23, 24, 25, 26, 27, 28]  # 8 testpoint
    for i in point_index:
        if i < 25:
            kalman1.x[j] = np.float32(landmarks[i].x)
            kalman1.x[j + 1] = np.float32(landmarks[i].y)

            j += 2
    j = 0
    for i in point_index:
        if i > 24:
            kalman2.x[j] = np.float32(landmarks[i].x)
            kalman2.x[j + 1] = np.float32(landmarks[i].y)
            j += 2


def resetkalman2(kalman2, prevlandmarks):
    num_point = 4  # 4 Points for each Kalman
    num_dimension = 2  # 2 Dimension(x,y)

    instabilityl_current_measurement = np.ones((num_point * num_dimension, 1))
    j = 0
    point_index = [25, 26, 27, 28]  # 8testpoint

    for i in point_index:
        kalman2.x[j] = np.float32(prevlandmarks[i].x)
        kalman2.x[j+1] = np.float32(prevlandmarks[i].y)
        j += num_point
