# # Author: Bingzhen Lyu
# # Date: 2022/1/18
#

import numpy as np

def all_points(opints_kalman, landmarks):
    # todo: set variable
    unit_num_state = 8  # x，y，dx，dy, ddx, ddy, dddx, dddy
    num_point = 33  # 33 Points
    num_state = unit_num_state * num_point
    num_dimension = 2  # 2 Dimension(x,y)
    current_measurement = np.ones((num_point*num_dimension, 1))
    current_prediction = np.ones((num_point, 2))
    j = 0
    for i in range(num_point):
        current_measurement[j] = np.float32(landmarks[i].x)
        current_measurement[j+1] = np.float32(landmarks[i].y)
        j += 2
    # print(current_measurement)
    # print("-------------------------------")

    opints_kalman.correct(np.array(current_measurement, np.float32))
    prediction = opints_kalman.predict()

    j = 0

    for i in range(current_prediction.shape[0]):
        current_prediction[i] = prediction[j][0], prediction[j+1][0]
        j += unit_num_state
    # print("prediction", current_prediction)
    # print("-------------------------------")
    return current_prediction


#
# -----------33 Kalman in a list----------
# def all_points(list_kalman, landmarks):
#
#     current_prediction = np.ones((33, 2))
#
#     for i in range(len(landmarks)):
#         kalman_i = list_kalman[i][0]
#         # print("kalman", kalman_i)
#         current = np.array([[np.float32(landmarks[i].x)], [np.float32(landmarks[i].y)]])
#         # if i == 11:
#         #     print("current_SHOULDER:", current)
#         kalman_i.correct(current)
#         # print("new_current_measurement", current_measurement)
#         prediction = list_kalman[i][0].predict()
#         current_prediction[i] = prediction[0], prediction[1]
#
#     return current_prediction
