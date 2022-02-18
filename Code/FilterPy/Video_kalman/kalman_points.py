# # Author: Bingzhen Lyu
# # Date: 2022/1/18

import numpy as np


def all_points(all_kalman, landmarks):
    # todo: set variable
    unit_num_state = 6  # x，y，dx，dy,ddx,ddy
    num_point = 33  # 33 Points
    # num_state = unit_num_state * num_point
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
    all_kalman.predict()
    all_kalman.update(current_measurement)

    prediction = all_kalman.x

    j = 0

    for i in range(current_prediction.shape[0]):
        current_prediction[i] = prediction[j][0], prediction[j+1][0]
        j += unit_num_state
    # print("prediction", current_prediction)
    # print("-------------------------------")
    return current_prediction


def wrong_points(all_kalman, old_point):
    # todo: set variable
    unit_num_state = 6  # x，y，dx，dy,ddx,ddy
    num_point = 33  # 33 Points
    # num_state = unit_num_state * num_point
    # num_dimension = 2  # 2 Dimension(x,y)
    # current_measurement = np.ones((num_point*num_dimension, 1))
    current_prediction = np.ones((num_point, 2))

    # j = 0
    # for i in range(num_point):
    #     current_measurement[j] = np.float32(old_point[i][0])
    #     current_measurement[j+1] = np.float32(old_point[i][1])
    #     j += 2

    # print(current_measurement)
    # print("-------------------------------")

    all_kalman.predict()
    prediction = all_kalman.x
    # prediction = all_kalman.get_update()[0]
    # prediction = all_kalman.get_prediction()[0]
    # prediction = all_kalman.get_update(current_measurement)[0]

    j = 0

    for i in range(current_prediction.shape[0]):
        current_prediction[i] = prediction[j][0], prediction[j+1][0]
        j += unit_num_state
    # print("prediction", current_prediction)
    # print("-------------------------------")
    return current_prediction
