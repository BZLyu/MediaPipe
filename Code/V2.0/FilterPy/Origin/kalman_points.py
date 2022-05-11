# # Author: Bingzhen Lyu
# # Date: 2022/1/18

import numpy as np


def all_points(all_kalman, landmarks):
    # todo: set variable
    unit_num_state = 4  # x，y，dx，dy
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


# -----------------------------Points Kalman-------------
# def hand_points(up_kalman, down_kalman, landmarks, last_wrist, accesspoints_up,
#                 accesspoints_down, change_up, change_down):
#     # todo: set variable
#     unit_num_state = 4  # x，y，dx，dy
#     num_point = 4  # 4 Points of hand
#     # num_state = unit_num_state * num_point
#     num_dimension = 2  # 2 Dimension(x,y)
#     current_measurement = np.ones((num_point*num_dimension, 1))
#     current_prediction = np.ones((num_point, 2))
#
#     # 15 left_wrist-current
#     current_measurement[0] = np.float32(landmarks[15].x)
#     current_measurement[1] = np.float32(landmarks[15].y)
#     # 17 left_pinky-current
#     current_measurement[2] = np.float32(landmarks[17].x)
#     current_measurement[3] = np.float32(landmarks[17].y)
#     # 19 left_index-current
#     current_measurement[4] = np.float32(landmarks[19].x)
#     current_measurement[5] = np.float32(landmarks[19].y)
#     # 21 left_thumb-current
#     current_measurement[6] = np.float32(landmarks[21].x)
#     current_measurement[7] = np.float32(landmarks[21].y)
#
#     # print("current_measurement", current_measurement)
#     # print("-------------------------------")
#     # todo: Switch kalman
#     # last elbow_y - current elbow_y . Check if it is greater or less than 0
#     # >0 down, <=0 up
#
#     current_elbow_y = landmarks[15].y
#
#     if change_down is False and change_up is False:  # Initial
#         accesspoints_down = accesspoints_up = current_measurement
#     direction = last_wrist - current_elbow_y
#     error = abs(direction) / current_elbow_y
#     # print("direction：", direction)
#     # print("error:", error)
#     if direction <= 0 and error > 0.005:  # up
#         if change_up is False:  #
#             print("change to up! ")
#             accesspoints_up = current_measurement
#             change_down = False
#             change_up = True
#             down_kalman.correct(np.array(accesspoints_down, np.float32))
#
#         up_kalman.correct(np.array(current_measurement, np.float32))
#         prediction = up_kalman.predict()
#
#     else:  # down
#         if change_down is False:  #
#             print("change to down!")
#             accesspoints_down = current_measurement
#             change_up = False
#             change_down = True
#             up_kalman.correct(np.array(accesspoints_up, np.float32))
#
#         down_kalman.correct(np.array(current_measurement, np.float32))
#         prediction = down_kalman.predict()
#
#     j = 0
#     for i in range(current_prediction.shape[0]):
#         current_prediction[i] = prediction[j][0], prediction[j+1][0]
#         j += unit_num_state
#
#     # print("prediction", current_prediction)
#     # print("-------------------------------")
#
#     last_wrist = current_elbow_y
#
#     return current_prediction, last_wrist, accesspoints_up, accesspoints_down, change_up, change_down


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
