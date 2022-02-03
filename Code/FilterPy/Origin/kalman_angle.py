# Author: Bingzhen Lyu 
# Date: 2022/1/18

import numpy as np
import math


def get_elbow_angle(elbow_angle_kalman, angle):

    current_measurement = np.array(np.float32(angle))
    elbow_angle_kalman.correct(current_measurement)
    # print('measurement_angle: ', current_measurement)
    current_prediction = elbow_angle_kalman.predict()
    # print(current_prediction)
    new_angle = float(current_prediction[0])
    # print('prediction_angle: ', new_angle)

    return new_angle


# -----------------------------2 Kalman hand -------------
def hand_angle_2kalman(up_kalman, down_kalman, landmarks, last_wrist, accesspoints_up, accesspoints_down,
                       change_up, change_down):
    # todo: set variable
    num_point = 4  # 4 Points of hand
    num_dimension = 1  # 1 Angle
    current_measurement = np.ones((num_point*num_dimension, 1))
    rad_prediction = np.ones((num_point*num_dimension, 1))
    current_prediction = np.ones((num_point, 2))

    # 15 left_wrist-current
    y, x = landmarks[15].y, landmarks[15].x
    print("x:", x, "y:", y)
    rad = math.atan2(y, x)
    # angle = (np.abs(rad * 180.0/np.pi))
    current_measurement[0] = np.float32(rad)

    # 17 left_pinky-current
    y, x = landmarks[17].y, landmarks[17].x
    print("x:", x, "y:", y)
    rad = math.atan2(y, x)
    # angle = (np.abs(rad * 180.0/np.pi))
    current_measurement[1] = np.float32(rad)

    # 19 left_index-current
    y, x = landmarks[19].y, landmarks[19].x
    print("x:", x, "y:", y)
    rad = math.atan2(y, x)
    # angle = (np.abs(rad * 180.0/np.pi))
    current_measurement[2] = np.float32(rad)

    # 21 left_thumb-current
    y, x = landmarks[21].y, landmarks[21].x
    print("x:", x, "y:", y)
    rad = math.atan2(y, x)
    # angle = (np.abs(rad * 180.0/np.pi))
    current_measurement[3] = np.float32(rad)

    # print("current_measurement", current_measurement)
    # print("-------------------------------")
    # todo: Switch kalman
    # last elbow_y - current elbow_y. Check if it is greater or less than 0
    # >0 down, <=0 up

    current_elbow_y = landmarks[15].y

    if change_down is False and change_up is False:  # Initial
        accesspoints_down = accesspoints_up = current_measurement
    direction = last_wrist - current_elbow_y
    last_wrist = current_elbow_y
    error = abs(direction) / last_wrist
    # print("direction：", direction)
    # print("error:", error)
    if direction <= 0:  # up
        if change_up is False:  #
            print("change to up! ")
            up_kalman.correct(np.array(accesspoints_up, np.float32))

            print("last_accesspoints_up:", accesspoints_up)
            accesspoints_up = current_measurement
            print("current_accesspoints_up", accesspoints_up)

            up_kalman.correct(np.array(current_measurement, np.float32))

            change_down = False
            change_up = True
            down_kalman.correct(np.array(accesspoints_down, np.float32))
        else:
            up_kalman.correct(np.array(current_measurement, np.float32))
            down_kalman.correct(np.array(accesspoints_down, np.float32))
        print("up_prediction:")
        prediction = up_kalman.predict()

    else:  # down
        if change_down is False:  #
            print("change to down!")
            down_kalman.correct(np.array(accesspoints_down, np.float32))
            print("last_accesspoints_down:", accesspoints_down)
            accesspoints_down = current_measurement
            print("current_accesspoints_down", accesspoints_down)
            down_kalman.correct(np.array(accesspoints_down, np.float32))
            change_up = False
            change_down = True
            up_kalman.correct(np.array(accesspoints_up, np.float32))
        else:
            down_kalman.correct(np.array(current_measurement, np.float32))
            up_kalman.correct(np.array(accesspoints_up, np.float32))
        print("down_prediction:")
        prediction = down_kalman.predict()

    j = 0
    for i in range(rad_prediction.shape[0]):
        rad_prediction[i] = prediction[j]
        j += 2
    # print("prediction:\n", rad_prediction)

    #     # 15 left_wrist-current
    # x = rcos（θ), # y = rsin（θ）
    j = 0
    r = math.sqrt((landmarks[15].x**2)+(landmarks[15].y**2))
    current_prediction[j] = r*math.cos(rad_prediction[j]), r*math.sin(rad_prediction[j])
    j += 1
    #     # 17 left_pinky-current
    r = math.sqrt((landmarks[17].x**2)+(landmarks[17].y**2))
    current_prediction[j] = r*math.cos(rad_prediction[j]), r*math.sin(rad_prediction[j])
    j += 1
    #     # 19 left_index-current
    r = math.sqrt((landmarks[19].x**2)+(landmarks[19].y**2))
    current_prediction[j] = r*math.cos(rad_prediction[j]), r*math.sin(rad_prediction[j])
    j += 1
    #     # 21 left_thumb-current
    r = math.sqrt((landmarks[21].x**2)+(landmarks[21].y**2))
    current_prediction[j] = r*math.cos(rad_prediction[j]), r*math.sin(rad_prediction[j])
    # print("current_prediction:\n", current_prediction)

    print("prediction", current_prediction)
    print("-------------------------------")


    return current_prediction, last_wrist, accesspoints_up, accesspoints_down, change_up, change_down


# --------------------1 Kalman-hand-Angle---------------------------
# def get_hand_angle(hand_kalman, landmarks):
#
#     # todo: set variable
#     # unit_num_state = 2  # x，y，dx，dy
#     num_point = 4  # 4 Points of hand
#     # num_state = unit_num_state * num_point
#     num_dimension = 1  # 1 Angle
#     current_measurement = np.ones((num_point*num_dimension, 1))
#     current_prediction = np.ones((num_point, 2))
#
#     # 15 left_wrist-current
#     rad = math.atan2(landmarks[15].y, landmarks[15].x)
#     # angle = (np.abs(rad * 180.0/np.pi))
#     current_measurement[0] = np.float32(rad)
#
#     # 17 left_pinky-current
#     rad = math.atan2(landmarks[17].y, landmarks[17].x)
#     # angle = (np.abs(rad * 180.0/np.pi))
#     current_measurement[1] = np.float32(rad)
#
#     # 19 left_index-current
#     rad = math.atan2(landmarks[19].y, landmarks[19].x)
#     # angle = (np.abs(rad * 180.0/np.pi))
#     current_measurement[2] = np.float32(rad)
#
#     # 21 left_thumb-current
#     rad = math.atan2(landmarks[21].y, landmarks[21].x)
#     # angle = (np.abs(rad * 180.0/np.pi))
#     current_measurement[3] = np.float32(rad)
#
#     hand_kalman.correct(np.array(current_measurement, np.float32))
#     print('measurement_angle: ', current_measurement)
#     prediction = hand_kalman.predict()
#     print('prediction_angle: ', prediction)
#
#     #     # 15 left_wrist-current
#     # x = rcos（θ), # y = rsin（θ）
#     r = math.sqrt((landmarks[15].x**2)+(landmarks[15].y**2))
#     current_prediction[0] = r*math.cos(prediction[0]), r*math.sin(prediction[0])
#
#     #     # 17 left_pinky-current
#     r = math.sqrt((landmarks[17].x**2)+(landmarks[17].y**2))
#     current_prediction[1] = r*math.cos(prediction[2]), r*math.sin(prediction[2])
#
#     #     # 19 left_index-current
#     r = math.sqrt((landmarks[19].x**2)+(landmarks[19].y**2))
#     current_prediction[2] = r*math.cos(prediction[4]), r*math.sin(prediction[4])
#
#     #     # 21 left_thumb-current
#     r = math.sqrt((landmarks[21].x**2)+(landmarks[21].y**2))
#     current_prediction[3] = r*math.cos(prediction[6]), r*math.sin(prediction[6])
#     # print("current_prediction:\n", current_prediction)
#
#     return current_prediction
