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


def get_hand_angle(hand_kalman, landmarks):

    # todo: set variable
    num_point = 4  # 4 Points of hand
    num_dimension = 1  # 1 Angle
    current_measurement = np.ones((num_point*num_dimension, 1))
    current_prediction = np.ones((num_point, 2))

    # 15 left_wrist-current
    rad = math.atan2(landmarks[15].y, landmarks[15].x)
    # angle = (np.abs(rad * 180.0/np.pi))
    current_measurement[0] = np.float32(rad)

    # 17 left_pinky-current
    rad = math.atan2(landmarks[17].y, landmarks[17].x)
    # angle = (np.abs(rad * 180.0/np.pi))
    current_measurement[1] = np.float32(rad)

    # 19 left_index-current
    rad = math.atan2(landmarks[19].y, landmarks[19].x)
    # angle = (np.abs(rad * 180.0/np.pi))
    current_measurement[2] = np.float32(rad)

    # 21 left_thumb-current
    rad = math.atan2(landmarks[21].y, landmarks[21].x)
    # angle = (np.abs(rad * 180.0/np.pi))
    current_measurement[3] = np.float32(rad)

    hand_kalman.correct(np.array(current_measurement, np.float32))
    print('measurement_angle: ', current_measurement)
    prediction = hand_kalman.predict()
    print('prediction_angle: ', prediction)

    #     # 15 left_wrist-current
    # x = rcos（θ), # y = rsin（θ）
    r = math.sqrt((landmarks[15].x**2)+(landmarks[15].y**2))
    current_prediction[0] = r*math.cos(prediction[0]), r*math.sin(prediction[0])

    #     # 17 left_pinky-current
    r = math.sqrt((landmarks[17].x**2)+(landmarks[17].y**2))
    current_prediction[1] = r*math.cos(prediction[2]), r*math.sin(prediction[2])

    #     # 19 left_index-current
    r = math.sqrt((landmarks[19].x**2)+(landmarks[19].y**2))
    current_prediction[2] = r*math.cos(prediction[4]), r*math.sin(prediction[4])

    #     # 21 left_thumb-current
    r = math.sqrt((landmarks[21].x**2)+(landmarks[21].y**2))
    current_prediction[3] = r*math.cos(prediction[6]), r*math.sin(prediction[6])
    # print("current_prediction:\n", current_prediction)

    return current_prediction
