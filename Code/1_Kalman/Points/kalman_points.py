# Author: Bingzhen Lyu 
# Date: 2022/1/18
import set_kalman_Points
import numpy as np
import cv2


def all_points(landmarks, mp_pose):

    kalman = set_kalman_Points.set_kalman()

# shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
# landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

    list_kalman = np.ones((33, 1), cv2.KalmanFilter)
    current_measurement = np.ones((33, 2))
    current_prediction = np.ones((33, 2))

    # make 33 kalman
    for i in range(len(list_kalman)):
        list_kalman[i] = kalman
    # put the position into current_measurement
    # for i in range(len(landmarks)):
# current_measurement [[0.93523204][0.6457451 ]]
#         current_measurement[i] = [np.float32(landmarks[i].x), np.float32(landmarks[i].y)]

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
        # if i == 11:
        #     print("prediction_SHOULDER", current_prediction[i])

    return current_prediction
