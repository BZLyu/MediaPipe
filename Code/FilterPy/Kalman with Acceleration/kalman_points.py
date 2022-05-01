# # Author: Bingzhen Lyu
# # Date: 2022/1/18

import numpy as np
import time
import set_kalman


def all_points(all_kalman, prevlandmarks, landmarks, prevTime):
    # todo: set variable
    unit_num_state = 6  # x，y，dx，dy,ddx,ddy
    num_point = 33  # 33 Points
    # num_state = unit_num_state * num_point
    num_dimension = 2  # 2 Dimension(x,y)

    current_measurement = np.ones((num_point*num_dimension, 1))
    last_measurement = np.ones((num_point*num_dimension, 1))

    current_prediction = np.ones((num_point, 2))

    # todo : Determine if Mediapipe is right
    right = 1
    j = 0
    for i in range(num_point):
        current_measurement[j] = np.float32(landmarks[i].x)
        current_measurement[j+1] = np.float32(landmarks[i].y)
        j += 2

    j = 0
    for i in range(num_point):
        last_measurement[j] = np.float32(prevlandmarks[i].x)
        last_measurement[j+1] = np.float32(prevlandmarks[i].y)
        j += 2
            # change Q
    currTime = time.time()
    t = currTime-prevTime
    for i in range(current_measurement.shape[0]):
        v = abs(current_measurement[i]-last_measurement[i])/t

        if v > 0.0073:# 0.0075-78%
            right = 0
            break

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
