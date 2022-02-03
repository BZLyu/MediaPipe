# Author: Bingzhen Lyu 
# Date: 2022/1/18

import numpy as np


def get_mew_angle(kalman, angle):

    current_measurement = np.array(np.float32(angle))
    kalman.correct(current_measurement)
    # print('measurement_angle: ', current_measurement)
    current_prediction = kalman.predict()
    # print(current_prediction)
    new_angle = float(current_prediction[0])
    # print('prediction_angle: ', new_angle)

    return new_angle
