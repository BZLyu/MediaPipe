# Author: Bingzhen Lyu 
# Date: 2022/2/7
import numpy as np
import math
import statistics

# TODO: change test Point


def initial(landmarks, prediction, real_points):
    mediapipe_list = [0]

    mediapipe_list_x = [landmarks[26].x]  # Test Point

    mediapipe_list_y = [landmarks[26].y]  # Test Point

    for i in range(len(mediapipe_list)):
        mediapipe_list[i] = tuple(np.multiply((mediapipe_list_x[i], mediapipe_list_y[i]), [1920, 1080]))

    kalman_list = [prediction[0]]

    for i in range(len(kalman_list)):
        kalman_list[i] = tuple(np.multiply((kalman_list[i]), [1920, 1080]))

    # TODO: change compare point
    real_list = [(real_points[2]+real_points[22])/2]  # Real point

    return mediapipe_list, kalman_list, real_list

