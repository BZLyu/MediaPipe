# Author: Bingzhen Lyu 
# Date: 2022/2/7
import numpy as np
import math
import statistics


'''
kalman_each_better = 0
kalman_each_upper_better = 0
kalman_each_lower_better = 0
'''

def initial(landmarks, prediction, real_points):
    mediapipe_list = [0]

    mediapipe_list_x = [landmarks[12].x]

    mediapipe_list_y = [landmarks[12].y]

    for i in range(len(mediapipe_list)):
        mediapipe_list[i] = tuple(np.multiply((mediapipe_list_x[i], mediapipe_list_y[i]), [1920, 1080]))

    kalman_list = [prediction[0]]

    for i in range(len(kalman_list)):
        kalman_list[i] = tuple(np.multiply((kalman_list[i]), [1920, 1080]))


    real_list = [real_points[9]]

    return mediapipe_list, kalman_list, real_list

