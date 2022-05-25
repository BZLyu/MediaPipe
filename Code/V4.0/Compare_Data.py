# Author: Bingzhen Lyu 
# Date: 2022/2/7
import numpy as np
import math

Total = 0

sum_erro_k = 0
sum_k_upper = 0
sum_k_lower = 0

sum_erro_m = 0
sum_m_upper = 0
sum_m_lower = 0

kalman_median_better = 0
kalman_median_upper_better = 0
kalman_median_lower_better = 0

list_k_error = []
list_k_upper_error = []
list_k_lower_error = []
list_m_error = []
list_m_upper_error = []
list_m_lower_error = []

kalman_mean_better = 0
kalman_mean_upper_better = 0
kalman_mean_lower_better = 0
'''
kalman_each_better = 0
kalman_each_upper_better = 0
kalman_each_lower_better = 0
'''

def get_kalman_median_better():
    list_k_error.sort()
    new_list_k_error = list_k_error.copy()



def compare(landmarks, prediction, real_points):

    mediapipe_list, kalman_list, real_list = initial(landmarks, prediction, real_points)
    erro_k_frame = [0, 0, 0, 0, 0, 0, 0, 0]
    erro_m_frame = [0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(erro_k_frame)):

        diff_m = math.sqrt(math.pow((mediapipe_list[i][0] - real_list[i][0]), 2) +
                               math.pow((mediapipe_list[i][1] - real_list[i][1]), 2))
        diff_k = math.sqrt(math.pow((kalman_list[i][0] - real_list[i][0]), 2) +
                           math.pow((kalman_list[i][1] - real_list[i][1]), 2))
        if i < 4:
            list_k_upper_error.append(diff_k)
            list_m_upper_error.append(diff_m)
        else:
            list_k_lower_error.append(diff_k)
            list_m_upper_error.append(diff_m)
        erro_k_frame[i] = diff_k
        erro_m_frame[i] = diff_m
        list_k_error.append(diff_k)
        list_m_error.append(diff_m)

    get_kalman_median_better()
    get_kalman_median_upper_better()
    get_kalman_median_lower_better()

    get_kalman_mean_better()
    get_kalman_mean_upper_better()
    get_kalman_mean_lower_better()

    get_kalman_each_better()
    get_kalman_each_upper_better()
    get_kalman_each_lower_better()

    return erro_k, erro_m


def initial(landmarks, prediction, real_points):
    mediapipe_list = [0, 0, 0, 0, 0, 0, 0, 0]

    mediapipe_list_x = [landmarks[11].x, landmarks[12].x, landmarks[23].x, landmarks[24].x, landmarks[25].x,
                        landmarks[26].x, landmarks[27].x, landmarks[28].x]

    mediapipe_list_y = [landmarks[11].y, landmarks[12].y, landmarks[23].y, landmarks[24].y, landmarks[25].y,
                        landmarks[26].y, landmarks[27].y, landmarks[28].y]

    for i in range(len(mediapipe_list)):
        mediapipe_list[i] = tuple(np.multiply((mediapipe_list_x[i], mediapipe_list_y[i]), [1920, 1080]))


    kalman_list = [prediction[0], prediction[1], prediction[2], prediction[3], prediction[4],
                   prediction[5], prediction[6], prediction[7]]

    for i in range(len(kalman_list)):
        kalman_list[i] = tuple(np.multiply((kalman_list[i]), [1920, 1080]))


    real_list = [real_points[6], real_points[9], (real_points[1]+real_points[4]+real_points[17])/3,
                 (real_points[11]+real_points[10]+real_points[0])/3, (real_points[14]+real_points[13])/2,
                 (real_points[2]+real_points[22])/2, (real_points[15]+real_points[18])/2,
                 (real_points[16]+real_points[8])/2]

    return mediapipe_list, kalman_list, real_list

