# Author: Bingzhen Lyu 
# Date: 2022/2/7
import numpy as np
import math


def compare(landmarks, prediction, real_points):

    #
    checklist = [0, 0, 0, 0, 0, 0, 0, 0]
    mediapipe_list, kalman_list, real_list = initial(landmarks, prediction, real_points)
    erro_k = [0, 0, 0, 0, 0, 0, 0, 0]
    erro_m = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(checklist)):

        diff_m = math.sqrt(math.pow((mediapipe_list[i][0] - real_points[i][0]), 2) +
                           math.pow((mediapipe_list[i][1] - real_points[i][1]), 2))
        erro_m[i] = diff_m

        diff_k = math.sqrt(math.pow((kalman_list[i][0] - real_points[i][0]), 2) +
                           math.pow((kalman_list[i][1] - real_points[i][1]), 2))
        erro_k[i] = diff_k

        if diff_k < diff_m:
            checklist[i] = 1


    count = checklist.count(1)
    if count >= 4:
        better_result = 'K'
    else:
        better_result = 'M'

    return better_result, erro_k, erro_m


def initial(landmarks, prediction, real_points):
    mediapipe_list = [0, 0, 0, 0, 0, 0, 0, 0]

    mediapipe_list_x = [landmarks[12].x, landmarks[11].x, landmarks[24].x, landmarks[23].x, landmarks[26].x,
                        landmarks[25].x, landmarks[28].x, landmarks[27].x]

    mediapipe_list_y = [landmarks[12].y, landmarks[11].y, landmarks[24].y, landmarks[23].y, landmarks[26].y,
                        landmarks[25].y, landmarks[28].y, landmarks[27].y]

    for i in range(len(mediapipe_list)):
        mediapipe_list[i] = tuple(np.multiply((mediapipe_list_x[i], mediapipe_list_y[i]), [1920, 1080]))


    kalman_list = [prediction[12], prediction[11], prediction[24], prediction[23], prediction[26],
                   prediction[25], prediction[28], prediction[27]]

    for i in range(len(kalman_list)):
        kalman_list[i] = tuple(np.multiply((kalman_list[i]), [1920, 1080]))


    real_list = [real_points[9], real_points[6], real_points[10], real_points[4], (real_points[2]+real_points[22])/2,
                 (real_points[14]+real_points[13])/2, (real_points[16]+real_points[8])/2,
                 (real_points[15]+real_points[18])/2]

    return mediapipe_list, kalman_list, real_list

