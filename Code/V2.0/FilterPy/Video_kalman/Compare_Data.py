# Author: Bingzhen Lyu 
# Date: 2022/2/7
import numpy as np
import math


def compare(landmarks, prediction, real_points):

    # a = tuple(np.multiply((landmarks[12].x, landmarks[12].y), [1920, 1080]))
    #
    # diff_m = math.sqrt((a[0] - real_points[9][0])**2 + (a[1] - real_points[9][1])**2)
    # a = tuple(np.multiply(prediction[12], [1920, 1080]))
    # diff_k = math.sqrt((a[0] - real_points[9][0])**2 + (a[1] - real_points[9][1])**2)
    #
    # if diff_k < diff_m:
    #     better_result = 'K'
    # else:
    #     better_result = 'M'

    #
    error_m = []
    error_k = []
    mediapipe_list, kalman_list, real_list = initial(landmarks, prediction, real_points)
    for i in range(len(checklist)):

        if np.isnan(real_points[i][0]) or np.isnan(real_points[i][1]):
            checklist[i] = 1
        else:
            diff_m = math.sqrt(math.pow((mediapipe_list[i][0] - real_points[i][0]), 2) +
                               math.pow((mediapipe_list[i][1] - real_points[i][1]), 2))

            diff_k = math.sqrt(math.pow((kalman_list[i][0] - real_points[i][0]), 2) +
                               math.pow((kalman_list[i][1] - real_points[i][1]), 2))
        error_k.append()

    #         if diff_k < diff_m:
    #             checklist[i] = 1
    #
    # count = checklist.count(1)
    # if count >= 2:
    #     better_result = 'K'
    # else:
    #     better_result = 'M'

    return better_result,


def initial(landmarks, prediction, real_points):
    mediapipe_list = [0, 0, 0, 0]

    mediapipe_list_x = [landmarks[12].x, landmarks[11].x, landmarks[24].x, landmarks[23].x]
    # a = (landmarks[28].x + landmarks[32].x + landmarks[30].x)/3
    # mediapipe_list_x[6] = a
    # a = (landmarks[27].x + landmarks[29].x + landmarks[31].x)/3
    # mediapipe_list_x[7] = a

    mediapipe_list_y = [landmarks[12].y, landmarks[11].y, landmarks[24].y, landmarks[23].y]
    #
    # a = (landmarks[28].y + landmarks[32].y + landmarks[30].y)/3
    # mediapipe_list_y[6] = a
    # a = (landmarks[27].y + landmarks[29].y + landmarks[31].y)/3
    # mediapipe_list_y[7] = a

    for i in range(len(mediapipe_list)):
        mediapipe_list[i] = tuple(np.multiply((mediapipe_list_x[i], mediapipe_list_y[i]), [1920, 1080]))



    kalman_list = [prediction[12], prediction[11], prediction[24], prediction[23]]

    # a = (prediction[28] + prediction[32] + prediction[30])/3
    # kalman_list[6] = a
    #
    # a = (prediction[27] + prediction[29] + prediction[31])/3
    # kalman_list[7] = a
    for i in range(len(kalman_list)):
        kalman_list[i] = tuple(np.multiply((kalman_list[i]), [1920, 1080]))


    real_list = [real_points[9], real_points[6], real_points[10], real_points[4]]
    #
    # if np.isnan(real_points[2]):
    #     a =
    # elif np.isnan(real_points[22]):
    # else:
    #     a = (real_points[2] + real_points[22])/2
    #     real_list[4] = a
    #     a = (real_points[14] + real_points[13])/2
    #     real_list[5] = a
    #     a = (real_points[16] + real_points[20] + real_points[8] + real_points[3] + real_points[12])/5
    #     real_list[6] = a
    #     a = (real_points[21] + real_points[15] + real_points[18] + real_points[7] + real_points[19])/5
    #     real_list[7] = a

    return mediapipe_list, kalman_list, real_list

