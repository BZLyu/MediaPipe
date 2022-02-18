# Author: Bingzhen Lyu 
# Date: 2022/1/17
#！！！ not change!

import cv2
import numpy as np


def mousemove(event, x, y, s, p):
    #
    global frame, current_measurement, measurements, last_measurement, current_prediction, last_prediction
    #
    last_measurement = current_measurement
    last_prediction = current_prediction
    current_x = np.float32(x)
    current_y = np.float32(y)
    #
    current_measurement = np.array([[current_x], [current_y]])
    print("current_measurement", current_measurement)
    kalman.correct(current_measurement)
    print("new_current_measurement", current_measurement)
    #
    current_prediction = kalman.predict()
    print("current_prediction", current_prediction)

    #
    lmx, lmy = last_measurement[0], last_measurement[1]
    #
    cmx, cmy = current_measurement[0], current_measurement[1]
    #
    lpx, lpy = last_prediction[0], last_prediction[1]
    #
    cpx, cpy = current_prediction[0], current_prediction[1]
    #
    cv2.line(frame, (int(lmx), int(lmy)), (int(cmx), int(cmy)), (0, 100, 0))
    #
    cv2.line(frame, (int(lpx), int(lpy)), (int(cpx), int(cpy)), (0, 0, 200))


if __name__ == '__main__':

    frame = np.zeros((800, 800, 3), np.uint8)

    last_measurement = current_measurement = np.array((2, 1), np.float32)

    last_predicition = current_prediction = np.zeros((2, 1), np.float32)

    cv2.namedWindow("kalman_tracker")

    cv2.setMouseCallback("kalman_tracker", mousemove)

    kalman = cv2.KalmanFilter(4, 2)
#
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
#
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
#
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03

    while True:
        cv2.imshow("kalman_maustracker", frame)
        if (cv2.waitKey(30) & 0xff) == ord('q'):  # 按 q 键退出
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
