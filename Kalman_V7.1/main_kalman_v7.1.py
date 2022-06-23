import math

import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import kalman_points
import set_kalman
import Compare_Data


def video():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture('/Users/stella/Desktop/Meidapipe/cut_1.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('/Users/stella/Desktop/Meidapipe/cut_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                          (frame_width, frame_height))
    points = np.load('/Users/stella/Desktop/Meidapipe/2d_transformed_ground_truth.npy')

    # TODO: set kalman filter
    all_kalman = set_kalman.set_kalman_all()

    # Set real points
    frame_pointer = 23776  # Start Frame
    first_frame = True
    start = 25576  # Start sampling
    end = 25776  # End sampling

    sum_ab_m = 0  # sum of MediaPipe Absolute Error
    sum_ab_k = 0  # sum of Kalman filter Absolute Error
    sum_mse_m = 0  # sum of MediaPipe MSE
    sum_mse_k = 0  # sum of Kalman filter MSE
    x = []  # t
    y1 = []  # x-axis MediaPipe Absolute Error
    y2 = []  # x-axis Kalman filter Absolute Error
    y3 = []  # y-axis MediaPipe Absolute Error
    y4 = []  # y-axis Kalman filter Absolute Error


    # Setup mediapipe instance
    # TODO: change model_complexity
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            if frame_pointer >= end:  # 126000
                break
            ret, frame = cap.read()
            if not ret:
                print("Stream end.")
                break
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            # Recolor image to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if first_frame is True:
                prevlandmarks = results.pose_landmarks.landmark
                kalman_points.reset_x(all_kalman, prevlandmarks)

                first_frame = False
            landmarks = results.pose_landmarks.landmark

            # TODO：get prediction point.

            prediction, prevlandmarks = kalman_points.all_points(all_kalman, prevlandmarks, landmarks)

            for i in range(len(prediction)):
                point_prediction = tuple(np.multiply(prediction[i], [1920, 1080]).astype(int))
                cv2.circle(image, point_prediction, 5, (0, 100, 0), -1)

            # TODO: show real Points.

            for i in points[frame_pointer]:

                if np.isnan(i[0]) or np.isnan(i[1]):
                    continue

                point_real = tuple(i.astype(int))
                cv2.circle(image, point_real, 5, (0, 255, 0), -1)

            '''--Sampling, plot--'''

            if start <= frame_pointer <= end:
                # TODO: Reset function
                if np.abs(all_kalman.x[6][0]) > 0.05 or np.abs(all_kalman.x[7][0]) > 0.05:
                    kalman_points.reset_x(all_kalman, prevlandmarks)

                mediapipe_list, kalman_list, real_list = Compare_Data.initial(landmarks, prediction,
                                                                              points[frame_pointer])

                k_x = kalman_list[0][0] - real_list[0][0]
                k_y = kalman_list[0][1] - real_list[0][1]
                m_x = mediapipe_list[0][0] - real_list[0][0]
                m_y = mediapipe_list[0][1] - real_list[0][1]

                y1.append(m_x)
                y2.append(k_x)

                y3.append(m_y)
                y4.append(k_y)

                d_m = math.sqrt(math.pow((mediapipe_list[0][0] - real_list[0][0]), 2) +
                                math.pow((mediapipe_list[0][1] - real_list[0][1]), 2))

                d_k = math.sqrt(math.pow((kalman_list[0][0] - real_list[0][0]), 2) +
                                math.pow((kalman_list[0][1] - real_list[0][1]), 2))

                sum_ab_m += d_m
                sum_mse_m += d_m ** 2
                sum_ab_k += d_k
                sum_mse_k += d_k ** 2
                mae_m = sum_ab_m / len(y1)
                mae_k = sum_ab_k / len(y1)
                rmse_m = (sum_mse_m / len(y1)) / 2
                rmse_k = (sum_mse_k / len(y1)) / 2

                t = (frame_pointer - start) / 200
                x.append(t)

            # chooses with 2/3 probability a 7, with 1/3 a 6 -> on average 6.67 steps forward
            arr = [7, 6, 7]
            frame_pointer += arr[int(np.random.randint(0, 3, 1))]

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Mediapipe mit Kalman", image)

            # exit

            out.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    fig, diff = plt.subplots()

    diff.plot(x, y1, label='mediapipe', color='coral')
    diff.plot(x, y2, label='kalman', color='green')
    title = "Knee with Reset, Complexity 1, sigma=0.085"
    diff.set_title(title, fontsize=17, fontweight='bold')
    diff.set_xlabel('ts', fontsize=20, fontweight='bold')
    diff.set_ylabel('Absolute Error_x', fontsize=20, fontweight='bold')
    diff.legend()
    # plt.savefig('./Knee with Reset Modul1, 0.085_x.pdf', dpi=600, format='pdf')
    plt.show()
    # plt.close()
    #
    fig, diff1 = plt.subplots()
    # diff1.plot(x, y4, label='real', color='blue')
    diff1.plot(x, y3, label='mediapipe', color='coral')
    diff1.plot(x, y4, label='kalman', color='green')
    diff1.set_title('Knee with Reset, Complexity 1, sigma=0.085', fontsize=17, fontweight='bold')
    diff1.set_xlabel('ts', fontsize=20, fontweight='bold')
    diff1.set_ylabel('Absolute Error_y', fontsize=20, fontweight='bold')
    diff1.legend()
    # plt.savefig('./Knee with Reset Modul1, 0.085_y.pdf', dpi=800, format='pdf')
    plt.show()

    print("mea_k= ", '%.3f' % mae_k)
    print("mea_m= ", '%.3f' % mae_m)
    print("rmse_k= ", '%.3f' % rmse_k)
    print("rmse_m= ", '%.3f' % rmse_m)


if __name__ == '__main__':
    video()
    print("End!")
