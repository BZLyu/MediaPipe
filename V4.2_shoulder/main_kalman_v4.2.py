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
    x =[]  # t
    y0=[]  # Diff_s
    y1=[]
    y2=[]
    cap = cv2.VideoCapture('/Users/stella/Desktop/Meidapipe/cut_1.mp4')  # D:cut_1.mp4, /Users/stella/Desktop/Meidapipe/cut_1.mp4
    # cap.set(CV_CAP_PROP_BUFFERSIZE,33)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('/Users/stella/Desktop/Meidapipe/cut_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
#     D:cut_2.avi
# /Users/stella/Desktop/Meidapipe/cut_2.avi
    points = np.load('/Users/stella/Desktop/Meidapipe/2d_transformed_ground_truth.npy')
    # D:transformed_ground_truth.npy
    # /Users/stella/Desktop/Meidapipe/2d_transformed_ground_truth.npy
    # TODO: set kalman filter
    all_kalman = set_kalman.set_kalman_all()

    # Set real points
    frame_pointer = 23776  # Start Picture
    first_frame = True
    start = 25576
    end = 25776
    sum_ab_m = 0
    sum_ab_k = 0
    sum_mse_m = 0
    sum_mse_k = 0


    Total = 0


    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
        while cap.isOpened():
            if frame_pointer >= end:# 126000
                break
            ret, frame = cap.read()
            if not ret:
                # print("Can't receive frame (stream end?). Exiting ...")
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
                kalman_points.frist_x(all_kalman, prevlandmarks)
                first_frame = False
            landmarks = results.pose_landmarks.landmark

# TODO: show prediction
            # TODO：get prediction point.//(test whether Mediapipe goes wrong.)

            prediction, prevlandmarks = kalman_points.all_points(all_kalman, prevlandmarks, landmarks)


            for i in range(len(prediction)):
                point_prediction = tuple(np.multiply(prediction[i], [1920, 1080]).astype(int))
                cv2.circle(image, point_prediction, 5, (0, 100, 0), -1)

# TODO: show which is better, Mediapipe with or without kalman

            Total += 1


# TODO: show real Points.

            for i in points[frame_pointer]:

                if np.isnan(i[0]) or np.isnan(i[1]):
                    continue

                point_real = tuple(i.astype(int))
                cv2.circle(image, point_real, 5, (0, 255, 0), -1)

            # chooses with 2/3 probability a 7, with 1/3 a 6 -> on average 6.67 steps forward
            '''--------plot'''
            if frame_pointer >=start and frame_pointer <=end:

                mediapipe_list, kalman_list, real_list = Compare_Data.initial(landmarks, prediction, points[frame_pointer])
                # print(leg)
                if frame_pointer < start+7:
                    init = real_list[0]
                for i in range(len(kalman_list)):
                    if real_list[0][1]-init[1] < 0:
                        d_real1 = - math.sqrt(math.pow((real_list[0][0] - init[0]), 2) +
                                           math.pow((real_list[0][1] - init[1]), 2))
                    else:
                        d_real1 = math.sqrt(math.pow((real_list[0][0] - init[0]), 2) +
                                                 math.pow((real_list[0][1] - init[1]), 2))

                    if real_list[0][1]-mediapipe_list[0][1] < 0:
                        d_m = -math.sqrt(math.pow((mediapipe_list[0][0] - real_list[0][0]), 2) +
                                           math.pow((mediapipe_list[0][1] - real_list[0][1]), 2))
                    else:
                        d_m = math.sqrt(math.pow((mediapipe_list[0][0] - real_list[0][0]), 2) +
                                           math.pow((mediapipe_list[0][1] - real_list[0][1]), 2))

                    if real_list[0][1] - kalman_list[0][1] < 0:
                        d_k = - math.sqrt(math.pow((kalman_list[0][0] - real_list[0][0]), 2)+
                                          math.pow((kalman_list[0][1] - real_list[0][1]), 2))
                    else:
                        d_k = math.sqrt(math.pow((kalman_list[0][0] - real_list[0][0]), 2) +
                                             math.pow((kalman_list[0][1] - real_list[0][1]), 2))
                    y0.append(d_real1)
                    y1.append(d_m)
                    y2.append(d_k)
                    sum_ab_m += abs(d_m)
                    sum_mse_m += abs(d_m) ** 2
                    sum_ab_k += abs(d_k)
                    sum_mse_k += abs(d_k) ** 2
                t = (frame_pointer-start) / 200
                x.append(t)


            '''--------plot'''
            arr = [7, 6, 7]
            frame_pointer += arr[int(np.random.randint(0, 3, 1))]

            # fps = cap.get(cv2.CAP_PROP_FPS)

            # cv2.putText(image, f'FPS:{int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # cv2.imshow("Mediapipe mit Kalman", image)


# TODO: exit

            out.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


    fig, diff = plt.subplots()

    diff.plot(x, y0, label='real',color='blue')
    diff.plot(x, y1, label='mediapipe', color='coral')
    diff.plot(x, y2, label='kalman', color='green')
    diff.set_title('')
    diff.set_xlabel('ts')
    diff.set_ylabel('distance')
    diff.legend()
    plt.savefig('./shoulder_without_reset_modul_2_0.028.jpg')
    plt.show()
    plt.close()
    #
    # fig, diff1 = plt.subplots()
    # diff1.plot(x, y1, label='mediapipe', color='coral')
    # diff1.plot(x, y2, label='kalman', color='green')
    # diff1.set_title('distance_left_leg_without_reset')
    # diff1.set_xlabel('ts')
    # diff1.set_ylabel('distance')
    # diff1.legend()
    # plt.savefig('./d_left_leg_without_reset1_frist10.jpg')
    # plt.show()

    mae_m = sum_ab_m/len(y1)
    mae_k = sum_ab_k/len(y2)
    rmse_m = (sum_mse_m/len(y1))/2
    rmse_k = (sum_mse_k/len(y2))/2

    print("mea_k= ", '%.5f'% mae_k)
    print("mea_m= ", '%.5f' % mae_m)
    print("rmse_k= ", '%.5f'% rmse_k)
    print("rmse_m= ", '%.5f'% rmse_m)

if __name__ == '__main__':



    video()
    print("End!")
