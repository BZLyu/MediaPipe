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
    y0 =[]  # Diff_s
    y1=[]
    y2=[]

    x1=[]
    z0=[]
    z1=[]
    z2=[]

    cap = cv2.VideoCapture('D:cut_1.mp4')  # D:cut_1.mp4, /Users/stella/Desktop/Meidapipe/cut_1.mp4
    # cap.set(CV_CAP_PROP_BUFFERSIZE,33)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('D:cut_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
#     D:cut_2.avi
# /Users/stella/Desktop/Meidapipe/cut_2.avi
    points = np.load('D:transformed_ground_truth.npy')
    # D:transformed_ground_truth.npy
    # /Users/stella/Desktop/Meidapipe/2d_transformed_ground_truth.npy
    # TODO: set kalman filter
    all_kalman = set_kalman.set_kalman_all()

    # Set real points
    frame_pointer = 23776  # Start Picture
    first_frame = True


    Total = 0


    list_k_error = []
    list_k_upper_error = []
    list_k_lower_error = []
    list_m_error = []
    list_m_upper_error = []
    list_m_lower_error = []

    kalman_median_better = 0
    kalman_upper_median_better = 0
    kalman_lower_median_better = 0

    kalman_mean_better = 0
    kalman_upper_mean_better = 0
    kalman_lower_mean_better = 0

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        while cap.isOpened():
            if frame_pointer >= 26376:#126000
                break
            ret, frame = cap.read()
            if not ret:
                # print("Can't receive frame (stream end?). Exiting ...")
                print(str_k_median)
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

            # for i in mp_pose.POSE_CONNECTIONS:
            #     start_point = tuple(np.multiply(prediction[i[0]], [1920, 1080]).astype(int))
            #     end_point = tuple(np.multiply(prediction[i[1]], [1920, 1080]).astype(int))
            #     cv2.line(image, start_point, end_point, (0, 100, 0), 2)

# TODO: show which is better, Mediapipe with or without kalman

            kalman_median_better, kalman_mean_better,kalman_upper_median_better, kalman_upper_mean_better,\
            kalman_lower_median_better, kalman_lower_mean_better = \
                Compare_Data.compare(landmarks, prediction, points[frame_pointer],list_k_upper_error, list_m_upper_error, list_k_lower_error,
            list_m_lower_error, list_k_error, list_m_error, kalman_median_better, kalman_mean_better, kalman_upper_median_better,
            kalman_upper_mean_better, kalman_lower_median_better,kalman_lower_mean_better)


            Total += 1
#             todo: Absolute error


            # str_k_mean = "Absolute Error Kalman in Mean: " + str(k_Mean)
            # str_m_mean = "Absolute Error Kalman in Mean: " + str(m_Mean)

            str_k_mean_better = "Kalman in Mean better: " + str(int(kalman_mean_better / Total * 100)) + "%"
            # str_k_mean_upper_better = "Upper Kalman Mean better: " + str(int(kalman_mean_upper_better/Total*100)) + "%"
            # str_k_mean_lower_better= "Lower Kalman Mean better: " + str(int(kalman_mean_lower_better/Total*100)) + "%"


            # str_ab_k = "Absolute Error Kalman each Frame: " + str(erro_k_frame)
            # str_ab_m = "Absolute Error Mediapip each Frame: " + str(erro_m_frame)


            # str_k_each_better = "Kalman each Frame better: " + str(int(kalman_each_better/Total*100))+"%"
            # str_k_upper_each_better = "Kalman upper each Frame better: " + str(int(kalman_each_upper_better/Total*100))+"%"
            # str_k_lower_each_better = "Kalman lower each Frame better: " + str(int(kalman_each_lower_better/Total*100))+"%"


            # str_ab_k_median = "Absolute Error Kalman in Median" + str(median_k)
            # str_ab_m_median = "Absolute Error Mediapip in Median" + str(median_m)

            str_k_median = "Kalman in Median better: " + str(int(kalman_median_better/Total*100))+"%"
            str_k_mean = "Kalman in Mean better: " + str(int(kalman_mean_better/Total*100))+"%"

            str_k_upper_median = "Kalman upper in Median better" + str(int(kalman_upper_median_better/Total*100))+"%"
            str_k_upper_mean = "Kalman upper in Mean better" + str(int(kalman_upper_mean_better / Total * 100)) + "%"

            str_k_lower_median = "Kalman lower in Median better" + str(int(kalman_lower_median_better/Total*100))+"%"
            str_k_lower_mean = "Kalman lower in Mean better" + str(int(kalman_lower_mean_better / Total * 100)) + "%"

            '''
            ----------------------show Error in Each Frame----------------------
            '''

            # cv2.putText(image, str_ab_k, (50, 200), cv2.FONT_HERSHEY_PLAIN,
            #                 2, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, str_ab_m, (50, 225), cv2.FONT_HERSHEY_PLAIN,
            #             2, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, str_k_each, (50, 250), cv2.FONT_HERSHEY_PLAIN,
            #             2, (255, 255, 255), 2, cv2.LINE_AA)
            '''
            ----------------------show Median----------------------
            '''
            # cv2.putText(image, str_ab_k_median, (50, 300), cv2.FONT_HERSHEY_PLAIN,
            #             2, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, str_ab_m_median, (50, 325), cv2.FONT_HERSHEY_PLAIN,
            #             2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_k_median, (50, 350), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_k_upper_median, (50, 375), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_k_lower_median, (50, 400), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)

            '''
            ----------------------show Mean----------------------
            '''
            # cv2.putText(image, str_k_mean, (50, 400), cv2.FONT_HERSHEY_PLAIN,
            #             2, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, str_m_mean, (50, 425), cv2.FONT_HERSHEY_PLAIN,
            #             2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_k_mean, (50, 450), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_k_upper_mean, (50, 475), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_k_lower_mean, (50, 500), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, "Pointer:"+str(frame_pointer), (50, 500), cv2.FONT_HERSHEY_PLAIN,
            #             2, (255, 255, 255), 2, cv2.LINE_AA)


# TODO: show real Points.

            for i in points[frame_pointer]:

                if np.isnan(i[0]) or np.isnan(i[1]):
                    continue

                point_real = tuple(i.astype(int))
                cv2.circle(image, point_real, 5, (0, 255, 0), -1)

            # chooses with 2/3 probability a 7, with 1/3 a 6 -> on average 6.67 steps forward
            '''--------plot'''
            if frame_pointer >=25576 and frame_pointer <=26076:

                mediapipe_list, kalman_list, real_list = Compare_Data.initial(landmarks, prediction, points[frame_pointer])
                # print(leg)
                if frame_pointer < 25576+7:
                    init = real_list[0]

                if real_list[0][1]-init[1] < 0:
                    real_diff1 = - math.sqrt(math.pow((real_list[0][0] - init[0]), 2) +
                                       math.pow((real_list[0][1] - init[1]), 2))
                else:
                    real_diff1 = math.sqrt(math.pow((real_list[0][0] - init[0]), 2) +
                                             math.pow((real_list[0][1] - init[1]), 2))

                if real_list[0][1]-mediapipe_list[0][1]< 0:
                    diff_m =-math.sqrt(math.pow((mediapipe_list[0][0] - real_list[0][0]), 2) +
                                       math.pow((mediapipe_list[0][1] - real_list[0][1]), 2))
                else:
                    diff_m = math.sqrt(math.pow((mediapipe_list[0][0] - real_list[0][0]), 2) +
                                       math.pow((mediapipe_list[0][1] - real_list[0][1]), 2))

                if real_list[0][1] - kalman_list[0][1] < 0:
                    diff_k = - math.sqrt(math.pow((kalman_list[0][0] - real_list[0][0]), 2)+
                                      math.pow((kalman_list[0][1] - real_list[0][1]), 2))
                else:
                    diff_k = math.sqrt(math.pow((kalman_list[0][0] - real_list[0][0]), 2) +
                                         math.pow((kalman_list[0][1] - real_list[0][1]), 2))

                # diff_m=mediapipe_list[0][0] - real_list[0][0]
                # diff_k=kalman_list[0][0] - real_list[0][0]
                # real_diff = init[0] - real_list[0][0]
                y0.append(real_diff1)
                y1.append(diff_m)
                y2.append(diff_k)

                t = (frame_pointer - 23776) / 200
                x.append(t)

            '''--------plot'''
            # if frame_pointer >= 124000 and frame_pointer <= 124200:
            #
            #     mediapipe_list, kalman_list, real_list = Compare_Data.initial(landmarks, prediction,
            #                                                                   points[frame_pointer])
            #     # print(leg)
            #     if frame_pointer == 123000:
            #         init = real_list[0]
            #     if real_list[0][1]-init[1] < 0:
            #         real_diff1 = - math.sqrt(math.pow((real_list[0][0] - init[0]), 2) +
            #                            math.pow((real_list[0][1] - init[1]), 2))
            #     else:
            #         real_diff1 = math.sqrt(math.pow((real_list[0][0] - init[0]), 2) +
            #                                  math.pow((real_list[0][1] - init[1]), 2))
            #
            #     diff_m = math.sqrt(math.pow((mediapipe_list[0][0] - real_list[0][0]), 2) +
            #                        math.pow((mediapipe_list[0][1] - real_list[0][1]), 2))
            #
            #     diff_k = math.sqrt(math.pow((kalman_list[0][0] - real_list[0][0]), 2) +
            #                        math.pow((kalman_list[0][1] - real_list[0][1]), 2))
            #
            #     # diff_m=mediapipe_list[0][0] - real_list[0][0]
            #     # diff_k=kalman_list[0][0] - real_list[0][0]
            #     # real_diff = init[0] - real_list[0][0]
            #     z0.append(real_diff1)
            #     z1.append(diff_m)
            #     z2.append(diff_k)
            #
            #     t = (frame_pointer - 23776) / 200
            #     x1.append(t)

            '''--------plot'''
            arr = [7, 6, 7]
            frame_pointer += arr[int(np.random.randint(0, 3, 1))]

            # fps = cap.get(cv2.CAP_PROP_FPS)

            # cv2.putText(image, f'FPS:{int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # cv2.imshow("Mediapipe mit Kalman", image)




            #str_erro_kalman="merro kalman:"+erro_kalman+"%"
            #str_erro_mediapipe="merro mediapipe:"+erro_mediapipe+"%"

            # cv2.putText(image, str_erro_kalman, (200, 50), cv2.FONT_HERSHEY_PLAIN,
            #            3, (255, 255, 255), 2)

# TODO: exit

            out.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(str_k_median)
    print(str_k_upper_median)
    print(str_k_lower_median)

    print(str_k_mean)
    print(str_k_upper_mean)
    print(str_k_lower_mean)

    fig, diff = plt.subplots()

    diff.plot(x, y0, label='real',color='blue')
    diff.plot(x, y1, label='mediapipe', color='coral')
    diff.plot(x, y2, label='kalman', color='green')
    diff.set_title('diff_left_leg_none_reset')
    diff.set_xlabel('ts')
    diff.set_ylabel('diff')
    diff.legend()
    plt.savefig('./diff_left_leg_none_reset_frist10.jpg')
    plt.show()
    plt.close()

    fig, diff1 = plt.subplots()
    diff1.plot(x, y1, label='mediapipe', color='coral')
    diff1.plot(x, y2, label='kalman', color='green')
    diff1.set_title('diff_left_leg_none_reset')
    diff1.set_xlabel('ts')
    diff1.set_ylabel('diff')
    diff1.legend()
    plt.savefig('./diff_left_leg_none_reset1_frist10.jpg')
    plt.show()
    # plt.close()

    # fig, diff3 = plt.subplots()
    #
    # diff3.plot(x1, z0, label='real')
    # diff3.plot(x1, z1, label='mediapipe')
    # diff3.plot(x1, z2, label='kalman')
    # diff3.set_title('diff_left_leg_none_reset')
    # diff3.set_xlabel('ts')
    # diff3.set_ylabel('diff')
    # diff3.legend()
    # plt.savefig('./diff_left_leg_none_reset_last10.jpg')
    # plt.show()
    # plt.close()
    #
    # fig, diff4 = plt.subplots()
    # diff4.plot(x1, z1, label='mediapipe')
    # diff4.plot(x1, z2, label='kalman')
    # diff4.set_title('diff_left_leg_none_reset')
    # diff4.set_xlabel('ts')
    # diff4.set_ylabel('diff')
    # diff4.legend()
    # plt.savefig('./diff_left_leg_none_reset1_last10.jpg')
    # plt.show()

    # print(str_k_each)

if __name__ == '__main__':



    video()
    print("End!")
