import cv2
import mediapipe as mp
import numpy as np
import math
import time
import kalman_points
import set_kalman
import Compare_Data


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    rad = np.abs(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    angle = (np.abs(rad * 180.0/np.pi))
    # return angle
    return angle


def video():

    cap = cv2.VideoCapture('D:cut_1.mp4')  # D:cut_1.mp4, /Users/stella/Desktop/Meidapipe/cut_1.mp4
    # cap.set(CV_CAP_PROP_BUFFERSIZE,33)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('D:cut_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    points = np.load('D:transformed_ground_truth.npy')
    # D:transformed_ground_truth.npy
    # /Users/stella/Desktop/Meidapipe/2d_transformed_ground_truth.npy
    # TODO: set kalman filter
    all_kalman = set_kalman.set_kalman_all()

    # Set real points
    frame_pointer = 23776  # Which Picture
    Total= 0
    kalman_each_better =0
    kalman_mediane_better =0
    list_k_error = []
    list_m_error = []


    prevTime = 0
    first_frame = True

    sum_erro_k = 0
    sum_erro_m = 0
    


    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        while cap.isOpened():
            if frame_pointer > 126000:
                break
            a = np.isnan(points[frame_pointer])
            # if True in a:
            #     frame_pointer += 1
            #     continue

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


            if results.pose_landmarks is None:
                arr = [6, 7, 7]
                frame_pointer += arr[int(np.random.randint(0, 3, 1))]
                # frame_pointer += 1
                continue
            if first_frame is True:
                prevlandmarks = results.pose_landmarks.landmark
                first_frame = False
            landmarks = results.pose_landmarks.landmark

# TODO: show prediction
            # TODO：get prediction point.//(test whether Mediapipe goes wrong.)
            prediction, prevlandmarks = kalman_points.all_points(all_kalman, prevlandmarks, landmarks, prevTime)


            for i in range(len(prediction)):
                point_prediction = tuple(np.multiply(prediction[i], [1920, 1080]).astype(int))
                cv2.circle(image, point_prediction, 5, (0, 100, 0), -1)

            for i in mp_pose.POSE_CONNECTIONS:
                start_point = tuple(np.multiply(prediction[i[0]], [1920, 1080]).astype(int))
                end_point = tuple(np.multiply(prediction[i[1]], [1920, 1080]).astype(int))
                cv2.line(image, start_point, end_point, (0, 100, 0), 2)

# TODO: show which is better, Mediapipe with or without kalman

            erro_k, erro_m = Compare_Data.compare(landmarks, prediction, points[frame_pointer])

            Total += 1

#             todo: Absolute error
            erro_k_frame = 0
            erro_m_frame = 0

            for i in range(len(erro_k)):
                erro_k_frame += erro_k[i]
                erro_m_frame += erro_m[i]
            sum_erro_k += erro_k_frame
            sum_erro_m += erro_m_frame

            str_ab_k = "Absolute Error Kalman each Frame: " + str(int(erro_k_frame))
            str_ab_m = "Absolute Error Mediapip each Frame: " + str(int(erro_m_frame))
            if erro_k_frame < erro_m_frame:
                kalman_each_better += 1
            str_k_each = "Better Absolute Error Kalman each Frame: " + str(int(kalman_each_better/Total*100))+"%"

            list_k_error.append(erro_k_frame)
            list_m_error.append(erro_m_frame)
            median_k = list_k_error[int(len(list_k_error)/2)]
            median_m = list_m_error[int(len(list_m_error)/2)]
            str_ab_k_median = "Absolute Error Kalman in Median" + str(int(median_k))
            str_ab_m_median = "Absolute Error Mediapip in Median" + str(int(median_m))

            if (median_k) < (median_m):
                kalman_mediane_better += 1
            str_k_median = "Better Absolute Error Kalman in Median: " + str(int(kalman_mediane_better/Total*100))+"%"

            cv2.putText(image, str_ab_k, (50, 200), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_ab_m, (50, 225), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_k_each, (50, 250), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, str_ab_k_median, (50, 300), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_ab_m_median, (50, 325), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str_k_median, (50, 350), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)


            cv2.putText(image, "Pointer:"+str(frame_pointer), (50, 500), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2, cv2.LINE_AA)


# TODO: show real Points.

            for i in points[frame_pointer]:

                if np.isnan(i[0]) or np.isnan(i[1]):
                    continue

                point_real = tuple(i.astype(int))
                cv2.circle(image, point_real, 5, (0, 255, 0), -1)

            # chooses with 2/3 probability a 7, with 1/3 a 6 -> on average 6.67 steps forward
            arr = [6, 7, 7]
            frame_pointer += arr[int(np.random.randint(0, 3, 1))]

            if frame_pointer < 0:
                break

            currTime = time.time()
            fps = 1/(currTime-prevTime)
            prevTime = currTime
            if fps < 1:
                # print(str_k_median)
                continue

            # fps = cap.get(cv2.CAP_PROP_FPS)

            cv2.putText(image, f'FPS:{int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Mediapipe mit Kalman", image)

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

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    video()
    print("End!")
