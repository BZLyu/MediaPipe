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

    cap = cv2.VideoCapture('D:cut_1.mp4')
    points = np.load('D:transformed_ground_truth.npy')
    # TODO: set kalman filter
    first_frame = True
    all_kalman = set_kalman.set_kalman_all()

    # Curl counter variables
    # counter = 0
    # stage = None

    # Set View
    front = False

    # Set real points
    frame_pointer = 23776  # Which Picture
    K = 0
    Tatol= 0
    prevTime = 0

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
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
                continue

            landmarks = results.pose_landmarks.landmark

            # length_shoulder = math.sqrt(((landmarks[12].x-landmarks[11].x)**2) +
            #                         ((landmarks[12].y-landmarks[11].y)**2))
            # length_hip = math.sqrt(((landmarks[24].x-landmarks[23].x)**2) +
            #                    ((landmarks[24].y-landmarks[23].y)**2))
            # if length_hip < length_shoulder*(1/2) or length_shoulder < length_hip or landmarks[13].x < landmarks[23].x:
            #     continue

# TODO:  show Mediapipe points
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


# TODO: show prediction

            # if first_frame is True:
            #
            #     prediction_old = prediction = kalman_points.all_points(all_kalman, landmarks)
            #     first_frame = False
            #
            # else:
            #     # TODO：test Mediapipe goes wrong.
            #     length_shoulder = math.sqrt(((landmarks[12].x-landmarks[11].x)**2) +
            #                                 ((landmarks[12].y-landmarks[11].y)**2))
            #     length_hip = math.sqrt(((landmarks[24].x-landmarks[23].x)**2) +
            #                            ((landmarks[24].y-landmarks[23].y)**2))
            #     last_hip = math.sqrt(((prediction_old[24][0]-prediction_old[23][0])**2) +
            #                          ((prediction_old[24][1]-prediction_old[23][1])**2))
            #     last_shoulder = math.sqrt(((prediction_old[12][0]-prediction_old[11][0])**2) +
            #                               ((prediction_old[12][1]-prediction_old[11][1])**2))
            #     # (7/12)
            #     if length_hip < length_shoulder*(1/3) or length_shoulder < length_hip or landmarks[13].x < landmarks[23].x:
            #         # print("26:", landmarks[26].x)
            #         # print("25:", landmarks[25].x)
            #         # print("Wrong!")
            #         prediction = kalman_points.wrong_points(all_kalman, prediction_old)
            #         prediction_old = prediction
            #
            #     else:
            prediction = kalman_points.all_points(all_kalman, landmarks)
            prediction_old = prediction


            for i in range(len(prediction)):
                point_prediction = tuple(np.multiply(prediction[i], [1920, 1080]).astype(int))
                cv2.circle(image, point_prediction, 5, (0, 100, 0), -1)

            for i in mp_pose.POSE_CONNECTIONS:
                start_point = tuple(np.multiply(prediction[i[0]], [1920, 1080]).astype(int))
                end_point = tuple(np.multiply(prediction[i[1]], [1920, 1080]).astype(int))
                cv2.line(image, start_point, end_point, (0, 100, 0), 2)

# TODO: show which is better, Mediapipe with or without kalman
            if frame_pointer > 126000:
                break
            better_results = Compare_Data.compare(landmarks, prediction, points[frame_pointer])
            Tatol += 1
            if better_results == 'K':
                cv2.putText(image, "Kalman better", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
                K += 1
            else:
                cv2.putText(image, "Mediapipe better", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
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
                continue
            cv2.putText(image, f'FPS:{int(fps)}', (20, 78), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

            cv2.imshow("Mediapipe mit Kalman", image)
# TODO: exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    success = K/Tatol
    print("Success=", success)


if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    video()
    print("End!")
