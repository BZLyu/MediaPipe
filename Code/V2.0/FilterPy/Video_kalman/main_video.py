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

    while cap.isOpened():

        ret, image = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image.flags.writeable = False
        #
        # image.flags.writeable = True
        # image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# TODO: show real Points.
#
#         for i in points[frame_pointer]:
#
#             if np.isnan(i[0]) or np.isnan(i[1]):
#                 continue
#
#             point_real = tuple(i.astype(int))
#             cv2.circle(image, point_real, 5, (0, 255, 0), -1)
#
#         # chooses with 2/3 probability a 7, with 1/3 a 6 -> on average 6.67 steps forward
#         arr = [6, 7, 7]
#         frame_pointer += arr[int(np.random.randint(0, 3, 1))]
#         if frame_pointer < 0:
#             break

        currTime = time.time()
        if currTime == prevTime:
            continue
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
