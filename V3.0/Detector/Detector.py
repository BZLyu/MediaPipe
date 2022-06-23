import cv2
import mediapipe as mp
import numpy as np
import math

import kalman_points
import set_kalman

import View


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    rad = np.abs(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    angle = (np.abs(rad * 180.0/np.pi))
    # return angle
    return angle


def video():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('/Users/stella/Desktop/1.mov')

    # todo: set kalman filter
    counter = 0
    stage = None
    front = False
    lose = 0
    Total = 0
    first_frame = True
    all_kalman = set_kalman.set_kalman_all()


    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            Total += 1
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor image to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if first_frame is True:
                if results.pose_landmarks is None:
                    lose += 1
                    continue
                first_landmarks = results.pose_landmarks.landmark
                kalman_points.frist_x(all_kalman, first_landmarks)
                first_frame = False

            if results.pose_landmarks is None:  # When an empty landmark is received
                lose += 1
                shoulder = [prediction[12][0], prediction[12][1]]
                elbow = [prediction[14][0], prediction[14][1]]
                wrist = [prediction[16][0], prediction[16][1]]
            else:
                landmarks = results.pose_landmarks.landmark
                #  When shoulder or elbow or wrist is empty, use data of kalman filter
                if landmarks[12] is None or landmarks[14] is None or landmarks[16] is None:
                    shoulder = [prediction[12][0], prediction[12][1]]
                    elbow = [prediction[14][0], prediction[14][1]]
                    wrist = [prediction[16][0], prediction[16][1]]
                else:
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)


            # Arm up judgment
            if angle > 160:
                stage = "down"
            if angle < 60 and stage == 'down':
                stage = "up"
                counter += 1

            # Visualize angle
            # TODO: show arm status
            # np.abs(math.atan2(c[1]-b[1], c[0]-b[0])
            # front = View.body_view(landmarks, stage, front)
            # if front is True:
            #     cv2.putText(image, "Position: Front", (150, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
            # else:
            #     cv2.putText(image, "Position: Lateral", (150, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)


            # TODO: show counter.

            # cv2.putText(image, "Number: "+str(counter), (150, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)


            # TODO: show all new points.

            prediction = kalman_points.all_points(all_kalman, landmarks)
            # show
            for i in range(len(prediction)):
                point_new = tuple(np.multiply(prediction[i], [1280, 720]).astype(int))
                # cv2.circle(image, point_new, 5, (0, 100, 0), -1)
            # todo: connect line
            for i in mp_pose.POSE_CONNECTIONS:
                start_point = tuple(np.multiply(prediction[i[0]], [1280, 720]).astype(int))
                end_point = tuple(np.multiply(prediction[i[1]], [1280, 720]).astype(int))
                # cv2.line(image, start_point, end_point, (100, 100, 0), 3)

            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Mediapipe Feed", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            rate = (lose/Total)*100

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print("Rate of Empty Landmark list: " + str(rate))

    print("Number of Dumbbell lifts: " + str(counter))

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    video()
    print("End!")
