import cv2
import mediapipe as mp
import numpy as np
import math

import kalman_points
import set_kalman

import kalman_angle

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
    # todo: set kalman filter
    kalman_elbow_angle = set_kalman.set_kalman_elbow_angle()

    all_kalman = set_kalman.set_kalman_all()

    angle_kalman = set_kalman.set_kalman_hand_angle()

    # Curl counter variables
    counter = 0
    new_counter = 0
    stage = None
    new_stage = None

    # Set View
    front = False

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():

            ret, frame = cap.read()

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

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            new_angle = kalman_angle.get_elbow_angle(kalman_elbow_angle, angle)

            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 45 and stage == 'down':
                stage = "up"
                counter += 1

            if new_angle > 160:
                new_stage = "down"
            if new_angle < 45 and new_stage == 'down':
                new_stage = "up"
                new_counter += 1

            # Visualize angle
# TODO: show arm status
            # np.abs(math.atan2(c[1]-b[1], c[0]-b[0])
            front = View.body_view(landmarks, stage, front)
            if front is True:
                cv2.putText(image, "Front", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Lateral", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)


# TODO: show counter.

            cv2.putText(image, str(counter), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(new_counter), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 0), 2, cv2.LINE_AA)


# TODO: show all new points.

            prediction = kalman_points.all_points(all_kalman, landmarks)
            # todo: change points of hand.
            hand_prediction = kalman_angle.get_hand_angle(angle_kalman, landmarks)

            # 15 left_wrist-current
            prediction[15] = hand_prediction[0]
            # 17 left_pinky-current
            prediction[17] = hand_prediction[1]
            # 19 left_index-current
            prediction[19] = hand_prediction[2]
            # 21 left_thumb-current
            prediction[21] = hand_prediction[3]

            # show
            for i in range(len(prediction)):
                point_new = tuple(np.multiply(prediction[i], [1280, 720]).astype(int))
                cv2.circle(image, point_new, 5, (0, 100, 0), -1)
# todo: connect line

            for i in mp_pose.POSE_CONNECTIONS:
                start_point = tuple(np.multiply(prediction[i[0]], [1280, 720]).astype(int))
                end_point = tuple(np.multiply(prediction[i[1]], [1280, 720]).astype(int))
                cv2.line(image, start_point, end_point, (0, 100, 0), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Mediapipe Feed", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    video()
    print("End!")
