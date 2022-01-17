import cv2
import mediapipe as mp
import numpy as np
import math
import set_kalman_2_Point
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


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
    kalman = set_kalman_2_Point.set_kalman()
    # Curl counter variables
    counter = 0
    stage = None

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

            # print (results)
            # cv2.imshow("Mediapipe Feed", frame)

            # Extrack landmarks

            if results.pose_landmarks is None:
                continue

            landmarks = results.pose_landmarks.landmark
            # landmarks[12].x
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            current_x1 = np.float32(shoulder[0])
            current_y1 = np.float32(shoulder[1])
            current_x2 = np.float32(elbow[0])
            current_y2 = np.float32(elbow[1])
            # print("current_x:", current_x, "current_y:", current_y)
            #
            current_measurement = np.array([[current_x1], [current_y1], [current_x2], [current_y2]])
            print("current_measurement", current_measurement)
            kalman.correct(current_measurement)
            # print("new_current_measurement", current_measurement)
            #
            current_prediction = kalman.predict()
            print("current_prediction", current_prediction)
            #
            # new_shoulder = [[float(current_prediction[0]),float(current_prediction[1])]]
            # print('new: ',new_shoulder)

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # current_measurement = np.array([angle])
            #  kalman.correct(current_measurement)
            #  current_prediction = kalman.predict()

            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 45 and stage == 'down':
                stage = "up"
                counter += 1
            # print (counter)

            # Visualize angle

            # position=tuple(np.multiply(elbow, [640,480]).astype(int))
            position = (100, 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, str(counter), position, font, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # print (landmarks)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Mediapipe Feed", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    video()
    print("End!")


