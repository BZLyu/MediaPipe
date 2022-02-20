import time

import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def video():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/Users/stella/Desktop/Meidapipe/cut_1.mov')
    prevTime = 0
# Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
        while cap.isOpened():

            ret, image = cap.read()
            if not ret:
                print("Ignoring empty caemra freme.")
                continue
            # Recolor image to RGB
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Make detection
            results = pose.process(image)

            # Recolor image to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extrack landmarks

            # if results.pose_landmarks is None:
            #     continue

            # landmarks = results.pose_landmarks.landmark
            #
            # length_shoulder = math.sqrt(((landmarks[12].x-landmarks[11].x)**2) +
            #                             ((landmarks[12].y-landmarks[11].y)**2))
            # length_hip = math.sqrt(((landmarks[24].x-landmarks[23].x)**2) +
            #                        ((landmarks[24].y-landmarks[23].y)**2))
            # if length_hip < length_shoulder*(1/2) or length_shoulder < length_hip or landmarks[13].x < landmarks[23].x:
            #     continue



            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            currTime = time.time()
            fps = 1/(currTime-prevTime)
            prevTime = currTime

            cv2.putText(image, f'FPS:{int(fps)}', (20, 78), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)


            cv2.imshow("Mediapipe Feed", image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    video()
    print("End!")
