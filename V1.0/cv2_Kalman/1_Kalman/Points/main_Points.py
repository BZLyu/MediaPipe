import cv2
import mediapipe as mp
import numpy as np
import math
import kalman_points


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

            prediction = kalman_points.all_points(landmarks, mp_pose)
            # print("prediction: ", prediction.shape)
            new_shoulder = prediction[13]
            # print("old_shoulder", shoulder)
            # print("new_shoulder", new_shoulder)

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 45 and stage == 'down':
                stage = "up"
                counter += 1
            # print (counter)

            # Visualize angle

            # position=tuple(np.multiply(elbow, [640,480]).astype(int))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, str(counter), (100, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # print (landmarks)

            # Render detections

            point_new = tuple(np.multiply(new_shoulder, [1280, 720]).astype(int))
            point_old = tuple(np.multiply(elbow, [1280, 720]).astype(int))
            cv2.circle(image, point_new, 10, (255, 255, 255), -1)
            cv2.circle(image, point_old, 10, (0, 0, 255), -1)

            # [640, 480]

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# float(new_shoulder[0]), float(new_shoulder[1])
# cv2.circle(输入的图片data,圆心位置,圆的半径,圆的颜色,圆形轮廓的粗细（如果为正）负数(-1)表示要绘制实心圆,圆边界的类型,中心坐标和半径值中的小数位数)
            # mp_drawing.draw_landmarks(image,)
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
