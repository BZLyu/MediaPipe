import cv2
import numpy as np
import time

def video():
    cap = cv2.VideoCapture('D:cut_1.mp4')
    prevTime =0
    while cap.isOpened():

        ret, image = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        currTime = time.time()
        if currTime == prevTime:
            continue
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        if fps < 1:
            continue
        # fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(image, f'FPS:{int(fps)}', (20, 78), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

        cv2.imshow("Mediapipe mit Kalman", image)
        # TODO: exit
        wait= int( (1 / int(fps)) * 1000)
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    video()
    print("end!")
