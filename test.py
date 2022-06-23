import cv2
import numpy as np
import time

from matplotlib import pyplot as plt


def video():
    points = np.load('D:transformed_ground_truth.npy')
    frame_pointer = 23776  # Start Picture
    x=[]
    y=[]
    while frame_pointer < 27776:#126000
        leg=(points[frame_pointer][2]+points[frame_pointer][22])/2
        # print(leg)
        if frame_pointer == 23776:
            init_point=leg

        diff1 = leg-init_point

        if diff1[1] < 0: # low
            diff = -np.sqrt(diff1[0]**2+diff1[1]**2)
        else:
            diff=np.sqrt(diff1[0]**2+diff1[1]**2)
        y.append(diff)
        t = (frame_pointer-23776)/200
        x.append(t)
        arr = [7, 6, 7]
        frame_pointer += arr[int(np.random.randint(0, 3, 1))]

    # fig = plt.figure(dpi=128, figsize=(10, 6))
    # plt.plot(dates, close, c='blue')
    plt.plot(x,y)
    # plt.xticks(range(23776, 126000, 200))
    plt.show()


if __name__ == '__main__':
    video()
    print("end!")
