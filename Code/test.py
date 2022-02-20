import cv2
import numpy as np

def video():

    a = [1,1,1,1,1,1,np.nan,1,1,1,1,1]
    b = np.isnan(a)
    print(b)
    if True in b:
        print("a has nan.")

if __name__ == '__main__':
    video()
    print("end!")
