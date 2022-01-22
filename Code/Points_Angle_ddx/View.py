# Author: Bingzhen Lyu 
# Date: 2022/1/18

import math


def body_view(landmarks, old_left_underarm, initial):
    # print("shoulder", landmarks[11].x, landmarks[11].y)
    # print("wrist", landmarks[15].x, landmarks[15].y)

    # 13 eblow, 15 wrist ,11 shoulder
    left_underarm = math.sqrt(((landmarks[13].x-landmarks[15].x)**2)+((landmarks[13].y-landmarks[15].y)**2))

    if initial is True:
        return True, left_underarm, False
    # how large is underarm change.
    wrist_is_left = landmarks[15].x - landmarks[11].x
    # print("diff_x_shoulder_wrist", wrist_is_left)

    diff = abs(old_left_underarm - left_underarm)

    distance_shoulder_wrist = math.sqrt(((landmarks[11].x-landmarks[15].x)**2)+((landmarks[11].y-landmarks[15].y)**2))
    # print("distance_shoulder_wrist", distance_shoulder_wrist)
    # print("diff", diff)

    if distance_shoulder_wrist < 0.060:
        if wrist_is_left < 0.06:
            return True, left_underarm, False
        return False, left_underarm, False
    elif diff > 0.063:
        return True, left_underarm, False

    return False, left_underarm, False
