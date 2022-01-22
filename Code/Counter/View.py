# Author: Bingzhen Lyu 
# Date: 2022/1/18

import math


def body_view(landmarks, stage, front):
    if stage == 'down':
        return front
    if landmarks[11].x < landmarks[15].x < landmarks[13].x:  # wrist is between elbow and shoulder
        front = False
    else: # 手臂长度 必须要手臂的 x 差不多，才是侧面
        left_upper_arm = math.sqrt(((landmarks[13].x-landmarks[11].x)**2)+((landmarks[13].y-landmarks[11].y)**2))
        print("x_:", landmarks[13].x-landmarks[11].x)
        if left_upper_arm > (landmarks[13].x-landmarks[11].x):
            front = True
        else:
            front = False
    return front

    # # 13 elbow, 15 wrist ,11 shoulder
    # left_underarm = math.sqrt(((landmarks[13].x-landmarks[15].x)**2)+((landmarks[13].y-landmarks[15].y)**2))
    #
    # # how large is underarm change.
    # elbow_is_left = landmarks[13].x - landmarks[11].x
    # # print("diff_x_shoulder_wrist", wrist_is_left)
    #
    # diff = abs(old_left_underarm - left_underarm)
    #
    # distance_shoulder_wrist = math.sqrt(((landmarks[11].x-landmarks[15].x)**2)+((landmarks[11].y-landmarks[15].y)**2))
    # # print("distance_shoulder_wrist", distance_shoulder_wrist)
    # # print("diff", diff)
    #
    # if distance_shoulder_wrist < 0.060:
    #     if wrist_is_left < 0.06:
    #         return True, left_underarm, False
    #     return False, left_underarm, False
    # elif diff > 0.063:
    #     return True, left_underarm, False
    #
    #
