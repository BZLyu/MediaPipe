# Author: Bingzhen Lyu 
# Date: 2022/1/18

import math

def body_view(landmarks, stage, front):
    # 13 elbow, 15 wrist ,11 shoulder

    if stage == 'down':  # when stage is down, the status of body view will not change.
        return front
    else:
        # 1/5 of the arm as a criterion, the elbow x is more than 1/5 of the arm as the side
        left_upper_arm = math.sqrt(((landmarks[13].x-landmarks[11].x)**2)+((landmarks[13].y-landmarks[11].y)**2))

        if (landmarks[13].x-landmarks[11].x) > left_upper_arm * 0.2:
            front = False
        else:
            front = True

    return front
