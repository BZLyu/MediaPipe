#Pose Classification Options

https://developers.google.com/ml-kit/vision/pose-detection/classifying-poses

##Pose classification and repetition counting with the k-NN algorithm

The Z Coordinate is an experimental value that is calculated for every landmark. It is measured in "image pixels" like the X and Y coordinates, but it is not a true 3D value. The Z axis is perpendicular to the camera and passes between a subject's hips. The origin of the Z axis is approximately the center point between the hips (left/right and front/back relative to the camera). Negative Z values are towards the camera; positive values are away from it. The Z coordinate does not have an upper or lower bound.



Z坐标是为每个地标计算的一个实验值。它与X和Y坐标一样以 "图像像素 "为单位进行测量，但它不是一个真正的三维值。Z轴垂直于相机，从被摄者的臀部之间穿过。Z轴的原点大约是臀部之间的中心点（相对于相机的左/右和前/后）。负的Z值是朝向摄像机的；正的值是远离摄像机的。Z坐标没有上限或下限。

This produces a set of pose landmarks to be used for training. We are not interested in the pose detection itself, since we will be training our own model in the next step.

The k-NN algorithm we've chosen for custom pose classification requires a feature vector representation for each sample and a metric to compute the distance between two vectors to find the target nearest to the pose sample. This means we must convert the pose landmarks we just obtained.

To convert pose landmarks to a feature vector, we use the pairwise distances between predefined lists of pose joints, such as the distance between wrist and shoulder, ankle and hip, and left and right wrists. Since the scale of images can vary, we normalized the poses to have the same torso size and vertical torso orientation before converting the landmarks.

这将产生一组用于训练的姿势地标。我们对姿势检测本身不感兴趣，因为我们将在下一步训练我们自己的模型。

我们选择的用于自定义姿势分类的k-NN算法需要为每个样本提供一个特征向量表示，以及一个计算两个向量之间距离的指标，以找到与姿势样本最近的目标。这意味着我们必须转换我们刚刚获得的姿势地标。

为了将姿势地标转换为特征向量，我们使用预定义的姿势关节列表之间的成对距离，如手腕和肩膀、脚踝和臀部、左手腕和右手腕之间的距离。由于图像的比例可能不同，在转换地标之前，我们将姿势归一化，使其具有相同的躯干尺寸和垂直躯干方向。



To count repetitions, we used another Colab algorithm to monitor the probability threshold of a target pose position. For example:

When the probability of the "down" pose class passes a given threshold for the first time, the algorithm marks that the "down" pose class is entered.
When the probability drops below the threshold, the algorithm marks that the "down" pose class has been exited and increases the counter.



为了计算重复次数，我们使用另一种Colab算法来监测目标姿势位置的概率阈值。比如说。

当 "向下 "姿势类的概率首次超过一个给定的阈值时，该算法标志着 "向下 "姿势类已经进入。
当概率下降到阈值以下时，该算法标志着 "向下 "姿势类已被退出，并增加计数器。

## Recognizing simple gestures by calculating landmark distance

When two or more landmarks are close to each other, they can be used to recognize gestures. For example, when the landmark for one or more fingers on a hand is close to the landmark for the nose, you can infer the user is most likely touching their face.

当两个或多个地标相互靠近时，它们可以被用来识别手势。例如，当一只手的一个或多个手指的地标靠近鼻子的地标时，你可以推断出用户很可能在触摸他们的脸。



## Recognizing a yoga pose with angle heuristics

You can identify a yoga pose by computing the angles of various joints. For example, Figure 2 below shows the Warrior II yoga pose. The approximate angles that identify this pose are written in:

你可以通过计算各种关节的角度来识别一个瑜伽姿势。例如，下面的图2显示了战士II的瑜伽姿势。识别这个姿势的大致角度写在。

This pose can be described as the following combination of approximate body part angles:

90 degree angle at both shoulders
180 degrees at both elbows
90 degree angle at the front leg and waist
180 degree angle at the back knee
135 degree angle at the waist
You can use the pose landmarks to compute these angles. For example, the angle at the right front leg and waist is the angle between the line from the right shoulder to the right hip, and the line from the right hip to the right knee.

Once you've computed all the angles needed to identify the pose, you can check to see if there's a match, in which case you've recognized the pose.

The code snippet below demonstrates how to use the X and Y coordinates to calculate the angle between two body parts. This approach to classification has some limitations. By only checking X and Y, the calculated angles vary according to the angle between the subject and the camera. You'll get the best results with a level, straight forward, head-on image. You could also try extending this algorithm by making use of the Z coordinate and see if it performs better for your use case.

这个姿势可以描述为以下身体部位角度的大致组合。

‚两肩呈90度角
两肘呈180度角
前腿和腰部90度角
后腿膝盖处180度角
腰部135度角
你可以使用姿势地标来计算这些角度。例如，右前腿和腰部的角度是右肩到右臀部的线，以及右臀部到右膝的线之间的角度。

一旦你计算了识别姿势所需的所有角度，你就可以检查是否有匹配的角度，在这种情况下，你就识别了这个姿势。

下面的代码片段演示了如何使用X和Y坐标来计算两个身体部位之间的角度。这种分类方法有一些局限性。通过只检查X和Y，计算出来的角度会根据被摄体和相机之间的角度而变化。你会在一个水平的、直线的、正面的图像中得到最好的结果。你也可以尝试通过利用Z坐标来扩展这个算法，看看它是否对你的用例表现得更好。