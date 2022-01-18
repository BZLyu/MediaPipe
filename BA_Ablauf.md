# Ablauf

# 第一周 11.8-11.12

安装配置，放弃 Openpose，初次尝试 Mediapipe

# 第二周 11.15-11.19

跟着教程用 pyhton做了写了可以计算举手次数的code。

## 底层逻辑

1. 用 opencv 打开摄像头，注意 mac 和 window 相比，多了一句代码
2. 把得到的数据流（image）从 BGR 转换成 RGB，因为 Meidapipe 只读取 RGB
3. 用 SSD 的方法来确定目标
4. Mediapipe 库中定义了 Landmark ，通过这样来取得身体点
5. 标记点
6. 用 ROI 的方式确定下个 image 的点
7. 连接各个点
8. 再把 RGB 转换成 BGR（输出）
9. 通过连接的线，来计算运动角度
10. 通过角度来确定姿势
11. 通过姿势变化的判断来确定举手次数

## 遇到的问题

1.  本来以为是我距离远了有些举手次数识别不到，结果和导师一起分析，因为在远处，我的身体构架也是能够识别的，也就是说，连接的线及形成的角度也是能够识别。和远近就没有关系了

2. 然后发现有几帧我的手已经抬起来了，但是没有识别到，构架还是停在原地，再观察，可能与我穿的白衣服和白墙有关，机器识别我的衣服和白墙颜色一样，出现错误，这个概率很大

3. 衣服相近，与环境探测不准确可以用 Camara 来改善（具体是什么 Camara，忘记了，好像是和温度有关）

4. 也有可能是 todside 死角

5. 也或许是我的角度设定的范围太小了。或许 360 度更好，不要再 180 之类的，可以有四个 象限

6. 想要知道 xyz,具体是什么，可否通过 再加一个数值来改进（现在只用了 xyz）

   

## 接下来研究的方向

1. 在 SSD 上加上 kalmanfilter  ([ Single Shot Detector (Object Detection)](https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11))
   1. kalmanfilter 对于手部滑动这种固定的，有规律的探测很有用处 (连续的动作)
   2. 所以要根据不同的动作加上不同的 filter
   3. 先用 kalmanfilter 在 Mediapipe 上做出来，再看看其他的 filter
   
2. 为什么大家都喜欢转换成 RGB, 而不直接用，从 Opencv 里面出来的 BGR image

3. （这段有些没听懂）ROI (Region of Interest ) 抓取得 sample 有些 verschieben 点。是说，有 verschieben 的问题还是 通过点的 verschieben 来确定运动规律。或者两点都有讲到，但是我记忆搞混了时间了

4. 之前的改进探测准确度

    

# 第三周 22.11-26.11

##看论文

1.  **A Kalman Filtering Based Data Fusion for Object Tracking**

   1. 把几个摄像头的数据结合在一起
   2. 各自 kalman 然后合在一起

2.   **Single Shot Multibox Detector With Kalman Filter for Online Pedestrian Detection in Video** -2019

   1. 使用 CNN 的 Kalman Filter: 优点比 KNN,SVM 的好，原因是使用 sklearn 图像分类他们不行
   2. 比 MGP(Motion Guided Propagation) 计算负荷低
   3. CNN + SSD 学习人的特征然后标签：SSD 生成每个物体的检测结果（坐标，宽度，高度，置信度分数），然后放入 Kalman 里优化结果，再用 NMS 来消除重叠的边界盒。
   4. SSD 的缺点：对远距离的人准确度低，经常漏检。

3.  **Kalman Filtering Based Object Tracking in Surveillance Video System** -2011

   1. 设计了一种，改进版 Kalman 提高了在追踪物被遮挡，突然消失的情况下，重新回到画面中，可以很快的继续追踪
   2. 本文提出的增强型卡尔曼滤波器还不够完美，因为它不能在运动模式（如正弦运动或人字形运动）的遮挡期间跟踪运动物体。本文提出的增强卡尔曼滤波器只能预测方向和速度固定的被遮挡运动物体的位置。

4. **An_Efficient_and_Robust_Tracking_System**

   1. 使用了 Simplex 算法，它是一种广泛的迭代代数程序，用于为每个问题确定至少一个最优解。作为一种线性优化方法，Simplex算法优化了一个受某些限制的函数。基于MD的成本函数，也称为统计距离，Mahalanobis Distance (MD). 这是通过重新调整成分的比例来实现的。
   2. 使用 NPV 的方法

5.  **A particle ﬁlter for joint detection and tracking of color objects**

   1. 彩色物体的粒子滤器。它的成功可以解释为它有能力应对多模态的测 量密度和非线性观测模型。
   2. 比 KM 好的点在于善于读取非线性的问题：km 是线性高斯估值的最优解
   3. 这种基于颜色的跟踪器使用 颜色直方图作为图像特征，遵循Comaniciu等人[9]提出 的流行的MeanShift跟踪器。由于颜色直方图对部分遮挡具有鲁棒性， 并且是尺度和旋转不变的，因此所产生的算法可以很好 地成功处理非刚性变形。
   4. 相反，当场景中存在几个具有相同颜色描述的物体时 （如足球比赛转播、蚂蚁或蜜蜂群的记录等），颜色 粒子过滤器的方法就会失败，因为粒子被不同的物体 吸引，计算出的状态估计值没有意义，或者粒子倾向 于只跟踪最合适的目标-这种现象被称为凝聚[10]。在这两种情况下，必须找到 替代方法。
   5. 我们开发了一个粒子过滤器，通过在算 法中整合新物体的检测和跟踪多个类似物体，扩展了[7 ,8]的彩色粒子过滤器。对进入场景的新目标的检测， 即跟踪初始化，被嵌入到粒子过滤器中，而不依赖于 外部目标检测算法。此外，所提出的算法可以跟踪多 个共享相同颜色描述并在场景中移动的物体。
   6. 我们在贝叶斯框架中提 出了联合检测和跟踪问题的概念性解决方案，并使用 顺序蒙特卡洛方法加以实现。他们提出了一个排除原则，以避免两个目标靠 近时凝聚到最佳目标上。
   7. 贝叶斯估计法

6.  **An_Efficient_and_Robust_Tracking_System_using_Kalman Filter** -2012

   1. 非线性滤波方法-Unscented Kalman Filter（UKF）
   2. 通过使用无痕变换直接对概率密度进行近似并准确估计状态的均值和协方差，很大程度上避免了线性化误差。因此，基于经典的卡尔曼曲线算法在移动目标跟踪中的缺陷，本文采用了基于无痕变换的无痕卡尔曼曲线算法来跟踪和估计移动目标。
   3. 其核心思想是利用两次无痕变换来获得近似的采样点（Sigma点）并减少线性化误差。同时，卡尔曼滤波是用来计算增益矩阵和更新状态估计和协方差估计的。
   4. 对视频移动目标跟踪的探索和应用有很大影响。UKF算法的提出是为了解决视频移动目标跟踪中使用的经典卡尔曼滤波方法的低精度和滤波分歧的问题。UKF是一种标准的卡尔曼滤波算法，它采取无损变换，将非线性系统方程应用于线性假设，使卡尔曼滤波中的非线性系统具有更高的精度和发散。论文从移动目标检测与跟踪的无跟踪卡尔曼滤波技术出发，结合国内外最新研究成果，实现了视频移动物体检测与跟踪的UKF算法。通过定性分析，特别是统计误差分析，结果表明UKF算法比卡尔曼滤波算法更准确。UKF算法基于UT变换和线性卡尔曼滤波框架，线性误差小，在其他应用中仍然发挥着巨大的影响。本文在细节上可能存在一些问题，希望今后能继续研究和改进

7.  **Data fusion of radar and image measurements for multi-objecttracking via Kalman filtering -2012**

   1. UKF算法的提出是为了解决视频移动目标跟踪中使用的经典卡尔曼滤波方法的低精度和滤波分歧的问题。UKF是一种标准的卡尔曼滤波算法，它采取无损变换，将非线性系统方程应用于线性假设，使卡尔曼滤波中的非线性系统具有更高的精度和发散。论文从移动目标检测与跟踪的无跟踪卡尔曼滤波技术出发，结合国内外最新研究成果，实现了视频移动物体检测与跟踪的UKF算法。通过定性分析，特别是统计误差分析，结果表明UKF算法比卡尔曼滤波算法更准确。UKF算法基于UT变换和线性卡尔曼滤波框架，线性误差小

8. **Embedded Architecture for Object Tracking using Kalman Filter-2015**

   1. 在这项研究中，我们提出了一个使用卡尔曼滤波器进行物体跟踪的并行架构，它可以在速度上进行优化。我们在Virtex 5 XC5VLX200T FPGA器件上实现了拟议的设计。当比较我们提出的卡尔曼滤波器设计的实施结果和在Xilinx

      Virtex-4 XC4VFX140 FPGA器件上提出的设计和(Hancey, 2008)在Xilinx Virtex-4 XC4VSX35 FPGA器件上提出的设计(见表)，我们发现我们的设计比其他实现方式快2倍。这些优化是通过专注于并行化和试图促进该过滤器的数学运算来实现的。

9.  Hand gesture tracking system using Adaptive Kalman Filter-2010

   1. 本文提出了一个解决这些难题的系统。该系统由两个主要阶段组成，即初始化和跟踪。在初始化过程中，手势者需要用他的手做重要的移动，但他身体的其他部分则允许小范围的移动。然后，通过结合皮肤颜色和运动线索实现手的检测，并在检测位置周围创建手的ROI。在跟踪阶段，利用恒定速度模型，我们开发了一个基于手的位置和速度的状态矢量，作为自适应卡尔曼滤波（AKF）的估计过程。根据估计的位置，我们通过扫描ROI的右角、左角和顶角的运动和皮肤像素来计算手的位移，并测量实际的手的位置。测量噪声来自于实际位置和估计位置之间的误差，它被送入AKF的测量更新方程。为了处理非线性运动，即手的速度在变化，AKF的测量噪声协方差和过程噪声协方差通过加速度大小进行自适应调整。
   2. 本文描述了一个用于手部检测和跟踪的系统。自适应卡尔曼滤波器被用来跟踪视频序列中的手势。在拟议的自适应卡尔曼滤波算法中，过程噪声协方差和测量噪声协方差使用基于加速度阈值的加权因子进行自适应调整。实验结果证明了拟议算法的稳健性和有效性，在45fps的速度下，总体成功率为97.78%。在未来，所提出的算法将与手势识别引擎相结合，产生一个实时的人机交互系统。

10.  Improved Tracking of Multiple Humans with Trajectory Prediction and Occlusion Modeling -1998

    1. 恢复非刚性物体的轨迹通常是假设没有结构（例如，跟踪作为点的主体）或使用领域约束（例如，已知的地平面或通过使用场景的俯视图折叠一个维度）。对于运动分析，需要有足够的关于物体形状和它们如何演变的信息。许多以前的方法不能提供这些信息，尽管它们使恢复轨迹的工作简单了很多。

       

11. Kalman Filter Algorithm for Sports Video Moving Target Tracking -2020

    1. 基于自适应卡尔曼滤波的改进型目标跟踪算法，可以减少干扰
    2. 分析了卡尔曼滤波跟踪算法的优缺点，研究了多特征融合卡尔曼滤波算法，并采用自适应方法调整各特征的权重。实验结果表明，在目标和背景颜色相似、目标被其他物体干扰、目标被部分遮挡、背景比较复杂的情况下，改进后的跟踪算法具有较高的跟踪精度和稳定的跟踪效果。
    3. 显著小于卡尔曼滤波算法。通过计算中心偏移距离，单特征卡尔曼滤波算法的平均跟踪偏差约为20像素，而多特征融合卡尔曼滤波算法的平均跟踪偏差约为9像素。可以得出，多特征融合卡尔曼滤波算法的目标跟踪轨迹更接近真实的目标轨迹，跟踪误差相对减小，这也说明多特征融合卡尔曼滤波算法比卡尔曼滤波算法的目标跟踪精度更高。

12.  Kernel-Based Object Tracking -2003

    1. 我们提出了一种新的目标表示和定位方法，这是对非刚性物体进行视觉跟踪的核心部分。基于特征直方图的目标表征通过各向同性核的空间掩蔽进行规范化。

13. Machine Learning-Based Multitarget Tracking of Motion in Sports Video -2021

    1. 本文通过机器学习算法跟踪体育视频中多个目标的运动，并深入研究其跟踪技术。在移动目标检测方面，对传统的检测算法进行了理论上的分析以及算法上的实现，在此基础上提出了四种帧间diﬀerence方法和背景平均法的融合算法，以解决帧间diﬀerence方法和背景diﬀerence方法的不足之处。-融合算法利用学习率实时更新背景，并结合形态学处理来修正前景，可以有效地应对背景的缓慢变化。根据智能视频监控系统对实时性、准确性和占用较少视频存储空间的要求，本文对该算法的精简版进行了改进。-实验结果表明，改进后的多目标跟踪算法有效地改善了基于卡尔曼滤波的算法，满足了智能视频监控场景下的实时性和准确性要求。

    

## 遇到的问题

1. x,y,z 怎么带入 kalmanfilter。什么带入噪音？什么带入 y？
2. 我觉得搞清第 1 点得弄清，SSD 是怎么运行的？
3. 通过 第一次 SSD 有了第1 ，2 个点的位置，然后用 kalmanfilter 算第3个点的位置。因为 kalmanfilter 是一个推测的点和一个测量的点相结合得到一个最可能的实际点。推测的点是通过，一个过去点，和现在的点推测下一个点
4. 卡尔曼是相同方向预测很不错，但是往回的时候是不是误差最大？
5. SSD 和 ROI 区别？ 是不是 ROI 是 SSD 的一部分
6.  什么是闭塞 occlusions, group formation

# 第四周 29.11-03.12

  ## 下周准备做的事 看论文

1. Evaluation von Filter-Ansätzen für die Positionsschätzung von Fahrzeugen mit den Werkzeugen der Sensitivitätsanalyse

## 和导师讨论

1. kalman 是 liner 的。所以我们想办法把不是 liner 的变成 linier 的。

2. 从 ssd得到的(x,y,z) 转换成 Kulgel koodinaten (因为是手臂沿着手肘旋转)，坐标系的转换。可以变成 liner。

3. 这里要注意，SSD 得到的是 房间里的 xyz 还是屏幕里的 xyz？（个人推测是屏幕里的 xyz）

4. 设定两个 kalman filter 一个从上到下，一个从下到上，这样解决了变方向的问题。$c_1=-,c_2=+$

4. 

5. 设定哪个点是起点？哪个点是终点？

6. ![20211201_105432](/Volumes/Life/OneDrive - stud.tu-darmstadt.de/Darmstadt/Arbeit/MedieapipePose/20211201_105432.jpg)

   ![20211201_105449](/Volumes/Life/OneDrive - stud.tu-darmstadt.de/Darmstadt/Arbeit/MedieapipePose/20211201_105449.jpg)

8. 抬手和放下手的速度，加速度 会变。

9. 手速不能太快 $f_{s}=\frac{1}{\Delta t}$ =30HZ 不然读取的图画信息太少 

9. ![2021-12-08 um 2.53.05 PM](/Volumes/Life/OneDrive - stud.tu-darmstadt.de/Darmstadt/Arbeit/MedieapipePose/2021-12-08 um 2.53.05 PM.png)

# 第五周06.12-10.12

1. 老师给的书（Evaluation von Filter-Ansätzen für die Positionsschätzung von Fahrzeugen mit den Werkzeugen der Sensitivitätsanalyse。 看了里面有 圆弧的例子，不需要转换坐标轴，有点麻烦，可以在写论文的时候，作为换坐标轴的对比。显示换坐标轴，方便许多。

# 第六周 13.12-17.12

## 发现

1. pose 里面有 3D 的现实坐标
2. 尝试了 Holistic,发现并没有区别

## 要做

- [ ] 看一下 kalman python 的模型

  - [ ] 我查到了 2 种模型：

    - [ ] 第一个是车匀加速运动，它是通过，坐标加上加速度，速度来建立了一个 4 维（也不是 4 维，是 4 个变量）的模型，然后有一些噪音，它忽略不计了。

    - [ ] 第二个是 跟踪鼠标的。鼠标是没有规律的。

      

- [ ] 带入 Code

- [x] 卡迪尔坐标轴->球坐标：

  - [x] 

  - ```python
    import numpy as np
    import math as m
    
    def cart2sph(x,y,z):
        XsqPlusYsq = x**2 + y**2
        r = m.sqrt(XsqPlusYsq + z**2)               # r
        elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
        az = m.atan2(y,x)                           # phi
        return r, elev, az
    
    def cart2sphA(pts):
        return np.array([cart2sph(x,y,z) for x,y,z in pts])
    
    def appendSpherical(xyz):
        np.hstack((xyz, cart2sphA(xyz)))
    ```

  - [ ] 这个不行，没有 cython.

  - ```cython
    cdef extern from "math.h":
        long double sqrt(long double xx)
        long double atan2(long double a, double b)
    
    import numpy as np
    cimport numpy as np
    cimport cython
    
    ctypedef np.float64_t DTYPE_t
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def appendSpherical(np.ndarray[DTYPE_t,ndim=2] xyz):
        cdef np.ndarray[DTYPE_t,ndim=2] pts = np.empty((xyz.shape[0],6))
        cdef long double XsqPlusYsq
        for i in xrange(xyz.shape[0]):
            pts[i,0] = xyz[i,0]
            pts[i,1] = xyz[i,1]
            pts[i,2] = xyz[i,2]
            XsqPlusYsq = xyz[i,0]**2 + xyz[i,1]**2
            pts[i,3] = sqrt(XsqPlusYsq + xyz[i,2]**2)
            pts[i,4] = atan2(xyz[i,2],sqrt(XsqPlusYsq))
            pts[i,5] = atan2(xyz[i,1],xyz[i,0])
        return pts
    ```

  - [x] 重新写了

  - ```python
    import numpy as np
    xyz=np.array([[2,2,3]])
    def appendSpherical_np(xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2) #r
        #ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up  # theta
        ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0]) # phi
        return ptsnew
      
      print(ptsnew)
      # [[2.         2.         3.         4.12310563 0.81482692 0.78539816]]
     
    ```


  1. Results.pose_landmarks 这个通过 SSD 得到坐标是作为观察值还是事实值-？我认为是观察值。所以得到的这个坐标可以更新吗？通过 Kalman 更新。更新后的值再反馈给图像，展示出来。那么需要考虑的是，之前，反馈的图像，是每一次 SSD 之后反馈的吗？还是通过一个 Array 把坐标都保存下来，然后一次性展示的呢？我认为是，每一次，因为，如果是摄像头的话，就是实时的，假设时间是无限的。所以，必不可以一次性保存，超过大小就崩溃了。就是说，每得到一个值就反馈一次。现在做的事，就是去看代码里面，是不是这样的呢？验证一下。

  2. 说到了，每次优化后展示，那么就是说，我每次展示之前，是可以再使用一次 Kalman更新一次坐标的。因为我的 Kalman 模型是通过现实 3D 笛卡尔坐标 转为 球坐标，来进行的。通过角度来设计的模型。那么我，需要找到，现实 3D 坐标 与 画面坐标 的相互转换公式。这样才能把更新的值，通过画面展示出来。这需要看代码。 

  3. 我先看是应该 SSD 之后得到的值还是，每一次 SSD 之后，用 Kalman 改善后，再带入 SSD。也就是说，是同时进行的吗？—没错，https://1.bp.blogspot.com/-J66lTDBjlgw/XzVwzgeQJ7I/AAAAAAAAGYM/WBIhbOqzi4ICUswEOHv8r7ItJIOJgL9iwCLcBGAsYHQ/s411/image11.jpg 这个图就说明了，每一个 Frame 会有一次 SSD。

  4. 也是说要看，SSD 的原理，是什么？通过先验框和预设框来确定探测到的东西。—>那么，怎么通过框到点的呢？因为 Mediapipe 图像输出里面 是有 点的位置的。Ploze Pose 用了 COCO 2020 keypoint Detection Task ，Blaze Face , PlazePalm 来通过探测到的框，确定了 33 个点。pose_landmark (https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark)，Ploze pose : https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html

  5. Pose_detectoin->pose_landmark （module 里面）

  6. 通过看代码，发现它使用了 **Tensorflow**(bptxt)，**tflite** 来写 SSD 和 Ploze Pose 来确定点

- 综上，我认为，不用管 Tensorflow 的部分，即，直接用 process 得出的 result 进行 Kalman。然后再更新数据库，进行下一次的 ssd 和 kalman。那么，现在需要想的是，Kalman 过后得到的是，新的球坐标的坐标，然后需要重新转换成笛卡尔坐标，（新的实际的 3D 坐标），然后需要验证的是，3D坐标变了后，图像 2D 坐标会跟着变吗？还是，需要找出他们之间的转换公式，手动转换。如果需要手动转换，需要学习使用Tensorflow 吗？

# 第七周 20.12-24.12

## 实现

1. Kalman 鼠标跟踪

   1. ```python
      class pwm:
          def __init__(self):
              self.desired_position = 0.0
              self.actual_position = 0
      
          def nextPWM(self, speed):
              self.desired_position += speed
              movement = round(self.desired_position - self.actual_position)
              self.actual_position += movement
              return movement
      ```

      ```
      function makeVelocityCalculator(e_init, callback) {
          var x = e_init.clientX,
              y = e_init.clientY,
              t = Date.now();
          return function(e) {
              var new_x = e.clientX,
                  new_y = e.clientY,
                  new_t = Date.now();
              var x_dist = new_x - x,
                  y_dist = new_y - y,
                  interval = new_t - t;
              var velocity = Math.sqrt(x_dist*x_dist+y_dist*y_dist)/interval;
              callback(velocity);
              // update values:
              x = new_x;
              y = new_y;
              t = new_t;
          };
      }
      ```

      鼠标速度
      
      可以直接移动距离微分！！以上程序都可以不用!❌
      
      

2. https://www.codenong.com/cs105985289/

- Übertragungsmatrix: https://de.wikipedia.org/wiki/Übergangsmatrix

## 想法

1. 或许可以不用球坐标，可以用 polar coordinate。极坐标就行

# 第九周 27.12-31.12 Urlaub

# 第十周 3.01-07.01

## 看过的网页：

1. https://www.kalmanfilter.net/CN/background_cn.html
2. https://docs.opencv.org/3.4/dd/d6a/classcv_1_1KalmanFilter.html
2. ** https://www.icode9.com/content-4-918215.html
2. 

## 捋一遍逻辑，建模型，写代码

1. 角度的求法
   1. 通过 3D 的 x,y,z->球坐标-> 角度
   2. 通过 2D 的 xy->polar coordinate 极坐标->角度
2. 确定好要用哪个 Kalman 的模型，需要哪些变量的带入？–自适应型，自己写 Kalman函数，无法带入，自适应型的 kalman 的参数需要自己定义(AKF)
3. 通过角度求导直接得出 Ableitung der Zustandsvektor 和 Zustandsvektor der länge
4. 什么是 $\Delta t$? -求Übertragungsmatrix **F**
5. Beobachtungsmatrix **H**=(1 0 0 0）
6. prozessrauschen Q=
7. 以上变量带入 Kalman 得到新的角度
8. 再从角度-> 新坐标-> 替换原来的坐标

## 代码：

1. Initiallized
   1. x~k~=[ $ \varphi$  $$ \dot{\varphi}$$   $\ddot{\varphi} $  $\dddot{\varphi}$ ]

2. Prediction
   1. Set ==$\Delta t $==
   2. P
   3.    ==B 控制矩阵 不为0 ，因为有加速度，且它变化==?? 但是老师给的 B 为 0，控制矩阵和噪声。我不会判断，问老师

## 想和老师沟通：

1. SSD 用来训练识别 手部，脸部的，已经用 google 的资料训练好了。直接调用。Results.pose_landmarks 这个通过 SSD 得到坐标是作为观察值 (Measurement) https://1.bp.blogspot.com/-J66lTDBjlgw/XzVwzgeQJ7I/AAAAAAAAGYM/WBIhbOqzi4ICUswEOHv8r7ItJIOJgL9iwCLcBGAsYHQ/s411/image11.jpg 这个图就说明了，每一个 Frame 会有一次 SSD。

2. 球坐标？-极坐标- 可以有 3D 的，怎么更新？(3D 转 2D 的转换函数不会找，因为需要学习 Tensorflow）Pose_detectoin->pose_landmark （module 里面）

3. $\Delta t$ 怎么设定？wiki 里例子的时间是通过随机间隔来模拟的
   1. 设定初始值时，截取系统时间，每次调用再截取一次系统时间，减去上次的系统时间
   
4. 初始值：
   1. x~k~=[ $ \varphi$  $$ \dot{\varphi}$$   $\ddot{\varphi} $  $\dddot{\varphi}$ $]^T$ ——(n*1)
   2. Durch  [Matrixexponential](https://de.wikipedia.org/wiki/Matrixexponential)   得到：Übergangsmatrix $F=\left[\begin{array}{ccc}1 & \Delta t &\frac{1}{2} \Delta t^{2} &\frac{1}{6}\Delta t^{3} \\ 0 & 1 &\Delta t &\frac{1}{2} \Delta t^{2}\\ 0 & 0 &1 & \Delta t\\ 0 &0&0&1 \end{array}\right]$—–(n*n)
   3. Messwert nur Winkel—$\varphi$  ,so Beobachtunhsmatix ist $H=\left[\begin{array}{llll}1 & 0 & 0 & 0\end{array}\right]$  —1*m
   4. Processrauschen vs. Dynamik der deterministischen Störung und Projektion auf den Systemzustand? G=? 
   
5. 怎么去验证？-通过鼠标运动，验证？

   ------

   

6. 关于 B 和 r 的讨论，r=B*a, 有 r 意味着，B 和 a 都不是 0，设置 B

6. 

## 解答

1. 不需要重新带入 SSD ，直接对比 Kalman 的数据
2. 用 2D
3. 使用包过后，不需要设定时间；但是如果要设定，我可定任意设定一个值，可大可小这样
4. 不用设定 B 直接设定 Q
5. 老师给了一篇论文，可以检测有没有提高准确性

## 要做的：

1. 把 Kalman 带入 Mediapip

   1. 我先用 x,y 坐标轴：
      1. 放入 landmarks: 先放 1个关键点 kalman ✓
      2. 先以图像为中心的坐标轴（原来的坐标轴）-之后以手肘为中心
      3. 角度的kalman  ✓

2. 之后会给仪器，给定身体上真实的坐标轴

   1. 考虑怎么把 3D 的转为 2D

      
   

# 第十一周 10.1-14.01

## 开始动笔写一部分论文

1. 论文 模板？ 排版，字体字号，latex 模板 ,30 页, 从哪些开始算？ Titil 算？ 文献算？

## 代码完善

1. 公式写出来，F 怎么算，Q 怎么算，r 怎么算

2. 我想优化所有33个Landmarke。我首先为1个点做了一个卡尔曼滤波器。如图所示。

   我有两个想法。
   1.建立33个卡尔曼过滤器。
   2.建立一个卡尔曼滤波器。每个地标有4种状态，共有33个Landmarke。那么x就是（4**33）*（1）Matrix。

   对于第二种想法，我不确定。因为如图片所示。如果我用2个地标建立一个卡尔曼滤波器。D看起来并不好。我怀疑的原因是：卡尔曼滤波器只能计算一个点，因为多个点不是线性的。

3. 全身的点连成新的线.

## Frage stellen: 老师讨论

Über Kalman-Filter von Armwinkel möchte ich einige Frage stellen. Vorher denke wir, dass wir zwei Kalman-Filter brauchen, um die Genauigkeit zu verbessern. Ein ist für nach oben gehen, andere ist für nach unter gehen. Wie kann man diese 2 Kalman-Filter aufbauen? Ich habe folgenden unvollständigen Idee:

1. Zwei unterschiedliche Kalman-Filter aufbauen. Ich setze Richtung oben ist positiv und Unter ist negativ. Die Winkel ist immer positiv in zwischen 30°-180° (Ideale Situation), die Beschleunigung und die Änderung der Beschleunigung kann positiv und negative sein. Dann ich setzt diese 3 Variable anfangen immer positiv sein. Wie Bild zeigt. Ich möchte wissen, ob ich ein richtig Modul bilden?![Winkel Kalman](/Volumes/Life/OneDrive - stud.tu-darmstadt.de/Darmstadt/Arbeit/Papers/Winkel Kalman.jpeg)

2. Zwei gleiche Kalman-Filter aufbauen. Sie sind in passende Zeitpunkt gewelchselt werden. Einen von sie wird aktiviert, wenn die Winkel sich vergrößern. Andere wird aktiviert, wenn die Winkel sich verkleinern.

3. Wie kann man die Zeitpunkt für Wechsel der Kalman-Filter richtig setzten? Meine Meinung nach ist die Zeitpunkt, wann Winkelgeschwindigkeit gleich 0 ist. Wie denkst du?

   -------------------------------------------------------------------------------

4. Und Ich möchte alle 33 Landmarken optimieren dann die verbinden, denn die optimierenden Ergebnisse kann im Bildschirm ersichtlich sein. Ich habe zuerst einen Kalman-Filter für 1 Punkt gemacht. Wie Bild zeigt.![x,y_Kalman](/Volumes/Life/OneDrive - stud.tu-darmstadt.de/Darmstadt/Arbeit/Papers/x,y_Kalman.jpeg)

5. Dann machte ich weiter. An Anfangen hatte Ich 2 Gedanken:
   1. 33 Kalman-Filter aufbauen.
   2. Ein Kalman-Filter aufbauen. Jede Landmarke hat 4 Status und es gibt 33 Landmarke. Dann State ist ein [4*33]x[1] Matrix.
      1. Für 2. Gedanken wird wie folgendes Bild zeigt. Falls ich einen Kalman-Filter mit 2 Landmarken aufbauen. D sieht nicht gut aus. Den Grund vermute ich: Kalman-Filter kann nur ein Punkt rechnen. Mehrere Punkte sind nicht passend. Aber Warum?

![2_Punkt_Kalman](/Volumes/Life/OneDrive - stud.tu-darmstadt.de/Darmstadt/Arbeit/Papers/2_Punkt_Kalman.jpeg)

\--------------------------------------------------------------------------------

Ich bin nicht sicher ob Optimierung für alle Punkt nötig ist. Denn durch Winkelzähler kann man schon die Unterschiede sehen.

------------------------------------------------------------

### 第二部分讨论

1. simulated annealing：
   1. zu urteilen, ob Ergbnis gut genug ist und ob zu nächte Status gehen 
   2. Aufforderung der Zustand ist/ 2 Voraussetzung: Rechnen bis (gegebene) genuge Mal oder Ergbnis genuge gut.
   3. 
2. **Fragen: ** wie kann Kalman Filler vorige Status bleiben ? oder 

## 读老师给得文献

Optimization and Filtering for Human Motion Capture

A Multi-layer Framework

1. 摘要 局部优化和排序已被广泛地应用于基于模型的三维人体运动捕捉。全局随机优化最近被提出来作为跟踪和初始化的有前途的替代解决方案。为了从优化和调整中获益，我们引入了一个多层框架，将随机优化、调整和局部优化结合起来。第一层依赖于交互式模拟退火（**simulated annealing**）和一些关于物理约束的弱先验信息，第二层则通过修正和局部优化对估计值进行修正，从而提高精确度，并在不对动力学施加限制的情况下逐步解决模糊问题。在我们的实验评估中，我们证明了多层框架的显著改进，并为完整的HumanEva-II数据集提供了定量的三维姿势跟踪结果。本文还包括全局随机优化与粒子修正、退火粒子修正和局部优化的比较。

## 构架

1. simulated annealing:
   1. 算法思想为：先从一个较高的初始温度出发，逐渐降低温度，直到温度降低到满足热平衡条件为止。在每个温度下，进行n轮搜索，每轮搜索时对旧解添加随机扰动生成新解，并按一定规则接受新解。





## 关于角度计算：

1. Artan2 ：unabhängig von positiv oder negative
2. https://stackoverflow.com/questions/7235839/calculating-the-angle-between-the-horizontal-and-a-line-through-two-points
3. https://pythonmana.com/2021/12/202112111704472145.html
4. https://blog.csdn.net/m0_37316917/article/details/102728869



## 讨论完要做的事：

1. 修正角度。30->45 ✓
2. 33 个点优化
   1. 分成小块 33 个一个点的,合成一个一个大的，其实 就是造 一个大的 kalman 然后位置更好确定。
   2. 分块矩阵 np.block ✗
   3. 无法确定能否成功，先用 2个点来试试
      1. 没有成功：
      2. 考虑：其实是卡尔曼滤波器多目标跟踪 （然后查到：多目标我们一般不会去用Kalman，而是用粒子滤波和PHD滤波。但也找到这个： https://blog.51cto.com/u_15287693/3026290
      3. 在检测很准的情况下，现在基于的深度学习的检测算法，比如ssd都可以较为准备的检测目标，但是不论检测有多准，漏检的问题是无法避免的，比如目标间相互遮挡，等等，甚至就是漏检。当发生这种情况时才能体现出卡尔曼滤波的作用，虽然我没有检测到，但是我能根据之前的运动状态估算出当前目标的位置，当目标再次出现时重新跟踪上目标，我认为这一点是kalman的意义所在。
      4. 并没有！！ 找到相关资料，是 H 设置有问题：https://www.kalmanfilter.net/stateUpdate.html
      5. 有个问题，就是在一个点的时候，**Q 设置不对，只能设为单位矩阵，按照算法算出，带入结果是错的。**
         1. 决定造 33 个 Kalman，放到 list 里面，index 0：(x,y), index 1 :每个位置对应一个 kalmanfilter，index 2: 预测的(x,y)
3. 两个相同的 kalman 来判定上下的标准
   1. vorzeichen ändern？
   1. State. 为什么只做到速度而不是做到加速度, 建模的时候要注意。看看文献找到答案
4. 正面，侧面的判定（通过 unteramt 的臂长变化）
5. 灭火原理，判定优化和不和标准。

# 第十二周 17.01-21.01





