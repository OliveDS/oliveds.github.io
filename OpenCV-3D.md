# OpenCV相机校准和3D重建

## 3D重建步骤

1. 相机校准：使用一组图像来推断相机的焦距和光学中心

2. 获取undistort图像：摆脱用于重建的图片中的镜头失真

3. 特征匹配：在两个图片之间寻找相似的特征并构建深度图

4. 重新投影点：使用深度图将像素重新投影到3D空间

5. 构建点云：生成包含3D空间中的点的新文件以进行可视化

## 相机校准
使用两张从不同角度拍摄的同一物体的照片, 就可以进行3D重建, 与人眼的原理类似. 具体的方法是在两张图片中寻找相同的东西, 并从位置差异中推断出深度, 也就是立体匹配.
为了进行立体匹配, 使两个图像具有完全相同的特征是很重要的. 也就是要求两张图片都不能有任何失真. 但实际上, 大多数相机镜头都会导致失真. 因此, 为了准确地进行立体匹配, 需要知道所使用相机的光学中心和焦距. 因此, 需要首先进行相机校准.

### findChessboardCorners

`cv2.findChessboardCorners`是`opencv`的一个函数, 可以用来寻找棋盘图的内角点位置. 该函数试图确定输入图像是否是棋盘模式，并确定角点的位置。如果所有角点都被检测到且它们都被以一定顺序排布，函数返回非零值，否则在函数不能发现所有角点或者记录它们地情况下，函数返回0. 函数形式为:

```python
# C++
bool findChessboardCorners(InputArray image, Size patternSize, OutputArray corners, int flags=CALIB_CB_ADAPTIVE_THRESH+CALIB_CB_NORMALIZE_IMAGE )
# Python
cv2.findChessboardCorners(image, patternSize[, corners[, flags]]) → retval, corners
```

|参数|含义|
|----------|----------------------|
|Image|输入的棋盘图，必须是8位的灰度或者彩色图像|
|pattern_size|棋盘图中每行和每列角点的个数|
|Corners|检测到的角点|
|corner_count|输出，角点的个数。如果不是NULL，函数将检测到的角点的个数存储于此变量|
|Flags|各种操作标志，可以是0或者下面值的组合：CV_CALIB_CB_ADAPTIVE_THRESH -使用自适应阈值（通过平均图像亮度计算得到）将图像转换为黑白图，而不是一个固定的阈值 CV_CALIB_CB_NORMALIZE_IMAGE -在利用固定阈值或者自适应的阈值进行二值化之前，先使用cvNormalizeHist来均衡化图像亮度 CV_CALIB_CB_FILTER_QUADS -使用其他的准则（如轮廓面积，周长，方形形状）来去除在轮廓检测阶段检测到的错误方块|

### calibrateCamera

`cv2.calibrateCamera`是`opencv`的一个函数, 用来标定模块,其函数形式为:

```python
# C++
double calibrateCamera(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags=0, TermCriteria criteria=TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON) )
# Python
cv2.calibrateCamera(objectPoints, imagePoints, imageSize[, cameraMatrix[, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]]]) → retval, cameraMatrix, distCoeffs, rvecs, tvecs
```

|参数|含义|
|----------|----------------------|
|objectPoints|世界坐标系中的点。在使用时，应该输入一个三维点的vector的vector，即vector<vector<Point3f>> objectPoints|
|imagePoints|对应的图像点。和objectPoints一样，应该输入std::vector<std::vector<cv::Point2f>> imagePoints型的变量|
|imageSize|图像的大小，在计算相机的内参数和畸变矩阵需要用到|
|cameraMatrix|内参数矩阵。输入一个cv::Mat cameraMatrix即可|
|distCoeffs|畸变矩阵。输入一个cv::Mat distCoeffs即可|
|rvecs|旋转向量；应该输入一个cv::Mat的vector，即vector<cv::Mat> rvecs因为每个vector<Point3f>会得到一个rvecs|
|tvecs|位移向量；和rvecs一样，也应该为vector<cv::Mat> tvecs|
|flags|标定时所采用的算法。可如下某个或者某几个参数： CV_CALIB_USE_INTRINSIC_GUESS：使用该参数时，在cameraMatrix矩阵中应该有fx,fy,cx,cy的估计值。否则的话，将初始化(cx,cy）图像的中心点，使用最小二乘估算出fx，fy。如果内参数矩阵和畸变居中已知的时候，应该标定模块中的solvePnP()函数计算外参数矩阵 CV_CALIB_FIX_PRINCIPAL_POINT：在进行优化时会固定光轴点。当CV_CALIB_USE_INTRINSIC_GUESS参数被设置，光轴点将保持在中心或者某个输入的值 CV_CALIB_FIX_ASPECT_RATIO：固定fx/fy的比值，只将fy作为可变量，进行优化计算。当CV_CALIB_USE_INTRINSIC_GUESS没有被设置，fx和fy将会被忽略。只有fx/fy的比值在计算中会被用到 CV_CALIB_ZERO_TANGENT_DIST：设定切向畸变参数（p1,p2）为零 CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6：对应的径向畸变在优化中保持不变 CV_CALIB_RATIONAL_MODEL：计算k4，k5，k6三个畸变参数。如果没有设置，则只计算其它5个畸变参数|

### 代码
```python
#! /usr/local/bin/python
# -*- coding: UTF-8 -*-
'''
Created by Omar Padierna "Para11ax" on Jan 1 2019
'''
import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
#============================================
# Camera calibration
#============================================
#Define size of chessboard target. 
chessboard_size = (7,5) # 使用7*5网格
#Define arrays to save detected points
obj_points = [] #3D points in real world space 
img_points = [] #3D points in image plane
#Prepare grid and points to display
objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
#read images
calibration_paths = glob.glob('./calibration_images/*')# 棋盘图像
#Iterate over images to find intrinsic matrix
for image_path in tqdm(calibration_paths):
	#Load image
	image = cv2.imread(image_path)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("Image loaded, Analizying...")
	#find chessboard corners
	ret,corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
	if ret == True:
		print("Chessboard detected!")
		print(image_path)
		#define criteria for subpixel accuracy
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		#refine corner location (to subpixel accuracy) based on criteria.
		cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
		obj_points.append(objp)
		img_points.append(corners)
#Calibrate camera # 校准相机
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,gray_image.shape[::-1], None, None)
#Save parameters into numpy file
#同一相机只需校准一次,保存参数文件,以备使用
np.save("./camera_params/ret", ret)
np.save("./camera_params/K", K)
np.save("./camera_params/dist", dist)
np.save("./camera_params/rvecs", rvecs)
np.save("./camera_params/tvecs", tvecs)
#Get exif data in order to get focal length. 
exif_img = PIL.Image.open(calibration_paths[0])
exif_data = {
	PIL.ExifTags.TAGS[k]:v
	for k, v in exif_img._getexif().items()
	if k in PIL.ExifTags.TAGS}
#Get focal length in tuple form
focal_length_exif = exif_data['FocalLength']
#Get focal length in decimal form
focal_length = focal_length_exif[0]/focal_length_exif[1]
#Save focal length
np.save("./camera_params/FocalLength", focal_length)
#Calculate projection error. 
mean_error = 0
for i in range(len(obj_points)):
	img_points2, _ = cv2.projectPoints(obj_points[i],rvecs[i],tvecs[i], K, dist)
	error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2)/len(img_points2)
	mean_error += error
total_error = mean_error/len(obj_points)
print (total_error)
```

### 运行结果

程序逐个加载每张棋盘照片, 检测棋盘角点, 并进行标定. 计算相机参数, 保存为`numpy`类型文件.

![calibrateCamera](calibrateCamera.png)

## 3D重建

### StereoSGBM
`StereoSGBM`是`OpenCV`提供的用于立体匹配的类,可将两幅由处在同一水平线的不同摄像机拍摄的图像进行匹配,比较物体在两幅图像中的相对位置,计算求得其视差图.

#### create()

`StereoSGBM`类中创建`StereoSGBM`对象的方法为`create()`

```python
# create() 创建StereoSGBM对象
# C++
static Ptr<StereoSGBM> cv::StereoSGBM::create (int minDisparity = 0,int  numDisparities = 16,int 	blockSize = 3,int P1 = 0,int P2 = 0,int disp12MaxDiff = 0,int preFilterCap = 0,int uniquenessRatio = 0,int speckleWindowSize = 0,int speckleRange = 0,int mode = StereoSGBM::MODE_SGBM)
# Python
retval = cv2.StereoSGBM_create([，minDisparity [，numDisparities [，blockSize [，P1 [，P2 [，disp12MaxDiff [，preFilterCap [，uniquenessRatio [，speckleWindowSize [，speckleRange [，mode]]]]]]]]]]]])
```

其参数如下.

|参数|含义|
|------|----------------|
|minDisparity|最小可能的差异值。通常情况下，它是零，但有时整流算法可能会改变图像，所以这个参数需要作相应的调整。|
|numDisparities|最大差异减去最小差异。该值总是大于零。在当前的实现中，该参数必须可以被16整除。|
|BLOCKSIZE|匹配的块大小。它必须是> = 1的奇数。通常情况下，它应该在3..11的范围内。|
|P1	|控制视差平滑度的第一个参数。|
|P2	|第二个参数控制视差平滑度。值越大，差异越平滑。P1是相邻像素之间的视差变化加或减1的惩罚。P2是相邻像素之间的视差变化超过1的惩罚。该算法需要P2> P1。请参见stereo_match.cpp示例，其中显示了一些相当好的P1和P2值（分别为8 number_of_image_channels SADWindowSize SADWindowSize和32 number_of_image_channels SADWindowSize SADWindowSize）。|
|disp12MaxDiff|左右视差检查中允许的最大差异（以整数像素为单位）。将其设置为非正值以禁用检查。|
|preFilterCap|预滤波图像像素的截断值。该算法首先计算每个像素的x导数，并通过[-preFilterCap，preFilterCap]间隔剪切其值。结果值传递给Birchfield-Tomasi像素成本函数。|
|uniquenessRatio|最佳（最小）计算成本函数值应该“赢”第二个最佳值以考虑找到的匹配正确的百分比保证金。通常，5-15范围内的值就足够了。|
|speckleWindowSize|平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。|
|speckleRange|每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了。|
|mode|将其设置为StereoSGBM :: MODE_HH以运行全尺寸双通道动态编程算法。它将消耗O（W H numDisparities）字节，这对640x480立体声很大，对于HD尺寸的图片很大。默认情况下，它被设置为false。|

#### compute()

`StereoSGBM`类中计算`StereoSGBM`的方法为`compute()`

```python
# compute() 计算StereoSGBM
# C++
public void compute(Mat left,Mat right,Mat disp)
# Python
disp = StereoSGBM.compute(left,right)
```

其参数如下.

|参数|含义|
|------|----------------|
|left|左目图像矩阵|
|right|右目图像矩阵|
|disp|StereoSGBM结果矩阵|

### reprojectImageTo3D

`reprojectImageTo3D`是`OpenCV`提供的根据一组差异图像构建3D空间的函数, 该函数变换一个单通道代表三维表面的三通道图像的视差图. 其函数形式为:

```python
# C++
void reprojectImageTo3D(InputArray disparity, OutputArray _3dImage, InputArray Q, bool handleMissingValues=false, int ddepth=-1 )
# Python
cv2.reprojectImageTo3D(disparity, Q[, _3dImage[, handleMissingValues[, ddepth]]]) → _3dImage¶
```
|参数|含义|
|------|----------------|
|disparity|视差图像。可以是8位无符号，16位有符号或者32位有符号的浮点图像。|
|_3dImage |和视差图同样大小的3通道浮点图像。_3dImage （x,y）位置（也是视察图的(x,y)）包含3D坐标点。|
|Q  |透视变换矩阵。 可以通过stereoRectify（）获得。|
|handleMissingValues|是否处理缺失值（即点差距不计算）。如果handleMissingValues​​=true，则具有最小视差对应的异常值（见StereoBM::operator符（）） 否则，具有非常大的Z值​​（目前设置为10000）转化为三维点的像素。|
|ddepth |可选的输出数组的深度。如果是-1，输出图像将有深度CV_32F。也可以设置ddepth的CV_16S，的CV_32S或CV_32F。|

### 代码

```python
#! /usr/local/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt 
#=====================================
# Function declarations
#=====================================
#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])
	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')
#Function that Downsamples image x number (reduce_factor) of times. 
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape
		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image
#=========================================================
# Stereo 3D reconstruction 
#=========================================================
#Load camera parameters
ret = np.load('./camera_params/ret.npy')
K = np.load('./camera_params/K.npy')
dist = np.load('./camera_params/dist.npy')
#Load pictures
img_1 = cv2.imread('left1.jpg') # 左目图像
cv2.imshow('Left Image', img_1)
img_2 = cv2.imread('right1.jpg') # 右目图像
cv2.imshow('Right Image', img_2)
#Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size and height. 
h,w = img_2.shape[:2]
#Get optimal camera matrix for better undistortion 
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
#Undistort images
img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)
#Downsample each image 3 times (because they're too big)
img_1_downsampled = downsample_image(img_1_undistorted,3)
img_2_downsampled = downsample_image(img_2_undistorted,3)
window_size = 5 # 匹配的块大小 > = 1的奇数
min_disp = 16 # 最小可能的差异值
num_disp = 192-min_disp # 最大差异减去最小差异
blockSize = window_size # 匹配的块大小
uniquenessRatio = 1 # 最佳（最小）计算成本函数值
speckleRange = 3 # 每个连接组件内的最大视差变化
speckleWindowSize = 3 # 平滑视差区域的最大尺寸
disp12MaxDiff = 200 # 左右视差检查中允许的最大差异
P1 = 600 # 控制视差平滑度的第一个参数
P2 = 2400 # 第二个参数控制视差平滑度
imgL = cv2.imread('left1.jpg') # 左目图像
cv2.imshow('Left Image', imgL)
imgR = cv2.imread('right1.jpg') # 右目图像
cv2.imshow('Right Image', imgR)
# 创建StereoSGBM对象并计算
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,blockSize = window_size,uniquenessRatio = uniquenessRatio,speckleRange = speckleRange,speckleWindowSize = speckleWindowSize,disp12MaxDiff = disp12MaxDiff,P1 = P1,P2 = P2)
disparity_map = stereo.compute(imgL, imgR).astype(np.float32) / 16.0 # 计算视差图
#Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
plt.imshow(disparity_map,'gray') #视差图
plt.show()
#生成点云图
print ("\nGenerating the 3D map...")
#Get new downsampled width and height 
h,w = img_2_downsampled.shape[:2]
#Load focal length. 
focal_length = np.load('./camera_params/FocalLength.npy')
#Perspective transformation matrix
#This transformation matrix is from the openCV documentation, didn't seem to work for me. 
Q = np.float32([[1,0,0,-w/2.0],
				[0,-1,0,h/2.0],
				[0,0,0,-focal_length],
				[0,0,1,0]])
#This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
#Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q2 = np.float32([[1,0,0,0],
				[0,-1,0,0],
				[0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
				[0,0,0,1]])
#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
#Get color points
colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)
#Get rid of points with value 0 (i.e no depth)
mask_map = disparity_map > disparity_map.min()
#Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = colors[mask_map]
#Define name for output file
output_file = 'reconstructed.ply'
#Generate point cloud 
print ("\n Creating the output file... \n")
create_output(output_points, output_colors, output_file)
```

### 运行结果

原双目图像

![left](left.png)

![right](right.png)

点云图像

![point-clouds](pointclouds.png)

鼠标移动到图像中的某点位置,下方显示该点的点云x,y坐标
