import cv2
import numpy as np
import os

from src.feature_detection import FeatureDetection
from src.feature_description import FeatureDescribe
from src.keypoint_orientation import kptsOrientation

from src.match import match_keypoints_nn
import time

def convert_to_rgb_from_grayscale(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image

def convert_to_grayscale(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

time1 = time.time()
image_folder_path = "images/sar-optical"

# Load images
str1 = os.path.join(image_folder_path, 'pair1.jpg')
str2 = os.path.join(image_folder_path, 'pair2.jpg')

img1 = cv2.imread(str1)
img2 = cv2.imread(str2)

#make images grayscale
img1_grayscale = convert_to_grayscale(img1)
img2_grayscale = convert_to_grayscale(img2)
img1_hw = img1_grayscale.shape
img2_hw = img2_grayscale.shape

print('RIFT2 feature detection')
key1, m1, eo1 = FeatureDetection(img1_grayscale, 4, 6, 5000)
key2, m2, eo2 = FeatureDetection(img2_grayscale, 4, 6, 5000)

print('RIFT2 main orientation calculation')
kpts1 = kptsOrientation(key1, m1, 1, 96)
kpts2 = kptsOrientation(key2, m2, 1, 96)

print('RIFT2 feature description')
des1 = FeatureDescribe(img1_hw, eo1, kpts1, 96, 6, 6)
des2 = FeatureDescribe(img2_hw, eo2, kpts2, 96, 6, 6)




time2 = time.time()
print("Total time: ", time2-time1)

###############
#MATCHING BELOW
from matplotlib import pyplot as plt


kp1 = kpts1.T
des1 = des1.T
kp2 = kpts2.T
des2 = des2.T


# Convert NumPy arrays to cv2.KeyPoint objects
kp1 = [cv2.KeyPoint(x=float(point[0]), y=float(point[1]), size=1) for point in kp1]
kp2 = [cv2.KeyPoint(x=float(point[0]), y=float(point[1]), size=1) for point in kp2]

# Ensure descriptors are of type CV_32F
if des1.dtype != np.float32:
    des1 = des1.astype(np.float32)
if des2.dtype != np.float32:
    des2 = des2.astype(np.float32)

# Match keypoints
points1, points2, mutual_matches = match_keypoints_nn(des1, des2, kp1, kp2, lowes_ratio=0.95, mutual=False)

print(points1.shape, points2.shape)

# Outlier removal using MAGSAC
H, mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 5.0)
matchesMask = mask.ravel().tolist()

# Select inlier keypoints and descriptors
inliers1 = [kp1[m.queryIdx] for i, m in enumerate(mutual_matches) if matchesMask[i]]
inliers2 = [kp2[m.trainIdx] for i, m in enumerate(mutual_matches) if matchesMask[i]]
des_inliers1 = np.array([des1[m.queryIdx] for i, m in enumerate(mutual_matches) if matchesMask[i]])
des_inliers2 = np.array([des2[m.trainIdx] for i, m in enumerate(mutual_matches) if matchesMask[i]])

# Draw matches
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, mutual_matches, None, **draw_params)
plt.imshow(img3), plt.show()

# Display the number of inliers and outliers
num_inliers = np.sum(mask)
num_outliers = len(mutual_matches) - num_inliers
print(f'Number of inliers: {num_inliers}')
print(f'Number of outliers: {num_outliers}')