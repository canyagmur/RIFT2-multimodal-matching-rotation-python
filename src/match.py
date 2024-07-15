import cv2
import numpy as np

def match_keypoints_nn(des1, des2, kp1, kp2, lowes_ratio=0.75, mutual=True):
    # BFMatcher with default params
    bf = cv2.BFMatcher()

    # Mutual Nearest Neighbor Matching
    matches1 = bf.knnMatch(des1, des2, k=2)
    

    if mutual:
        matches2 = bf.knnMatch(des2, des1, k=2)
        # Apply ratio test version 2 (mutual nearest neighbor check)
        good_matches1 = []
        for m, n in matches1:
            if m.distance < lowes_ratio * n.distance:
                good_matches1.append(m)

        good_matches2 = []
        for m, n in matches2:
            if m.distance < lowes_ratio * n.distance:
                good_matches2.append(m)

        # Mutual Nearest Neighbor
        mutual_matches = []
        for m in good_matches1:
            if any(m.queryIdx == n.trainIdx and m.trainIdx == n.queryIdx for n in good_matches2):
                mutual_matches.append(m)
    else:
        # Apply ratio test version 1
        mutual_matches = []
        for m, n in matches1:
            if m.distance < lowes_ratio * n.distance:
                mutual_matches.append(m)

    # Extract location of good matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in mutual_matches]).reshape(-1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in mutual_matches]).reshape(-1, 2)

    return points1, points2, mutual_matches

# Example usage
# Assuming kp1, kp2, des1, des2 are already defined
# points1, points2, mutual_matches = match_keypoints(des1, des2, kp1, kp2, lowes_ratio=0.95, mutual=True)
