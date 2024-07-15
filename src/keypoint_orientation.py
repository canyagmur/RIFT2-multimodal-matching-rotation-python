import numpy as np
import cv2

def kptsOrientation(key, im, is_ori, patch_size):
    if is_ori == 1:
        n = 24
        ORI_PEAK_RATIO = 0.8
        h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        gradient_x = cv2.filter2D(im, -1, h, borderType=cv2.BORDER_REPLICATE)
        gradient_y = cv2.filter2D(im, -1, h.T, borderType=cv2.BORDER_REPLICATE)
        gradientImg = np.sqrt(gradient_x**2 + gradient_y**2)
        temp_angle = np.arctan2(gradient_y, gradient_x)
        temp_angle = np.degrees(temp_angle)
        temp_angle[temp_angle < 0] += 360
        gradientAng = temp_angle


    feat_index = 0
    kpts = np.zeros((3, key.shape[1] * 6))
    for k in range(key.shape[1]):
        x = int(round(key[0, k]))
        y = int(round(key[1, k]))
        r = int(round(patch_size))

        x1 = max(1, x - r // 2)
        y1 = max(1, y - r // 2)
        x2 = min(x + r // 2, im.shape[1] - 1)
        y2 = min(y + r // 2, im.shape[0] - 1)

        if y2 - y1 != r or x2 - x1 != r:
            continue

        if is_ori == 1:
            angle = orientation(x, y, gradientImg, gradientAng, r, n, ORI_PEAK_RATIO)
            

            for i in range(len(angle)):
                kpts[:, feat_index] = [x, y, angle[i]]
                feat_index += 1
        else:
            kpts[:, feat_index] = [x, y, 0]
            feat_index += 1

    kpts = kpts[:, kpts[0, :] != 0]

    return kpts


from src.calculate_orientation_hist import calculate_orientation_hist

def orientation(x, y, gradientImg, gradientAng, patch_size, n, ORI_PEAK_RATIO):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(patch_size+1), int(patch_size+1)))
    Sa = se.astype(np.uint8)
    hist, max_value = calculate_orientation_hist(x, y, patch_size / 2, gradientImg, gradientAng, n, Sa)

    mag_thr = max_value * ORI_PEAK_RATIO
    ANG = []
    for k in range(n):
        k1 = n - 1 if k == 0 else k - 1
        k2 = 0 if k == n - 1 else k + 1
        if hist[k] > hist[k1] and hist[k] > hist[k2] and hist[k] > mag_thr:
            bin = k - 1 + 0.5 * (hist[k1] - hist[k2]) / (hist[k1] + hist[k2] - 2 * hist[k])
            if bin < 0:
                bin = n + bin
            elif bin >= n:
                bin = bin - n
            angle = (360 / n) * bin  # 0-360
            ANG.append(angle)
    return ANG






