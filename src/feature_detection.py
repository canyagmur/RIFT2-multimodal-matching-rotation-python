import numpy as np
import cv2

from src.phase_congruency.phasecong import phasecong

def FeatureDetection(im, s, o, npt):
    #expects image to be in grayscale
    m,_,_,_,_, eo,_ = phasecong(im, nscale=s, norient=o, minWaveLength=3, mult=1.6, sigmaOnf=0.75, g=3, k=1)
    a = np.max(m)
    b = np.min(m)
    m = (m - b) / (a - b)


    #added by me 
    m_image = (m * 255).astype(np.uint8)
    eo = np.transpose(eo, (1, 0, 2, 3))  # Now eo1 has shape (s, o, H, W)

    fast = cv2.FastFeatureDetector_create(threshold=1, nonmaxSuppression=True)
    keypoints = fast.detect(m_image, None)

    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    keypoints = keypoints[:npt]
    kpts = np.array([kp.pt for kp in keypoints]).T

    return kpts, m, eo
