import numpy as np
import cv2
from joblib import Parallel, delayed

def FeatureDescribe(img_hw, eo, kpts, patch_size, no, nbin):
    n = kpts.shape[1]
    yim, xim = img_hw
    CS = np.zeros((yim, xim, no))

    for j in range(no):
        for i in range(4):
            CS[:, :, j] += np.abs(eo[i][j])

    MIM = np.argmax(CS, axis=2)

    des = np.zeros((no * no * nbin, n))

    def process_keypoint(k):
        x = kpts[0, k]
        y = kpts[1, k]
        r = patch_size
        ang = kpts[2, k]

        patch = extract_patches(MIM, x, y, round(r / 2), ang)
        patch = cv2.resize(patch, (r + 1, r + 1), interpolation=cv2.INTER_LINEAR)
        h, _ = np.histogram(patch, bins=np.arange(1, no + 2))  # Correcting the histogram bins
        idx = np.argmax(h)
        patch_rot = patch - idx + 1
        patch_rot[patch_rot < 0] += no

        ys, xs = patch_rot.shape
        histo = np.zeros((no, no, nbin))

        for j in range(no):
            for i in range(no):
                clip = patch_rot[round((j) * ys / no):round((j + 1) * ys / no), round((i) * xs / no):round((i + 1) * xs / no)]
                histo[j, i, :] = np.histogram(clip, bins=np.arange(1, nbin + 2))[0]  # Ensuring correct histogram bins

        histo = histo.flatten()

        if np.linalg.norm(histo) != 0:
            histo = histo / np.linalg.norm(histo)

        return histo

    des = Parallel(n_jobs=-1)(delayed(process_keypoint)(k) for k in range(n))
    des = np.array(des).T

    return des

def extract_patches(img, x, y, s, t):
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    m = img.shape[2] if img.ndim == 3 else 1

    x = np.clip(np.round(x).astype(int), 0, w-1)
    y = np.clip(np.round(y).astype(int), 0, h-1)

    s = int(round(s))
    t = np.deg2rad(t)

    patchsize = s * 2 + 1
    xg, yg = np.meshgrid(np.arange(-s, s + 1), np.arange(-s, s + 1))
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xygrot = R @ np.vstack([xg.ravel(), yg.ravel()])
    xygrot[0, :] += x
    xygrot[1, :] += y

    xr = xygrot[0, :]
    yr = xygrot[1, :]
    xf = np.floor(xr).astype(int)
    yf = np.floor(yr).astype(int)
    xp = xr - xf
    yp = yr - yf

    patch = np.zeros((patchsize, patchsize, m))

    valid_mask = (xf >= 0) & (xf <= w - 2) & (yf >= 0) & (yf <= h - 2)
    xf = xf[valid_mask]
    yf = yf[valid_mask]
    xp = xp[valid_mask]
    yp = yp[valid_mask]

    ind1 = np.ravel_multi_index((yf, xf), (h, w))
    ind2 = np.ravel_multi_index((yf, xf + 1), (h, w))
    ind3 = np.ravel_multi_index((yf + 1, xf), (h, w))
    ind4 = np.ravel_multi_index((yf + 1, xf + 1), (h, w))

    for ch in range(m):
        imgch = img[:, :, ch] if m > 1 else img
        ivec = ((1 - yp) * (xp * imgch.ravel()[ind2] + (1 - xp) * imgch.ravel()[ind1]) +
                yp * (xp * imgch.ravel()[ind4] + (1 - xp) * imgch.ravel()[ind3]))
        temp = np.zeros((patchsize, patchsize))
        temp.ravel()[valid_mask] = ivec
        patch[:, :, ch] = temp

    return patch

