import numpy as np
import cv2

def calculate_orientation_hist(x, y, radius, gradient, angle, n, Sa):
    sigma = radius / 3
    radius_x_left = int(x - radius)
    radius_x_right = int(x + radius)
    radius_y_up = int(y - radius)
    radius_y_down = int(y + radius)
    
    # Ensure indices are within bounds
    radius_x_left = max(0, radius_x_left)
    radius_x_right = min(gradient.shape[1], radius_x_right + 1)
    radius_y_up = max(0, radius_y_up)
    radius_y_down = min(gradient.shape[0], radius_y_down + 1)
    
    sub_gradient = gradient[radius_y_up:radius_y_down, radius_x_left:radius_x_right]
    sub_angle = angle[radius_y_up:radius_y_down, radius_x_left:radius_x_right]

    X = np.arange(-(x - radius_x_left), (radius_x_right - x))
    Y = np.arange(-(y - radius_y_up), (radius_y_down - y))
    XX, YY = np.meshgrid(X, Y)

    gaussian_weight = np.exp(-(XX**2 + YY**2) / (2 * sigma**2))
    W1 = sub_gradient * gaussian_weight
    W = np.double(Sa[:W1.shape[0], :W1.shape[1]]) * np.double(W1)

    bin = np.round(sub_angle * n / 360).astype(int)
    bin[bin >= n] -= n
    bin[bin < 0] += n

    temp_hist = np.zeros(n)
    for i in range(n):
        wM = W[bin == i]
        if wM.size > 0:
            temp_hist[i] = np.sum(wM)

    hist = np.zeros(n)
    hist[0] = (temp_hist[n-2] + temp_hist[2]) / 16 + \
              4 * (temp_hist[n-1] + temp_hist[1]) / 16 + temp_hist[0] * 6 / 16
    hist[1] = (temp_hist[n-1] + temp_hist[3]) / 16 + \
              4 * (temp_hist[0] + temp_hist[2]) / 16 + temp_hist[1] * 6 / 16
    hist[2:n-2] = (temp_hist[0:n-4] + temp_hist[4:n]) / 16 + \
                  4 * (temp_hist[1:n-3] + temp_hist[3:n-1]) / 16 + temp_hist[2:n-2] * 6 / 16
    hist[n-2] = (temp_hist[n-4] + temp_hist[0]) / 16 + \
                4 * (temp_hist[n-3] + temp_hist[n-1]) / 16 + temp_hist[n-2] * 6 / 16
    hist[n-1] = (temp_hist[n-3] + temp_hist[1]) / 16 + \
                4 * (temp_hist[n-2] + temp_hist[0]) / 16 + temp_hist[n-1] * 6 / 16

    max_value = np.max(hist)
    return hist, max_value


# import numpy as np
# import cv2

# def calculate_orientation_hist(x, y, radius, gradient, angle, n, Sa):
#     sigma = radius / 3
#     radius_x_left = int(x-radius)
#     radius_x_right = int(x + radius)
#     radius_y_up = int(y - radius)
#     radius_y_down = int(y + radius)
    
#     sub_gradient = gradient[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]
#     sub_angle = angle[radius_y_up:radius_y_down+1, radius_x_left:radius_x_right+1]

#     X = np.arange(-(x - radius_x_left), (radius_x_right - x) + 1)
#     Y = np.arange(-(y - radius_y_up), (radius_y_down - y) + 1)

#     XX, YY = np.meshgrid(X, Y)

#     gaussian_weight = np.exp(-(XX**2 + YY**2) / (2 * sigma**2))
#     W1 = sub_gradient * gaussian_weight
#     W = np.double(Sa) * np.double(W1)

#     bin = np.round(sub_angle * n / 360).astype(int)
#     bin[bin >= n] -= n
#     bin[bin < 0] += n

#     temp_hist = np.zeros(n)
#     for i in range(n):
#         wM = W[bin == i]
#         if wM.size > 0:
#             temp_hist[i] = np.sum(wM)

#     hist = np.zeros(n)
#     hist[0] = (temp_hist[n-2] + temp_hist[2]) / 16 + \
#               4 * (temp_hist[n-1] + temp_hist[1]) / 16 + temp_hist[0] * 6 / 16
#     hist[1] = (temp_hist[n-1] + temp_hist[3]) / 16 + \
#               4 * (temp_hist[0] + temp_hist[2]) / 16 + temp_hist[1] * 6 / 16
#     hist[2:n-2] = (temp_hist[0:n-4] + temp_hist[4:n]) / 16 + \
#                   4 * (temp_hist[1:n-3] + temp_hist[3:n-1]) / 16 + temp_hist[2:n-2] * 6 / 16
#     hist[n-2] = (temp_hist[n-4] + temp_hist[0]) / 16 + \
#                 4 * (temp_hist[n-3] + temp_hist[n-1]) / 16 + temp_hist[n-2] * 6 / 16
#     hist[n-1] = (temp_hist[n-3] + temp_hist[1]) / 16 + \
#                 4 * (temp_hist[n-2] + temp_hist[0]) / 16 + temp_hist[n-1] * 6 / 16

#     max_value = np.max(hist)
#     return hist, max_value