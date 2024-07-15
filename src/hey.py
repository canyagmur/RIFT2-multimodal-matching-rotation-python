import numpy as np
import scipy.io

from calculate_orientation_hist import calculate_orientation_hist
# Load the MATLAB variables
matlab_data = scipy.io.loadmat('D:\Research Related\HasanHoca\RIFT2-multimodal-matching-rotation-python/third_party\RIFT2-multimodal-matching-rotation-master/matlab_variables.mat')

# Extract the variables
x = matlab_data['x'][0][0]
y = matlab_data['y'][0][0]
patch_size = matlab_data['patch_size'][0][0]
gradientImg = matlab_data['gradientImg']
gradientAng = matlab_data['gradientAng']
n = matlab_data['n'][0][0]
Sa = matlab_data['Sa']
matlab_hist = matlab_data['hist'][0]
matlab_max_value = matlab_data['max_value'][0][0]

print("x:", x)
print("y:", y)
print("patch_size:", patch_size)
print("n:", n)
print("Sa:", Sa.shape ,Sa)
print("hist:", matlab_hist)
print("max_value:", matlab_max_value)



# Run the Python function with the same inputs
python_hist, python_max_value = calculate_orientation_hist(x, y, patch_size / 2, gradientImg, gradientAng, n, Sa)

# Compare the results
print("MATLAB hist:", matlab_hist)
print("Python hist:", python_hist)
print("MATLAB max_value:", matlab_max_value)
print("Python max_value:", python_max_value)

# Check if they are close
hist_close = np.allclose(matlab_hist, python_hist, atol=1e-6)
max_value_close = np.isclose(matlab_max_value, python_max_value, atol=1e-6)

print("Histograms close:", hist_close)
print("Max values close:", max_value_close)
