import cv2
import numpy as np

# Read the image in grayscale
image = cv2.imread('meter-img.jpeg', 0)

# Apply Gaussian blur to the image to reduce noise and make edges smoother
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.Canny(image, 50, 60)
blurred = cv2.GaussianBlur(img, (3, 5), 0)

# Crop the image using the specified ROI
# roi_coordinates = [50:200, 100:300]

# Use morphological operations to close gaps and join edges
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

# Optionally, you can apply additional smoothing with Gaussian blur
smoothed = cv2.GaussianBlur(closed, (5, 5), 0)
cropped_image = smoothed[0:500, 320:1100]

cv2.imshow('Smoothed', cropped_image)
cv2.waitKey(0)
# cv2.destroyAllWindows()