import cv2
import numpy as np
from matplotlib import pyplot as plt


# 1. LOAD IMAGE
dataImage = cv2.imread("images.jpg")

# Check if image loaded correctly
if dataImage is None:
    print("Error loading image")
    exit()

# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(dataImage, cv2.COLOR_BGR2RGB)

# 2. THRESHOLDING (Brightness / B&W Conversion)

gray = cv2.cvtColor(dataImage, cv2.COLOR_BGR2GRAY)

# Apply binary threshold
_, threshold_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 3. NOISE REMOVAL USING FILTERS

# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(dataImage, (5, 5), 0)

# Median Blur
median_blur = cv2.medianBlur(dataImage, 5)

# Bilateral Filter
bilateral_filter = cv2.bilateralFilter(dataImage, 9, 75, 75)

# 4. EDGE DETECTION

edges = cv2.Canny(gray, 100, 200)

# 5. FEATURE DETECTION

# ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Draw keypoints
feature_image = cv2.drawKeypoints(
    dataImage,
    keypoints,
    None,
    color=(0, 255, 0),
    flags=0
)

# 6. PANORAMA STITCHING

img1 = cv2.imread("pano1.jpg")
img2 = cv2.imread("pano2.jpg")

panorama_images = [img1, img2]

# Create stitcher object
stitcher = cv2.Stitcher_create()

status, panorama = stitcher.stitch(panorama_images)

if status == cv2.STITCHER_OK:
    print("Panorama created successfully")
else:
    print("Error in panorama stitching")

# 7. DISPLAY RESULTS

plt.figure(figsize=(15,10))

# Original Image
plt.subplot(2,3,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

# Threshold Image
plt.subplot(2,3,2)
plt.imshow(threshold_img, cmap='gray')
plt.title("Threshold Image")
plt.axis("off")

# Blurred Image
plt.subplot(2,3,3)
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blur")
plt.axis("off")

# Edge Detection
plt.subplot(2,3,4)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")
plt.axis("off")

# Feature Detection
plt.subplot(2,3,5)
plt.imshow(cv2.cvtColor(feature_image, cv2.COLOR_BGR2RGB))
plt.title("Feature Detection")
plt.axis("off")

# Panorama
if status == cv2.STITCHER_OK:
    plt.subplot(2,3,6)
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title("Panorama")
    plt.axis("off")

plt.tight_layout()
plt.show()

print("hello Mithilesh")
print("hello ali")
