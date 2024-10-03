import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Load the image in grayscale
image_path = 'img.jpeg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to compute and display LBP
def plot_lbp(image, radius, n_points, method='uniform'):
    lbp = local_binary_pattern(image, n_points, radius, method=method)
    plt.imshow(lbp, cmap='gray')
    plt.title(f'LBP: Radius={radius}, Points={n_points}, Method={method}')
    plt.axis('off')
    plt.show()

# Experiment 1: Different radii and number of points
plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# LBP with different radii and points
plot_lbp(image, radius=1, n_points=8, method='uniform')
plot_lbp(image, radius=2, n_points=16, method='uniform')
plot_lbp(image, radius=3, n_points=24, method='uniform')
plot_lbp(image, radius=3, n_points=24, method='ror')  # Rotation invariant LBP

# Experiment 2: Different LBP methods
plot_lbp(image, radius=1, n_points=8, method='uniform')  # Uniform LBP
plot_lbp(image, radius=1, n_points=8, method='ror')      # Rotation-invariant LBP
plot_lbp(image, radius=1, n_points=8, method='var')      # Variance-based LBP

# Function to plot LBP histogram
def plot_histogram(lbp, n_bins='auto'):
    hist, bins = np.histogram(lbp.ravel(), bins=n_bins, range=(0, lbp.max() + 1))
    plt.bar(bins[:-1], hist, width=0.5, align='center')
    plt.title('LBP Histogram')
    plt.xlabel('LBP Value')
    plt.ylabel('Frequency')
    plt.show()

# Experiment 3: LBP histogram
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# Display LBP and its histogram
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(lbp, cmap='gray')
plt.title('LBP Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plot_histogram(lbp)

# Experiment 4: LBP on different image types (e.g., texture and face images)
texture_image = cv2.imread('texture_image.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with texture image path
face_image = cv2.imread('face_image.jpg', cv2.IMREAD_GRAYSCALE)        # Replace with face image path

# Apply LBP to texture and face images
plot_lbp(texture_image, radius=2, n_points=16, method='uniform')
plot_lbp(face_image, radius=1, n_points=8, method='uniform')

# Experiment 5: LBP on image regions (sliding window)
def plot_lbp_window(image, window_size=64):
    h, w = image.shape
    for i in range(0, h, window_size):
        for j in range(0, w, window_size):
            region = image[i:i+window_size, j:j+window_size]
            lbp = local_binary_pattern(region, 8, 1, method='uniform')

            plt.figure(figsize=(4, 4))
            plt.imshow(lbp, cmap='gray')
            plt.title(f'LBP on Region ({i}, {j})')
            plt.axis('off')
            plt.show()

# Apply LBP on smaller regions of the image
plot_lbp_window(image, window_size=64)

# Experiment 6: Multi-scale LBP comparison
scales = [1, 2, 3]  # Radii to explore different scales
lbp_images = []

for radius in scales:
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp_images.append(lbp)

# Display multi-scale LBP results
plt.figure(figsize=(15, 5))
for i, lbp_img in enumerate(lbp_images):
    plt.subplot(1, len(scales), i + 1)
    plt.imshow(lbp_img, cmap='gray')
    plt.title(f'Radius={scales[i]}')
    plt.axis('off')
plt.show()
