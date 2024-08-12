# Shivraj Aniruddha Bhalekar (BT22CSE024)


# DIP Assignment 4:-

# What are Bit-planes ?
# Bit planes are the binary layers of an image, where each bit in a pixel's binary representation contributes
# to the overall appearance. In an 8-bit grayscale image, the least significant bit (LSB) captures fine details
# and noise, while the most significant bit (MSB) captures the main features.
# Visualizing and manipulating these planes allows for enhancing specific features or reducing noise
# by reconstructing the image using selected bit planes.


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale mode
image = cv2.imread('img.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    print("Error loading image.")
else:
    print("Image loaded successfully.")

# List to store the 8 bit planes of the image
bit_planes = []

# Loop through each bit plane from 0 to 7
for i in range(8):
    # Initialize an empty array with the same shape as the image to store the bit plane
    bit_plane = np.zeros_like(image)

    # Extract the i-th bit plane using bitwise AND operation
    # (1 << i) shifts 1 to the left by 'i' bits to create a mask for the i-th bit
    bit_plane = cv2.bitwise_and(image, 1 << i)

    # Convert non-zero values to 255 (white) for better visualization of the bit plane
    bit_plane = np.where(bit_plane > 0, 255, 0).astype(np.uint8)

    # Append the extracted bit plane to the list
    bit_planes.append(bit_plane)

# Create a 2x4 grid of subplots to display all 8 bit planes
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()  # Flatten the 2x4 array into a 1D array for easy iteration

# Loop through each bit plane and display it in the corresponding subplot
for i in range(8):
    axes[i].imshow(bit_planes[i], cmap='gray')
    axes[i].set_title(f'Bit Plane {i}')  # Set the title to indicate the bit plane number
    axes[i].axis('off')  # Hide the axis for a cleaner visualization

plt.tight_layout()  # Adjust the spacing between subplots
plt.show()  # Display the bit planes

# Initialize an empty array to store the reconstructed image
# The reconstruction will be based on the most significant bit planes (4 to 7)
reconstructed_image = np.zeros_like(image)

# Loop through bit planes 4 to 7 and add their contributions to the reconstructed image
for i in range(4, 8):
    # Add the weighted bit plane to the reconstructed image
    # (1 << i) shifts the bit plane back to its original value in the pixel
    reconstructed_image += bit_planes[i] * (1 << i)

# Display the reconstructed image using only the most significant bit planes
plt.figure(figsize=(10, 7))
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image Using Bit Planes 4 to 7')  # Set the title
plt.axis('off')  # Hide the axis for a cleaner visualization
plt.show()  # Display the reconstructed image
