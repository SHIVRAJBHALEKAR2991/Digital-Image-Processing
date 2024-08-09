import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('img.jpeg')
print(img.shape)

# Extract the first layer (channel)
img_layer1 = img[:, :, 0]

# Create new images with the same shape, but all zeros
base_red = np.zeros_like(img)
base_blue = np.zeros_like(img)
base_green = np.zeros_like(img)

# Assign the first layer to the new images, keeping the other layers as zeros
base_red[:, :, 2] = img_layer1  # Red channel
base_blue[:, :, 1] = img_layer1  # Blue channel
base_green[:, :, 0] = img_layer1  # Green channel

print(img.shape)

# Display the images
cv.imshow('Base Red', base_red)
cv.imshow('Base Blue', base_blue)
cv.imshow('Base Green', base_green)
cv.imshow('RGB Image', img)
cv.imshow('First Layer', img_layer1)
cv.waitKey(0)
cv.destroyAllWindows()
