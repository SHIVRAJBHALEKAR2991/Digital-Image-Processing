# Shivraj Aniruddha Bhalekar (BT22CSE024)


# DIP Assignment 3:-

import cv2 as cv

# Reading the image
img = cv.imread('img.jpeg')

# Displaying the actual image
cv.imshow('image', img)

# Converting the RGB image into a binary format using thresholding
_, binary_image_rgb = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
cv.imshow('binary_image_rgb', binary_image_rgb)

# Converting the image to grayscale
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Converting the grayscale image into a binary format using thresholding
_, binary_image_gray = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
cv.imshow('binary_image_gray', binary_image_gray)

# Wait for a key press indefinitely or for a given amount of time in milliseconds
cv.waitKey(0)