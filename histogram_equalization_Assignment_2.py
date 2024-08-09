import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram_equalization(image):
    """Perform histogram equalization on a grayscale image."""
    # Step 1: Calculate the histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Step 2: Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Step 3: Normalize the CDF
    cdf_normalized = cdf * (255 / cdf[-1])

    # Step 4: Use the CDF to map the old values to the new ones
    image_equalized = cdf_normalized[image.flatten().astype(int)]

    # Step 5: Reshape the image to its original shape
    image_equalized = image_equalized.reshape(image.shape)

    return image_equalized.astype(np.uint8)


# Example usage
if __name__ == "__main__":
    image = cv2.imread('img.jpeg', 0)  # Load a grayscale image
    equalized_image = histogram_equalization(image)

    # Display the original and equalized images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Histogram Equalized Image")
    plt.imshow(equalized_image, cmap='gray')
    plt.show()
