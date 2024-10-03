import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('img.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply 2D wavelet transform (Decomposition)
coeffs2 = pywt.dwt2(image, 'haar')
LL, (LH, HL, HH) = coeffs2

# Watermarking - Create a simple watermark (you can replace this with an actual image)
watermark = np.zeros_like(LL)
cv2.putText(watermark, 'Shivraj', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)

# Embed the watermark in the LL subband (e.g., adding it to the approximation coefficients)
LL_watermarked = LL + 0.1 * watermark  # The factor (0.1) controls the watermark intensity

# Reconstruct the image using inverse wavelet transform (with watermark)
coeffs2_watermarked = (LL_watermarked, (LH, HL, HH))
reconstructed_image = pywt.idwt2(coeffs2_watermarked, 'haar')

# Plot the original subband images and the watermarked reconstructed image
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Plot original subbands
ax[0, 0].imshow(LL, cmap='gray')
ax[0, 0].set_title('Approximation (LL)')

ax[0, 1].imshow(LH, cmap='gray')
ax[0, 1].set_title('Horizontal Detail (LH)')

ax[0, 2].imshow(HL, cmap='gray')
ax[0, 2].set_title('Vertical Detail (HL)')

ax[1, 0].imshow(HH, cmap='gray')
ax[1, 0].set_title('Diagonal Detail (HH)')

# Plot watermarked LL and reconstructed image
ax[1, 1].imshow(LL_watermarked, cmap='gray')
ax[1, 1].set_title('Watermarked LL Subband')

ax[1, 2].imshow(reconstructed_image, cmap='gray')
ax[1, 2].set_title('Reconstructed Image with Watermark')

plt.tight_layout()
plt.show()
