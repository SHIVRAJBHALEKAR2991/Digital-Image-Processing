import cv2
import numpy as np
from collections import defaultdict

# Function to calculate the frequency of each pixel value in the image
def calculate_frequencies(image):
    freq = defaultdict(int)
    for pixel in image.flatten():
        freq[pixel] += 1
    return freq

# Function to perform the Shannon-Fano coding
def shannon_fano(freq):
    # Sort the frequency dictionary
    sorted_freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)

    def divide_set(items):
        total = sum([item[1] for item in items])
        acc = 0
        for i in range(len(items)):
            acc += items[i][1]
            if acc >= total / 2:
                return items[:i + 1], items[i + 1:]

    def recursive_shannon_fano(items):
        if len(items) == 1:
            return {items[0][0]: ""}
        left, right = divide_set(items)
        left_codes = recursive_shannon_fano(left)
        right_codes = recursive_shannon_fano(right)

        for k in left_codes:
            left_codes[k] = "0" + left_codes[k]
        for k in right_codes:
            right_codes[k] = "1" + right_codes[k]

        left_codes.update(right_codes)
        return left_codes

    return recursive_shannon_fano(sorted_freq)

# Function to encode the image using Shannon-Fano codes
def encode_image(image, shannon_fano_codes):
    encoded_image = ''.join([shannon_fano_codes[pixel] for pixel in image.flatten()])
    return encoded_image

# Function to decode the encoded image back to the original image
def decode_image(encoded_image, shannon_fano_codes, image_shape):
    reverse_codes = {v: k for k, v in shannon_fano_codes.items()}
    decoded_image = []
    code = ""
    for bit in encoded_image:
        code += bit
        if code in reverse_codes:
            decoded_image.append(reverse_codes[code])
            code = ""
    return np.array(decoded_image).reshape(image_shape)


image = cv2.imread('img.jpeg', cv2.IMREAD_GRAYSCALE)

# Calculate frequencies
freq = calculate_frequencies(image)

# Generate Shannon-Fano codes
shannon_fano_codes = shannon_fano(freq)
print(shannon_fano_codes)
# Encode the image
encoded_image = encode_image(image, shannon_fano_codes)

# Decode the image
decoded_image = decode_image(encoded_image, shannon_fano_codes, image.shape)


# Display the original and decoded images
cv2.imshow('Original Image', image)
cv2.imshow('Decoded Image', decoded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()