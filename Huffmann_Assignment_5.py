# Shivraj Aniruddha Bhalekar (BT22CSE024)


# DIP Assignment 5:-
import heapq
from collections import defaultdict, Counter
import numpy as np
import cv2

class Node:
    def __init__(self, frequency, symbol, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    def __lt__(self, other):
        return self.frequency < other.frequency

def calculate_frequency(data):
    return dict(Counter(data))

def build_huffman_tree(frequency):
    heap = [Node(frequency, symbol) for symbol, frequency in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = Node(left.frequency + right.frequency, None, left, right)
        heapq.heappush(heap, merged)

    return heap[0]

def build_codes(node, current_code=''):
    codes = {}
    if node is not None:
        if node.symbol is not None:
            codes[node.symbol] = current_code
        codes.update(build_codes(node.left, current_code + '0'))
        codes.update(build_codes(node.right, current_code + '1'))
    return codes

def huffman_encoding(image):
    flat_image = image.flatten()
    frequency = calculate_frequency(flat_image)
    huffman_tree = build_huffman_tree(frequency)
    huffman_codes = build_codes(huffman_tree)

    encoded_data = ''.join(huffman_codes[pixel] for pixel in flat_image)
    return encoded_data, huffman_codes, image.shape

def huffman_decoding(encoded_data, huffman_codes, shape):
    reversed_codes = {v: k for k, v in huffman_codes.items()}
    decoded_data = []
    code = ''

    for bit in encoded_data:
        code += bit
        if code in reversed_codes:
            decoded_data.append(reversed_codes[code])
            code = ''

    return np.array(decoded_data, dtype=np.uint8).reshape(shape)

# Example usage
image_path = 'img.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original Image', image)

# Huffman Encoding
encoded_data, huffman_codes, shape = huffman_encoding(image)
print(huffman_codes)
# Huffman Decoding
decoded_image = huffman_decoding(encoded_data, huffman_codes, shape)

# Show and save the decoded image
cv2.imshow('Decoded Image', decoded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
