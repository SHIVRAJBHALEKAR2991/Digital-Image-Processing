#Shivraj Aniruddha Bhalekar (BT22CSE024)

#Arithemtic Coding

# Dip assignment 7 :=

import numpy as np
import cv2
from collections import defaultdict


class ArithmeticCoding:
    def __init__(self, symbols):
        """
        Initialize the ArithmeticCoding class with a list of symbols.

        Parameters:
        symbols (list): A list of symbols (e.g., pixel values from an image).
        """
        self.symbols = symbols
        self.probabilities = self.calculate_probabilities()  # Calculate symbol probabilities
        self.intervals = self.calculate_intervals()  # Calculate symbol intervals

    def calculate_probabilities(self):
        """
        Calculate the probability of each symbol in the list.

        Returns:
        dict: A dictionary where keys are symbols and values are their probabilities.
        """
        total_count = len(self.symbols)
        freq_dict = defaultdict(int)

        # Count the frequency of each symbol
        for symbol in self.symbols:
            freq_dict[symbol] += 1

        # Calculate probability by dividing frequency by the total number of symbols
        return {k: v / total_count for k, v in freq_dict.items()}

    def calculate_intervals(self):
        """
        Calculate the intervals for each symbol based on its probability.

        Returns:
        dict: A dictionary where keys are symbols and values are tuples (low, high)
              representing the intervals.
        """
        intervals = {}
        low = 0.0

        # Sort symbols by their probabilities and calculate intervals
        for symbol, probability in sorted(self.probabilities.items()):
            high = low + probability
            intervals[symbol] = (low, high)
            low = high  # Update low for the next symbol's interval
        return intervals

    def encode(self, data):
        """
        Encode the input data using arithmetic coding.

        Parameters:
        data (list): The data to be encoded (e.g., flattened image).

        Returns:
        float: The encoded value representing the entire data sequence.
        """
        low = 0.0
        high = 1.0

        # Narrow down the interval range based on the data sequence
        for symbol in data:
            symbol_low, symbol_high = self.intervals[symbol]
            range_ = high - low
            high = low + range_ * symbol_high
            low = low + range_ * symbol_low

        # Return the final value within the interval
        return (low + high) / 2

    def decode(self, code, length):
        """
        Decode the encoded value back into the original data sequence.

        Parameters:
        code (float): The encoded value.
        length (int): The length of the original data sequence.

        Returns:
        list: The decoded data sequence.
        """
        decoded_data = []

        # Reconstruct the original sequence by iterating over its length
        for _ in range(length):
            for symbol, (symbol_low, symbol_high) in self.intervals.items():
                if symbol_low <= code < symbol_high:
                    decoded_data.append(symbol)
                    range_ = symbol_high - symbol_low
                    code = (code - symbol_low) / range_
                    break
        return decoded_data


# Load a grayscale image
image = cv2.imread('img.jpeg', cv2.IMREAD_GRAYSCALE)

# Display the original image
cv2.imshow('Original Image', image)

# Get the actual dimensions of the image
height, width = image.shape

# Flatten the image to a 1D array of pixel values
image = image.flatten()

# Initialize the ArithmeticCoding class with unique symbols from the image
unique_symbols = np.unique(image)
arithmetic_coding = ArithmeticCoding(unique_symbols)

# Encode the image using arithmetic coding
encoded_value = arithmetic_coding.encode(image)
print(f'Encoded value: {encoded_value}')

# Decode the encoded value back into the original image
decoded_image = arithmetic_coding.decode(encoded_value, len(image))

# Reshape the decoded image back to its original dimensions
decoded_image = np.array(decoded_image, dtype=np.uint8).reshape((height, width))

print(image)
print(decoded_image)
# Display the decoded image
cv2.imshow('Decoded Image', decoded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
