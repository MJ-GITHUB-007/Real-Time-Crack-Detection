import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('data/train/Negative/00001.jpg', cv2.IMREAD_GRAYSCALE)

# Apply FFT
f_transform = np.fft.fft2(image)

f_transform_shifted = np.fft.fftshift(f_transform)
print(f_transform_shifted.shape)

magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)  # Add 1 to avoid log(0)
print(magnitude_spectrum.shape)

# Display the original and frequency domain images
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
