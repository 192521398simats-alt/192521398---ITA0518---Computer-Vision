# ==============================================================================
# QUESTION 3: Prewitt vs Sobel Comparison
# Description: Generates a noisy target image in code and evaluates Prewitt vs 
#              Sobel operators under identical conditions.
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Input Image directly inside the script
base_img_13 = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(base_img_13, (40, 40), (260, 260), 160, -1)
cv2.circle(base_img_13, (150, 150), 70, 70, -1)
cv2.line(base_img_13, (10, 290), (290, 10), 240, 4)

# Add mild noise to highlight structural differences in operator performance
noise = np.random.normal(0, 12, base_img_13.shape).astype(np.float32)
img_13 = np.clip(base_img_13.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# 2. Prewitt Operator (Uniform 1s smoothing weighting)
kernel_px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
kernel_py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
prewitt_x = cv2.filter2D(img_13, cv2.CV_64F, kernel_px)
prewitt_y = cv2.filter2D(img_13, cv2.CV_64F, kernel_py)
prewitt_mag = cv2.magnitude(prewitt_x, prewitt_y)

# 3. Sobel Operator (Weighted central pixel = 2 for Gaussian smoothing effect)
sobel_x = cv2.Sobel(img_13, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_13, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)

# 4. Difference Map (|Sobel - Prewitt|)
difference_map = cv2.absdiff(sobel_mag, prewitt_mag)

# 5. Display Outputs
titles = [
    "Noisy Input Image",
    "Prewitt Edge Magnitude",
    "Sobel Edge Magnitude",
    "Absolute Difference\n(|Sobel - Prewitt|)"
]
images = [img_13, prewitt_mag, sobel_mag, difference_map]

plt.figure(figsize=(16, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()
