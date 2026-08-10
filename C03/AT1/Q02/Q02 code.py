# ==============================================================================
# QUESTION 12: Gradient-Based Edge Detection Study
# Description: Generates an image, calculates spatial gradients (Gx, Gy), and 
#              derives the gradient magnitude and orientation.
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Input Image directly inside the script
img_12 = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img_12, (50, 50), (250, 250), 200, -1)
cv2.circle(img_12, (150, 150), 50, 80, -1)
cv2.line(img_12, (0, 150), (300, 150), 255, 3)

# 2. Compute Horizontal (Gx) and Vertical (Gy) Gradients using Sobel
grad_x = cv2.Sobel(img_12, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img_12, cv2.CV_64F, 0, 1, ksize=3)

# 3. Compute Gradient Magnitude M(x,y) and Direction Angle
magnitude = cv2.magnitude(grad_x, grad_y)
orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)

# 4. Log Gradient metrics at a specific edge boundary pixel (y=150, x=50)
sample_y, sample_x = 150, 50
print(f"--- Sample Gradient Metrics at Pixel ({sample_x}, {sample_y}) ---")
print(f"Horizontal Gradient (Gx) : {grad_x[sample_y, sample_x]:.2f}")
print(f"Vertical Gradient (Gy)   : {grad_y[sample_y, sample_x]:.2f}")
print(f"Gradient Magnitude       : {magnitude[sample_y, sample_x]:.2f}")
print(f"Gradient Orientation     : {orientation[sample_y, sample_x]:.2f}°\n")

# 5. Display Outputs
titles = [
    "Original Input Image",
    "Horizontal Gradient |Gx|",
    "Vertical Gradient |Gy|",
    "Gradient Magnitude M(x,y)"
]
images = [img_12, np.abs(grad_x), np.abs(grad_y), magnitude]

plt.figure(figsize=(16, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()
