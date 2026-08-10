# ==============================================================================
# QUESTION 11: Effect of Smoothing on Edge Detection
# Description: Generates an image, adds noise, and compares Canny edge detection
#              performance before and after Gaussian smoothing.
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Input Image directly inside the script
img_11 = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img_11, (40, 40), (260, 260), 180, -1)
cv2.circle(img_11, (150, 150), 65, 60, -1)
cv2.line(img_11, (20, 20), (280, 280), 255, 4)

# 2. Add Gaussian noise to simulate real-world signal interference
noise = np.random.normal(0, 28, img_11.shape).astype(np.float32)
noisy_img = np.clip(img_11.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# 3. Edge Detection WITHOUT Smoothing (high false-positive rate due to noise)
edges_noisy = cv2.Canny(noisy_img, 50, 150)

# 4. Edge Detection WITH Gaussian Smoothing (suppresses noise, connects edges)
smoothed_img = cv2.GaussianBlur(noisy_img, (5, 5), 1.5)
edges_smoothed = cv2.Canny(smoothed_img, 50, 150)

# 5. Display Outputs
titles = [
    "1. Noisy Input Image",
    "2. Canny (No Smoothing)\n[False Noise Edges]",
    "3. Smoothed Image (Gaussian)",
    "4. Canny (After Smoothing)\n[Clean Edge Boundaries]"
]
images = [noisy_img, edges_noisy, smoothed_img, edges_smoothed]

plt.figure(figsize=(16, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()
