#Perform Sharpening of Image using Gradient masking.
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Generate a synthetic medium-to-low size image directly in memory
def generate_sample_image(width=360, height=240):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width, height), (200, 200, 200), -1)
    cv2.rectangle(img, (40, 40), (160, 200), (40, 40, 40), -1)
    cv2.circle(img, (260, 120), 55, (80, 80, 80), -1)
    cv2.putText(img, 'GRADIENT', (55, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.line(img, (0, 0), (width, height), (20, 20, 20), 2)
    return img

# Load image and convert to grayscale
img_color = generate_sample_image(width=360, height=240)
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Calculate Sobel gradients along X and Y axes
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Compute gradient magnitude mask: |Gx| + |Gy|
gradient_mask = cv2.magnitude(sobel_x, sobel_y)

# Normalize gradient mask to range [0, 1]
gradient_mask_norm = gradient_mask / np.max(gradient_mask)

# Apply Gaussian blur to smooth the gradient mask
smooth_gradient_mask = cv2.GaussianBlur(gradient_mask_norm, (5, 5), 0)

# Compute Laplacian response for high-frequency detail enhancement
laplacian_mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
laplacian_response = cv2.filter2D(gray.astype(np.float32), -1, laplacian_mask)

# Multiply smoothed gradient mask with Laplacian response
gradient_boost = laplacian_response * smooth_gradient_mask

# Add gradient-boosted response to original image
sharpened = gray.astype(np.float32) + gradient_boost

# Clip pixel values to valid range [0, 255] and convert to uint8
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Convert gradient mask for visualization display
gradient_display = cv2.convertScaleAbs(gradient_mask)

# Display original, gradient mask, and sharpened image in Google Colab
print("--- 1. Original Grayscale Image (360x240) ---")
cv2_imshow(gray)

print("\n--- 2. Sobel Gradient Mask ---")
cv2_imshow(gradient_display)

print("\n--- 3. Gradient Mask Sharpened Image ---")
cv2_imshow(sharpened)
