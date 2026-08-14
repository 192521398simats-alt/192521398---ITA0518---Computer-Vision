#Perform Sharpening of Image using Laplacian mask with negative center coefficient.
import cv2
import numpy as np
import urllib.request
from google.colab.patches import cv2_imshow

# 1. Download sample image from web
image_url = 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800&auto=format&fit=crop'
req = urllib.request.urlopen(image_url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

# 2. Resize to Low-to-Medium Size (400x267 pixels)
low_med_img = cv2.resize(img, (400, 267), interpolation=cv2.INTER_AREA)

# 3. Convert to Grayscale
gray = cv2.cvtColor(low_med_img, cv2.COLOR_BGR2GRAY)

# 4. Define Laplacian Mask with Negative Center Coefficient
# Center coefficient is -4 (4-neighbor connectivity)
laplacian_mask = np.array([[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]], dtype=np.float32)

# 5. Apply Spatial Filtering using cv2.filter2D
laplacian_response = cv2.filter2D(gray.astype(np.float32), -1, laplacian_mask)

# 6. Sharpening formula for Negative Center Coefficient: Original - Laplacian
sharpened = gray.astype(np.float32) - laplacian_response

# 7. Clip pixel values to [0, 255] and convert back to uint8
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
laplacian_display = cv2.convertScaleAbs(laplacian_response)

# 8. Display Results in Google Colab
print("--- Original Image (400x267) ---")
cv2_imshow(gray)

print("\n--- Laplacian Edge Response ---")
cv2_imshow(laplacian_display)

print("\n--- Sharpened Image (Negative Center) ---")
cv2_imshow(sharpened)
