#Perform Edge detection using Sobel Matrix along X axis
import cv2
import numpy as np
import urllib.request
from google.colab.patches import cv2_imshow

# 1. Download a new high-contrast sample image (Taj Mahal)
# Perfect for showcasing vertical edges along the X-axis
image_url = 'https://images.unsplash.com/photo-1564507592333-c60657eea523?w=800&auto=format&fit=crop'

req = urllib.request.urlopen(image_url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

# 2. Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Perform Sobel Edge Detection along the X-axis (dx=1, dy=0)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)

# 4. Convert float results back to 8-bit integer image (0-255)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)

# 5. Display Results in Google Colab
print("--- New Original Image ---")
cv2_imshow(img)

print("\n--- Sobel Edge Detection (X-Axis) ---")
cv2_imshow(sobel_x_abs)
