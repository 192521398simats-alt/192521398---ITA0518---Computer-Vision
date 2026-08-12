#Perform Edge detection using Sobel Matrix along Y axis
import cv2
import numpy as np
import urllib.request
from google.colab.patches import cv2_imshow

# 1. Download a sample image rich in horizontal lines
image_url = 'https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=800&auto=format&fit=crop'

req = urllib.request.urlopen(image_url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

# 2. Convert image to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Perform Sobel Edge Detection along the Y-axis
# dx=0, dy=1 detects horizontal edges (changes along vertical Y-axis)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)

# 4. Convert float results back to unsigned 8-bit integers (0 - 255)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# 5. Display Results in Google Colab
print("--- Original Image ---")
cv2_imshow(img)

print("\n--- Sobel Edge Detection (Y-Axis) ---")
cv2_imshow(sobel_y_abs)
