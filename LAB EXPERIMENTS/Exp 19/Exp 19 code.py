#Perform Edge detection using Sobel Matrix along XY axis
import cv2
import numpy as np
import urllib.request
from google.colab.patches import cv2_imshow

# 1. Download sample image from web
image_url = 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800&auto=format&fit=crop'
req = urllib.request.urlopen(image_url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

# 2. Resize image to Medium Size (Width: 500px, Height: 333px)
medium_img = cv2.resize(img, (500, 333), interpolation=cv2.INTER_AREA)

# 3. Convert medium image to Grayscale
gray = cv2.cvtColor(medium_img, cv2.COLOR_BGR2GRAY)

# 4. Calculate Sobel gradients along X and Y axes
sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)

# 5. Take absolute values
abs_sobel_x = cv2.convertScaleAbs(sobel_x)
abs_sobel_y = cv2.convertScaleAbs(sobel_y)

# 6. Combine both X and Y gradients (Sobel XY)
sobel_xy = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# 7. Display Results in Google Colab
print("--- Medium Size Original Image (500x333) ---")
cv2_imshow(medium_img)

print("\n--- Sobel X (Vertical Edges) ---")
cv2_imshow(abs_sobel_x)

print("\n--- Sobel Y (Horizontal Edges) ---")
cv2_imshow(abs_sobel_y)

print("\n--- Combined Sobel XY (Full Edge Map) ---")
cv2_imshow(sobel_xy)
