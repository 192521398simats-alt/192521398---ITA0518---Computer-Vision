#Perform Sharpening of Image using Laplacian mask implemented with an extension of diagonal neighbors.
import cv2
import numpy as np
import urllib.request
from google.colab.patches import cv2_imshow

# 1. Image Download with User-Agent Header (Prevents HTTP 403 / 404 errors)
image_url = 'https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=800&auto=format&fit=crop'

try:
    # Adding User-Agent makes Python request look like a real browser
    req_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(image_url, headers=req_headers)
    
    with urllib.request.urlopen(req) as response:
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

except Exception as e:
    print(f"Network error ({e}). Generating fallback geometric image...")
    img = None

# Fallback: Generate a high-contrast test image directly in memory if link fails
if img is None:
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (200, 200, 200), -1)
    cv2.circle(img, (150, 150), 70, (50, 50, 50), -1)
    cv2.line(img, (0, 0), (300, 300), (255, 255, 255), 5)

# 2. Resize to Low-to-Medium Size (360x240 pixels)
small_med_img = cv2.resize(img, (360, 240), interpolation=cv2.INTER_AREA)

# 3. Convert to Grayscale
gray = cv2.cvtColor(small_med_img, cv2.COLOR_BGR2GRAY)

# 4. Define Extended Laplacian Mask (8-Neighbors, Negative Center: -8)
laplacian_8_neighbor = np.array([[ 1,  1,  1],
                                 [ 1, -8,  1],
                                 [ 1,  1,  1]], dtype=np.float32)

# 5. Apply Spatial Filtering
laplacian_response = cv2.filter2D(gray.astype(np.float32), -1, laplacian_8_neighbor)

# 6. Sharpening Formula: Original - Laplacian
sharpened = gray.astype(np.float32) - laplacian_response
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
laplacian_display = cv2.convertScaleAbs(laplacian_response)

# 7. Display Results in Google Colab
print("--- Original Grayscale Image (360x240) ---")
cv2_imshow(gray)

print("\n--- Extended Laplacian Response (8-Neighbors) ---")
cv2_imshow(laplacian_display)

print("\n--- Sharpened Image (Diagonal Extension) ---")
cv2_imshow(sharpened)
