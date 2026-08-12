#Perform Edge detection using canny method
import cv2
import numpy as np
# import urllib.request # Not needed if reading from local file
from google.colab.patches import cv2_imshow

# 1. Read the image from a local file
# Assuming the user has uploaded 'sample.jpg' to the Colab environment
image_path = '/content/sample.jpg' # Use the local image file
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Check if image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path}. Please ensure it's uploaded.")
else:
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Apply Gaussian Blur to smooth out fine noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Perform Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # 5. Display the results in Google Colab
    print("-- Original Image ---")
    cv2_imshow(img)

    print("\n--- Canny Edge Detection Result ---")
    cv2_imshow(edges)
