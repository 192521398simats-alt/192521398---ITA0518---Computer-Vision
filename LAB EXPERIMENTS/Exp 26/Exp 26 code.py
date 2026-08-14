#Insert water marking to the image using OpenCV.
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Generate a new synthetic landscape image directly in memory
def generate_sample_image(width=360, height=240):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw sky background gradient
    for y in range(120):
        img[y, :] = (230 - y//2, 180 - y//3, 100)
        
    # Draw sun
    cv2.circle(img, (280, 50), 30, (0, 215, 255), -1)
    
    # Draw mountains using polygons
    mountain1 = np.array([[0, 180], [100, 70], [200, 180]], np.int32)
    mountain2 = np.array([[120, 180], [240, 90], [360, 180]], np.int32)
    cv2.fillPoly(img, [mountain1], (100, 80, 60))
    cv2.fillPoly(img, [mountain2], (70, 60, 40))
    
    # Draw ground field
    cv2.rectangle(img, (0, 180), (width, height), (50, 150, 50), -1)
    
    return img

# Load image
img_color = generate_sample_image(width=360, height=240)

# Create a blank overlay layer for the watermark with same dimensions
watermark_overlay = img_color.copy()

# Add text to the overlay layer
watermark_text = "WATER MARK"
cv2.putText(watermark_overlay, watermark_text, (30, 130), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

# Set transparency alpha value (0.0 transparent -> 1.0 opaque)
alpha = 0.35

# Blend original image and watermark overlay layer together
watermarked_img = cv2.addWeighted(watermark_overlay, alpha, img_color, 1 - alpha, 0)

# Display original and watermarked images in Google Colab
print("--- 1. Original Image (360x240) ---")
cv2_imshow(img_color)

print("\n--- 2. Watermarked Image (Alpha = 0.35) ---")
cv2_imshow(watermarked_img)
