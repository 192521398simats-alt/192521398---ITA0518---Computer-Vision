#Perform Sharpening of Image using unsharp masking.
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# ==========================================
# 1. GENERATE AN INBUILT MEDIUM-TO-LOW SIZE IMAGE
# (Generates a 360x240 image directly in memory)
# ==========================================
def generate_sample_image(width=360, height=240):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add background pattern
    cv2.rectangle(img, (0, 0), (width, height), (220, 220, 220), -1)
    
    # Draw dark shapes with fine edge details
    cv2.rectangle(img, (40, 40), (160, 200), (50, 50, 50), -1)
    cv2.circle(img, (260, 120), 60, (90, 90, 90), -1)
    
    # Add fine text/line elements to demonstrate sharpening
    cv2.putText(img, 'UNSHARP', (60, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(img, (0, 0), (width, height), (30, 30, 30), 2)
    return img

# Load inbuilt image and convert to Grayscale
img_color = generate_sample_image(width=360, height=240)
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# ==========================================
# 2. PERFORM UNSHARP MASKING
# ==========================================
# Step A: Blur the original image using Gaussian Blur
blurred = cv2.GaussianBlur(gray, (9, 9), 10.0)

# Step B: Create the Unsharp Mask (Original - Blurred)
# Use float32 to prevent underflow when subtracting
mask = gray.astype(np.float32) - blurred.astype(np.float32)

# Step C: Add the weighted mask to the original image
# k is the sharpening scaling factor (amount)
k = 1.5
sharpened = gray.astype(np.float32) + (k * mask)

# Step D: Clip pixel values to [0, 255] and convert back to uint8
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
mask_display = cv2.convertScaleAbs(mask)

# ==========================================
# 3. DISPLAY RESULTS IN GOOGLE COLAB
# ==========================================
print("--- 1. Original Grayscale Image (360x240) ---")
cv2_imshow(gray)

print("\n--- 2. Unsharp Mask (Edges & High-Frequency Details) ---")
cv2_imshow(mask_display)

print("\n--- 3. Sharpened Image (Original + k * Mask) ---")
cv2_imshow(sharpened)
