#Do Cropping, Copying and pasting image inside another image using OpenCV.
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Generate synthetic Source Image (Image 1: Geometric Pattern)
def generate_source_image(width=300, height=300):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width, height), (180, 180, 180), -1)
    cv2.circle(img, (150, 150), 80, (0, 0, 255), -1)
    cv2.putText(img, 'CROP', (110, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img

# Generate synthetic Target Image (Image 2: Landscape Pattern)
def generate_target_image(width=400, height=300):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width, 180), (230, 200, 150), -1)
    cv2.rectangle(img, (0, 180), (width, height), (100, 180, 100), -1)
    return img

# Load source and target images directly in memory
source_img = generate_source_image(width=300, height=300)
target_img = generate_target_image(width=400, height=300)

# Step 1: Define Cropping Region (y1:y2, x1:x2) from Source Image
crop_x1, crop_y1 = 70, 70
crop_x2, crop_y2 = 230, 230

# Crop the patch using NumPy array slicing
cropped_patch = source_img[crop_y1:crop_y2, crop_x1:crop_x2]

# Step 2: Define Paste Location (y1:y2, x1:x2) in Target Image
crop_h, crop_w, _ = cropped_patch.shape
paste_x1, paste_y1 = 120, 50
paste_x2, paste_y2 = paste_x1 + crop_w, paste_y1 + crop_h

# Step 3: Copy and Paste cropped patch into Target Image
pasted_result = target_img.copy()
pasted_result[paste_y1:paste_y2, paste_x1:paste_x2] = cropped_patch

# Display original images, cropped patch, and final pasted result in Google Colab
print("--- 1. Source Image ---")
cv2_imshow(source_img)

print("\n--- 2. Cropped Patch ---")
cv2_imshow(cropped_patch)

print("\n--- 3. Target Image with Cropped Patch Pasted Inside ---")
cv2_imshow(pasted_result)
