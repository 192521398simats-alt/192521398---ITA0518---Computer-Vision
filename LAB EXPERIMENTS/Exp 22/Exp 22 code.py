#Perform Sharpening of Image using Laplacian mask with positive center coefficient.
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# ==========================================
# 1. GENERATE AN INBUILT MEDIUM-SIZE IMAGE
# (Generates a geometric image in memory, no internet required)
# ==========================================
def generate_test_image(size=(400, 400)):
    """Generates a medium-sized synthetic image with clear edges."""
    # Create black canvas
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Draw gray rectangle
    cv2.rectangle(img, (50, 50), (350, 350), (180, 180, 180), -1)
    
    # Draw darker gray circle
    cv2.circle(img, (200, 200), 100, (80, 80, 80), -1)
    
    # Draw a diagonal white line
    cv2.line(img, (0, 0), (400, 400), (255, 255, 255), 4)
    
    return img

# Generate and convert to Grayscale
img_color = generate_test_image()
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# ==========================================
# 2. DEFINE LAPLACIAN MASK (POSITIVE CENTER)
# ==========================================
# Standard 4-neighbor Laplacian mask with +4 at the center
laplacian_mask_positive = np.array([[ 0, -1,  0],
                                     [-1,  4, -1],
                                     [ 0, -1,  0]], dtype=np.float32)

# ==========================================
# 3. APPLY SHARPENING
# ==========================================
# Step 3a: Apply spatial filter using cv2.filter2D.
# We use float32 to handle negative gradient values before combination.
laplacian_response = cv2.filter2D(gray.astype(np.float32), -1, laplacian_mask_positive)

# Step 3b: Sharpening Formula for Positive Center: Original + Laplacian
sharpened = gray.astype(np.float32) + laplacian_response

# Step 3c: Post-Processing
# Use np.clip to ensure pixel values stay in valid [0, 255] range,
# then convert back to unsigned 8-bit integers.
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Convert the raw filter response for display purposes
laplacian_display = cv2.convertScaleAbs(laplacian_response)

# ==========================================
# 4. DISPLAY RESULTS IN GOOGLE COLAB
# ==========================================
print("--- Inbuilt Original Grayscale Image (400x400) ---")
cv2_imshow(gray)

print("\n--- Laplacian Edge Response (Edges Highlighted) ---")
cv2_imshow(laplacian_display)

print("\n--- Sharpened Image (Original + Laplacian) ---")
cv2_imshow(sharpened)
