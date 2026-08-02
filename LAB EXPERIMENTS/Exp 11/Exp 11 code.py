# Affine Transformation Pipeline
# Demonstrates 2D Affine Transformation mapping:
# 1. Synthesizes a structured test pattern with a reference grid.
# 2. Defines 3 non-collinear source points and their corresponding target points.
# 3. Computes the 2x3 Affine Transformation Matrix (M).
# 4. Warps the spatial coordinate grid using cv2.warpAffine.


import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 1. GENERATE SYNTHETIC TEST IMAGE WITH GRID LINES ---
height, width = 500, 500
image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light grey background

# Draw a background grid to clearly visualize spatial distortion
grid_spacing = 50
for x in range(0, width, grid_spacing):
    cv2.line(image, (x, 0), (x, height), (200, 200, 200), 1)
for y in range(0, height, grid_spacing):
    cv2.line(image, (0, y), (width, y), (200, 200, 200), 1)

# Draw a centered square object
cv2.rectangle(image, (150, 150), (350, 350), (40, 100, 220), -1)  # Blue-orange box
cv2.rectangle(image, (150, 150), (350, 350), (0, 0, 0), 3)        # Black border
cv2.circle(image, (250, 250), 30, (255, 255, 255), -1)             # Center white marker

# --- 2. DEFINE 3-POINT CORRESPONDENCES ---
# Source points (Top-Left, Top-Right, Bottom-Left corners of the central square)
pts_src = np.float32([
    [150, 150],  # Point 1: Top-Left
    [350, 150],  # Point 2: Top-Right
    [150, 350]   # Point 3: Bottom-Left
])

# Target points (Shifted, rotated, scaled, and sheared mapping coordinates)
pts_dst = np.float32([
    [100, 180],  # Shifted Top-Left
    [380, 120],  # Shifted & Rotated Top-Right
    [180, 420]   # Sheared Bottom-Left
])

# --- 3. COMPUTE MATRIX & WARP IMAGE ---
# Calculate the 2x3 Affine Transformation Matrix M
M = cv2.getAffineTransform(pts_src, pts_dst)

# Apply the spatial warp transformation
warped_image = cv2.warpAffine(image, M, (width, height), borderValue=(240, 240, 240))

# Annotate source and destination reference points for visualization
annotated_src = image.copy()
annotated_dst = warped_image.copy()

colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255)]  # Red, Green, Blue markers
for i in range(3):
    pt_s = tuple(pts_src[i].astype(int))
    pt_d = tuple(pts_dst[i].astype(int))
    cv2.circle(annotated_src, pt_s, 8, colors[i], -1)
    cv2.circle(annotated_dst, pt_d, 8, colors[i], -1)

# --- 4. VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

axes[0].imshow(cv2.cvtColor(annotated_src, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Original Image\n(With 3 Source Reference Points)", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(annotated_dst, cv2.COLOR_BGR2RGB))
axes[1].set_title("2. Affine Transformed Image\n(Rotation + Scale + Shear + Translation)", fontsize=11, fontweight='bold')
axes[1].axis('off')

plt.suptitle("2D Affine Transformation Mapping via 3-Point Correspondences", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# --- 5. LOG COMPUTED MATRICES ---
print("\n" + "="*60)
print("             COMPUTED AFFINE TRANSFORMATION MATRIX             ")
print("="*60)
print(f"Matrix M (2x3):\n{M}")
print("="*60 + "\n")
