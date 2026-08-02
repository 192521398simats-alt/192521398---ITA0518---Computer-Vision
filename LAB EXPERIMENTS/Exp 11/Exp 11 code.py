# ==============================================================================
# 4-Point Perspective Transformation (Homography) Pipeline
# ==============================================================================
# Demonstrates 3x3 Perspective Transformation mapping:
# 1. Synthesizes a distorted target image (e.g., a document shot at an angle).
# 2. Defines 4 non-collinear corner source points and target destination coordinates.
# 3. Computes the 3x3 Perspective Transformation Matrix (H) via cv2.getPerspectiveTransform.
# 4. Corrects the spatial grid distortion using cv2.warpPerspective.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 1. GENERATE SYNTHETIC DISTORTED DOCUMENT SCENE ---
height, width = 600, 600
image = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark background surface

# Draw a synthetic "tilted" document card on the surface
doc_pts = np.array([[120, 150], [420, 80], [500, 480], [180, 520]], dtype=np.int32)
cv2.fillPoly(image, [doc_pts], (240, 240, 240))  # White paper base
cv2.polylines(image, [doc_pts], True, (0, 0, 0), 2)  # Paper boundary

# Draw dummy text lines on the distorted paper
cv2.line(image, (160, 200), (390, 150), (60, 60, 60), 4)
cv2.line(image, (170, 250), (410, 200), (60, 60, 60), 4)
cv2.line(image, (180, 300), (430, 250), (60, 60, 60), 4)
cv2.line(image, (200, 400), (460, 350), (0, 0, 200), 6)  # Red signature line

# --- 2. DEFINE 4-POINT CORRESPONDENCES ---
# Source points: The 4 corners of the distorted paper (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
pts_src = np.float32([
    [120, 150],  # Top-Left corner
    [420, 80],   # Top-Right corner
    [500, 480],  # Bottom-Right corner
    [180, 520]   # Bottom-Left corner
])

# Target points: Flattened, rectangular top-down perspective (300x400 output box)
out_w, out_h = 300, 400
pts_dst = np.float32([
    [0, 0],          # Top-Left target
    [out_w, 0],      # Top-Right target
    [out_w, out_h],  # Bottom-Right target
    [0, out_h]       # Bottom-Left target
])

# --- 3. COMPUTE HOMOGRAPHY MATRIX & WARP PERSPECTIVE ---
# Calculate the 3x3 Perspective Transformation Matrix H
H = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Apply the 3D-to-2D perspective warp
warped_document = cv2.warpPerspective(image, H, (out_w, out_h))

# Annotate source points on a copy of the original image for visualization
annotated_src = image.copy()
colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

for i in range(4):
    pt = tuple(pts_src[i].astype(int))
    cv2.circle(annotated_src, pt, 9, colors[i], -1)

# --- 4. VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

axes[0].imshow(cv2.cvtColor(annotated_src, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Original Distorted Image\n(With 4 Corner Source Points)", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(warped_document, cv2.COLOR_BGR2RGB))
axes[1].set_title("2. Perspective Corrected Image\n(Flattened Top-Down View)", fontsize=11, fontweight='bold')
axes[1].axis('off')

plt.suptitle("4-Point Perspective Transformation (Homography)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# --- 5. LOG COMPUTED MATRICES ---
print("\n" + "="*60)
print("          COMPUTED PERSPECTIVE TRANSFORMATION MATRIX           ")
print("="*60)
print(f"Matrix H (3x3):\n{H}")
print("="*60 + "\n")
