# ==============================================================================
# Homography Transformation via Direct Linear Transform (DLT)
# ==============================================================================
# 1. Synthesizes a scene with a quadrilateral plane (e.g., planar poster/billboard).
# 2. Maps 4 point correspondences between the source plane and destination plane.
# 3. Solves for the 3x3 Homography Matrix H using cv2.findHomography.
# 4. Applies spatial warping via cv2.warpPerspective.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 1. GENERATE SYNTHETIC SCENE WITH PLANAR OBJECT ---
height, width = 500, 500
image = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark background

# Synthesize a tilted target billboard plane (quadrilateral region)
pts_src = np.float32([
    [100, 120],  # Top-Left corner
    [380, 60],   # Top-Right corner
    [450, 420],  # Bottom-Right corner
    [140, 460]   # Bottom-Left corner
])

# Draw planar polygon object with a inner target cross
cv2.fillPoly(image, [pts_src.astype(np.int32)], (220, 220, 220))
cv2.polylines(image, [pts_src.astype(np.int32)], True, (0, 0, 0), 3)
cv2.line(image, tuple(pts_src[0].astype(int)), tuple(pts_src[2].astype(int)), (200, 0, 0), 3)
cv2.line(image, tuple(pts_src[1].astype(int)), tuple(pts_src[3].astype(int)), (0, 0, 200), 3)

# --- 2. DEFINE TARGET RECTANGULAR DESTINATION GRID ---
out_w, out_h = 300, 300
pts_dst = np.float32([
    [0, 0],          # Top-Left
    [out_w, 0],      # Top-Right
    [out_w, out_h],  # Bottom-Right
    [0, out_h]       # Bottom-Left
])

# --- 3. COMPUTE HOMOGRAPHY MATRIX & WARP IMAGE ---
# Compute the 3x3 Homography Matrix H using standard RANSAC/DLT solver
H, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)

# Warp source plane to destination plane
rectified_plane = cv2.warpPerspective(image, H, (out_w, out_h))

# Annotate source points
annotated_src = image.copy()
colors = [(255, 0, 0), (0, 255, 0), (0, 165, 255), (255, 255, 0)]
for i in range(4):
    cv2.circle(annotated_src, tuple(pts_src[i].astype(int)), 8, colors[i], -1)

# --- 4. VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

axes[0].imshow(cv2.cvtColor(annotated_src, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Source Planar Object\n(With 4 Tracked Corner Points)", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(rectified_plane, cv2.COLOR_BGR2RGB))
axes[1].set_title("2. Homography Rectified Output\n(Normalized Frontal Plane View)", fontsize=11, fontweight='bold')
axes[1].axis('off')

plt.suptitle("Planar Rectification via 3x3 Homography Matrix Estimation", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# --- 5. LOG MATRICES ---
print("\n" + "="*60)
print("             ESTIMATED 3x3 HOMOGRAPHY MATRIX (H)               ")
print("="*60)
print(H)
print("="*60 + "\n")
