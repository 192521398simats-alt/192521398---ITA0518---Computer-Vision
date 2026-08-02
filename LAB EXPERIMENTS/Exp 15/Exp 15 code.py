# ==============================================================================
# Direct Linear Transformation (DLT) Algorithm Implementation
# ==============================================================================
# Solves Ah = 0 using Singular Value Decomposition (SVD) from scratch:
# 1. Constructs matrix A from 4 point correspondences.
# 2. Computes SVD to extract the optimal Homography Matrix H.
# 3. Warps the image plane using the manually derived H matrix.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 1. SVD-BASED DIRECT LINEAR TRANSFORMATION (DLT) SOLVER ---
def compute_homography_dlt(pts_src, pts_dst):
    """
    Computes 3x3 Homography Matrix H using DLT algorithm via SVD.
    
    Parameters:
        pts_src: (N, 2) numpy array of source coordinates (x, y)
        pts_dst: (N, 2) numpy array of target coordinates (x', y')
    Returns:
        H: (3, 3) normalized Homography Matrix
    """
    num_pts = pts_src.shape[0]
    if num_pts < 4:
        raise ValueError("DLT requires at least 4 point correspondences.")
        
    A = []
    for i in range(num_pts):
        x, y = pts_src[i][0], pts_src[i][1]
        u, v = pts_dst[i][0], pts_dst[i][1]
        
        # Build 2 rows of matrix A for point pair i
        row1 = [-x, -y, -1,  0,  0,  0, u*x, u*y, u]
        row2 = [ 0,  0,  0, -x, -y, -1, v*x, v*y, v]
        
        A.append(row1)
        A.append(row2)
        
    A = np.array(A, dtype=np.float64)
    
    # Solve Ah = 0 using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)
    
    # H is the last row of Vt (corresponding to the smallest singular value)
    H = Vt[-1].reshape((3, 3))
    
    # Normalize matrix so that H[2, 2] = 1
    return H / H[2, 2]


# --- 2. GENERATE SYNTHETIC INPUT SCENE ---
height, width = 500, 500
image = np.ones((height, width, 3), dtype=np.uint8) * 35  # Dark background

# Define source quadrilateral points (distorted planar document)
pts_src = np.float32([
    [120, 140],  # Top-Left
    [400, 80],   # Top-Right
    [460, 430],  # Bottom-Right
    [150, 470]   # Bottom-Left
])

# Draw paper structure with interior features
cv2.fillPoly(image, [pts_src.astype(np.int32)], (230, 230, 230))
cv2.polylines(image, [pts_src.astype(np.int32)], True, (0, 0, 0), 3)
cv2.circle(image, (270, 280), 40, (0, 0, 200), -1)  # Synthetic red mark

# --- 3. DEFINE DESTINATION GRID & EXECUTE CUSTOM DLT ---
out_w, out_h = 300, 300
pts_dst = np.float32([
    [0, 0],          # Top-Left
    [out_w, 0],      # Top-Right
    [out_w, out_h],  # Bottom-Right
    [0, out_h]       # Bottom-Left
])

# Run custom DLT function
H_dlt = compute_homography_dlt(pts_src, pts_dst)

# Apply spatial warping using custom-built H
rectified_image = cv2.warpPerspective(image, H_dlt, (out_w, out_h))


# --- 4. VISUALIZATION PIPELINE ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Original Distorted Frame\n(Input Source Points)", fontsize=11, fontweight='bold')
axes[0].axis('off')

# Highlight source point markers
colors = [(255, 0, 0), (0, 255, 0), (255, 165, 0), (0, 0, 255)]
for i in range(4):
    axes[0].plot(pts_src[i][0], pts_src[i][1], marker='o', markersize=8, color=np.array(colors[i])/255.0)

axes[1].imshow(cv2.cvtColor(rectified_image, cv2.COLOR_BGR2RGB))
axes[1].set_title("2. DLT Rectified Result\n(Frontal Perspective View)", fontsize=11, fontweight='bold')
axes[1].axis('off')

plt.suptitle("Planar Rectification via Direct Linear Transformation (DLT)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# Log matrix output
print("\n" + "="*65)
print("        SOLVED HOMOGRAPHY MATRIX (H) VIA CUSTOM SVD-DLT          ")
print("="*65)
print(H_dlt)
print("="*65 + "\n")
