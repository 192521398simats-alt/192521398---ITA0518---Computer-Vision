# ==============================================================================
# QUESTION 4: Effect of Image Scaling on Feature Detection
# Description: Generates a high-contrast textured image in code and resizes it 
#              to analyze feature count variations at multiple scales.
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Input Image with rich corners & shapes directly inside the script
img_14 = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img_14, (30, 30), (130, 130), 220, -1)
cv2.circle(img_14, (220, 80), 40, 150, -1)
cv2.polylines(img_14, [np.array([[180, 180], [260, 270], [120, 270]])], True, 255, 3)
cv2.putText(img_14, 'SCALE', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)

# 2. Define scales to test
scales = [1.0, 0.75, 0.5, 0.25]
orb = cv2.ORB_create(nfeatures=500)

plt.figure(figsize=(16, 4))
print("--- Scale Analysis Log ---")

# 3. Downscale image and run ORB feature detection on each scale
for idx, scale in enumerate(scales):
    new_w = int(img_14.shape[1] * scale)
    new_h = int(img_14.shape[0] * scale)
    scaled_img = cv2.resize(img_14, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    keypoints = orb.detect(scaled_img, None)
    img_kp = cv2.drawKeypoints(scaled_img, keypoints, None, color=(0, 255, 0))
    
    print(f"Scale: {scale:4.2f}x | Resolution: {new_w}x{new_h} | Detected Keypoints: {len(keypoints)}")
    
    plt.subplot(1, len(scales), idx + 1)
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"Scale: {scale}x\n({len(keypoints)} Features)", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()
