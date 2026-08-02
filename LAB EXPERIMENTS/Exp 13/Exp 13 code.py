# ==============================================================================
# Frame-by-Frame Perspective Transformation on Video
# ==============================================================================
# Demonstrates 3x3 Homography transformation across a temporal video stream:
# 1. Synthesizes an MP4 video containing a moving tilted document.
# 2. Tracks/defines 4 corner points per frame.
# 3. Applies cv2.warpPerspective frame-by-frame.
# 4. Encodes and displays the output MP4 natively in Google Colab.
# ==============================================================================

import cv2
import numpy as np
from IPython.display import HTML
from base64 import b64encode

# --- 1. SYNTHESIZE AN INPUT VIDEO WITH PERSPECTIVE DISTORTION ---
input_video_path = 'input_distorted_doc.mp4'
output_video_path = 'output_perspective_corrected.mp4'

fps = 30
duration_sec = 3
total_frames = fps * duration_sec
frame_w, frame_h = 600, 600

# Initialize VideoWriter with mp4v codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_input = cv2.VideoWriter(input_video_path, fourcc, fps, (frame_w, frame_h))

# Generate synthetic frames where the document slightly oscillates
for t in range(total_frames):
    frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 40  # Dark background
    
    # Introduce small periodic movement over time
    shift_x = int(15 * np.sin(2 * np.pi * t / fps))
    shift_y = int(10 * np.cos(2 * np.pi * t / fps))
    
    # Dynamic 4 corner source points for the distorted paper
    pts_src = np.array([
        [120 + shift_x, 150 + shift_y],  # Top-Left
        [420 + shift_x, 80 - shift_y],   # Top-Right
        [500 + shift_x, 480 + shift_y],  # Bottom-Right
        [180 + shift_x, 520 - shift_y]   # Bottom-Left
    ], dtype=np.int32)
    
    # Draw paper and internal features
    cv2.fillPoly(frame, [pts_src], (240, 240, 240))
    cv2.polylines(frame, [pts_src], True, (0, 0, 0), 2)
    
    # Draw horizontal text lines
    p1_tl, p2_tr = pts_src[0], pts_src[1]
    p4_bl, p3_br = pts_src[3], pts_src[2]
    
    for line_ratio in [0.2, 0.4, 0.6, 0.8]:
        start_pt = (1 - line_ratio) * p1_tl + line_ratio * p4_bl
        end_pt = (1 - line_ratio) * p2_tr + line_ratio * p3_br
        pt_a = tuple((start_pt + [20, 0]).astype(int))
        pt_b = tuple((end_pt - [20, 0]).astype(int))
        color = (0, 0, 200) if line_ratio == 0.8 else (60, 60, 60)
        cv2.line(frame, pt_a, pt_b, color, 3)

    out_input.write(frame)

out_input.release()
print(f"Generated synthetic input video: {input_video_path}")


# --- 2. VIDEO PROCESSING PIPELINE: FRAME-BY-FRAME HOMOGRAPHY ---
cap = cv2.VideoCapture(input_video_path)

# Target flattened output dimensions
out_w, out_h = 300, 400
pts_dst = np.float32([
    [0, 0],          # Top-Left target
    [out_w, 0],      # Top-Right target
    [out_w, out_h],  # Bottom-Right target
    [0, out_h]       # Bottom-Left target
])

# Initialize VideoWriter for the transformed video stream
out_perspective = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Track dynamic source points per frame
    shift_x = int(15 * np.sin(2 * np.pi * frame_count / fps))
    shift_y = int(10 * np.cos(2 * np.pi * frame_count / fps))
    
    pts_src = np.float32([
        [120 + shift_x, 150 + shift_y],
        [420 + shift_x, 80 - shift_y],
        [500 + shift_x, 480 + shift_y],
        [180 + shift_x, 520 - shift_y]
    ])
    
    # 1. Compute 3x3 Perspective Matrix H for current frame
    H = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
    # 2. Warp current frame to flattened top-down view
    warped_frame = cv2.warpPerspective(frame, H, (out_w, out_h))
    
    # 3. Write transformed frame to output stream
    out_perspective.write(warped_frame)
    frame_count += 1

cap.release()
out_perspective.release()
print(f"Finished perspective transformation on {frame_count} video frames.")


# --- 3. INLINE VIDEO PLAYER FOR GOOGLE COLAB ---
def play_video_in_colab(file_path):
    """Encodes mp4 file to Base64 to render HTML5 video inline in Colab."""
    mp4 = open(file_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f"""
    <video width=350 controls autoplay loop>
        <source src="{data_url}" type="video/mp4">
    </video>
    """)

print("\nTransformed Video Output (Perspective Corrected Stream):")
play_video_in_colab(output_video_path)
