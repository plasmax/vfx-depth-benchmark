"""
Testing out MegaSaM using a Runpod.io 48GB VRAM instance
https://github.com/mega-sam/mega-sam

First, I had to install miniconda

```bash
cd /workspace
wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
git clone --recursive https://github.com/mega-sam/mega-sam.git
cd ./mega-sam/
/workspace/miniconda3/bin/conda env create -f environment.yml
wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
/workspace/miniconda3/bin/conda install xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2

```

"""

import os
import sys
import subprocess
import argparse
import cv2
import numpy as np
import glob
from pathlib import Path
import shutil

def run_command(command, env=None):
    """Runs a shell command and handles errors."""
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        sys.exit(1)

def extract_frames(video_path, output_dir):
    """Extracts frames from a video file."""
    print(f"Extracting frames from {video_path} to {output_dir}...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save as 5-digit zero-padded filename (e.g., 00000.jpg)
        cv2.imwrite(os.path.join(output_dir, f"{count:05d}.jpg"), frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames.")

def render_video(npz_path, output_video_path):
    """Renders the final NPZ output to an MP4 video."""
    print(f"Rendering video to {output_video_path}...")
    data = np.load(npz_path)
    images = data['images']  # RGB images
    depths = data['depths']  # Estimated Depths

    # Prepare output video writer
    h, w, _ = images[0].shape
    # Double width for side-by-side comparison
    out_size = (w * 2, h)
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, out_size)

    for i in range(len(images)):
        img = images[i] # RGB
        depth = depths[i]

        # Normalize depth for visualization (invert for disparity-like view or keep as depth)
        # Using 1/depth for disparity visualization usually looks better
        disp = 1.0 / (depth + 1e-6)
        disp_norm = (disp - disp.min()) / (disp.max() - disp.min())
        disp_vis = (disp_norm * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

        # Concatenate Image and Depth
        combined = np.hstack((img, disp_color))
        out.write(combined)

    out.release()
    print("Video rendering complete.")

def main():
    parser = argparse.ArgumentParser(description="Run MegaSam Inference on a Video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input .mov or .mp4 file")
    parser.add_argument("--output_path", type=str, default="assets/videos/MegaSaM/output.mp4", help="Path for final video output")
    args = parser.parse_args()

    # 1. Setup Paths and Scene Name
    video_path = args.video_path
    scene_name = os.path.splitext(os.path.basename(video_path))[0]
    repo_root = os.getcwd()
    
    # Directory to store extracted frames
    data_dir = os.path.join(repo_root, "inputs", scene_name)
    
    # Checkpoints (Modify these if your paths differ)
    ckpt_depth_anything = "Depth-Anything/checkpoints/depth_anything_vitl14.pth"
    ckpt_raft = "cvd_opt/raft-things.pth"
    ckpt_megasam = "checkpoints/megasam_final.pth"

    # 2. Extract Frames
    extract_frames(video_path, data_dir)

    # 3. Run DepthAnything
    print("--- Step 1/5: Running DepthAnything ---")
    da_out_dir = os.path.join("Depth-Anything/video_visualization", scene_name)
    cmd_da = (
        f"python Depth-Anything/run_videos.py --encoder vitl "
        f"--load-from {ckpt_depth_anything} "
        f"--img-path {data_dir} "
        f"--outdir {da_out_dir}"
    )
    run_command(cmd_da)

    # 4. Run UniDepth
    print("--- Step 2/5: Running UniDepth ---")
    # Need to add UniDepth to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{os.path.join(repo_root, 'UniDepth')}"
    
    uni_out_dir = "UniDepth/outputs"
    cmd_uni = (
        f"python UniDepth/scripts/demo_mega-sam.py "
        f"--scene-name {scene_name} "
        f"--img-path {data_dir} "
        f"--outdir {uni_out_dir}"
    )
    run_command(cmd_uni, env=env)

    # 5. Run Camera Tracking (Droid-SLAM)
    print("--- Step 3/5: Running Camera Tracking ---")
    # Note: test_demo.py expects relative paths for depth inputs usually, or we pass absolute
    # The script arguments: --mono_depth_path, --metric_depth_path
    cmd_track = (
        f"python camera_tracking_scripts/test_demo.py "
        f"--datapath {data_dir} "
        f"--weights {ckpt_megasam} "
        f"--scene_name {scene_name} "
        f"--mono_depth_path {os.path.join(repo_root, 'Depth-Anything/video_visualization')} "
        f"--metric_depth_path {os.path.join(repo_root, 'UniDepth/outputs')} "
        f"--disable_vis"
    )
    run_command(cmd_track)

    # 6. Run Optical Flow (RAFT)
    print("--- Step 4/5: Running Optical Flow ---")
    cmd_flow = (
        f"python cvd_opt/preprocess_flow.py "
        f"--datapath {data_dir} "
        f"--model {ckpt_raft} "
        f"--scene_name {scene_name} "
        f"--mixed_precision"
    )
    run_command(cmd_flow)

    # 7. Run CVD Optimization
    print("--- Step 5/5: Running CVD Optimization ---")
    cmd_cvd = (
        f"python cvd_opt/cvd_opt.py "
        f"--scene_name {scene_name} "
        f"--w_grad 2.0 --w_normal 5.0"
    )
    run_command(cmd_cvd)

    # 8. Render Final Video
    final_npz = os.path.join("outputs_cvd", f"{scene_name}_sgd_cvd_hr.npz")
    if os.path.exists(final_npz):
        render_video(final_npz, args.output_path)
        print(f"SUCCESS: Video saved to {args.output_path}")
    else:
        print(f"ERROR: Could not find output file {final_npz}")

if __name__ == "__main__":
    main()