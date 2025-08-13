import mujoco
import mujoco.viewer
import numpy as np
import mediapy as media
import os
from pathlib import Path
import sys


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig


def visualize_and_record_trajectory(trajectory_path, output_video_path="output.mp4", 
                                  fps=30, playback_speed=1.0):
    """
    Visualize robot trajectory and save as video using mediapy.
    
    Args:
        trajectory_path: Path to .npy trajectory file
        output_video_path: Path to save the output video
        fps: Frames per second for the video
        playback_speed: Speed multiplier for visualization
    """
    
    # Load trajectory
    print(f"Loading trajectory from {trajectory_path}")
    trajectory = np.load(trajectory_path, allow_pickle=True)[:, 1:20]

    print(f"Trajectory shape: {trajectory.shape}")
    
    # Setup environment and viewer
    config = UnitreeGo2EnvConfig()
    env = UnitreeGo2Env(config)
    
    # Get MuJoCo model and data
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    
    # Launch viewer
    with mujoco.Renderer(model, 480, 640) as renderer:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'track')
    
        print("Starting visualization and recording...")
        
        # Calculate frame timing
        dt = config.dt
        frame_delay = dt / playback_speed
        
        # Record frames
        frames = []
        
        for i in range(trajectory.shape[0]):
            # Set robot state
            data.qpos[:] = trajectory[i]
            data.qvel[:] = 0.0
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Update viewer
            renderer.update_scene(data, camera=camera_id)
            
            # # Capture frame for video
            if output_video_path:
                # Get viewer image
                img = renderer.render() 
                if img is not None:
                    # Convert RGBA to RGB for mediapy (remove alpha channel)
                    img_rgb = img[:, :, :3]  # Take only RGB channels
                    frames.append(img_rgb)
            
            # # Control playback speed
            # import time
            # time.sleep(frame_delay)
                
            # Progress indicator
            if i % 100 == 0:
                print(f"Processed {i}/{len(trajectory)} frames")
        
        print("Visualization completed!")
        
        # Save video using mediapy if frames were captured
        if frames and output_video_path:
            print(f"Saving video to {output_video_path}")
            
            # Convert frames to numpy array
            frames_array = np.array(frames)
            print(f"Video shape: {frames_array.shape}")
            
            # Save video using mediapy
            media.write_video(output_video_path, frames_array, fps=fps)
            
            print(f"Video saved successfully! Total frames: {len(frames)}")
            print(f"Video saved at: {os.path.abspath(output_video_path)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Go2 robot trajectory")
    parser.add_argument("--trajectory_path", help="Path to .npy trajectory file")
    parser.add_argument("--output", "-o", default="output.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    parser.add_argument("--no-video", action="store_true", help="Only visualize, don't record video")
    
    args = parser.parse_args()

    visualize_and_record_trajectory(
        args.trajectory_path, 
        args.output, 
        args.fps, 
        args.speed
    )




