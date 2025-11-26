import argparse
import pathlib
import os
import time
import pickle

import numpy as np

from general_motion_retargeting import HandMotionRetargeting
from general_motion_retargeting import HandMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--robot_hand",
        choices=["shadow", "allegro", "inspire", "leap", 
                 "ability", "svh"],
        default="inspire",
        help="Target robot hand type for dex retargeting.",
    )
    
    parser.add_argument(
        "--hand_type",
        choices=["left", "right", "both"],
        default="both",
        help="Which hand(s) to retarget.",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the retargeted hand motion.",
    )

    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted motion to keep the same as the source motion.",
    )

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    # Set default robot directory
    ROBOT_DIR = HERE / ".." / "assets" / "robots" / "hands"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )


    # Initialize hand retargeting
    retarget = HandMotionRetargeting(
        robot_name=args.robot_hand,
        robot_dir=ROBOT_DIR,
        hand_type=args.hand_type,
    )
    

    # hand_viewer = HandMotionViewer(
    #         robot_type=args.robot_hand,
    #         robot_dir=ROBOT_DIR,
    #         hand_type=args.hand_type,
    #         motion_fps=aligned_fps,
    #         record_video=args.record_video,
    #         video_path=f"videos/{args.robot_hand}_{args.hand_type}_{pathlib.Path(args.smplx_file).stem}.mp4",
    #     )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Start processing
    i = 0
    print("[Main] Starting retargeting...")

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual processing FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Get current frame
        smplx_data = smplx_data_frames[i]

        # retarget
        qpos = retarget.retarget(smplx_data)
        
        # visualize
        # hand_viewer.step(
        #     joint_positions=qpos,
        #     rate_limit=args.rate_limit,
        # )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle
        dof_pos = np.array(qpos_list)
        motion_data = {
            "fps": aligned_fps,
            "dof_pos": dof_pos,  # the only thing that matters for Dex hand
            "hand_type": args.hand_type,
            "robot_hand": args.robot_hand,
        }

        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    # hand_viewer.close()