import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from scipy.spatial.transform import Rotation as R

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

from rich import print


class HandMotionRetargeting:
    """Hand Motion Retargeting using dex-retargeting library.
    
    Retargets SMPL-X hand motion data to robot hands.
    """
    
    def __init__(
        self,
        robot_name: str,
        robot_dir: Path,
        hand_type: str = "both",  # "left", "right", or "both",
        retargeting_type: str = "dexpilot",
        verbose: bool = True,
    ) -> None:
        """
        Initialize Hand Motion Retargeting.
        
        Args:
            robot_name: Name of the target robot (e.g., "inspire", "shadow", "allegro")
            hand_type: Which hand(s) to retarget - "left", "right", or "both"
            retargeting_type: Type of retargeting algorithm ("dexpilot", "vector", "position")
            verbose: Print detailed information
        """
        robot_name = RobotName[robot_name.lower()]
        self.hand_type = HandType[hand_type.lower()]
        self.retargeting_type = RetargetingType[retargeting_type.lower()]
        self.verbose = verbose

        RetargetingConfig.set_default_urdf_dir(robot_dir)
        self.config_paths = {}
        self.retargeters = {}
        
        if self.hand_type == HandType.left:
            left_config_path = get_default_config_path(
                robot_name, self.retargeting_type, HandType.left
            )
            
            self.config_paths["left"] = left_config_path
            left_config = RetargetingConfig.load_from_file(left_config_path)
            self.retargeters["left"] = left_config.build()
            
            if verbose:
                print(f"[HandRetarget] Initialized left hand retargeting for {robot_name}")
                print(f"[HandRetarget] Config: {left_config_path}")
                print(f"[HandRetarget] Left hand DoFs: {len(self.retargeters['left'].optimizer.robot.dof_joint_names)}")
        
        if self.hand_type == HandType.right:
            right_config_path = get_default_config_path(
                robot_name, self.retargeting_type, HandType.right
            )
            
            self.config_paths["right"] = right_config_path
            right_config = RetargetingConfig.load_from_file(right_config_path)
            self.retargeters["right"] = right_config.build()
            
            if verbose:
                print(f"[HandRetarget] Initialized right hand retargeting for {robot_name}")
                print(f"[HandRetarget] Config: {right_config_path}")
                print(f"[HandRetarget] Right hand DoFs: {len(self.retargeters['right'].optimizer.robot.dof_joint_names)}")
        
        if not self.retargeters:
            raise RuntimeError("Failed to initialize any hand retargeters")
        
        # Define SMPL-X hand joint mapping
        self.smplx_hand_joints = {
            "left": [
                "left_wrist",
                "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4",
                "left_index1", "left_index2", "left_index3", "left_index4",
                "left_middle1", "left_middle2", "left_middle3", "left_middle4",
                "left_ring1", "left_ring2", "left_ring3", "left_ring4",
                "left_pinky1", "left_pinky2", "left_pinky3", "left_pinky4"
            ],
            "right": [
                "right_wrist",
                "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4",
                "right_index1", "right_index2", "right_index3", "right_index4",
                "right_middle1", "right_middle2", "right_middle3", "right_middle4",
                "right_ring1", "right_ring2", "right_ring3", "right_ring4",
                "right_pinky1", "right_pinky2", "right_pinky3", "right_pinky4"
            ]
        }
    
    def extract_hand_positions(
        self, 
        smplx_data: Dict,
        hand: str = "left"
    ) -> Optional[np.ndarray]:
        """
        Extract hand joint positions from SMPL-X data.
        
        Args:
            smplx_data: Dictionary containing SMPL-X body data with joint positions
            hand: Which hand to extract - "left" or "right"
        
        Returns:
            Hand joint positions array of shape (n_joints, 3) or None if extraction fails
        """
        if hand not in self.smplx_hand_joints:
            return None
        
        hand_joint_names = self.smplx_hand_joints[hand]
        hand_positions = []
        
        for joint_name in hand_joint_names:
            if joint_name in smplx_data:
                pos, _ = smplx_data[joint_name]
                hand_positions.append(np.asarray(pos))
        
        if len(hand_positions) == 0:
            return None
        
        return np.array(hand_positions)
    
    def _prepare_retargeting_input(
        self,
        hand_positions: np.ndarray,
        retargeter: SeqRetargeting
    ) -> np.ndarray:
        """
        Prepare hand positions for retargeting based on retargeting type.
        
        Args:
            hand_positions: Raw hand joint positions (n_joints, 3)
            retargeter: The retargeting object
        
        Returns:
            Processed reference values for retargeting
        """
        retargeting_type = retargeter.optimizer.retargeting_type
        indices = retargeter.optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            # Use absolute positions
            ref_value = hand_positions[indices, :]
        else:
            # Use relative vectors (for VECTOR or DEXPILOT types)
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = (
                hand_positions[task_indices, :] - hand_positions[origin_indices, :]
            )
        
        return ref_value
    
    def retarget(self, smplx_data: Dict) -> Dict[str, np.ndarray]:
        """
        Retarget a single frame of hand motion.
        
        Args:
            smplx_data: SMPL-X data for one frame
        
        Returns:
            Dictionary mapping hand name to joint positions
        """
        results = {}
        
        for hand in self.retargeters:
            hand_positions = self.extract_hand_positions(smplx_data, hand)
                
            if hand_positions is None:
                if self.verbose:
                    print(f"[HandRetarget] Warning: Could not extract {hand} hand positions")
                continue

            if hand_positions.shape[0] < 21:
                hand_positions = np.pad(hand_positions, ((0, 21 - hand_positions.shape[0]), (0, 0)))

            # Prepare input based on retargeting type
            retargeter = self.retargeters[hand]
            ref_value = self._prepare_retargeting_input(hand_positions, retargeter)
            
            # Retarget using dex-retargeting
            target_qpos = retargeter.retarget(ref_value)
            results[hand] = target_qpos
        
        return results
