import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
import numpy as np
from loop_rate_limiters import RateLimiter
from rich import print
from dex_retargeting.constants import RobotName, ROBOT_NAME_MAP
from urdf2mjcf.convert import convert_urdf_to_mjcf

class HandMotionViewer:
    """
    HandMotionViewer
    ----------------
    Mujoco viewer for Dex hand URDF.

    Works similarly to RobotMotionViewer but:
      - no floating base
      - only joint DOFs
      - loads URDF instead of XML
    """

    def __init__(
        self,
        robot_type,
        robot_dir,
        hand_type,
        motion_fps=30,
        transparent_hand=0,
        record_video=False,
        video_path=None,
        video_width=640,
        video_height=480,
        keyboard_callback=None,
    ):
        robot_name = ROBOT_NAME_MAP[RobotName[robot_type]]
        self.urdf_path = robot_dir / robot_name / f"{robot_name}_{hand_type}.urdf"

        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.record_video = record_video

        print(f"[green]Loading URDF: {self.urdf_path}")

        # ------------------------------------------------------------
        #      LOAD URDF -> MJCF -> MODEL
        # ------------------------------------------------------------
        
        # Convert URDF to MJCF XML string
        convert_urdf_to_mjcf(self.urdf_path)
        xml_path = self.urdf_path.with_suffix(".xml")

        collision_folder = xml_path.parent / "meshes/collision"

        assets = {}
        for f in collision_folder.glob("*.obj"):
            print(f.name)
            assets[f.name] = f.read_bytes()

        self.model = mj.MjModel.from_xml_path(str(xml_path), assets=assets)
        
        self.data = mj.MjData(self.model)

        # Make one step to initialize positions
        mj.mj_forward(self.model, self.data)

        # ------------------------------------------------------------
        #      VIEWER CREATION
        # ------------------------------------------------------------
        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=keyboard_callback,
        )

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_hand

        # Center camera on hand
        self.viewer.cam.distance = 0.25
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 140
        self.viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])

        # ------------------------------------------------------------
        #      VIDEO RECORDING
        # ------------------------------------------------------------
        if self.record_video:
            assert video_path is not None
            self.video_path = video_path
            video_dir = os.path.dirname(video_path)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            print(f"[yellow]Recording video to {video_path}")

            self.mp4_writer = imageio.get_writer(
                video_path, fps=self.motion_fps
            )

            self.renderer = mj.Renderer(
                self.model, width=video_width, height=video_height
            )

    # ------------------------------------------------------------------
    #      STEP WITH JOINT ANGLES
    # ------------------------------------------------------------------
    def step(self, dof_pos, rate_limit=True):
        """
        dof_pos: array-like of joint angles (rad) for all DOFs in model.qpos
                 except Mujoco has base pos (3) + base quat (4) if floating
                 but your Dex URDF has NO floating base → qpos matches DOFs directly.
        """

        self.data.qpos[:] = dof_pos
        mj.mj_forward(self.model, self.data)

        self.viewer.sync()
        if rate_limit:
            self.rate_limiter.sleep()

        if self.record_video:
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)

    # ------------------------------------------------------------------
    def close(self):
        self.viewer.close()
        time.sleep(0.3)

        if self.record_video:
            self.mp4_writer.close()
            print(f"[green]Video saved to {self.video_path}")
