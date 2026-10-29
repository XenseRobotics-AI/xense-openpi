"""Elite-only XMV adapter: 20D action layout, assembly and cached SDK feedback.

This module is imported only when visualization is enabled. It opens no SDK
connections and never calls robot.get_observation(), IK or a motion command.
The already connected lerobot Elite driver owns every RTSI connection.
"""

import json
import time
import xml.etree.ElementTree as ET

from lerobot.robots.bi_elite_cs66_rt.gripper_snapshots import TaccapGripperSnapshotReader
import numpy as np
from xense_client.visualization import StateSample
from xense_motion_viewer import RobotDescription
from xense_motion_viewer import RobotState
from xense_motion_viewer import ToolFrame
from xense_motion_viewer import poses_from_xyz_rotation6d
from xense_motion_viewer.urdf import xyz_rpy

SIDES = ("left", "right")


class EliteVisualizationAdapter:
    def __init__(self, robot, *, max_state_age=0.25):
        self.robot = robot
        self.max_state_age = max_state_age
        self._stamps = {}
        self._changed_at = {}
        self._discontinuity = False
        self._gripper_readers = {}
        self._assembly = getattr(robot, "_assembly", None)
        if self._assembly is None:
            from libpyelite.assembly import Assembly

            if robot.config.native_assembly is None:
                raise ValueError(
                    "Elite visualization requires native_assembly to align policy, SDK and URDF world frames"
                )
            self._assembly = Assembly(robot.config.native_assembly)
        assembly = self._assembly
        self._world_from_base = {side: assembly.world_from_base(side) for side in SIDES}
        self._joint_names = {}
        for side in SIDES:
            root = ET.parse(assembly.control_model_path(side)).getroot()
            self._joint_names[side] = tuple(
                f"{side}_{joint.attrib['name']}"
                for joint in root.findall("joint")
                if joint.attrib["type"] != "fixed" and joint.find("mimic") is None
            )
            if len(self._joint_names[side]) != 6:
                raise ValueError("Expected six Elite controller joints per arm")
        profile = json.loads((assembly.root / "simulation_profile.json").read_text())
        self._calibration = profile["native_collision_gripper_calibration"]
        overrides = getattr(robot.config, "native_collision_gripper_calibration", None)
        if overrides is not None:
            self._calibration = overrides
        arm_names = tuple(name for side in SIDES for name in self._joint_names[side])
        gripper_names = tuple(name for side in SIDES for name in self._calibration[side])
        independent = {
            joint.attrib["name"]
            for joint in ET.parse(assembly.urdf).getroot().findall("joint")
            if joint.attrib["type"] != "fixed" and joint.find("mimic") is None
        }
        defaults = {
            key: value
            for key, value in assembly.viewer_joint_defaults().items()
            if key in independent and key not in gripper_names
        }
        tools = {}
        for side in SIDES:
            tool = getattr(robot.config, f"{side}_tool_transform", None)
            if tool is not None:
                tools[side] = ToolFrame(f"{side}_tool0", np.asarray(tool).reshape(4, 4))
        self._description = RobotDescription(
            "elite_station", assembly.urdf, arm_names + gripper_names, joint_defaults=defaults, tool_frames=tools
        )

    def describe(self):
        return self._description

    def bind_state_readers(self):
        """Bind once after connection; subsequent samples only read SDK caches.

        Use the Elite snapshot adapter even when station collision checking is
        disabled. TacCapFollower itself has no get_gripper_snapshot method.
        """
        existing = getattr(self.robot, "_station_gripper_readers", {})
        self._gripper_readers = {
            side: existing[side] if side in existing else TaccapGripperSnapshotReader(self.robot._gripper[side])
            for side in SIDES
        }

    def decode_actions(self, actions):
        values = np.asarray(actions)
        if values.ndim != 2 or values.shape[1] != 20:
            raise ValueError(f"Expected Elite [steps, 20] actions, got {values.shape}")
        return {
            "left": poses_from_xyz_rotation6d(values[:, :9]),
            "right": poses_from_xyz_rotation6d(values[:, 10:19]),
        }

    def _arm_sample(self, side):
        rtsi = self.robot._rtsi[side]
        if rtsi is None or not rtsi.isConnected():
            raise RuntimeError(f"{side} RTSI is disconnected")
        for _ in range(5):
            stamp = float(rtsi.getTimestamp())
            joints = np.asarray(rtsi.getActualJointPositions(), dtype=float)
            tcp = np.asarray(rtsi.getActualTCPPose(), dtype=float)
            if stamp != rtsi.getTimestamp():
                continue
            if (
                not np.isfinite(stamp)
                or stamp <= 0
                or joints.shape != (6,)
                or tcp.shape != (6,)
                or not np.isfinite(joints).all()
                or not np.isfinite(tcp).all()
            ):
                raise ValueError(f"Invalid {side} SDK state")
            now = time.monotonic()
            if stamp < self._stamps.get(side, stamp):
                self._discontinuity = True
            if self._stamps.get(side) != stamp:
                self._changed_at[side] = now
            self._stamps[side] = stamp
            if now - self._changed_at[side] > self.max_state_age:
                raise RuntimeError(f"{side} SDK state is stale")
            # Elite getActualTCPPose is XYZ + roll/pitch/yaw, in base frame.
            world_from_tcp = self._world_from_base[side] @ xyz_rpy(tcp[:3], tcp[3:])
            return stamp, joints, world_from_tcp
        raise RuntimeError(f"Cannot read a coherent {side} SDK sample")

    def read_state(self, received_at):
        self._discontinuity = False
        positions, tcp_poses, ids = {}, {}, []
        for side in SIDES:
            stamp, joints, pose = self._arm_sample(side)
            positions.update(zip(self._joint_names[side], joints, strict=True))
            tcp_poses[side] = pose
            ids.append(f"{side}:{stamp:.9f}")
            gripper = self.robot._gripper[side]
            if gripper is None:
                raise RuntimeError(f"{side} gripper state is unavailable")
            if side not in self._gripper_readers:
                raise RuntimeError("Bind Elite gripper state readers after connecting the robot")
            sample = self._gripper_readers[side].read()
            if (
                not sample.valid
                or sample.sequence <= 0
                or not np.isfinite(sample.position)
                or not 0 <= time.monotonic() - sample.sampled_at <= self.max_state_age
            ):
                raise RuntimeError(f"{side} gripper state is invalid or stale")
            ids.append(f"{side}-gripper:{sample.sequence}")
            for joint, knots in self._calibration[side].items():
                knots = np.asarray(knots, dtype=float)
                positions[joint] = float(np.interp(sample.position, knots[:, 0], knots[:, 1]))
        state = RobotState(received_at, positions, tcp_poses=tcp_poses)
        return StateSample(state, "|".join(ids), "SDK world TCP; joints in rad; measured grippers", self._discontinuity)
