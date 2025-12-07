# Copyright (c) 2025, Tamas Foldi and Istvan Fodor
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from . import mdp

##
# Pre-defined configs
##

from isaaclab_cello.robots.cello import CELLO_CONFIG  # isort:skip


@configclass
class CelloReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to Cello
        self.scene.robot = CELLO_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.scene.robot.init_state.pos = (0.17, 0.0, 0.2)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link6"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link6"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link6"]

        # >> 1. POZÍCIÓ JUTALOM NÖVELÉSE
        self.rewards.end_effector_position_tracking.weight = -1.0
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.5

        # >> 2. POSE JUTALOM KIKAPCSOLÁSA
        self.rewards.end_effector_orientation_tracking = None

        # >> 3. REGULÁRIS BÜNTETÉSEK NÖVELÉSE (A simaság érdekében)
        self.rewards.action_rate.weight = -0.001
        self.rewards.joint_vel.weight = -0.0005

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[r"joint\d+$"],
            scale=0.2,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "link6"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        self.commands.ee_pose.ranges.yaw = (math.pi, math.pi)
        self.commands.ee_pose.ranges.pos_x = (-0.65, -0.15)
        # self.commands.ee_pose.ranges.pos_x=(-0.65, -0.35)
        # self.commands.ee_pose.ranges.pos_y=(-0.0, 0.0)
        self.commands.ee_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.ee_pose.ranges.pos_z = (0.15, 0.5)
        self.commands.ee_pose.debug_vis = True


@configclass
class CelloReachEnvCfg_PLAY(CelloReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
