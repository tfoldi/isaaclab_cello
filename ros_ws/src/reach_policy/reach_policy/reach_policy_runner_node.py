# Copyright (c) 2025, Tamas Foldi and Istvan Fodor
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import (  # Used for EE position and Goal position (Note: PointStamped is replaced by PoseStamped)
    Pose,
    PoseStamped,
)
from rclpy.node import Node
from robo_interfaces.msg import (
    SetAngle,  # Used for sending joint commands to actual robot
)
from sensor_msgs.msg import JointState  # Used for both command and state messages

# --- CONFIGURATION ---
# ATTENTION: Replace the policy path and joint names!
CONTROL_FREQ = 60.0  # Hz (Training frequency)


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7_left", "joint7_right"]
# OBS_DIM is confirmed to be 29
OBS_DIM = 29
ACTION_DIM = 6  # Action size: (delta_joint_position for 6 joints)


class ReachPolicyRunnerNode(Node):
    def __init__(self):
        super().__init__("reach_policy_runner_node")

        self.policy_path = self.declare_parameter("policy_path", "").value
        self.get_logger().info(f"Reach Policy Runner node started. Loading policy from: {self.policy_path}")

        if self.policy_path == "":
            self.get_logger().error("Parameter 'policy_path' must be provided to load a TorchScript policy.")
            self.destroy_node()
            rclpy.shutdown()
            raise SystemExit(1)

        # 1. Load Policy
        try:
            self.policy = torch.jit.load(self.policy_path, map_location="cpu")

        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}")
            raise  # Re-raise the exception to terminate the node if loading fails

        # Storage for the last raw measured joint state (used internally for filtering)
        self.raw_joint_state = JointState()
        self.raw_joint_state.name = []
        self.raw_joint_state.position = []
        self.raw_joint_state.velocity = []

        # Filtered joint state (6 dimensions) used for observation and commanding
        self.filtered_joint_pos = np.zeros(len(JOINT_NAMES))
        self.filtered_joint_vel = np.zeros(len(JOINT_NAMES))

        # State variables for the 29-feature observation
        # 6+6 + 3+3 + 3 + 4+4 = 29
        self.ee_pos = np.zeros(3)  # Current End-Effector Position (3)
        self.target_pos = [-0.2, -0.2, 0.3]  # Target Position (3)
        self.pos_error = np.zeros(3)  # Positional Error (3)
        self.ee_ori = np.array([0.0, 0.0, 0.0, 1.0])  # Current EE Orientation (4-Quat)
        self.target_ori = np.array([0.0, 0.0, 0.0, 1.0])  # Target Orientation (4-Quat)
        self.previous_action = np.zeros(6)

        self.default_joint_positions = np.array([0.014, -0.0034, -0.005, 0.0209, -0.0053, -0.0035, -0.015, -0.015])

        # 2. ROS2 Communication
        # Subscriber 1: Robot joint state (position and velocity)
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",  # Robot's current JointState topic
            self._joint_state_callback,
            10,
        )

        # Subscriber 2: End-Effector Position (Pose includes Pos and Ori)
        self.ee_pos_sub = self.create_subscription(
            Pose,
            "/ee_pose",  # Assuming a more general pose topic name
            self._ee_pose_callback,
            10,
        )

        # Subscriber 3: Target Position (Pose includes Pos and Ori)
        # CHANGED: Now subscribes to PoseStamped for Rviz2 visualization support
        self.target_pos_sub = self.create_subscription(PoseStamped, "/target_pose", self._target_pose_callback, 10)

        # Publisher: Send joint commands
        self.joint_command_pub = self.create_publisher(JointState, "/joint_command", 10)

        self.set_angle_pub = self.create_publisher(SetAngle, "/set_angle_topic", 10)

        # 3. RL Loop (Timer)
        self.timer = self.create_timer(1.0 / CONTROL_FREQ, self._rl_loop)
        self.get_logger().info(f"RL control loop started at {CONTROL_FREQ} Hz.")

    # --- HELPER FUNCTION (Joint State Filtering) ---
    def _filter_joint_states(self, msg: JointState):
        """
        Filters the incoming JointState message to extract only the position and velocity
        for the 6 joints defined in JOINT_NAMES, in the correct order. This ignores extra joints.
        """
        # Create a map from joint name to its index in the incoming message
        joint_map = {name: i for i, name in enumerate(msg.name)}

        # Storage for the filtered data
        filtered_pos = []
        filtered_vel = []

        for name in JOINT_NAMES:
            if name in joint_map:
                index = joint_map[name]
                if index >= len(msg.position) or index >= len(msg.velocity):
                    self.get_logger().debug(
                        f"Index {index} for joint '{name}' is out of bounds in the JointState message."
                    )
                # Extract the data at the corresponding index
                filtered_pos.append(msg.position[index] if index < len(msg.position) else 0.0)
                filtered_vel.append(msg.velocity[index] if index < len(msg.velocity) else 0.0)
            else:
                # If a required joint is missing, log a warning and use a zero value
                self.get_logger().debug(f"Required joint '{name}' not found in /joint_states message. Using 0.0.")
                filtered_pos.append(0.0)
                filtered_vel.append(0.0)

        # Update the state variables used for observation and commanding
        self.filtered_joint_pos = np.array(filtered_pos)
        self.filtered_joint_vel = np.array(filtered_vel)

    # --- CALLBACK FUNCTION (Joint State) ---

    def _joint_state_callback(self, msg: JointState):
        """
        Updates the raw joint state and calls the filter to extract the relevant 6 joints.
        """
        # Store the raw message (optional, but good for debugging)
        self.raw_joint_state.name = msg.name
        self.raw_joint_state.position = msg.position
        self.raw_joint_state.velocity = msg.velocity

        # Filter the incoming data to get the 6 required joints
        self._filter_joint_states(msg)

    # --- CALLBACK FOR END-EFFECTOR POSE (Position and Orientation) ---
    def _ee_pose_callback(self, msg: Pose):
        """Updates the current End-Effector position and orientation."""
        self.ee_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.ee_ori = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self._calculate_errors()

    # --- CALLBACK FOR TARGET POSE (Position and Orientation) ---
    def _target_pose_callback(self, msg: PoseStamped):
        """
        Updates the target position and orientation from a PoseStamped message.
        The policy only uses the pose data, ignoring the header/frame.
        """
        pose = msg.pose
        self.target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        self.target_ori = np.array([
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ])
        self._calculate_errors()

    # --- ERROR CALCULATION ---

    def _calculate_errors(self):
        """Calculates the positional error (EE Pos - Target Pos)."""
        # Positional Error (3 terms)
        self.pos_error = self.ee_pos - self.target_pos

    # --- MAIN RL LOGIC ---

    def _get_observation(self) -> torch.Tensor:
        """
        Creates the observation vector matching the POLICY requirements (29 features).
        Expected order: [ joint_pos (6), joint_vel (6), ee_pos (3), target_pos (3), pos_error (3), ee_ori (4), target_ori (4) ]
        Total: 6 + 6 + 3 + 3 + 3 + 4 + 4 = 29
        """
        # Use the pre-filtered 6-dimensional arrays
        joint_pos = self.filtered_joint_pos - self.default_joint_positions  # (8)
        joint_vel = self.filtered_joint_vel  # (8)

        # Construct the 29-feature observation vector

        """
        [INFO] Observation Manager: <ObservationManager> contains 1 groups.
        +-------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (29,)) |
        +-------------+---------------------------+-------------+
        |    Index    | Name                      |    Shape    |
        +-------------+---------------------------+-------------+
        |      0      | joint_pos                 |     (8,)    |
        |      1      | joint_vel                 |     (8,)    |
        |      2      | pose_command              |     (7,)    |
        |      3      | actions                   |     (6,)    |
        +-------------+---------------------------+-------------+

        """

        obs_np = np.concatenate([
            joint_pos,  # 6 -- 8
            joint_vel,  # 6 -- 8
            # self.ee_pos,  # 3
            self.target_pos,  # 3
            # self.pos_error,  # 3
            # self.ee_ori,  # 4 (Quaternion)
            self.target_ori,  # 4 (Quaternion)
            self.previous_action,
        ])

        # component_sizes = ", ".join(
        #     f"{name}:{len(component)}"
        #     for name, component in (
        #         ("joint_pos", joint_pos),
        #         ("joint_vel", joint_vel),
        #         ("target_pos", self.target_pos),
        #         ("target_ori", self.target_ori),
        #         ("previous_action", self.previous_action),
        #     )
        # )
        # self.get_logger().info(f"Observation component sizes -> {component_sizes}")

        if len(obs_np) != OBS_DIM:
            # This check is now an explicit failure condition if filtering failed
            self.get_logger().error(
                f"FATAL: Observation dimension mismatch: {len(obs_np)} != {OBS_DIM}. Check filtering logic or policy"
                " configuration."
            )
            # Must return a tensor to avoid crash, but the data is bad
            return torch.zeros(1, OBS_DIM, dtype=torch.float32)

        # Add a batch dimension (unsqueeze(0)) and convert to float32 tensor
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

        return obs_tensor

    def _rl_loop(self):
        """The main loop for running the policy and sending the command."""

        # 1. Get Observation
        obs = self._get_observation()

        if torch.isnan(obs).any().item():
            self.get_logger().debug("Some observations were nan, skipping")
            return

        dist = np.linalg.norm(self.ee_pos - self.target_pos)
        self.get_logger().debug(f"Distance: {dist}")
        if dist > 0.05:
            # 2. Calculate Action using the policy
            with torch.no_grad():
                actions_raw = self.policy.forward(obs)[0].cpu()
                if torch.isnan(actions_raw).any().item():
                    self.get_logger().debug("Prediction has nan, skipping")
                    return

                actions_raw = actions_raw.numpy().flatten()
                self.previous_action = actions_raw

            # 3. Process Action (Scaling)
            action_scale = 0.05  # Confirmed by your environment config
            # TODO: apply some smoothing based on distance
            # action_scale = np.clip(action_scale * dist, 0.03, 0.2)
            target_pos = self.default_joint_positions.copy()
            target_pos[:6] += actions_raw * action_scale

            # self.get_logger().info(f"Current POS: {current_pos}")
            # self.get_logger().info(f"Arm Target POS: {target_pos}")
            # self.get_logger().info(f"Target POS: {np.concatenate([self.target_pos, self.target_ori])}")
            # 4. Send ROS2 message (as JointState!)
            command_msg = JointState()

            command_msg.name = JOINT_NAMES  # Use the explicit 6 names we care about
            command_msg.position = target_pos.tolist()

            command_msg.header.stamp = self.get_clock().now().to_msg()

            self.joint_command_pub.publish(command_msg)

            self.set_angle_pub.publish(
                SetAngle(
                    servo_id=[0, 1, 2, 3, 4, 5],
                    target_angle=[
                        radians_to_degrees(target_pos[0]),
                        radians_to_degrees(target_pos[1]),
                        radians_to_degrees(target_pos[2]),
                        radians_to_degrees(target_pos[3]),
                        radians_to_degrees(target_pos[4]),
                        radians_to_degrees(target_pos[5]),
                    ],
                    time=[100, 100, 100, 100, 100, 100],
                )
            )


def radians_to_degrees(radians):
    return radians * (180 / math.pi)


def main(args=None):
    rclpy.init(args=args)
    node = ReachPolicyRunnerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
