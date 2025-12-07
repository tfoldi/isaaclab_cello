import time
from typing import Optional

import numpy as np
import rclpy
import torch
import torch.nn as nn
from geometry_msgs.msg import (  # Used for EE position and Goal position (Note: PointStamped is replaced by PoseStamped)
    Pose,
    PoseStamped,
)
from rclpy.node import Node
from sensor_msgs.msg import JointState  # Used for both command and state messages

# --- CONFIGURATION ---
# ATTENTION: Replace the policy path and joint names!
POLICY_PATH = "/home/tfoldi/Developer/nvidia/isaaclab_cello/isaaclab_cello/logs/rsl_rl/reach_cello/2025-11-25_12-19-59/model_999.pt"
CONTROL_FREQ = 60.0  # Hz (Training frequency)
# This list MUST exactly match the names of the 6 joints the policy was trained on
# The filtering logic will IGNORE joint7_left and joint8_right from the /joint_states topic.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# OBS_DIM is confirmed to be 29
OBS_DIM = 29
ACTION_DIM = 6  # Action size: (delta_joint_position for 6 joints)


# --- POLICY MODEL ARCHITECTURE (STRUCTURAL FIX APPLIED) ---
# The component is named 'actor' to match the key naming convention in the checkpoint file.
class PolicyModel(nn.Module):
    # hidden_sizes is confirmed to be [64, 64]
    def __init__(self, obs_dim, action_dim, hidden_sizes=[64, 64]):
        super().__init__()

        layers = []
        in_size = obs_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size

        # Final layer maps to action space
        layers.append(nn.Linear(in_size, action_dim))

        # This name must match the prefix in the checkpoint keys ('actor.0.weight')
        self.actor = nn.Sequential(*layers)

    def act(self, observations):
        """Forward pass to compute actions."""
        return self.actor(observations)


class ReachPolicyRunnerNode(Node):
    def __init__(self):
        super().__init__("reach_policy_runner_node")
        self.get_logger().info(f"Reach Policy Runner node started. Loading policy from: {POLICY_PATH}")

        # 1. Load Policy
        try:
            # First, instantiate the model architecture
            self.policy_model = PolicyModel(OBS_DIM, ACTION_DIM)

            # Load the full checkpoint dictionary
            checkpoint = torch.load(POLICY_PATH, map_location="cpu")

            policy_state_dict = None

            # Case 2: Common RSL-RL case: full Actor-Critic state dict is under 'model_state_dict'
            if "model_state_dict" in checkpoint:
                full_state_dict = checkpoint["model_state_dict"]
                self.get_logger().info(
                    "Found full Actor-Critic weights under 'model_state_dict'. Filtering Actor keys..."
                )

                # Filter the state dictionary to keep only keys starting with 'actor.'
                policy_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith("actor.")}

            # Fallback for other checkpoint structures
            elif "actor_state_dict" in checkpoint:
                policy_state_dict = checkpoint["actor_state_dict"]
                self.get_logger().info("Found policy weights under 'actor_state_dict'.")
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                full_state_dict = checkpoint["model"]
                self.get_logger().info("Found full Actor-Critic weights under 'model'. Filtering Actor keys...")
                policy_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith("actor.")}
            else:
                key_list = list(checkpoint.keys())
                raise KeyError(f"Policy checkpoint file structure not recognized. Found keys: {key_list}")

            if policy_state_dict:
                # Load the filtered/actor-only state dictionary into the model
                self.policy_model.load_state_dict(policy_state_dict)
                self.get_logger().info("Policy state dict successfully loaded after filtering/renaming.")

            self.policy = self.policy_model  # Set the running policy to the loaded model
            self.policy.eval()  # Set to evaluation mode

        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}")
            raise  # Re-raise the exception to terminate the node if loading fails

        # Storage for the last raw measured joint state (used internally for filtering)
        self.raw_joint_state = JointState()
        self.raw_joint_state.name = []
        self.raw_joint_state.position = []
        self.raw_joint_state.velocity = []

        # Filtered joint state (6 dimensions) used for observation and commanding
        self.filtered_joint_pos = np.zeros(ACTION_DIM)
        self.filtered_joint_vel = np.zeros(ACTION_DIM)

        # State variables for the 29-feature observation
        # 6+6 + 3+3 + 3 + 4+4 = 29
        self.ee_pos = np.zeros(3)  # Current End-Effector Position (3)
        self.target_pos = np.zeros(3)  # Target Position (3)
        self.pos_error = np.zeros(3)  # Positional Error (3)
        self.ee_ori = np.array([0.0, 0.0, 0.0, 1.0])  # Current EE Orientation (4-Quat)
        self.target_ori = np.array([0.0, 0.0, 0.0, 1.0])  # Target Orientation (4-Quat)

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
                # Extract the data at the corresponding index
                filtered_pos.append(msg.position[index])
                filtered_vel.append(msg.velocity[index])
            else:
                # If a required joint is missing, log a warning and use a zero value
                self.get_logger().warn(f"Required joint '{name}' not found in /joint_states message. Using 0.0.")
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
        self.ee_ori = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
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
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
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
        joint_pos = self.filtered_joint_pos  # (6)
        joint_vel = self.filtered_joint_vel  # (6)

        # Construct the 29-feature observation vector
        obs_np = np.concatenate([
            joint_pos,  # 6
            joint_vel,  # 6
            self.ee_pos,  # 3
            self.target_pos,  # 3
            self.pos_error,  # 3
            self.ee_ori,  # 4 (Quaternion)
            self.target_ori,  # 4 (Quaternion)
        ])

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

        # 2. Calculate Action using the policy
        with torch.no_grad():
            actions_raw = self.policy.act(obs)[0].cpu().numpy().flatten()

        # 3. Process Action (Scaling)
        action_scale = 0.2  # Confirmed by your environment config
        # Use the filtered joint positions (now guaranteed to be (6,))
        current_pos = self.filtered_joint_pos

        delta_pos = actions_raw * action_scale
        # This addition should now work: (6,) + (6,) = (6,)
        target_pos = current_pos + delta_pos

        # 4. Send ROS2 message (as JointState!)
        command_msg = JointState()

        command_msg.name = JOINT_NAMES  # Use the explicit 6 names we care about
        command_msg.position = target_pos.tolist()

        command_msg.header.stamp = self.get_clock().now().to_msg()

        self.joint_command_pub.publish(command_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReachPolicyRunnerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
