import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState # Used for both command and state messages
from geometry_msgs.msg import Pose, PointStamped # Used for EE position and Goal position
import torch
import numpy as np
import time
from typing import Optional

# --- CONFIGURATION ---
# ATTENTION: Replace the policy path and joint names!
POLICY_PATH = "/path/to/your/model_XXXXX.pt" 
CONTROL_FREQ = 60.0 # Hz (Training frequency)
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"] 
OBS_DIM = 24 # The size of the full observation vector from training (Matches: 6+6+3+3+6 = 24)

class ReachPolicyRunnerNode(Node):

    def __init__(self):
        super().__init__('reach_policy_runner_node')
        self.get_logger().info(f"Reach Policy Runner node started. Loading policy from: {POLICY_PATH}")

        # 1. Load Policy 
        try:
            checkpoint = torch.load(POLICY_PATH, map_location='cpu')
            self.policy = checkpoint['model'] 
            self.policy.eval() # Set to evaluation mode
            self.get_logger().info("Policy loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}")
            raise

        # Storage for the last measured joint state
        self.last_joint_state = JointState()
        self.last_joint_state.names = JOINT_NAMES 
        self.last_joint_state.position = [0.0] * len(JOINT_NAMES)
        self.last_joint_state.velocity = [0.0] * len(JOINT_NAMES)
        
        # New members for the missing 6 observations (EE Position and Target Position)
        self.ee_pos = np.zeros(3)      # Current End-Effector Position (World Frame)
        self.target_pos = np.zeros(3)  # Target Position (World Frame)
        
        # The Goal Error as defined in your original script (6 terms)
        # Goal error = [ pos_err_x, pos_err_y, pos_err_z, ori_err_x, ori_err_y, ori_err_z ]
        self.goal_error = np.zeros(6) 
        
        # 2. ROS2 Communication
        # Subscriber 1: Robot joint state (position and velocity)
        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/joint_states', # Robot's current JointState topic
            self._joint_state_callback, 
            10
        )
        
        # Subscriber 2: End-Effector Position (Assuming this comes from a forward kinematics node)
        self.ee_pos_sub = self.create_subscription(
            Pose, 
            '/ee_position', 
            self._ee_pos_callback, 
            10
        )
        
        # Subscriber 3: Target Position (The goal from the user/planner)
        # THIS IS WHERE THE GOAL POSITION IS SPECIFIED externally via the 'position' field of a Pose message.
        self.target_pos_sub = self.create_subscription(
            Pose, # Using Pose as requested (x, y, z are in msg.position)
            '/target_position', 
            self._target_pos_callback, 
            10
        )

        # Publisher: Send joint commands
        self.joint_command_pub = self.create_publisher(
            JointState, 
            '/joint_command', 
            10
        )
        
        # 3. RL Loop (Timer)
        self.timer = self.create_timer(1.0 / CONTROL_FREQ, self._rl_loop)
        self.get_logger().info(f"RL control loop started at {CONTROL_FREQ} Hz.")


    # --- CALLBACK FUNCTION (Joint State) ---

    def _joint_state_callback(self, msg: JointState):
        """Updates the last known joint state, copying position and velocity."""
        self.last_joint_state.names = msg.names 
        self.last_joint_state.position = msg.position
        self.last_joint_state.velocity = msg.velocity
        # NOTE: If the joint order in msg.names differs from JOINT_NAMES, you MUST reorder them here!

    # --- CALLBACK FOR END-EFFECTOR POSITION ---
    def _ee_pos_callback(self, msg: Pose):
        """Updates the current End-Effector position based on the incoming Pose message."""
        self.ee_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        # Re-calculate goal error when EE position updates
        self._calculate_goal_error()

    # --- CALLBACK FOR TARGET POSITION (THE GOAL) ---
    def _target_pos_callback(self, msg: Pose):
        """Updates the target position based on the incoming Pose message."""
        # The goal position is extracted from the 'position' field of the geometry_msgs/Pose
        self.target_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        # Re-calculate goal error when the target updates
        self._calculate_goal_error()

    # --- GOAL ERROR CALCULATION ---
    
    def _calculate_goal_error(self):
        """Calculates the goal error vector [pos_err, ori_err] based on current EE and Target pos/ori."""
        
        # 1. Position Error (3 terms)
        # Error is defined as (Current EE Position - Target Position)
        pos_error_vector = self.ee_pos - self.target_pos
        self.goal_error[0:3] = pos_error_vector 
        
        # 2. Orientation Error (3 terms)
        # Since we are using simple position reaching, orientation error is typically ignored/set to zero.
        self.goal_error[3:6] = np.zeros(3) # Placeholder for orientation error terms 

    # --- MAIN RL LOGIC ---
    
    def _get_observation(self) -> torch.Tensor:
        """
        Creates the observation vector matching the POLICY requirements (24 features).
        The order MUST match the Isaac Lab training environment's observation stack:
        [ joint_pos (6), joint_vel (6), ee_pos (3), target_pos (3), goal_error (6) ]
        """
        joint_pos = np.array(self.last_joint_state.position)
        joint_vel = np.array(self.last_joint_state.velocity)
        
        # Construct the observation vector (6 + 6 + 3 + 3 + 6 = 24)
        obs_np = np.concatenate([
            joint_pos,
            joint_vel,
            self.ee_pos,      # 3 features
            self.target_pos,  # 3 features
            self.goal_error   # 6 features
        ])

        if len(obs_np) != OBS_DIM:
             self.get_logger().error(f"Observation dimension mismatch: {len(obs_np)} != {OBS_DIM}. Check ROS2 subscribers.")

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
        action_scale = 0.2 # Confirmed by your environment config
        current_pos = np.array(self.last_joint_state.position)

        delta_pos = actions_raw * action_scale
        target_pos = current_pos + delta_pos
        
        # 4. Send ROS2 message (as JointState!)
        command_msg = JointState()
        
        command_msg.names = self.last_joint_state.names 
        command_msg.position = target_pos.tolist() 
        
        command_msg.header.stamp = self.get_clock().now().to_msg()
        
        self.joint_command_pub.publish(command_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReachPolicyRunnerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
