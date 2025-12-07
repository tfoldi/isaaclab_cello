To execute the `reach_policy_runner_node` node, first you must train and export the cello model. See the main folder for instructions.

When executing the node, use the `policy_path` argument to point to the policy.

Example: `ros2 run reach_policy reach_policy_runner_node --ros-args -p policy_path:=../logs/rsl_rl/reach_cello/2025-12-04_11-26-31/exported/policy.pt`
