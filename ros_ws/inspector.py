import torch

policy_path = "/home/tfoldi/Developer/nvidia/isaaclab_cello/isaaclab_cello/logs/rsl_rl/reach_cello/2025-11-25_12-19-59/model_999.pt"

try:
    checkpoint = torch.load(policy_path, map_location="cpu")
    weights = checkpoint.get("model_state_dict", checkpoint)

    # 1. Input Layer (actor.0)
    input_weight = weights["actor.0.weight"]

    # OBS_DIM is the second dimension of the first weight matrix (input features)
    obs_dim = input_weight.shape[1]

    # First hidden layer size is the first dimension of the first weight matrix (output features)
    h1_size = input_weight.shape[0]

    # 2. Output Layer (actor.4 in your case)
    output_weight = weights["actor.4.weight"]
    action_dim = output_weight.shape[0]

    # Collect hidden sizes by finding consecutive linear layers
    hidden_sizes = [h1_size]
    i = 2
    while f"actor.{i}.weight" in weights:
        hidden_sizes.append(weights[f"actor.{i}.weight"].shape[0])
        i += 2

    # Remove the action layer size which was mistakenly added
    hidden_sizes.pop()

    print(f"Confirmed OBS_DIM: {obs_dim}")
    print(f"Confirmed ACTION_DIM: {action_dim}")
    print(f"Confirmed Hidden Sizes: {hidden_sizes}")

except Exception as e:
    print(f"Error inspecting checkpoint: {e}")
