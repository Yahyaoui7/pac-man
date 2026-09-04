import os
import sys
import torch
import numpy as np

# Ensure old_pacman is in sys.path
sys.path.insert(0, "/home/mouad/Desktop/WeCode/1337/old_pacman")

from AI_arena.player.player_env import PacmanPlayerEnv
from AI_arena.models.cnn_player import PlayerActorCritic

FEATURE_NAMES = [
    "00: player_dir_UP",
    "01: player_dir_DOWN",
    "02: player_dir_LEFT",
    "03: player_dir_RIGHT",
    "04: ghost1_edible",
    "05: ghost2_edible",
    "06: ghost3_edible",
    "07: ghost4_edible",
    "08: ghost1_timer",
    "09: ghost2_timer",
    "10: ghost3_timer",
    "11: ghost4_timer",
    "12: valid_action_UP",
    "13: valid_action_DOWN",
    "14: valid_action_LEFT",
    "15: valid_action_RIGHT",
    "16: normal_pellets_remaining_frac",
    "17: power_pellets_remaining_frac",
    "18: player_powered_timer_norm",
    "19: ghost1_bfs_dist_norm",
    "20: ghost2_bfs_dist_norm",
    "21: ghost3_bfs_dist_norm",
    "22: ghost4_bfs_dist_norm",
    "23: nearest_pp_dist_norm",
    "24: nearest_np_dist_norm",
    "25: directional_pellet_lookahead_UP",
    "26: directional_pellet_lookahead_DOWN",
    "27: directional_pellet_lookahead_LEFT",
    "28: directional_pellet_lookahead_RIGHT",
    "29: local_danger_UP",
    "30: local_danger_DOWN",
    "31: local_danger_LEFT",
    "32: local_danger_RIGHT",
    "33: delta_nearest_pellet_dist",
    "34: delta_nearest_ghost_dist",
    "35: delta_nearest_pp_dist",
    "36: steps_since_pellet_norm",
    "37: same_action_count_norm",
    "38: ghost1_rel_dx",
    "39: ghost1_rel_dy",
    "40: ghost2_rel_dx",
    "41: ghost2_rel_dy",
    "42: ghost3_rel_dx",
    "43: ghost3_rel_dy",
    "44: ghost4_rel_dx",
    "45: ghost4_rel_dy",
    "46: surrounded_danger_dist3_norm",
    "47: surrounded_danger_dist5_norm",
    "48: dead_end_flag",
    "49: junction_flag",
]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def render_grid_exact_floats(
    matrix: np.ndarray, col_width: int = 6, decimals: int = 2
) -> str:
    """Render a 2D numpy tensor matrix with exact float numbers across all rows and columns."""
    lines = []
    h, w = matrix.shape

    # Header row with column indices
    col_hdr = "     " + "".join(f"{col:^{col_width}d}" for col in range(w))
    lines.append(col_hdr)
    lines.append("     " + "-" * (w * col_width))

    for r in range(h):
        row_str = f"{r:2d} | "
        for c in range(w):
            val = float(matrix[r, c])
            fmt_val = f"{val:^{col_width}.{decimals}f}"
            row_str += fmt_val
        lines.append(row_str)
    return "\n".join(lines)


def run_dump():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PacmanPlayerEnv(stage=1, start_pellets=(5,), seed=42)
    obs = env.reset()

    model = PlayerActorCritic().to(device)
    model_path = (
        "/home/mouad/Desktop/WeCode/1337/old_pacman/AI_arena/models/player_rl.pt"
    )
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        state_dict = (
            ckpt["model_state_dict"]
            if (isinstance(ckpt, dict) and "model_state_dict" in ckpt)
            else ckpt
        )
        conv_w = state_dict.get("backbone.cnn.0.weight")
        if conv_w is not None and conv_w.shape[1] == 5:
            new_w = torch.zeros(64, 6, 3, 3, device=conv_w.device, dtype=conv_w.dtype)
            new_w[:, :5] = conv_w
            state_dict["backbone.cnn.0.weight"] = new_w
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    gru_state = None
    output_lines = []

    output_lines.append(
        "========================================================================================================================"
    )
    output_lines.append(
        "                PAC-MAN RL MODEL: 5-STEP FULL TENSOR GRID DUMP (WITH ZERO-PADDING [25x50])"
    )
    output_lines.append(
        "========================================================================================================================\n"
    )

    maze_h = len(env.maze)
    maze_w = len(env.maze[0])
    output_lines.append(f"Active Maze Size : {maze_w}x{maze_h} (Width x Height)")
    output_lines.append(
        f"CNN Tensor Shape : [1, 5, 25, 50] (5 Channels, 25 Rows x 50 Cols WITH PADDING)"
    )
    output_lines.append(f"Active Maze Region: Rows 0..{maze_h-1}, Cols 0..{maze_w-1}")
    output_lines.append(f"Zero-Padded Region: Rows {maze_h}..24, Cols {maze_w}..49\n")

    for step_idx in range(1, 6):
        grid, extra, valid_mask = obs
        grid_cuda = grid.to(device)
        extra_cuda = extra.to(device)
        valid_mask_cuda = valid_mask.to(device)

        with torch.no_grad():
            logits, value, gru_state = model(grid_cuda, extra_cuda, hidden=gru_state)
            masked_logits = logits.masked_fill(~valid_mask_cuda, -1e8)
            probs = torch.softmax(masked_logits, dim=-1).cpu().numpy()[0]
            val_est = value.item()
            action = torch.argmax(masked_logits, dim=-1).item()

        next_obs, reward, done, info, action_taken = env.step(action)

        output_lines.append(
            "########################################################################################################################"
        )
        output_lines.append(
            f"                                                      STEP {step_idx}"
        )
        output_lines.append(
            "########################################################################################################################"
        )
        output_lines.append(
            f"Player Pos         : ({env.player.grid_x}, {env.player.grid_y}) | Direction: {env.player.direction}"
        )
        output_lines.append(
            f"Ghosts Status      : "
            + ", ".join(
                [
                    f"G{i}@({g.grid_x},{g.grid_y},edible={getattr(g,'is_edible',False)},prison={getattr(g,'in_prison',False)})"
                    for i, g in enumerate(env.ghosts)
                ]
            )
        )
        output_lines.append(
            f"Valid Actions Mask : {valid_mask[0].tolist()} -> Valid: {[ACTION_NAMES[i] for i, v in enumerate(valid_mask[0].tolist()) if v]}"
        )
        output_lines.append(f"Model Action Chosen: {action} ({ACTION_NAMES[action]})")
        output_lines.append(
            f"Action Probabilities: UP={probs[0]:.4f}, DOWN={probs[1]:.4f}, LEFT={probs[2]:.4f}, RIGHT={probs[3]:.4f}"
        )
        output_lines.append(f"Value Estimate     : {val_est:.4f}")
        output_lines.append(f"Step Reward        : {reward:.4f}")
        output_lines.append(f"Reward Breakdown   : {info.get('breakdown', {})}\n")

        grid_np = grid[0].numpy()  # full shape (5, 25, 50)

        output_lines.append(
            "------------------------------------------------------------------------------------------------------------------------"
        )
        output_lines.append(
            "                      FULL SPATIAL CNN GRID TENSORS INCLUDING ZERO PADDING (SHAPE: 25x50)"
        )
        output_lines.append(
            "------------------------------------------------------------------------------------------------------------------------"
        )

        # Channel 0: Maze Topology
        output_lines.append(
            f"\n[CHANNEL 0: Raw Maze Bitmask Topology / 15.0] (Full Tensor 25x50 with Padding)"
        )
        output_lines.append(
            render_grid_exact_floats(grid_np[0], col_width=6, decimals=2)
        )

        # Channel 1: Normal Pellets
        output_lines.append(
            f"\n[CHANNEL 1: Normal Pellets Grid] (Full Tensor 25x50 with Padding: 1.00 = Pellet, 0.00 = Empty/Padded)"
        )
        output_lines.append(
            render_grid_exact_floats(grid_np[1], col_width=6, decimals=2)
        )

        # Channel 2: Power Pellets
        output_lines.append(
            f"\n[CHANNEL 2: Power Pellets Grid] (Full Tensor 25x50 with Padding: 1.00 = Power Pellet, 0.00 = Empty/Padded)"
        )
        output_lines.append(
            render_grid_exact_floats(grid_np[2], col_width=6, decimals=2)
        )

        # Channel 3: Player Position Heatmap Patch (3x3)
        output_lines.append(
            f"\n[CHANNEL 3: Player Position Heatmap Patch] (Full Tensor 25x50 with Padding: Center=1.00, Orthogonal=0.50, Diagonal=0.25)"
        )
        output_lines.append(
            render_grid_exact_floats(grid_np[3], col_width=6, decimals=2)
        )

        # Channel 4: Ghost Positions Signed Heatmap Patch (3x3)
        output_lines.append(
            f"\n[CHANNEL 4: Ghost Positions Signed Patch] (Full Tensor 25x50 with Padding: +1.00 = Dangerous, -1.00 = Edible)"
        )
        output_lines.append(
            render_grid_exact_floats(grid_np[4], col_width=6, decimals=2)
        )

        # Channel 5: Walkable Path & Active Map Mask Grid
        if len(grid_np) > 5:
            output_lines.append(
                f"\n[CHANNEL 5: Walkable Path & Active Map Mask Grid] (Full Tensor 25x50 with Padding: 1.00 = Walkable Inside Maze, 0.00 = Wall / Padded Border)"
            )
            output_lines.append(
                render_grid_exact_floats(grid_np[5], col_width=6, decimals=2)
            )

        output_lines.append(
            "\n------------------------------------------------------------------------------------------------------------------------"
        )
        output_lines.append(
            "                                            50 EXTRA VECTOR FEATURES ([1, 50])"
        )
        output_lines.append(
            "------------------------------------------------------------------------------------------------------------------------"
        )
        extra_list = extra[0].tolist()
        for fname, val in zip(FEATURE_NAMES, extra_list):
            output_lines.append(f"  {fname:<38} : {val:10.4f}")
        output_lines.append("\n")

        obs = next_obs
        if done:
            output_lines.append("Episode Terminated.")
            break

    out_file1 = "/home/mouad/Desktop/WeCode/1337/old_pacman/5step_full_grids_dump.txt"
    out_file2 = "/home/mouad/Desktop/WeCode/1337/old_pacman/steps_dump.txt"
    content = "\n".join(output_lines)
    with open(out_file1, "w") as f:
        f.write(content)
    with open(out_file2, "w") as f:
        f.write(content)
    print(
        f"Successfully wrote FULL 25x50 padded tensor grid dump to {out_file1} and {out_file2}"
    )


if __name__ == "__main__":
    run_dump()
