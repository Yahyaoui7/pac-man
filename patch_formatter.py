import sys
with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/data/formatter.py", "r") as f:
    content = f.read()

target = """        # Channel 4: Signed Ghost positions (Positive = dangerous, Negative = edible)
        for idx in range(min(GHOST_COUNT, len(ghost_states))):
            gst = ghost_states[idx]
            if gst.get("in_prison", False):
                continue
            gx, gy = gst["grid_x"], gst["grid_y"]
            gy = max(0, min(CNN_HEIGHT - 1, gy))
            gx = max(0, min(CNN_WIDTH - 1, gx))
            is_edible = gst.get("is_edible", False)
            ObservationFormatter._paint_signed_ghost_patch(
                grid[0, 4], gy, gx, height, width, is_edible=is_edible
            )

        # Channel 5: Walkable Path & Active Map Mask (1.0 = Walkable cell inside active maze, 0.0 = Wall or Padded region)
        grid[0, 5, :height, :width] = (maze_tensor != 15).float()"""

replacement = """        # Channels 4-7: Signed Ghost positions (one channel per ghost)
        for idx in range(min(GHOST_COUNT, len(ghost_states))):
            gst = ghost_states[idx]
            if gst.get("in_prison", False):
                continue
            gx, gy = gst["grid_x"], gst["grid_y"]
            gy = max(0, min(CNN_HEIGHT - 1, gy))
            gx = max(0, min(CNN_WIDTH - 1, gx))
            is_edible = gst.get("is_edible", False)
            ObservationFormatter._paint_signed_ghost_patch(
                grid[0, 4 + idx], gy, gx, height, width, is_edible=is_edible
            )

        # Channel 8: Walkable Path & Active Map Mask (1.0 = Walkable cell inside active maze, 0.0 = Wall or Padded region)
        grid[0, 8, :height, :width] = (maze_tensor != 15).float()"""

if target in content:
    content = content.replace(target, replacement)
    with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/data/formatter.py", "w") as f:
        f.write(content)
    print("Success")
else:
    print("Failed to find target")
