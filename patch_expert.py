import sys
with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/ghosts/ghost_expert.py", "r") as f:
    content = f.read()

target = """    def choose_actions(self, env: Any) -> GhostExpertDecision:
        \"\"\"Return optimal labels for all ghosts given the current env state.\"\"\"
        if env.movement is None or env.player is None or env.maze is None:
            raise RuntimeError("Environment must be reset before expert use.")

        movement = env.movement
        width = len(env.maze[0])
        player_cell = (env.player.grid_y, env.player.grid_x)

        # BFS distances from the player — used for both hunting and fleeing
        player_dists = movement.bfs_distances(player_cell)

        labels: list[int] = []
        all_scores: list[tuple[float, ...]] = []

        for ghost in env.ghosts:
            label, scores = self._label_for_ghost(
                ghost, movement, player_cell, player_dists, width
            )
            labels.append(label)
            all_scores.append(scores)

        return GhostExpertDecision(
            labels=tuple(labels),
            scores=tuple(all_scores),
        )"""

replacement = """    def choose_actions(self, env: Any) -> GhostExpertDecision:
        \"\"\"Return optimal labels for all ghosts given the current env state.\"\"\"
        if env.movement is None or env.player is None or env.maze is None:
            raise RuntimeError("Environment must be reset before expert use.")

        movement = env.movement
        width = len(env.maze[0])
        player_cell = (env.player.grid_y, env.player.grid_x)
        player_dir = getattr(env.player, "direction", 0)

        labels: list[int] = []
        all_scores: list[tuple[float, ...]] = []

        # Target offsets:
        # Ghost 0: 0 steps (direct)
        # Ghost 1: 1 step ahead
        # Ghost 2: 2 steps ahead
        # Ghost 3: 4 steps ahead
        offsets = [0, 1, 2, 4]

        for i, ghost in enumerate(env.ghosts):
            steps = offsets[i] if i < len(offsets) else 0
            
            target_cell = player_cell
            if steps > 0:
                try:
                    d_name = DIRECTIONS[player_dir]
                except (IndexError, TypeError):
                    d_name = DIRECTIONS[0]
                
                dy, dx = DELTAS[d_name]
                ty, tx = target_cell
                for _ in range(steps):
                    if movement.can_move(ty, tx, d_name):
                        ty += dy
                        tx += dx
                    else:
                        break
                target_cell = (ty, tx)

            target_dists = movement.bfs_distances(target_cell)

            label, scores = self._label_for_ghost(
                ghost, movement, target_cell, target_dists, width
            )
            labels.append(label)
            all_scores.append(scores)

        return GhostExpertDecision(
            labels=tuple(labels),
            scores=tuple(all_scores),
        )"""

if target in content:
    content = content.replace(target, replacement)
    with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/ghosts/ghost_expert.py", "w") as f:
        f.write(content)
    print("Success")
else:
    print("Failed to find target")
