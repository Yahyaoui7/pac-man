import re

with open("AI_arena/ghosts/ghost_collector.py", "r") as f:
    content = f.read()

target = """        while written < samples:
            # ── Reset episode ──
            env.reset()
            episode_step = 0
            done = False

            while not done and written < samples and episode_step < MAX_STEPS_PER_EPISODE:
                assert env.maze is not None
                assert env.pellets is not None
                assert env.player is not None
                assert env.movement is not None

                # ── Build observation (shared formatter — same as inference) ──
                ghost_state_dicts = _ghost_states(env)
                grid, extra_features, _valid_player, valid_ghost_actions = (
                    ObservationFormatter.format_observation(
                        maze=env.maze,
                        pellets=env.pellets,
                        player_pos=(env.player.grid_x, env.player.grid_y),
                        player_direction=env.player.direction,
                        ghost_states=ghost_state_dicts,
                        movement=env.movement,
                        device="cpu",
                    )
                )

                # ── Ghost expert labels ──
                decision = ghost_expert.choose_actions(env)
                labels = list(decision.labels)

                # ── Optionally skip fully-imprisoned steps ──
                all_in_prison = all(
                    getattr(g, "in_prison", False) for g in env.ghosts
                )
                if all_in_prison and not keep_prison_steps:
                    # Advance game with player expert action and continue
                    try:
                        player_decision = player_expert.choose_action(env)
                        player_action = player_decision.action
                    except RuntimeError:
                        player_action = 0
                    _, _, done, _, _ = env.step(player_action)
                    episode_step += 1
                    continue

                # ── Deduplicate ──
                key = _record_key(grid, extra_features, labels)
                if key not in seen:
                    seen.add(key)

                    record = {
                        "grid": grid[0].tolist(),
                        "extra_features": extra_features[0].tolist(),
                        # valid_ghost_actions shape: (GHOST_COUNT, ACTION_COUNT)
                        "valid_actions": valid_ghost_actions.tolist(),
                        "labels": labels,
                        "episode_id": episode_id,
                        "episode_step": episode_step,
                        "maze_width": len(env.maze[0]),
                        "maze_height": len(env.maze),
                    }
                    stream.write(json.dumps(record, separators=(",", ":")) + "\\n")
                    written += 1

                    if written % 500 == 0 or written == samples:
                        pct = 100 * written / samples
                        print(
                            f"  [{pct:5.1f}%] {written}/{samples} samples"
                            f"  (episode {episode_id}, step {episode_step})"
                        )

                # ── Advance game using the player expert ──
                try:
                    player_decision = player_expert.choose_action(env)
                    player_action = player_decision.action
                except RuntimeError:
                    player_action = 0

                _, _, done, _, _ = env.step(player_action)
                episode_step += 1

            episode_id += 1

    print(
        f"\\nDone — wrote {written} records from {episode_id} episodes"
        f" to {destination}"
    )"""

replacement = """        while written < samples:
            # ── Periodically reset to get a new maze layout ──
            if written % 50 == 0:
                env.reset()
                
            assert env.maze is not None
            assert env.pellets is not None
            assert env.player is not None
            assert env.movement is not None
            
            maze = env.maze
            height = len(maze)
            width = len(maze[0])
            
            # Find all walkable cells
            walkable_cells = [
                (y, x) for y in range(height) for x in range(width)
                if maze[y][x] != 15
            ]
            if not walkable_cells:
                env.reset()
                continue
                
            # ── Randomize Player ──
            py, px = rng.choice(walkable_cells)
            env.player.grid_y = py
            env.player.grid_x = px
            env.player.direction = rng.choice([0, 1, 2, 3])
            
            # 20% chance of being powered (frightened mode)
            is_powered = rng.random() < 0.2
            env.player.powered_mode = True if is_powered else None
            env.player.powered_timer = 10.0 if is_powered else 0.0

            # ── Randomize Ghosts ──
            for g in env.ghosts:
                gy, gx = rng.choice(walkable_cells)
                g.grid_y = gy
                g.grid_x = gx
                g.direction = rng.choice([0, 1, 2, 3])
                g.in_prison = False
                if is_powered:
                    g.is_edible = rng.choice([True, False])
                else:
                    g.is_edible = False

            # ── Build observation ──
            ghost_state_dicts = _ghost_states(env)
            grid, extra_features, _valid_player, valid_ghost_actions = (
                ObservationFormatter.format_observation(
                    maze=env.maze,
                    pellets=env.pellets,
                    player_pos=(env.player.grid_x, env.player.grid_y),
                    player_direction=env.player.direction,
                    ghost_states=ghost_state_dicts,
                    movement=env.movement,
                    device="cpu",
                )
            )

            # ── Ghost expert labels ──
            decision = ghost_expert.choose_actions(env)
            labels = list(decision.labels)

            # ── Deduplicate ──
            key = _record_key(grid, extra_features, labels)
            if key not in seen:
                seen.add(key)

                record = {
                    "grid": grid[0].tolist(),
                    "extra_features": extra_features[0].tolist(),
                    "valid_actions": valid_ghost_actions.tolist(),
                    "labels": labels,
                    "episode_id": written // 50,
                    "episode_step": written % 50,
                    "maze_width": width,
                    "maze_height": height,
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\\n")
                written += 1

                if written % 500 == 0 or written == samples:
                    pct = 100 * written / samples
                    print(
                        f"  [{pct:5.1f}%] {written}/{samples} samples"
                    )

    print(
        f"\\nDone — wrote {written} randomized records to {destination}"
    )"""

if target in content:
    content = content.replace(target, replacement)
    with open("AI_arena/ghosts/ghost_collector.py", "w") as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("Target not found. Doing fuzzy search...")
    # Just a simple find to see where it deviates
    print(repr(content[-2000:]))
