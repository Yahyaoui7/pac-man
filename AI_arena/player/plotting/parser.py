"""Training log text parser for extracting metric numpy arrays."""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np

UPD_RE = re.compile(r"Upd\s+(\d+)/\d+")
TOT_EP_RE = re.compile(r"Tot Ep:\s*(\d+)")
EPOCH_RWD_RE = re.compile(r"Averge?\s+Epoch Rwd:\s*([+-]?\d+\.?\d*)")
AVG_RWD_RE = re.compile(r"(?<!Epoch )Avg Rwd:\s*([+-]?\d+\.?\d*)")
AVG_PELLETS_RE = re.compile(r"Avg Pellets:\s*(\d+\.?\d*)\s*\(\s*(\d+\.?\d*)%\)")
MAX_PELLETS_RE = re.compile(r"Max (?:Epoch )?Pellets:\s*(\d+)\s*\(\s*(\d+\.?\d*)%\)")
LOSS_RE = re.compile(r"Loss \(P/V\):\s*([+-]?\d+\.\d+)/(\d+\.\d+)")
TIME_RE = re.compile(r"Time:\s*([\d.]+)s\s*\(\s*([\d.]+)s/upd\)")
OSC_PCT_RE = re.compile(r"Osc%:\s*(\d+\.?\d*)%")
MAZE_AREA_RE = re.compile(
    r"Avg Maze Area:\s*(\d+\.?\d*)\s*\(\s*(\d+\.?\d*)x(\d+\.?\d*)\)"
)

# Reward-breakdown regexes
BD_STEP_RE = re.compile(r"\|\s*Step:\s*([+-]?\d+\.?\d*)")
BD_OSC_RE = re.compile(r"\|\s*Osc:\s*([+-]?\d+\.?\d*)")
BD_PELLET_RE = re.compile(r"\|\s*Pellet:\s*([+-]?\d+\.?\d*)")
BD_SUPER_RE = re.compile(r"\|\s*Super:\s*([+-]?\d+\.?\d*)")
BD_GHOST_RE = re.compile(r"\|\s*Ghost:\s*([+-]?\d+\.?\d*)")
BD_COMPLETE_RE = re.compile(r"\|\s*Complete:\s*([+-]?\d+\.?\d*)")
BD_DEATH_RE = re.compile(r"\|\s*Death:\s*([+-]?\d+\.?\d*)")
BD_BFS_RE = re.compile(r"\|\s*BFS:\s*([+-]?\d+\.?\d*)")


def parse_log(path: Path) -> dict[str, np.ndarray]:
    """Parse log file lines into metric arrays."""
    updates, tot_eps = [], []
    epoch_rwds, avg_rwds = [], []
    avg_pellets, avg_pcts = [], []
    max_pellets, max_pcts = [], []
    policy_losses, value_losses = [], []
    sec_per_upd = []
    maze_areas, maze_widths, maze_heights = [], [], []
    osc_pcts = []

    bd_step, bd_osc, bd_pellet, bd_super = [], [], [], []
    bd_ghost, bd_complete, bd_death, bd_bfs = [], [], [], []

    seen: set[int] = set()
    n_epoch_rwd_found = 0
    n_maze_found = 0
    n_bd_found = 0
    n_osc_found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            m_upd = UPD_RE.search(line)
            m_tot = TOT_EP_RE.search(line)
            m_avg_rwd = AVG_RWD_RE.search(line)
            m_avg_pel = AVG_PELLETS_RE.search(line)
            m_max_pel = MAX_PELLETS_RE.search(line)
            m_loss = LOSS_RE.search(line)
            if not (
                m_upd and m_tot and m_avg_rwd and m_avg_pel and m_max_pel and m_loss
            ):
                continue
            upd = int(m_upd.group(1))

            updates.append(upd)
            tot_eps.append(int(m_tot.group(1)))
            avg_rwds.append(float(m_avg_rwd.group(1)))
            avg_pellets.append(float(m_avg_pel.group(1)))
            avg_pcts.append(float(m_avg_pel.group(2)))
            max_pellets.append(int(m_max_pel.group(1)))
            max_pcts.append(float(m_max_pel.group(2)))
            policy_losses.append(float(m_loss.group(1)))
            value_losses.append(float(m_loss.group(2)))

            m_ep = EPOCH_RWD_RE.search(line)
            if m_ep:
                epoch_rwds.append(float(m_ep.group(1)))
                n_epoch_rwd_found += 1
            else:
                epoch_rwds.append(0.0)

            m_time = TIME_RE.search(line)
            sec_per_upd.append(float(m_time.group(2)) if m_time else 0.0)

            m_maze = MAZE_AREA_RE.search(line)
            if m_maze:
                maze_areas.append(float(m_maze.group(1)))
                maze_widths.append(float(m_maze.group(2)))
                maze_heights.append(float(m_maze.group(3)))
                n_maze_found += 1
            else:
                maze_areas.append(0.0)
                maze_widths.append(0.0)
                maze_heights.append(0.0)

            m_osc = OSC_PCT_RE.search(line)
            if m_osc:
                osc_pcts.append(float(m_osc.group(1)))
                n_osc_found += 1
            else:
                osc_pcts.append(0.0)

            m_step = BD_STEP_RE.search(line)
            m_b_osc = BD_OSC_RE.search(line)
            m_pellet = BD_PELLET_RE.search(line)
            m_super = BD_SUPER_RE.search(line)
            m_ghost = BD_GHOST_RE.search(line)
            m_complete = BD_COMPLETE_RE.search(line)
            m_death = BD_DEATH_RE.search(line)
            m_bfs = BD_BFS_RE.search(line)

            if (
                m_step
                and m_b_osc
                and m_pellet
                and m_super
                and m_ghost
                and m_complete
                and m_death
                and m_bfs
            ):
                bd_step.append(float(m_step.group(1)))
                bd_osc.append(float(m_b_osc.group(1)))
                bd_pellet.append(float(m_pellet.group(1)))
                bd_super.append(float(m_super.group(1)))
                bd_ghost.append(float(m_ghost.group(1)))
                bd_complete.append(float(m_complete.group(1)))
                bd_death.append(float(m_death.group(1)))
                bd_bfs.append(float(m_bfs.group(1)))
                n_bd_found += 1
            else:
                bd_step.append(0.0)
                bd_osc.append(0.0)
                bd_pellet.append(0.0)
                bd_super.append(0.0)
                bd_ghost.append(0.0)
                bd_complete.append(0.0)
                bd_death.append(0.0)
                bd_bfs.append(0.0)

    if not updates:
        raise ValueError(f"No update lines found in '{path}'.")

    data = {
        "updates": np.array(updates, dtype=int),
        "tot_eps": np.array(tot_eps, dtype=int),
        "epoch_rwds": np.array(epoch_rwds, dtype=float),
        "avg_rwds": np.array(avg_rwds, dtype=float),
        "avg_pellets": np.array(avg_pellets, dtype=float),
        "avg_pcts": np.array(avg_pcts, dtype=float),
        "max_pellets": np.array(max_pellets, dtype=int),
        "max_pcts": np.array(max_pcts, dtype=float),
        "policy_losses": np.array(policy_losses, dtype=float),
        "value_losses": np.array(value_losses, dtype=float),
        "sec_per_upd": np.array(sec_per_upd, dtype=float),
        "maze_areas": np.array(maze_areas, dtype=float),
        "maze_widths": np.array(maze_widths, dtype=float),
        "maze_heights": np.array(maze_heights, dtype=float),
        "osc_pcts": np.array(osc_pcts, dtype=float),
        "bd_step": np.array(bd_step, dtype=float),
        "bd_osc": np.array(bd_osc, dtype=float),
        "bd_pellet": np.array(bd_pellet, dtype=float),
        "bd_super": np.array(bd_super, dtype=float),
        "bd_ghost": np.array(bd_ghost, dtype=float),
        "bd_complete": np.array(bd_complete, dtype=float),
        "bd_death": np.array(bd_death, dtype=float),
        "bd_bfs": np.array(bd_bfs, dtype=float),
        "has_epoch_rwds": (n_epoch_rwd_found == len(updates)),
        "has_maze": (n_maze_found == len(updates)),
        "has_breakdown": (n_bd_found == len(updates)),
        "has_osc": (n_osc_found == len(updates)),
    }
    return data
