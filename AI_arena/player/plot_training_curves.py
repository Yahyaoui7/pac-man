"""
Generate a numbered RL training report with plots and a README.

Usage:
    python -m AI_arena.player.plot_training_curves [logfile] [--title "My run"]

Each call creates:
    reports/report_NNN/
        00_overview.png
        01_avg_reward.png
        02_pellet_completion.png
        03_value_loss.png
        README.md
"""

import argparse
import re
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    raise SystemExit("ERROR: matplotlib not installed. Run: uv pip install matplotlib")

# ─── Log parsing ────────────────────────────────────────────────────────────

LOG_PATTERN = re.compile(
    r"Upd\s+(\d+)/\d+.*?"
    r"Tot Ep:\s*(\d+).*?"
    r"Avg Rwd:\s*([+-]?\d+\.?\d*).*?"
    r"Avg Pellets:\s*(\d+\.?\d*)\s*\(\s*(\d+\.?\d*)%\).*?"
    r"Max Pellets:\s*(\d+)\s*\(\s*(\d+\.?\d*)%\).*?"
    r"Loss \(P/V\):\s*([+-]?\d+\.\d+)/(\d+\.\d+)"
)


def parse_log(path: Path) -> dict[str, np.ndarray]:
    updates, tot_eps = [], []
    avg_rwds, avg_pellets, avg_pcts = [], [], []
    max_pellets, max_pcts = [], []
    policy_losses, value_losses = [], []

    seen: set[int] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            upd = int(m.group(1))
            if upd in seen:
                continue
            seen.add(upd)
            updates.append(upd)
            tot_eps.append(int(m.group(2)))
            avg_rwds.append(float(m.group(3)))
            avg_pellets.append(float(m.group(4)))
            avg_pcts.append(float(m.group(5)))
            max_pellets.append(int(m.group(6)))
            max_pcts.append(float(m.group(7)))
            policy_losses.append(float(m.group(8)))
            value_losses.append(float(m.group(9)))

    return {
        "updates": np.array(updates),
        "tot_eps": np.array(tot_eps),
        "avg_rwds": np.array(avg_rwds),
        "avg_pellets": np.array(avg_pellets),
        "avg_pcts": np.array(avg_pcts),
        "max_pellets": np.array(max_pellets),
        "max_pcts": np.array(max_pcts),
        "policy_losses": np.array(policy_losses),
        "value_losses": np.array(value_losses),
    }


# ─── Helpers ────────────────────────────────────────────────────────────────


def smooth(arr: np.ndarray, window: int = 30) -> np.ndarray:
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    pad = window // 2
    return np.convolve(np.pad(arr, pad, mode="edge"), kernel, mode="valid")[: len(arr)]


def _style(ax: plt.Axes, fig: plt.Figure) -> None:
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#444")


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")
    return p


# ─── Plots ──────────────────────────────────────────────────────────────────


def plot_all(data: dict[str, np.ndarray], out: Path) -> dict[str, float]:
    upd = data["updates"]
    RAW_A, RAW_C, SM_C, MAX_C = 0.18, "#5b9bd5", "#1a4a7a", "#e05c00"

    # 01 — Avg Reward
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(upd, data["avg_rwds"], alpha=RAW_A, color=RAW_C)
    ax.plot(upd, data["avg_rwds"], alpha=0.3, color=RAW_C, linewidth=0.8)
    ax.plot(upd, smooth(data["avg_rwds"]), color=SM_C, linewidth=2, label="Smoothed")
    ax.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set(
        title="Average Episode Reward (20-ep sliding window)",
        xlabel="PPO Update",
        ylabel="Avg Reward",
    )
    ax.legend(facecolor="#2a2a4e", labelcolor="white")
    _style(ax, fig)
    _save(fig, out, "01_avg_reward.png")

    # 02 — Pellet Completion
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(upd, data["avg_pcts"], alpha=RAW_A, color="#44bb77")
    ax.plot(upd, data["avg_pcts"], alpha=0.3, color="#44bb77", linewidth=0.8)
    ax.plot(
        upd,
        smooth(data["avg_pcts"]),
        color="#1a7a44",
        linewidth=2,
        label="Avg pellet %",
    )
    ax.plot(
        upd, data["max_pcts"], color=MAX_C, linewidth=1, alpha=0.6, label="Max pellet %"
    )
    ax.set(
        title="Pellet Completion % (20-ep sliding window)",
        xlabel="PPO Update",
        ylabel="Completion %",
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.legend(facecolor="#2a2a4e", labelcolor="white")
    _style(ax, fig)
    _save(fig, out, "02_pellet_completion.png")

    # 03 — Value Loss
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(upd, data["value_losses"], alpha=RAW_A, color="#e07b55")
    ax.plot(upd, data["value_losses"], alpha=0.3, color="#e07b55", linewidth=0.8)
    ax.plot(
        upd,
        smooth(data["value_losses"]),
        color="#aa2200",
        linewidth=2,
        label="Smoothed",
    )
    ax.set(title="Value Network Loss", xlabel="PPO Update", ylabel="Value Loss")
    ax.legend(facecolor="#2a2a4e", labelcolor="white")
    _style(ax, fig)
    _save(fig, out, "03_value_loss.png")

    # 00 — Combined overview
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        "Pac-Man RL Training Summary", color="white", fontsize=14, fontweight="bold"
    )
    for ax in axes:
        ax.set_facecolor("#242444")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
    axes[0].fill_between(upd, data["avg_rwds"], alpha=0.2, color=RAW_C)
    axes[0].plot(upd, smooth(data["avg_rwds"]), color=SM_C, linewidth=2)
    axes[0].axhline(0, color="white", linestyle="--", linewidth=0.7, alpha=0.4)
    axes[0].set_ylabel("Avg Reward", color="white")
    axes[1].fill_between(upd, data["avg_pcts"], alpha=0.2, color="#44bb77")
    axes[1].plot(
        upd, smooth(data["avg_pcts"]), color="#1a7a44", linewidth=2, label="Avg %"
    )
    axes[1].plot(
        upd, data["max_pcts"], color=MAX_C, linewidth=1, alpha=0.6, label="Max %"
    )
    axes[1].set_ylabel("Pellet %", color="white")
    axes[1].legend(facecolor="#2a2a4e", labelcolor="white", fontsize=8)
    axes[2].fill_between(upd, data["value_losses"], alpha=0.2, color="#e07b55")
    axes[2].plot(upd, smooth(data["value_losses"]), color="#aa2200", linewidth=2)
    axes[2].set_ylabel("Value Loss", color="white")
    axes[2].set_xlabel("PPO Update", color="white")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out, "00_overview.png")

    # Derived stats
    sm_rwd = smooth(data["avg_rwds"])
    best_sm_idx = int(np.argmax(sm_rwd))
    return {
        "total_updates": int(len(upd)),
        "total_episodes": int(data["tot_eps"][-1]),
        "best_avg_rwd": float(data["avg_rwds"].max()),
        "best_avg_rwd_upd": int(upd[int(np.argmax(data["avg_rwds"]))]),
        "best_sm_rwd": float(sm_rwd[best_sm_idx]),
        "best_sm_rwd_upd": int(upd[best_sm_idx]),
        "best_avg_pct": float(data["avg_pcts"].max()),
        "best_avg_pct_upd": int(upd[int(np.argmax(data["avg_pcts"]))]),
        "best_max_pct": float(data["max_pcts"].max()),
        "best_max_pct_upd": int(upd[int(np.argmax(data["max_pcts"]))]),
        "final_avg_rwd": float(data["avg_rwds"][-1]),
        "final_avg_pct": float(data["avg_pcts"][-1]),
        "final_max_pct": float(data["max_pcts"][-1]),
        "start_upd": int(upd[0]),
        "end_upd": int(upd[-1]),
    }


# ─── README generator ───────────────────────────────────────────────────────

README_TEMPLATE = """\
# Training Report {num:03d} — {title}

Generated: {date}  
Log file: `{log_file}`

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| PPO Updates | {start_upd} → {end_upd} ({total_updates} logged) |
| Total Episodes | {total_episodes} |
| Rollout Steps / Update | 512 |
| PPO Epochs | 4 |
| Mini-batch Size | 64 |
| Learning Rate | 3e-4 |
| Gamma (γ) | 0.99 |
| GAE Lambda (λ) | 0.95 |
| Clip ε | 0.2 |
| Entropy Coef | 0.02 |
| Value Coef | 0.5 |
| Max Grad Norm | 0.5 |

---

## Observation Format

Each step the model receives **3 tensors**:

### 1. Grid (`[1, C, H, W]` CNN input)
| Channel | Content |
|---------|---------|
| 0 | Walls (maze structure) |
| 1 | Pellets (normal, value 1) |
| 2 | Super-pellets (value 2) |
| 3 | Player position |
| 4 | Ghosts (all combined) |
| 5 | BFS distance-to-player heatmap (normalized) |

### 2. Extra Features (`[1, F]` MLP input)
| Feature | Description |
|---------|-------------|
| player_grid_x | Normalized column position |
| player_grid_y | Normalized row position |
| remaining_pellets | Fraction of pellets left |
| ghost_count | Number of active ghosts |
| ghost_min_dist | Closest ghost BFS distance (normalized) |
| powered_mode | 1 if player is in powered/attack mode |

### 3. Valid Actions (`[1, 4]` mask)
Binary mask over `[UP, DOWN, LEFT, RIGHT]` — invalid moves are masked to −∞ before softmax.

---

## Reward System

| Event | Reward |
|-------|--------|
| Every step (base penalty) | **−0.2** |
| Oscillating move (reversed within 6 steps) | **−0.3** *(active from report_002)* |
| First visit to a new grid tile | **+0.5** |
| Pellet eaten | **+5.0** |
| Super-pellet eaten | **+15.0** |
| Ghost eaten (in powered mode) | **+30.0** |
| Level completed | **+100.0** |
| Pac-Man died | **−20.0** |

**Net examples:**
- Step forward into a new pellet tile: `−0.2 + 0.5 + 5.0 = +5.3`
- Step forward into a new empty tile: `−0.2 + 0.5 = +0.3`
- Step forward into an already-visited tile: `−0.2`
- Oscillating move (back-track): `−0.2 − 0.3 = −0.5` *(active from report_002)*

---

## Results Summary

| Metric | Value | At Update |
|--------|-------|-----------|
| Best raw avg reward | {best_avg_rwd:.1f} | {best_avg_rwd_upd} |
| Best smoothed avg reward | {best_sm_rwd:.1f} | {best_sm_rwd_upd} |
| Best avg pellet % | {best_avg_pct:.1f}% | {best_avg_pct_upd} |
| Best max pellet % | {best_max_pct:.1f}% | {best_max_pct_upd} |
| Final avg reward | {final_avg_rwd:.1f} | {end_upd} |
| Final avg pellet % | {final_avg_pct:.1f}% | {end_upd} |
| Final max pellet % | {final_max_pct:.1f}% | {end_upd} |

---

## Plots

| File | Description |
|------|-------------|
| `00_overview.png` | Combined 3-panel summary (reward / pellets / value loss) |
| `01_avg_reward.png` | Average episode reward over updates |
| `02_pellet_completion.png` | Pellet completion % (avg & max) |
| `03_value_loss.png` | Value network loss |

![Overview](00_overview.png)

---

## Notes / Observations

<!-- Add manual notes about this run here -->
- Training data covers updates {start_upd}–{end_upd} ({total_episodes} episodes).
- Log window size: last 20 completed episodes per update.
- Oscillation penalty introduced in this run to combat node-to-node back-and-forth behavior.
"""


def write_readme(out: Path, num: int, title: str, log_file: str, stats: dict) -> None:
    content = README_TEMPLATE.format(
        num=num,
        title=title,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        log_file=log_file,
        **stats,
    )
    p = out / "README.md"
    p.write_text(content, encoding="utf-8")
    print(f"  Saved: {p}")


# ─── Report numbering ────────────────────────────────────────────────────────


def next_report_dir(reports_root: Path) -> tuple[int, Path]:
    """Return (report_number, path) for the next auto-numbered report."""
    reports_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [
            d
            for d in reports_root.iterdir()
            if d.is_dir() and d.name.startswith("report_")
        ],
        key=lambda d: d.name,
    )
    num = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    out = reports_root / f"report_{num:03d}"
    out.mkdir(exist_ok=True)
    return num, out


# ─── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a numbered RL training report."
    )
    parser.add_argument(
        "logfile",
        nargs="?",
        default="RL_logs.txt",
        help="Path to the training log file (default: RL-logs.txt)",
    )
    parser.add_argument(
        "--title", default="PPO Stage-1", help="Short title for this report"
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Root reports directory (default: reports/)",
    )
    args = parser.parse_args()

    log_path = Path(args.logfile)
    reports_root = Path(args.reports_dir)

    print(f"Parsing: {log_path}")
    data = parse_log(log_path)
    print(f"Found {len(data['updates'])} unique update entries.")

    num, out_dir = next_report_dir(reports_root)
    print(f"Creating report #{num:03d} → {out_dir}/")

    stats = plot_all(data, out_dir)
    write_readme(out_dir, num, args.title, str(log_path), stats)

    print(f"\n✓ Report {num:03d} ready at: {out_dir}/")
    print(f"  Episodes : {stats['total_episodes']}")
    print(f"  Updates  : {stats['start_upd']}→{stats['end_upd']}")
    print(
        f"  Best avg reward : {stats['best_avg_rwd']:.1f}  (upd {stats['best_avg_rwd_upd']})"
    )
    print(
        f"  Best pellet %   : {stats['best_max_pct']:.1f}% (upd {stats['best_max_pct_upd']})"
    )


if __name__ == "__main__":
    main()
