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
        04_reward_trend.png
        05_reward_vs_pellets.png
        06_epoch_vs_window_reward.png   (only if epoch reward is in the log)
        README.md

Log line formats supported (fields are matched independently, so order
and presence don't matter — old and new log formats both parse fine):

    Upd 1724/4000 | Tot Ep: 3105 | Averge Epoch Rwd: 31.0 | Max Epoch Pellets: 35 (89.7%) \
    | Avg Pellets: 39.5 (69.5%) | Avg Rwd: -20.5 | Loss (P/V): -0.0000/2.0153 \
    | Time: 3256.0s (2.42s/upd)
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
#
# Each field is matched independently with its own regex and searched for
# anywhere in the line. This makes parsing robust to fields being reordered
# or added/removed between log format revisions (which is exactly what
# happened here: "Max Pellets" -> "Max Epoch Pellets", and a new
# "Averge Epoch Rwd" field was inserted before Max/Avg Pellets).

UPD_RE = re.compile(r"Upd\s+(\d+)/\d+")
TOT_EP_RE = re.compile(r"Tot Ep:\s*(\d+)")
# "Averge" is the typo actually present in the logger — kept as-is, but also
# tolerate the correct spelling in case it gets fixed later.
EPOCH_RWD_RE = re.compile(r"Averge?\s+Epoch Rwd:\s*([+-]?\d+\.?\d*)")
AVG_RWD_RE = re.compile(r"(?<!Epoch )Avg Rwd:\s*([+-]?\d+\.?\d*)")
AVG_PELLETS_RE = re.compile(r"Avg Pellets:\s*(\d+\.?\d*)\s*\(\s*(\d+\.?\d*)%\)")
MAX_PELLETS_RE = re.compile(r"Max (?:Epoch )?Pellets:\s*(\d+)\s*\(\s*(\d+\.?\d*)%\)")
LOSS_RE = re.compile(r"Loss \(P/V\):\s*([+-]?\d+\.\d+)/(\d+\.\d+)")
TIME_RE = re.compile(r"Time:\s*([\d.]+)s\s*\(\s*([\d.]+)s/upd\)")


def parse_log(path: Path) -> dict[str, np.ndarray]:
    updates, tot_eps = [], []
    epoch_rwds, avg_rwds = [], []
    avg_pellets, avg_pcts = [], []
    max_pellets, max_pcts = [], []
    policy_losses, value_losses = [], []
    sec_per_upd = []

    seen: set[int] = set()
    n_epoch_rwd_found = 0
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
                continue  # not a training-update line

            upd = int(m_upd.group(1))
            if upd in seen:
                continue
            seen.add(upd)

            updates.append(upd)
            tot_eps.append(int(m_tot.group(1)))
            avg_rwds.append(float(m_avg_rwd.group(1)))
            avg_pellets.append(float(m_avg_pel.group(1)))
            avg_pcts.append(float(m_avg_pel.group(2)))
            max_pellets.append(int(m_max_pel.group(1)))
            max_pcts.append(float(m_max_pel.group(2)))
            policy_losses.append(float(m_loss.group(1)))
            value_losses.append(float(m_loss.group(2)))

            m_epoch = EPOCH_RWD_RE.search(line)
            if m_epoch:
                epoch_rwds.append(float(m_epoch.group(1)))
                n_epoch_rwd_found += 1
            else:
                epoch_rwds.append(np.nan)

            m_time = TIME_RE.search(line)
            sec_per_upd.append(float(m_time.group(2)) if m_time else np.nan)

    has_epoch_rwd = n_epoch_rwd_found > 0
    return {
        "updates": np.array(updates),
        "tot_eps": np.array(tot_eps),
        "epoch_rwds": np.array(epoch_rwds) if has_epoch_rwd else None,
        "avg_rwds": np.array(avg_rwds),
        "avg_pellets": np.array(avg_pellets),
        "avg_pcts": np.array(avg_pcts),
        "max_pellets": np.array(max_pellets),
        "max_pcts": np.array(max_pcts),
        "policy_losses": np.array(policy_losses),
        "value_losses": np.array(value_losses),
        "sec_per_upd": np.array(sec_per_upd),
    }


# ─── Helpers ────────────────────────────────────────────────────────────────


def smooth(arr: np.ndarray, window: int = 150) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr
    # nan-safe: fill nans by interpolation before smoothing so a few missing
    # entries don't poison the whole convolution window
    if np.isnan(arr).any():
        idx = np.arange(len(arr))
        good = ~np.isnan(arr)
        arr = np.interp(idx, idx[good], arr[good])
    kernel = np.ones(window) / window
    pad = window // 2
    return np.convolve(np.pad(arr, pad, mode="edge"), kernel, mode="valid")[: len(arr)]


def rolling_slope(x: np.ndarray, y: np.ndarray, window: int = 100) -> np.ndarray:
    """Local trend (units of y per unit of x) via a sliding linear fit,
    centered on each point. Positive = improving, negative = declining."""
    n = len(y)
    if n < 4:
        return np.zeros(n)
    half = max(2, window // 2)
    slopes = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        if hi - lo < 4:
            slopes[i] = 0.0
            continue
        xs, ys = x[lo:hi], y[lo:hi]
        slopes[i] = np.polyfit(xs, ys, 1)[0]
    return slopes


def segment_trends(
    upd: np.ndarray, slope: np.ndarray, flat_thresh: float
) -> list[tuple[int, int, str, float]]:
    """Collapse a per-update slope series into contiguous
    (start_upd, end_upd, label, mean_slope) segments."""

    def label_of(s: float) -> str:
        if s > flat_thresh:
            return "improving"
        if s < -flat_thresh:
            return "declining"
        return "plateau"

    labels = [label_of(s) for s in slope]
    segments = []
    seg_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[seg_start]:
            seg_slopes = slope[seg_start:i]
            segments.append(
                (
                    int(upd[seg_start]),
                    int(upd[i - 1]),
                    labels[seg_start],
                    float(seg_slopes.mean()),
                )
            )
            seg_start = i
    # merge segments shorter than 3% of the run into neighbors' label context
    # (kept simple: just drop noise segments under 15 updates long from the
    # headline list, they'll still be reflected in the plot shading)
    min_len = max(15, int(0.01 * (upd[-1] - upd[0] + 1)))
    return [s for s in segments if (s[1] - s[0]) >= min_len] or segments


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


TREND_COLORS = {"improving": "#1a7a44", "plateau": "#666666", "declining": "#aa2200"}
TREND_ALPHA = 0.16


def _shade_trends(ax: plt.Axes, segments: list[tuple[int, int, str, float]]) -> None:
    for start, end, label, _ in segments:
        ax.axvspan(start, end, color=TREND_COLORS[label], alpha=TREND_ALPHA, lw=0)


# ─── Plots ──────────────────────────────────────────────────────────────────


def plot_all(data: dict[str, np.ndarray], out: Path) -> dict:
    upd = data["updates"]
    RAW_A, RAW_C, SM_C, MAX_C = 0.18, "#5b9bd5", "#1a4a7a", "#e05c00"
    has_epoch = data["epoch_rwds"] is not None

    sm_avg_rwd = smooth(data["avg_rwds"])
    trend_window = max(50, len(upd) // 20)
    slope = rolling_slope(upd.astype(float), sm_avg_rwd, window=trend_window)
    # Self-normalizing threshold: the bottom ~40% of observed slope
    # magnitudes counts as "flat", regardless of the reward scale. This
    # adapts automatically instead of relying on a fixed constant that
    # would need retuning for every reward-shaping change.
    abs_slope = np.abs(slope)
    slope_thresh = float(np.percentile(abs_slope, 40)) if len(abs_slope) else 0.0
    segments = segment_trends(upd, slope, flat_thresh=slope_thresh)

    # 01 — Avg Reward (sliding-window, as before) with trend shading
    fig, ax = plt.subplots(figsize=(11, 4))
    _shade_trends(ax, segments)
    ax.fill_between(upd, data["avg_rwds"], alpha=RAW_A, color=RAW_C)
    ax.plot(upd, data["avg_rwds"], alpha=0.3, color=RAW_C, linewidth=0.8)
    ax.plot(upd, sm_avg_rwd, color=SM_C, linewidth=2, label="Smoothed")
    ax.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set(
        title="Average Episode Reward (20-ep sliding window) — green/red = trend",
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

    # 04 — Reward trend (rolling slope) — shows *when* reward is rising vs stuck
    fig, ax = plt.subplots(figsize=(11, 3.5))
    _shade_trends(ax, segments)
    ax.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.plot(upd, slope, color="#dddddd", linewidth=1.5)
    ax.set(
        title=f"Reward Trend — local slope of smoothed reward (window={trend_window} upd)",
        xlabel="PPO Update",
        ylabel="Δ Reward / Update",
    )
    handles = [
        plt.Line2D(
            [0], [0], color=TREND_COLORS[k], lw=8, alpha=0.5, label=k.capitalize()
        )
        for k in ("improving", "plateau", "declining")
    ]
    ax.legend(
        handles=handles, facecolor="#2a2a4e", labelcolor="white", loc="upper right"
    )
    _style(ax, fig)
    _save(fig, out, "04_reward_trend.png")

    # 05 — Reward vs pellet completion (dual axis) — is reward tracking the
    # actual objective (pellets) or drifting on other shaping terms?
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax2 = ax1.twinx()
    ax1.plot(upd, sm_avg_rwd, color=SM_C, linewidth=2, label="Avg reward (smoothed)")
    ax2.plot(
        upd,
        smooth(data["avg_pcts"]),
        color="#44bb77",
        linewidth=2,
        alpha=0.9,
        label="Avg pellet % (smoothed)",
    )
    ax1.set_ylabel("Avg Reward", color=SM_C)
    ax2.set_ylabel("Avg Pellet %", color="#44bb77")
    ax1.set_xlabel("PPO Update")
    ax1.set_title("Reward vs. Pellet Completion — do they move together?")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    for a in (ax1, ax2):
        a.set_facecolor("#1a1a2e")
        a.tick_params(colors="white")
        a.title.set_color("white")
        a.xaxis.label.set_color("white")
    fig.patch.set_facecolor("#1a1a2e")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        facecolor="#2a2a4e",
        labelcolor="white",
        loc="lower right",
    )
    _save(fig, out, "05_reward_vs_pellets.png")

    # correlation between smoothed reward and smoothed pellet % — tells us
    # whether reward increases are actually coming from eating more pellets
    reward_pellet_corr = float(np.corrcoef(sm_avg_rwd, smooth(data["avg_pcts"]))[0, 1])

    # 06 — Epoch (instantaneous) reward vs sliding-window average reward.
    # Only produced if the log has the new "Averge Epoch Rwd" field.
    epoch_volatility = None
    if has_epoch:
        epoch_rwds = data["epoch_rwds"]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(
            upd,
            epoch_rwds,
            alpha=0.35,
            color="#c77dff",
            linewidth=0.8,
            label="Epoch reward (raw, per-update)",
        )
        ax.plot(
            upd,
            smooth(epoch_rwds),
            color="#7b2cbf",
            linewidth=1.6,
            label="Epoch reward (smoothed)",
        )
        ax.plot(
            upd,
            sm_avg_rwd,
            color=SM_C,
            linewidth=2,
            label="20-ep sliding window avg (smoothed)",
        )
        ax.set(
            title="Instantaneous Epoch Reward vs. Sliding-Window Average",
            xlabel="PPO Update",
            ylabel="Reward",
        )
        ax.legend(facecolor="#2a2a4e", labelcolor="white")
        _style(ax, fig)
        _save(fig, out, "06_epoch_vs_window_reward.png")
        # gap between raw epoch signal and the smoothed window average —
        # large gap/spread = policy still unstable episode-to-episode
        epoch_volatility = float(
            np.nanstd(epoch_rwds - np.interp(upd, upd, data["avg_rwds"]))
        )

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
    _shade_trends(axes[0], segments)
    axes[0].fill_between(upd, data["avg_rwds"], alpha=0.2, color=RAW_C)
    axes[0].plot(upd, sm_avg_rwd, color=SM_C, linewidth=2)
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
    best_sm_idx = int(np.argmax(sm_avg_rwd))
    total_span = upd[-1] - upd[0] + 1
    trend_totals = {"improving": 0, "plateau": 0, "declining": 0}
    for start, end, label, _ in segments:
        trend_totals[label] += end - start + 1
    trend_pct = {k: 100.0 * v / total_span for k, v in trend_totals.items()}

    diagnosis = _build_diagnosis(
        trend_pct, reward_pellet_corr, epoch_volatility, segments, sm_avg_rwd
    )

    return {
        "total_updates": int(len(upd)),
        "total_episodes": int(data["tot_eps"][-1]),
        "best_avg_rwd": float(data["avg_rwds"].max()),
        "best_avg_rwd_upd": int(upd[int(np.argmax(data["avg_rwds"]))]),
        "best_sm_rwd": float(sm_avg_rwd[best_sm_idx]),
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
        "has_epoch_rwd": has_epoch,
        "epoch_volatility": epoch_volatility,
        "reward_pellet_corr": reward_pellet_corr,
        "segments": segments,
        "trend_pct": trend_pct,
        "diagnosis": diagnosis,
    }


def _build_diagnosis(
    trend_pct: dict[str, float],
    corr: float,
    epoch_volatility: float | None,
    segments: list[tuple[int, int, str, float]],
    sm_avg_rwd: np.ndarray,
) -> list[str]:
    """Heuristic, human-readable notes on what the curves suggest."""
    notes = []

    if trend_pct.get("plateau", 0) > 50:
        notes.append(
            f"Reward is flat for ~{trend_pct['plateau']:.0f}% of the run. "
            "The policy has likely converged to whatever the current reward "
            "shape rewards most easily — more updates alone are unlikely to "
            "move it further; this is the point to change the reward system "
            "or curriculum rather than keep training."
        )
    elif trend_pct.get("declining", 0) > 25:
        notes.append(
            f"Reward is declining across ~{trend_pct['declining']:.0f}% of the run. "
            "That usually means an unstable update (LR too high, value loss "
            "diverging, or a shaping term that's easy to exploit in a way "
            "that eventually collapses behavior) rather than something more "
            "training will fix."
        )
    else:
        notes.append(
            f"Reward is still trending up for a meaningful share of the run "
            f"(~{trend_pct.get('improving', 0):.0f}% improving). "
            "Worth letting this configuration keep training before changing "
            "anything."
        )

    if not np.isnan(corr):
        if corr > 0.6:
            notes.append(
                f"Reward and pellet completion move together closely (corr={corr:.2f}). "
                "Reward increases are coming from actually eating more pellets — "
                "the shaping is aligned with the real objective."
            )
        elif corr < 0.3:
            notes.append(
                f"Reward and pellet completion are weakly correlated (corr={corr:.2f}). "
                "Reward is moving for reasons other than pellet progress — check "
                "how much of it comes from the first-visit-tile bonus, the "
                "oscillation penalty, or the per-step cost; the agent may be "
                "optimizing those instead of clearing the maze."
            )
        else:
            notes.append(
                f"Reward and pellet completion are moderately correlated (corr={corr:.2f})."
            )

    if epoch_volatility is not None:
        window_spread = float(np.nanstd(sm_avg_rwd)) + 1e-6
        if epoch_volatility > 1.5 * window_spread:
            notes.append(
                f"Per-episode reward is noisy relative to the smoothed trend "
                f"(episode-to-window std ≈ {epoch_volatility:.1f}). Individual "
                "episodes still swing a lot — consider whether the maze "
                "size/seed randomization is creating too much variance per "
                "update, or whether more PPO epochs/rollout steps would help "
                "it settle."
            )

    return notes


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

## Reward Trend Diagnostics

Reward-vs-pellet correlation (smoothed): **{reward_pellet_corr:.2f}**
{epoch_volatility_line}

Time spent in each trend regime (by update count):

| Regime | % of run |
|--------|----------|
| Improving | {improving_pct:.0f}% |
| Plateau | {plateau_pct:.0f}% |
| Declining | {declining_pct:.0f}% |

### Segments

| Updates | Regime | Avg slope (Δreward/upd) |
|---------|--------|---------------------------|
{segment_rows}

### What this run suggests

{diagnosis_bullets}

---

## Plots

| File | Description |
|------|-------------|
| `00_overview.png` | Combined 3-panel summary (reward / pellets / value loss) |
| `01_avg_reward.png` | Average episode reward, shaded by trend regime |
| `02_pellet_completion.png` | Pellet completion % (avg & max) |
| `03_value_loss.png` | Value network loss |
| `04_reward_trend.png` | Local reward slope — where training is/isn't progressing |
| `05_reward_vs_pellets.png` | Reward vs pellet completion, dual-axis |
{epoch_plot_row}

![Overview](00_overview.png)

---

## Notes / Observations

<!-- Add manual notes about this run here -->
- Training data covers updates {start_upd}–{end_upd} ({total_episodes} episodes).
- Log window size: last 20 completed episodes per update.
- Oscillation penalty introduced in this run to combat node-to-node back-and-forth behavior.
"""


def write_readme(out: Path, num: int, title: str, log_file: str, stats: dict) -> None:
    segment_rows = "\n".join(
        f"| {start}–{end} | {label} | {slope:+.4f} |"
        for start, end, label, slope in stats["segments"]
    )
    diagnosis_bullets = "\n".join(f"- {n}" for n in stats["diagnosis"])
    epoch_volatility_line = (
        f"Episode-to-window reward volatility (std): **{stats['epoch_volatility']:.1f}**"
        if stats.get("epoch_volatility") is not None
        else "*(Log has no per-epoch reward field — add `Averge Epoch Rwd` to the "
        "logger to unlock instantaneous-vs-smoothed volatility diagnostics.)*"
    )
    epoch_plot_row = (
        "| `06_epoch_vs_window_reward.png` | Instantaneous epoch reward vs the smoothed sliding-window average |"
        if stats.get("has_epoch_rwd")
        else ""
    )

    content = README_TEMPLATE.format(
        num=num,
        title=title,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        log_file=log_file,
        segment_rows=segment_rows,
        diagnosis_bullets=diagnosis_bullets,
        epoch_volatility_line=epoch_volatility_line,
        epoch_plot_row=epoch_plot_row,
        improving_pct=stats["trend_pct"].get("improving", 0.0),
        plateau_pct=stats["trend_pct"].get("plateau", 0.0),
        declining_pct=stats["trend_pct"].get("declining", 0.0),
        **{
            k: v
            for k, v in stats.items()
            if k
            not in (
                "segments",
                "trend_pct",
                "diagnosis",
                "epoch_volatility",
                "has_epoch_rwd",
            )
        },
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
    if data["epoch_rwds"] is not None:
        print("  -> per-epoch reward field detected, extra diagnostic plot enabled.")

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
    print(f"  Reward/pellet correlation: {stats['reward_pellet_corr']:.2f}")
    for note in stats["diagnosis"]:
        print(f"  · {note}")


if __name__ == "__main__":
    main()
