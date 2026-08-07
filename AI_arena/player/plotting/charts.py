"""Diagnostic chart generation functions using matplotlib."""

from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    raise SystemExit("ERROR: matplotlib not installed.")


BG_DARK = "#1a1a2e"
CARD_DARK = "#16213e"
TEXT_LIGHT = "#e94560"
ACCENT_CYAN = "#00fff5"
ACCENT_GREEN = "#00ff87"
ACCENT_AMBER = "#ffb703"
ACCENT_CORAL = "#ff4d6d"
ACCENT_PURPLE = "#b5179e"
ACCENT_BLUE = "#48cae4"
TEXT_MUTED = "#8d99ae"

ALL_ACCENTS = [
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_CORAL, ACCENT_PURPLE,
    ACCENT_BLUE, "#90be6d", "#f94144", "#f8961e", "#f9c74f",
    "#43aa8b", "#577590", "#7209b7", "#4cc9f0", "#ff006e",
]


def setup_dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": CARD_DARK,
        "axes.edgecolor": "#2a2a4a",
        "axes.labelcolor": "#d0d0e0",
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "grid.color": "#2a2a4a",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "text.color": "#ffffff",
        "font.family": "sans-serif",
        "font.size": 11,
    })


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def smooth_x(x: np.ndarray, window: int = 5) -> np.ndarray:
    if len(x) < window:
        return x
    return x[window - 1 :]


def plot_avg_reward(data: dict[str, np.ndarray], out_path: Path) -> None:
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    upd = data["updates"]
    raw = data["avg_rwds"]

    ax.plot(upd, raw, color=ACCENT_CYAN, alpha=0.3, label="Raw (Per-Update)")
    if len(raw) >= 5:
        sm = smooth(raw, 5)
        ax.plot(smooth_x(upd, 5), sm, color=ACCENT_CYAN, linewidth=2.5, label="5-Update Moving Avg")

    ax.set_title("Average Window Reward Across Training Updates", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Update")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend(facecolor=CARD_DARK, edgecolor="#2a2a4a")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_pellet_completion(data: dict[str, np.ndarray], out_path: Path) -> None:
    setup_dark_style()
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    upd = data["updates"]
    avg_pct = data["avg_pcts"]
    max_pct = data["max_pcts"]

    ax1.plot(upd, avg_pct, color=ACCENT_GREEN, linewidth=2, label="Avg Pellet Collection (%)")
    ax1.plot(upd, max_pct, color=ACCENT_AMBER, linewidth=1.5, linestyle="--", label="Max Pellet Collection (%)")
    ax1.set_title("Pellet Collection & Completion Percentage", fontsize=14, pad=12, fontweight="bold")
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Collection %")
    ax1.set_ylim(0, 105)
    ax1.grid(True)
    ax1.legend(facecolor=CARD_DARK, edgecolor="#2a2a4a", loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_value_loss(data: dict[str, np.ndarray], out_path: Path) -> None:
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    upd = data["updates"]
    v_loss = data["value_losses"]

    ax.plot(upd, v_loss, color=ACCENT_CORAL, linewidth=2, label="Value Loss")
    if len(v_loss) >= 5:
        ax.plot(smooth_x(upd, 5), smooth(v_loss, 5), color="#ff758f", linewidth=2.5, label="5-Update Moving Avg")

    ax.set_title("PPO Value Loss Convergence", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Update")
    ax.set_ylabel("Value Loss")
    ax.grid(True)
    ax.legend(facecolor=CARD_DARK, edgecolor="#2a2a4a")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_oscillation_pct(data: dict[str, np.ndarray], out_path: Path) -> None:
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    upd = data["updates"]
    osc = data["osc_pcts"]

    ax.plot(upd, osc, color=ACCENT_CORAL, alpha=0.3, label="Raw Oscillation %")
    if len(osc) >= 5:
        sm = smooth(osc, 5)
        ax.plot(smooth_x(upd, 5), sm, color=ACCENT_CORAL, linewidth=2.5, label="5-Update Moving Avg")

    ax.axhline(5.0, color=ACCENT_GREEN, linestyle=":", alpha=0.8, label="Target Baseline (<5%)")
    ax.set_title("Movement Oscillation Percentage", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Update")
    ax.set_ylabel("Oscillation Rate (%)")
    ax.set_ylim(0, max(100.0, np.max(osc) * 1.1) if len(osc) > 0 else 100)
    ax.grid(True)
    ax.legend(facecolor=CARD_DARK, edgecolor="#2a2a4a")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_all(data: dict[str, np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_avg_reward(data, out_dir / "01_avg_reward.png")
    plot_pellet_completion(data, out_dir / "02_pellet_completion.png")
    plot_value_loss(data, out_dir / "03_value_loss.png")
    if data["has_osc"]:
        plot_oscillation_pct(data, out_dir / "13_oscillation_pct.png")
