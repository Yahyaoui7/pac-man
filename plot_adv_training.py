import matplotlib.pyplot as plt
import re
import argparse
from pathlib import Path

def plot_logs(log_file_path):
    log_path = Path(log_file_path)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    updates = []
    p_rews = []
    g_rews = []
    win_pcts = []
    die_pcts = []
    
    # Regex to match the log line format:
    # U 0001 [GHOST WARMUP] | Ep 0001 | P_Rew: -204.6 | G_Rew:  302.3 | Win%:  49.5% | Die%: 100.0%
    pattern = re.compile(
        r"U\s+(\d+)\s+\[.*?\]\s+\|\s+Ep\s+\d+\s+\|\s+P_Rew:\s+([-0-9.]+)\s+\|\s+G_Rew:\s+([-0-9.]+)\s+\|\s+Win%:\s+([0-9.]+)%\s+\|\s+Die%:\s+([0-9.]+)%"
    )
    
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                updates.append(int(match.group(1)))
                p_rews.append(float(match.group(2)))
                g_rews.append(float(match.group(3)))
                win_pcts.append(float(match.group(4)))
                die_pcts.append(float(match.group(5)))
                
    if not updates:
        print(f"No valid log lines found in {log_file_path}.")
        return

    # Create figure with 2 subplots (Rewards and Rates)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top subplot: Rewards
    ax1.plot(updates, p_rews, label="Player Reward", color='dodgerblue', linewidth=2)
    ax1.plot(updates, g_rews, label="Ghosts Reward", color='crimson', linewidth=2)
    ax1.set_ylabel("Reward")
    ax1.set_title("Adversarial Training Progress", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Bottom subplot: Win/Die percentages
    ax2.plot(updates, win_pcts, label="Player Win %", color='forestgreen', linewidth=2)
    ax2.plot(updates, die_pcts, label="Player Die %", color='darkorchid', linewidth=2)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xlabel("Update (U) Number")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    output_img = "adv_training_plot.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated plot! Saved as: {output_img}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot adversarial training logs.")
    parser.add_argument("--log", type=str, default="adv_training_log.txt", help="Path to the log file")
    args = parser.parse_args()
    
    plot_logs(args.log)
