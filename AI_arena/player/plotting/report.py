"""Automated markdown report generator for training evaluation."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
import numpy as np


def generate_markdown_report(
    data: dict[str, np.ndarray],
    out_dir: Path,
    log_name: str,
    custom_title: str | None = None,
) -> None:
    """Write README.md report detailing performance and diagnostic charts."""
    title = custom_title or f"Training Report — {log_name}"
    n_upd = len(data["updates"])
    start_upd = data["updates"][0]
    end_upd = data["updates"][-1]
    tot_eps = data["tot_eps"][-1]
    peak_pellet_pct = float(np.max(data["avg_pcts"]))
    last_pellet_pct = float(data["avg_pcts"][-1])
    peak_max_pct = float(np.max(data["max_pcts"]))
    avg_osc = float(np.mean(data["osc_pcts"])) if data["has_osc"] else 0.0

    content = f"""# {title}

**Generated at**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Log file**: `{log_name}`  
**Total Updates Analyzed**: {n_upd} (Updates #{start_upd} -> #{end_upd})  
**Total Completed Episodes**: {tot_eps}

---

## Executive Summary

- **Peak Window Pellet Collection**: `{peak_pellet_pct:.1f}%`
- **Latest Window Pellet Collection**: `{last_pellet_pct:.1f}%`
- **Max Single Episode Collection**: `{peak_max_pct:.1f}%`
- **Average Oscillation Rate**: `{avg_osc:.1f}%`

---

## Performance Diagnostic Charts

### 1. Window Average Reward
![Average Reward](01_avg_reward.png)

### 2. Pellet Collection Percentage
![Pellet Completion](02_pellet_completion.png)

### 3. PPO Value Loss Convergence
![Value Loss](03_value_loss.png)
"""
    if data["has_osc"]:
        content += """
### 4. Movement Oscillation Rate
![Oscillation Rate](13_oscillation_pct.png)
"""

    (out_dir / "README.md").write_text(textwrap.dedent(content), encoding="utf-8")
