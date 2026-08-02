# Training Report 003 — PPO Stage-1 — High-Exploration Corridor Expansion

Generated: 2026-08-02 08:46  
Status: **Ready for Continuation Run**  
Base Checkpoint Restored: `reports/report_002/player_rl_stage1.pt`

---

## 1. Dead-End & Oscillation Bottleneck Diagnosis

Previous runs plateaued at ~30 pellets because Pac-Man was trapped by dead-end mechanics:

1. **Dead-End Reversal Requirement**: In Pac-Man maze topology, every corridor end is a dead-end. To exit a dead-end corridor after eating pellets, Pac-Man **must U-turn**.
2. **The Oscillation Penalty Bottleneck**: Penalizing U-turns (`is_oscillating: -0.3`) punished Pac-Man every time it tried to exit a dead-end corridor. Consequently, Pac-Man learned **never to enter deep corridors**, restricting itself to the starting hub (~30 pellets).
3. **The Solution**:
   - **Removed U-turn / Oscillation penalty** so Pac-Man can freely exit dead-end corridors.
   - **Boosted New Tile Exploration Bonus to `+2.0`** (up from `+1.0`) to aggressively pull Pac-Man into unexplored corridors.
   - **Increased Pellet Reward to `+5.0`** (net `+6.9` on new pellet tile) to make clearing entire maze quadrants extremely lucrative.
   - **Mild Base Step Penalty (`-0.1`)** allowing Pac-Man to traverse 30+ cleared corridor steps to reach secondary quadrants with zero net deficit.

---

## 2. Updated High-Exploration Reward Matrix

| Event | Report 002 | Failed Oscillation Run | Calibrated Report 003 | Rationale & Net Impact |
|-------|------------|------------------------|-----------------------|------------------------|
| Base Step Penalty | **−0.1** | −0.15 | **−0.1** | Mild step penalty allowing cross-maze corridor traversal. |
| U-Turn / Oscillation | **0.0** | −0.3 | **0.0** | **Removed** to allow Pac-Man to exit dead-end corridors freely. |
| New Tile Exploration | **+1.0** | +1.0 | **+2.0** | **Doubled** to strongly pull Pac-Man into new corridors. |
| Pellet Eaten | **+3.0** | +4.0 / +6.0 | **+5.0** | **High Pellet Bonus** (+6.9 net per new pellet tile). |
| Super-pellet Eaten | **+5.0** | +6.0 | **+10.0** | Scaled proportionally. |
| Level Completed | **+100.0** | +100.0 | **+100.0** | Maintained. |
| Pac-Man Died | **−20.0** | −20.0 | **−20.0** | Maintained. |

---

## 3. Continuation Execution Command

Resume continuation training directly from the restored Report 002 model:

```bash
.venv/bin/python3 -m AI_arena.player.player_training --stage 1 --num-updates 5000 --save-interval 50
```
*(Do **NOT** use `--fresh` so training resumes from `player_rl_stage1.pt`).*
