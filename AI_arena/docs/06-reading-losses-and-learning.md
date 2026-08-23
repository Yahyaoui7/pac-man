# 06 — Reading Losses, LR & Entropy: "Is the model learning?"

Last updated: **2026-08-23**

Losses are **diagnostics, not scoreboards**. This doc explains what each
number in the log line means, what healthy looks like *for this project*, and
how to combine them with the eval score into a verdict.

## The evidence hierarchy (in order of trust)

1. **`EVAL @NNN:` score** — greedy policy on 20 fixed seeds. The only
   ground truth. Needs ≥2–3 points to call a trend.
2. **SURV telemetry trends** (`Esc↑ Corn↓ CDth↓ MinD↑`) — leading indicators.
3. Rolling training averages (`Avg Rwd`, `Avg Pellets`) — noisy, maze-mix
   contaminated.
4. **Losses & entropy — last.** They tell you *why* learning stalls or
   diverges, almost never *whether* it happens.

Rule: never change LR/entropy because a loss "looks big". Change them when a
symptom below appears alongside flat/declining eval scores.

## The three numbers in `Ent: x | Loss (P/V): p/v`

### Policy loss `p` — the clipped surrogate

`p = −min(ratio·A, clip(ratio, 0.8, 1.2)·A).mean()` with advantages normalized
to std 1 each update.

| Reading | Interpretation |
|---|---|
| ~0 ± 0.01 (**our normal**: −0.007…+0.010) | Healthy. Clipping keeps updates tiny; near-zero is the *steady state*, not stagnation |
| Consistently \|p\| > 0.03–0.05 | Ratios drifting far from 1 → stale rollouts or LR too high for batch size; expect eval instability |
| Large swings correlated with value-loss spikes | Update instability — usually LR or a reward-scale change |

The sign is nearly meaningless on its own; magnitude relative to your own
history is what matters.

### Value loss `v` — smooth L1 vs returns

`v = smooth_l1(values, returns)`. For errors ≫ 1 this is ≈ mean absolute
prediction error, so:

> **v ≈ 40 means the critic misses return by ~40 reward units on average.**

Context for our scale: single outcomes are death −350 / milestones +20…200 /
completion +5000, so episode-return std is in the hundreds. v ∈ 27–55 (our
observed range) is therefore small relative to what's being predicted —
healthy. Absolute "low" v is unreachable and not a goal.

| Reading | Interpretation |
|---|---|
| Stable plateau (ours: 27–55) | Critic tracking; fine even if never decreasing |
| Sudden ×2 jump after you edit rewards | Expected — return distribution changed. Wait ~50 upd before judging anything |
| Sustained growth / spikes ×5 | LR too high or advantage/return bug; halve LR |
| Slow decline over hundreds of updates | Critic refining — usually accompanies eval gains |

Better than raw v: explained variance `1 − Var(ret−V)/Var(ret)`
(>0 critic useful, <0 worse than predicting the mean). Not logged yet —
candidate future addition.

### Entropy `Ent` — how random the policy is

Mean entropy over masked action distribution, in nats.

- **ln(4) ≈ 1.386** = uniform/random policy (theoretical max)
- **→ 0** = fully deterministic

Healthy trajectory: starts high, declines *slowly* over thousands of episodes
while eval rises. Typical landing zone for a converged Pac-Man policy:
0.2–0.6.

| Symptom | Diagnosis | Fix |
|---|---|---|
| Ent < ~0.1 while eval stalls | Premature convergence — policy locked in before finding survival behaviors | Raise `ENTROPY_COEF` 0.015 → 0.03 temporarily, or lower LR |
| Ent collapses within tens of updates | Same, faster — often too-high LR + low entropy bonus acting together | Halve LR first |
| Ent pinned ≥1.2 for hundreds of updates | Not committing; reward signal too sparse/diluted to prefer actions | Lower `ENTROPY_COEF`; check that intended reward terms are non-zero in breakdown |
| Ent oscillates with eval cycles | Normal exploration/exploitation breathing — ignore unless eval declines |

Note: entropy bonus enters the loss as `−ENTROPY_COEF · Ent`, so its pressure
is weak at 0.015 — mostly an anti-collapse guard.

**Known failure mode (observed Aug 2026): Ent pinned ≈ 0.10 even at
`ENTROPY_COEF = 0.04`.** Two causes compound: (1) the entropy gradient
vanishes as the softmax saturates — a bigger coefficient pushes on a rope;
(2) in a harsh environment, exploratory deviations die quickly, so the
policy gradient actively re-sharpens. When this happens, stop leaning on the
entropy bonus and force exploration at data-collection time instead:
`ROLLOUT_EPSILON` (ε-uniform mixture during rollouts, PPO-correct log-probs —
see [01-training-loop.md](01-training-loop.md)).

## Learning rate symptom table

Our setting: `LEARNING_RATE=2e-4`, minibatch ≈ 64 frames, 2 PPO epochs,
clip 0.2.

| Symptom pattern | Likely cause | Action |
|---|---|---|
| Eval gains appear then vanish within 20–50 upd; v spiking | LR too high | Halve (2e-4 → 1e-4) |
| Everything flat 200+ upd, Ent mid-range (0.5–1.0), SURV flat too | LR too low **or** reward plateau | Check SURV first; only then try 1.5× LR |
| p regularly >0.05, eval choppy | Updates too aggressive per batch | Lower LR or raise `MINIBATCH_SEQS` |
| Smooth slow gains across ≥3 evals | Don't touch anything | — |

After any LR/reward change: discard the next 1–2 evals as transition noise.

## Worked example — this run (upd 1–181, ~2370 eps)

```
EVAL baseline (pre-run) : 9.0  pellet 39.0% death 100%
EVAL @050               : 13.9 pellet 43.9% death 100% esc 100%[9] MinD 9.76
EVAL @100               : 15.6 pellet 45.6% death 100% esc 100%[4] MinD 10.17 ← best saved
EVAL @150               : 14.6 pellet 44.6% death 100% esc  67%[9] MinD 9.42
P loss ≈ ±0.008 · V loss 27–55 stable · Osc% ~12–15 · death 100% everywhere
```

Verdicts:
- **Learning: yes** — +4.9 then +1.7 on identical seeds is far above noise;
  the @150 dip (−1.0) is within variance (esc n=9).
- **But**: every point of gain came from pellets. Survival metrics frozen —
  death 100% in all evals, `Truncated: 0%`, `Survival: +0.0` (the +200
  truncation bonus has still never fired).
- Losses healthy; no reason to touch LR or entropy.
- Timeline note: last meaningful improvement was @100 → stall warning fires at
  update ~350 if no eval beats 17.6 by then.

## Decision routine at every EVAL checkpoint

1. Score vs previous: ≥ +2 meaningful; ±1 noise; ≤ −2 twice in a row = regression, investigate.
2. Death% moved? If still 100% after several evals while score plateaus → the
   binding constraint is survival, and time alone won't fix it.
3. Check SURV deltas (Esc/Corn/CDth/MinD), not just score.
4. Glance at `Ent`: collapsing? pinned?
5. Only if symptoms above match: adjust one variable (LR *or* entropy *or*
   reward term), never two at once.

For survival specifically, before touching constants remember the pre-built
disabled shapers in `rewards.py`: `_evasion_skill_reward`,
`_threat_mastery_reward`, `_dense_survival_reward` — designed exactly for the
"dies in chases" phase we're in (see [03-rewards.md](03-rewards.md)).
