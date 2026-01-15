```markdown
# NOTES_RF — Relative Feasibility (RF) for FJSP (concise)

This note explains the RF gating idea in the context of Flexible Job Shop Scheduling (FJSP) and how the demo uses it.

Overview
- RF (Relative Feasibility) is a decision controller that decides *when* to spend expensive verification time (e.g., CP‑SAT) on a B&B node.
- RF is NOT a solver or relaxation — it is a lightweight ROI rule: probe only when expected savings > probe cost.

Why FJSP is a good showcase
- Jobs have operation sequences (precedence) and each operation has a small set of eligible machines.
- Pruning a high‑level node can remove a large subtree (many sequences), so a successful CP proof is high‑value.
- OR‑Tools interval models (precedence + AddNoOverlap) are strong verifiers for these subproblems.

RF signals (used in demo)
- Slack: (incumbent − lb) / incumbent — how close LB is to current UB.
- Time factor: urgency based on remaining time and estimated remaining nodes.
- Depth penalty: prefer verifying higher‑level nodes (pruning high saves more).
- Recent probe history: adaptive behavior can increase/decrease probe rate.

Illustrative workflow
1. At each node compute LB and RF score.
2. If LB ≥ incumbent → cheap prune.
3. Else if RF < θ_prune → run probe with budget τ:
   - Stage 1: heuristic greedy completion (respects precedence) — fast, used to update incumbent only.
   - Stage 2: OR‑Tools CP‑SAT interval model (optional) — used to prove prunes (conservative mode).
4. If CP proves infeasibility under tightened bound (makespan ≤ incumbent−1) → accept prune.
   If CP finds feasible solution < incumbent → update incumbent.
   Otherwise → expand node.

Safety modes
- Conservative (demo default): accept prunes only when CP proves infeasibility (no heuristic-only prunes).
- Experimental: allow heuristic prunes with periodic audits.

Key knobs
- τ (tau): per‑probe time budget (seconds)
- θ_prune: RF threshold to trigger probing
- num_workers: CP‑SAT threads
- preseed / no_greedy: initial greedy UB

Metrics to collect (for tuning)
- nodes_expanded, nodes_pruned, proof_prunes
- probe_time_total, probe_attempts, probe_successes, probe_time_fraction
- rf_score distribution (histogram)
- false_prunes (if running in audit/experimental mode)

Minimal probe API (recommended)
- Probe(ops, partial, time_limit, incumbent) -> {status, makespan?}
  - Status: 'FEASIBLE' (provides makespan), 'INFEASIBLE' (proved no solution <= ub), 'UNKNOWN'

Practical advice
- Start conservative: small τ (0.4–1.2s), modest θ_prune (0.04–0.12).
- Monitor probe_time_fraction — keep probes from dominating (>30% is often too high).
- Use `--no_greedy` to stress the hybrid controller or `--force_probe` to demonstrate CP at root.
- Use monotonic timing in demos to avoid runtime/resume artifacts (demo uses `time.monotonic()`).

Research & engineering opportunities
- Learned RF predictor (features → probe ROI).
- Bandit allocation of probe budgets across nodes.
- Caching CP results for isomorphic subproblems.
- Better cheap relaxations (LBs) to reduce unnecessary probes.

This file is intentionally short — link it from README for visitors.
```