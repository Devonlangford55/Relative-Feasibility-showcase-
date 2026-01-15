```markdown
# RF‑FJSP Demo — Relative Feasibility (RF) showcase

A focused, demo that demonstrates the Relative Feasibility (RF) gating idea applied to a Flexible Job Shop Scheduling (FJSP) style instance.

This repository contains a clean, conservative demo:
- Script: `rf_fjsp_demo_clean.py`
  - Multi‑stage probing: precedence‑aware greedy local completion → OR‑Tools CP‑SAT interval verification (optional).
  - Conservative by default: prunes are accepted only when CP proves infeasibility under a tightened incumbent bound.
  - Robust timing: uses a monotonic clock for elapsed/timeouts and wall‑clock time only for logging.
  - Outputs: `instance.json`, `trace.csv`, `hybrid_summary.json`, `outputs.zip`.

Goals
- Be reproducible and easy to run locally or in Colab.
- Demonstrate RF decision‑making and safe CP verification.
- Produce clear, auditable artifacts for reporting and tuning.

Prerequisites
- Python 3.8+
- Optional (for CP verification): OR‑Tools
  - Install: `pip install ortools`
  - In Colab: install and then Restart runtime (or use the provided `--force_probe` to test).

Quick start (local)

1. Create a virtual environment (optional but recommended)
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt   # if you create one; OR just install ortools optionally
   ```

2. Smoke test (no OR‑Tools required)
   ```
   python rf_fjsp_demo_clean.py --jobs 20 --machines 6 --time 30 --outdir outputs_smoke
   ```

3. Showcase with OR‑Tools CP probes (recommended)
   ```
   pip install ortools
   # In notebooks: restart runtime after install
   python rf_fjsp_demo_clean.py --run_cpsat --num_workers 2 --jobs 40 --machines 8 --time 300 --tau 1.2 --theta_prune 0.06 --outdir outputs_cpsat
   ```

Helpful demo flags
- `--run_cpsat` : enable OR‑Tools CP probes (requires ortools).
- `--num_workers N` : CP‑SAT search workers.
- `--no_greedy` : start with no greedy preseed (force exploration).
- `--force_probe` : run one CP probe at root (useful to demonstrate verify timings).
- `--tau` : per‑probe time budget (seconds). Typical: 0.4–2.0.
- `--theta_prune` : RF threshold that triggers probing. Typical: 0.04–0.15.
- `--max_probe_fraction` : stop probing if probes exceed this fraction of elapsed run (default 0.3).
- `--trace_sample_rate` : how often nodes are written to `trace.csv` (keeps files small).

Primary outputs (in `--outdir`)
- `instance.json` — saved instance (seeded) used for reproduction
- `trace.csv` — per‑node trace (timestamp, elapsed, lb, rf_score, actions, verify info)
- `hybrid_summary.json` — final summary:
  - `makespan`, `nodes_expanded`, `nodes_pruned`, `probe_time_total`, `probe_attempts`, `probe_successes`, `elapsed`, start/end timestamps
- `outputs.zip` — zipped artifact bundle

How to interpret results (short)
- `makespan`: best found makespan (lower is better).
- `nodes_expanded` vs `nodes_pruned`: shows how much B&B explored vs how much RF+CP removed.
- `probe_time_total` and `probe_time_fraction = probe_time_total / elapsed`: how much time probes consumed — balance this to get ROI.
- `probe_successes / probe_attempts`: probe effectiveness (finding improvements or proofs).
- `trace.csv`: inspect per‑node RF decisions, verify times and statuses (`INFEASIBLE`, `FEASIBLE`, `UNKNOWN`).

Tuning guidance (practical)
- If probes consume almost all time but rarely help:
  - Lower `--tau` and/or raise `--theta_prune`.
- If probes rarely run but could help:
  - Lower `--theta_prune` or use `--force_probe` to test root proof potential.
- If greedy preseed hides search value:
  - Use `--no_greedy` to force exploration.
- Typical starting defaults: `--tau 0.6 --theta_prune 0.06`.

Colab tips
- After `pip install ortools` restart the runtime before running `--run_cpsat`.
- Use `--force_probe` initially to demonstrate CP solving without long runs.
- Consider mounting Drive for larger output persistence.

Troubleshooting: clock & runtime issues
- The demo uses `time.monotonic()` for elapsed/timeouts and `time.time()` only for wall timestamps.
- If you see very large `elapsed` values (Unix timestamps) in `hybrid_summary.json`, restart your environment and re‑run. The demo writes both `start_timestamp` and `end_timestamp` for clarity.

Safety & reproducibility
- The default mode is conservative — prunes require CP proof. This avoids false prunes.
- All runs are seeded (use `--seed`) and `instance.json` is saved for reproduction.

License & contributing
- This demo is intended to be distributed under Apache‑2.0 (see LICENSE in repo root).
- If you accept contributions, consider requiring DCO (`git commit -s ...`) to simplify contributor licensing.

Next steps (suggested)
- Add a small parameter‑sweep driver to find good `tau` / `theta_prune` defaults for your instance class.
- Add a Colab notebook that runs the demo, shows the three plots (nodes vs time, RF histogram, verification spikes) and provides an interactive slider for tau/theta_prune.

If you’d like, I can generate:
- a README with badges and quick badges for Colab,
- a Colab notebook that runs the showcase and visualizes `trace.csv`,
- a short parameter sweep driver.

Tell me which and I’ll produce the files next.
```
