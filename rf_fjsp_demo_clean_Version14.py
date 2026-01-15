#!/usr/bin/env python3
"""
rf_fjsp_demo_clean.py

Clean, repo-ready demo for Relative Feasibility (RF) on a Flexible Job Shop Scheduling (FJSP)-style instance.

Goals:
- Conservative by default: accept prunes only when CP-SAT proves infeasibility under a tightened incumbent bound.
- Multi-stage probing: fast precedence-aware greedy completion first, then optional OR-Tools CP-SAT interval verification.
- Robust timing: use time.monotonic() for elapsed/timeouts and time.time() only for human-facing timestamps.
- Proper probe accounting and reproducible outputs: instance.json, trace.csv, hybrid_summary.json, outputs.zip.
- Small, well-documented CLI for quick smoke and showcase runs.

Usage (examples)
  # smoke (no OR-Tools)
  python rf_fjsp_demo_clean.py --jobs 20 --machines 6 --time 30 --outdir outputs_smoke

  # showcase with OR-Tools (install ortools first)
  pip install ortools
  python rf_fjsp_demo_clean.py --run_cpsat --num_workers 2 --jobs 40 --machines 8 --time 300 --tau 1.2 --theta_prune 0.06 --outdir outputs_cpsat

Notes
- This file is intentionally focused on the demo code only (no README, CONTRIBUTING, etc).
- For Colab: after pip installing ortools, restart the runtime before running with --run_cpsat.
"""
from __future__ import annotations
import argparse
import csv
import heapq
import json
import math
import os
import random
import signal
import sys
import time
import zipfile
from collections import namedtuple
from typing import Dict, List, Optional

# Optional OR-Tools
try:
    from ortools.sat.python import cp_model  # type: ignore
    HAS_ORTOOLS = True
except Exception:
    HAS_ORTOOLS = False

# Graceful shutdown
shutdown_requested = False
def _sig_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print("Signal received — will checkpoint and exit gracefully.", flush=True)
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

Node = namedtuple("Node", ["index", "partial", "machine_loads", "job_ends", "makespan", "depth", "remaining", "id"])

TRACE_HEADER = [
    "timestamp_wall", "elapsed", "node_id", "index", "depth", "lb", "incumbent",
    "slack", "time_factor", "rf_score", "action", "verify_used", "verify_time",
    "verify_status", "node_dur"
]

# --------------------------
# Instance generator (FJSP-like)
# --------------------------
def generate_fjsp_instance(machines: int, jobs: int, min_ops: int, max_ops: int, seed: int = 20260107) -> Dict:
    rng = random.Random(seed)
    jobs_list = []
    ops_flat = []
    op_id = 0
    for j in range(jobs):
        k = rng.randint(min_ops, max_ops)
        seq = []
        for o in range(k):
            r = rng.random()
            if r < 0.08:
                p = rng.randint(80, 220)
            elif r < 0.35:
                p = rng.randint(30, 80)
            else:
                p = rng.randint(5, 30)
            elig_k = rng.randint(max(1, machines//4), min(machines, max(2, machines//2)))
            eligible = sorted(rng.sample(range(machines), elig_k))
            seq.append({"id": op_id, "proc_time": p, "eligible": eligible})
            ops_flat.append({"job": j, "id": op_id, "proc_time": p, "eligible": eligible})
            op_id += 1
        jobs_list.append({"job_id": j, "operations": seq})
    return {"machines": machines, "jobs": jobs_list, "ops_flat": ops_flat, "meta": {"seed": seed, "jobs": jobs, "ops_total": op_id}}

# --------------------------
# Greedy preseeding (respects precedence)
# --------------------------
def greedy_preseed_schedule(ops_flat: List[Dict], machines: int):
    # Build job sequences
    jobs = {}
    for op in ops_flat:
        jobs.setdefault(op["job"], []).append(op)
    # scheduling state
    machine_loads = [0.0]*machines
    job_ends = {j:0.0 for j in jobs}
    assign = {}
    remaining = {op["id"] for op in ops_flat}
    # schedule in waves: pick first unassigned op per job, schedule by LPT among ready
    while remaining:
        ready = []
        for j, seq in jobs.items():
            for op in seq:
                if op["id"] in remaining:
                    ready.append(op)
                    break
        # largest processing time first
        ready.sort(key=lambda x: -x["proc_time"])
        for op in ready:
            if op["id"] not in remaining:
                continue
            best_m = min(op["eligible"], key=lambda m: max(machine_loads[m], job_ends[op["job"]]) + op["proc_time"])
            start = max(machine_loads[best_m], job_ends[op["job"]])
            finish = start + op["proc_time"]
            machine_loads[best_m] = finish
            job_ends[op["job"]] = finish
            assign[op["id"]] = best_m
            remaining.remove(op["id"])
    makespan = max(machine_loads) if machine_loads else 0.0
    return float(makespan), assign

# --------------------------
# Local probe: greedy completion (respects precedence)
# --------------------------
def local_probe_complete(ops_flat: List[Dict], partial: Dict[int,int], machines: int, time_limit: float) -> Optional[float]:
    if time_limit <= 0:
        return None
    t0 = time.monotonic()
    machine_loads = [0.0]*machines
    job_ends = {}
    assigned = set()
    for op in ops_flat:
        job_ends[op["job"]] = job_ends.get(op["job"], 0.0)
        if op["id"] in partial:
            m = partial[op["id"]]
            start = max(machine_loads[m], job_ends[op["job"]])
            finish = start + op["proc_time"]
            machine_loads[m] = finish
            job_ends[op["job"]] = finish
            assigned.add(op["id"])
    remaining_set = {op["id"] for op in ops_flat if op["id"] not in assigned}
    jobs_seq = {}
    for op in ops_flat:
        jobs_seq.setdefault(op["job"], []).append(op)
    # schedule remaining respecting precedence
    while remaining_set:
        if time.monotonic() - t0 >= time_limit:
            break
        ready = []
        for j, seq in jobs_seq.items():
            for op in seq:
                if op["id"] in remaining_set:
                    ready.append(op)
                    break
        ready.sort(key=lambda x: -x["proc_time"])
        for op in ready:
            if op["id"] not in remaining_set:
                continue
            best_m = min(op["eligible"], key=lambda m: max(machine_loads[m], job_ends[op["job"]]) + op["proc_time"])
            start = max(machine_loads[best_m], job_ends[op["job"]])
            finish = start + op["proc_time"]
            machine_loads[best_m] = finish
            job_ends[op["job"]] = finish
            remaining_set.remove(op["id"])
            if time.monotonic() - t0 >= time_limit:
                break
    return float(max(machine_loads)) if machine_loads else None

# --------------------------
# CP verifier (interval model with precedence + AddNoOverlap)
# --------------------------
def cp_verify_fjsp(ops_flat: List[Dict], machines: int, partial: Dict[int,int], time_limit: float,
                   incumbent: Optional[float], num_workers: int = 1) -> Optional[Dict]:
    if not HAS_ORTOOLS or time_limit <= 0:
        return None
    try:
        model = cp_model.CpModel()
        n = len(ops_flat)
        horizon = int(sum(op["proc_time"] for op in ops_flat) + 10)
        starts = [model.NewIntVar(0, horizon, f"s_{i}") for i in range(n)]
        ends = [model.NewIntVar(0, horizon, f"e_{i}") for i in range(n)]
        intervals_per_machine = {m: [] for m in range(machines)}
        presence = {}
        # create optional intervals only for eligible machines
        for idx, op in enumerate(ops_flat):
            dur = int(op["proc_time"])
            bools = []
            for m in op["eligible"]:
                b = model.NewBoolVar(f"x_{idx}_{m}")
                bools.append(b)
                iv = model.NewOptionalIntervalVar(starts[idx], dur, ends[idx], b, f"int_{idx}_{m}")
                intervals_per_machine[m].append(iv)
                presence[(idx, m)] = b
            model.Add(sum(bools) == 1)
            model.Add(ends[idx] == starts[idx] + dur)
        # map op id -> index in ops_flat
        id_to_idx = {op["id"]: i for i, op in enumerate(ops_flat)}
        for op_id, mm in partial.items():
            if op_id in id_to_idx:
                i = id_to_idx[op_id]
                if (i, mm) in presence:
                    model.Add(presence[(i, mm)] == 1)
                else:
                    # impossible fixed assignment -> model infeasible
                    model.AddFalseConstraint()
        # precedence constraints per job
        job_groups = {}
        for i, op in enumerate(ops_flat):
            job_groups.setdefault(op["job"], []).append(i)
        for seq in job_groups.values():
            for a, b in zip(seq, seq[1:]):
                model.Add(starts[b] >= ends[a])
        # machine no-overlap
        for m in range(machines):
            if intervals_per_machine[m]:
                model.AddNoOverlap(intervals_per_machine[m])
        makespan = model.NewIntVar(0, horizon, "makespan")
        for i in range(n):
            model.Add(makespan >= ends[i])
        model.Minimize(makespan)
        # If incumbent provided and we want to *prove* prune, enforce ub (search for solution <= ub)
        if incumbent is not None and math.isfinite(incumbent):
            ub = max(0, int(math.floor(incumbent)) - 1)
            model.Add(makespan <= ub)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.num_search_workers = max(1, int(num_workers))
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL:
            return {"status": "OPTIMAL", "makespan": float(solver.Value(makespan))}
        elif status == cp_model.FEASIBLE:
            return {"status": "FEASIBLE", "makespan": float(solver.Value(makespan))}
        elif status == cp_model.INFEASIBLE:
            return {"status": "INFEASIBLE", "makespan": None}
        else:
            return {"status": "UNKNOWN", "makespan": None}
    except Exception as e:
        print("cp_verify error:", e, file=sys.stderr)
        return None

# --------------------------
# RF scoring
# --------------------------
def lower_bound(node: Node, remaining_proc_sum: float, machines: int, max_job_sum: float) -> float:
    load_lb = (sum(node.machine_loads) + remaining_proc_sum) / machines
    return max(node.makespan, load_lb, max_job_sum)

def rf_score(lb: float, incumbent: float, depth: int, time_left: float, per_node_est: float, alpha: float = 0.02) -> float:
    if not math.isfinite(incumbent):
        return 0.0
    slack = 0.0 if lb >= incumbent else max(0.0, (incumbent - lb) / max(1.0, incumbent))
    time_factor = time_left / (per_node_est + time_left) if (per_node_est + time_left) > 0 else 0.0
    return slack * time_factor / (1.0 + alpha * depth)

# --------------------------
# IO helpers
# --------------------------
def write_trace_header(path: str):
    first = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if first:
        w.writerow(TRACE_HEADER)
    return f, w

def flush_json_atomic(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def safe_zip(outdir: str, name: str = "outputs.zip"):
    zpath = os.path.join(outdir, name)
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(outdir):
            for fn in files:
                if fn == os.path.basename(zpath):
                    continue
                z.write(os.path.join(root, fn), arcname=os.path.relpath(os.path.join(root, fn), outdir))
    print("Zipped outputs to", zpath, flush=True)

# --------------------------
# Hybrid B&B driver
# --------------------------
def run_demo(ops_flat: List[Dict], machines: int, time_budget: float, tau: float, theta_prune: float,
             outdir: str, run_cpsat: bool, num_workers: int, no_greedy: bool,
             force_probe: bool = False, max_probe_fraction: float = 0.30, trace_sample_rate: int = 1):
    # clocks
    start_monotonic = time.monotonic()
    start_wall = time.time()

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "instance.json"), "w") as f:
        json.dump({"machines": machines, "ops_flat": ops_flat, "meta": {"start_wall": start_wall}}, f, indent=2)

    n = len(ops_flat)
    # cheap per-job sum LB helper
    job_sums = {}
    for op in ops_flat:
        job_sums.setdefault(op["job"], 0.0)
        job_sums[op["job"]] += op["proc_time"]
    max_job_sum = max(job_sums.values()) if job_sums else 0.0

    # preseed
    if not no_greedy:
        incumbent, _ = greedy_preseed_schedule(ops_flat, machines)
        print("Greedy preseed incumbent:", incumbent, flush=True)
    else:
        incumbent = float("inf")
        print("No greedy preseed; incumbent=inf", flush=True)

    root = Node(0, {}, tuple([0.0]*machines), {j:0.0 for j in set(op["job"] for op in ops_flat)}, 0.0, 0, n, 1)
    id_ctr = 1
    def next_id():
        nonlocal id_ctr; id_ctr += 1; return id_ctr

    remaining_total = sum(o["proc_time"] for o in ops_flat)
    heap = [(lower_bound(root, remaining_total, machines, max_job_sum), next_id(), root)]
    trace_path = os.path.join(outdir, "trace.csv")
    trace_file, trace_writer = write_trace_header(trace_path)

    # counters
    nodes_expanded = 0; nodes_pruned = 0; nodes_capped = 0
    probe_time_total = 0.0; probe_attempts = 0; probe_successes = 0; proof_prunes = 0
    ema = 0.01; node_counter = 0

    # optional forced root probe (helps demo)
    if force_probe and run_cpsat and HAS_ORTOOLS:
        t0 = time.monotonic()
        cp_res = cp_verify_fjsp(ops_flat, machines, {}, tau, incumbent, num_workers=num_workers)
        t_probe = time.monotonic() - t0
        probe_time_total += t_probe
        probe_attempts += 1
        if cp_res and cp_res.get("status") == "FEASIBLE" and cp_res.get("makespan") is not None and cp_res.get("makespan") + 1e-9 < incumbent:
            incumbent = float(cp_res.get("makespan"))
            probe_successes += 1
        print("Forced root probe:", cp_res, "time:", round(t_probe,3), "incumbent:", incumbent, flush=True)

    try:
        while heap:
            if shutdown_requested:
                print("Shutdown requested — stopping.", flush=True); break
            elapsed = time.monotonic() - start_monotonic
            if elapsed >= time_budget:
                print("Time budget exhausted.", flush=True); break
            cb, _, node = heapq.heappop(heap)
            node_start = time.monotonic()
            remaining_sum = sum(ops_flat[i]["proc_time"] for i in range(node.index, n))
            lb = lower_bound(node, remaining_sum, machines, max_job_sum)
            nodes_est = max(1, min(node.remaining, 200000))
            per_node = max(1e-6, ema)
            time_left = max(0.0, time_budget - (time.monotonic() - start_monotonic))
            time_factor = time_left / (per_node * nodes_est + time_left) if (per_node * nodes_est + time_left) > 0 else 0.0
            rf = rf_score(lb, incumbent, node.depth, time_left, per_node)
            tau_alloc = tau
            action = "consider"; verify_used = False; verify_time = 0.0; verify_status = ""

            # cheap LB prune
            if lb >= incumbent - 1e-9:
                action = "prune_by_lb"; nodes_pruned += 1
                # trace and continue
                trace_writer.writerow([time_wall(start_wall, start_monotonic), elapsed, node.id, node.index, node.depth, lb, incumbent,
                                       max(0.0,(incumbent-lb)/max(1.0,incumbent)), time_factor, rf, action, False, 0.0, "", 0.0])
                trace_file.flush()
                continue

            # RF gating
            if rf < theta_prune:
                # guard: don't probe if probes already used too much time
                if (probe_time_total / max(1e-9, max(1e-9, elapsed))) > max_probe_fraction:
                    action = "probe_skipped_max_fraction"
                else:
                    verify_used = True
                    probe_attempts += 1
                    t0 = time.monotonic()
                    # local probe first
                    local_best = local_probe_complete(ops_flat, node.partial, machines, min(0.4, tau_alloc*0.25))
                    verify_time = time.monotonic() - t0
                    probe_time_total += verify_time
                    if local_best is not None and local_best + 1e-9 < incumbent:
                        incumbent = float(local_best)
                        probe_successes += 1
                        verify_status = "local_improve"
                        action = "probe_found_better_local"
                    else:
                        # run CP verify if requested and available
                        if run_cpsat and HAS_ORTOOLS:
                            t1 = time.monotonic()
                            cp_res = cp_verify_fjsp(ops_flat, machines, node.partial, tau_alloc, incumbent, num_workers=num_workers)
                            vtime = time.monotonic() - t1
                            probe_time_total += vtime
                            verify_time += vtime
                            if cp_res is not None:
                                status = cp_res.get("status")
                                verify_status = status
                                if status == "INFEASIBLE":
                                    # proven prune: model unsat under makespan <= incumbent-1
                                    action = "prune_verified_infeasible"
                                    nodes_pruned += 1
                                    proof_prunes += 1
                                    # write trace and continue
                                    trace_writer.writerow([time_wall(start_wall, start_monotonic), time.monotonic()-start_monotonic, node.id, node.index, node.depth, lb, incumbent,
                                                           max(0.0,(incumbent-lb)/max(1.0,incumbent)), time_factor, rf, action, True, round(verify_time,6), verify_status, 0.0])
                                    trace_file.flush()
                                    continue
                                elif status == "FEASIBLE":
                                    ms = cp_res.get("makespan")
                                    if ms is not None and ms + 1e-9 < incumbent:
                                        incumbent = float(ms)
                                        probe_successes += 1
                                        action = "probe_found_better_cp"
                                    else:
                                        action = "probe_inconclusive_feasible"
                                else:
                                    action = "probe_inconclusive"
                            else:
                                action = "probe_failed"
                        else:
                            # no CP available
                            if local_best is None:
                                action = "probe_local_noresult"
                            else:
                                action = "probe_local_result"
            elif rf < 0.6:
                action = "cap"
                nodes_capped += 1
            else:
                action = "expand"

            # Expand node
            if action.startswith("prune"):
                trace_writer.writerow([time_wall(start_wall, start_monotonic), time.monotonic()-start_monotonic, node.id, node.index, node.depth, lb, incumbent,
                                       max(0.0,(incumbent-lb)/max(1.0,incumbent)), time_factor, rf, action, verify_used, round(verify_time,6), verify_status, 0.0])
                trace_file.flush()
                continue

            nodes_expanded += 1
            if node.index < n:
                j = node.index
                p = ops_flat[j]["proc_time"]
                order = sorted(ops_flat[j]["eligible"], key=lambda mm: node.machine_loads[mm])
                for mm in order:
                    if mm not in ops_flat[j]["eligible"]:
                        continue
                    new_partial = dict(node.partial); new_partial[ops_flat[j]["id"]] = mm
                    new_machine_loads = list(node.machine_loads)
                    new_job_ends = dict(node.job_ends)
                    start_time = max(new_machine_loads[mm], new_job_ends.get(ops_flat[j]["job"], 0.0))
                    finish = start_time + p
                    new_machine_loads[mm] = finish
                    new_job_ends[ops_flat[j]["job"]] = finish
                    new_mk = max(new_machine_loads)
                    child = Node(j+1, new_partial, tuple(new_machine_loads), new_job_ends, new_mk, node.depth+1, max(0, node.remaining-1), node.id*10 + (mm+1))
                    if child.index == n:
                        if child.makespan + 1e-9 < incumbent:
                            incumbent = float(child.makespan)
                    else:
                        cb = lower_bound(child, sum(ops_flat[i]["proc_time"] for i in range(child.index, n)), machines, max_job_sum)
                        heapq.heappush(heap, (cb, next_id(), child))

            node_dur = time.monotonic() - node_start
            ema = 0.9 * ema + 0.1 * node_dur
            node_counter += 1
            if node_counter % trace_sample_rate == 0:
                trace_writer.writerow([time_wall(start_wall, start_monotonic), time.monotonic()-start_monotonic, node.id, node.index, node.depth, lb, incumbent,
                                       max(0.0,(incumbent-lb)/max(1.0,incumbent)), time_factor, rf, action, verify_used, round(verify_time,6), verify_status, round(node_dur,6)])
                trace_file.flush()
    except Exception as e:
        print("Exception in main loop:", e, file=sys.stderr)
    finally:
        elapsed_total = time.monotonic() - start_monotonic
        summary = {
            "start_timestamp": start_wall,
            "end_timestamp": start_wall + elapsed_total,
            "elapsed": elapsed_total,
            "makespan": float(incumbent) if math.isfinite(incumbent) else None,
            "nodes_expanded": nodes_expanded,
            "nodes_pruned": nodes_pruned,
            "nodes_capped": nodes_capped,
            "probe_time_total": probe_time_total,
            "probe_attempts": probe_attempts,
            "probe_successes": probe_successes,
            "proof_prunes": proof_prunes
        }
        flush_json_atomic(os.path.join(outdir, "hybrid_summary.json"), summary)
        try:
            trace_file.close()
        except Exception:
            pass
        safe_zip(outdir)
        print("Final summary:", summary, flush=True)
        return summary

# --------------------------
# small helpers
# --------------------------
def time_wall(start_wall: float, start_monotonic: float) -> float:
    # compute a human-facing wall-clock timestamp for current monotonic time
    return start_wall + (time.monotonic() - start_monotonic)

# --------------------------
# CLI
# --------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RF FJSP demo (clean, conservative, monotonic timing)")
    p.add_argument("--machines", type=int, default=8)
    p.add_argument("--jobs", type=int, default=40)
    p.add_argument("--min_ops", type=int, default=3)
    p.add_argument("--max_ops", type=int, default=6)
    p.add_argument("--time", type=float, default=300.0)
    p.add_argument("--tau", type=float, default=1.2)
    p.add_argument("--theta_prune", type=float, default=0.06)
    p.add_argument("--outdir", type=str, default="outputs_fjsp_demo_clean")
    p.add_argument("--run_cpsat", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_greedy", action="store_true")
    p.add_argument("--force_probe", action="store_true", help="run one CP probe at root (demo)")
    p.add_argument("--max_probe_fraction", type=float, default=0.30)
    p.add_argument("--trace_sample_rate", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260107)
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    inst = generate_fjsp_instance(args.machines, args.jobs, args.min_ops, args.max_ops, seed=args.seed)
    ops_flat = inst["ops_flat"]
    print("OR-Tools available:", HAS_ORTOOLS, "run_cpsat flag:", args.run_cpsat, flush=True)
    if args.run_cpsat and not HAS_ORTOOLS:
        print("Warning: --run_cpsat requested but ortools unavailable. Install ortools and restart runtime if in notebooks.", flush=True)
    _ = run_demo(ops_flat, args.machines, args.time, args.tau, args.theta_prune,
                 args.outdir, args.run_cpsat, args.num_workers, args.no_greedy,
                 force_probe=args.force_probe, max_probe_fraction=args.max_probe_fraction,
                 trace_sample_rate=args.trace_sample_rate)

if __name__ == "__main__":
    main()