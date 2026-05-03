"""
Main Runner — Demonstrates the complete AIOps remediation pipeline.

Usage:
    python run_simulation.py

This script runs the full pipeline end-to-end and prints results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from config import get_default_config
from backend.pipeline import AIOpsRemediationPipeline


def main():
    print("=" * 70)
    print("  Adaptive Loss-Based Autonomous Remediation")
    print("  in Simulated AIOps Systems")
    print("=" * 70)
    print()

    # Initialize pipeline with default config
    config = get_default_config()
    pipeline = AIOpsRemediationPipeline(config)

    # Run simulation
    print("[1/10] Generating synthetic telemetry...")
    print("[2/10] Injecting realistic incidents...")
    print("[3/10] Running anomaly detection...")
    print("[4/10] Computing severity scores...")
    print("[5/10] Predicting failure probabilities...")
    print("[6/10] Estimating stability scores...")
    print("[7/10] Computing operational loss...")
    print("[8/10] Running adaptive optimization...")
    print("[9/10] Selecting and simulating remediations...")
    print("[10/10] Evaluating remediation effectiveness...")
    print()

    results = pipeline.run(
        num_services=5,
        duration_seconds=3600,
        with_incidents=True,
        run_optimization=True
    )

    # Print results
    print("-" * 70)
    print("  SIMULATION RESULTS")
    print("-" * 70)
    print(f"  Services simulated:      {results['num_services']}")
    print(f"  Incidents injected:      {results['num_incidents_injected']}")
    print(f"  Anomalies detected:      {results['total_anomalies_detected']}")
    print(f"  Remediations executed:   {results['total_remediations']}")
    print(f"  Avg cumulative loss:     {results['avg_cumulative_loss']:.4f}")
    print(f"  Max cumulative loss:     {results['max_cumulative_loss']:.4f}")
    print()

    # Evaluation metrics
    eval_m = results["evaluation_metrics"]
    print("-" * 70)
    print("  EVALUATION METRICS")
    print("-" * 70)
    print(f"  Loss Reduction Rate (LRR):         {eval_m.get('LRR', 0):.4f}")
    print(f"  Remediation Success Rate (RSR):     {eval_m.get('RSR', 0):.4f}")
    print(f"  False Remediation Rate (FRR):       {eval_m.get('FRR', 0):.4f}")
    print(f"  Stability Improvement Score (SIS):  {eval_m.get('SIS', 0):.4f}")
    print(f"  Mean Time To Recovery (MTTR):       {eval_m.get('MTTR', 0):.2f}s")
    print()

    # Optimizer state
    opt = results["optimizer_params"]
    print("-" * 70)
    print("  OPTIMIZED LOSS WEIGHTS")
    print("-" * 70)
    print(f"  alpha (severity):     {opt['alpha']:.4f}")
    print(f"  beta  (downtime):     {opt['beta']:.4f}")
    print(f"  gamma (failure):      {opt['gamma']:.4f}")
    print(f"  delta (stability):    {opt['delta']:.4f}")
    print(f"  Sum:                  {sum(opt.values()):.4f}")
    print()

    # Memory
    mem = results["memory_summary"]
    print("-" * 70)
    print("  OPERATIONAL MEMORY")
    print("-" * 70)
    print(f"  Total records:           {mem['total_records']}")
    print(f"  Unique services:         {mem['unique_services']}")
    print(f"  Remediations stored:     {mem['remediations_stored']}")
    print(f"  Successful remediations: {mem['successful_remediations']}")
    print()

    # Per-service summary
    print("-" * 70)
    print("  PER-SERVICE SUMMARY")
    print("-" * 70)
    full_state = pipeline.get_full_state()
    for svc, st in full_state["service_states"].items():
        print(f"\n  {svc}:")
        print(f"    Avg severity:    {st['avg_severity']:.4f}")
        print(f"    Max severity:    {st['max_severity']:.4f}")
        print(f"    Avg failure P:   {st['avg_failure_prob']:.4f}")
        print(f"    Avg stability:   {st['avg_stability']:.4f}")
        print(f"    Loss:            {st['cumulative_loss']:.4f}")
        print(f"    Downtime frac:   {st['downtime_fraction']:.4f}")
        print(f"    Anomalies:       {st['num_anomalies']}")
        print(f"    Remediations:    {st['num_remediations']}")

    print("\n" + "=" * 70)
    print("  Simulation complete. Use the API for interactive exploration:")
    print("    python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 7860")
    print("=" * 70)

    # Save full results to JSON
    output_path = os.path.join(os.path.dirname(__file__), "simulation_results.json")
    with open(output_path, "w") as f:
        json.dump(full_state, f, indent=2, default=str)
    print(f"\n  Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
