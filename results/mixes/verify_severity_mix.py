#!/usr/bin/env python3
"""
Verification script for Patient Mix Sensitivity Study

This script tests the severity_mix parameter implementation by:
1. Generating instances with different severity mixes
2. Verifying DRG distribution accuracy
3. Verifying constant utilization across mixes
4. Checking normalization logic
"""

import sys
import os

# Get the directory of this script (mixes directory)
mixes_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (repository root)
repo_root = os.path.dirname(mixes_dir)

# Add repository root to path (for Utils module)
sys.path.insert(0, repo_root)
# Add mixes directory to path
sys.path.insert(0, mixes_dir)

# Now import from the mixes directory
from instance_setup import generate_patient_data_log
import numpy as np

# Define severity mixes to test
SEVERITY_MIXES = {
    'Φ_Neuro (High-Complexity)': (0.7, 0.2, 0.1),
    'Φ_Biasfree (High-Turnover)': (0.33, 0.34, 0.33),
    'Baseline (Default)': None
}

def verify_drg_distribution(DRG, severity_mix, tolerance=0.02):
    """
    Verify that the actual DRG distribution matches the expected severity_mix.
    
    Args:
        DRG: Dictionary mapping patient IDs to DRG codes
        severity_mix: Tuple (E65A%, E65B%, E65C%) or None for baseline
        tolerance: Maximum allowed deviation
        
    Returns:
        bool: True if distribution is within tolerance
    """
    from collections import Counter
    
    # Count actual DRG distribution
    drg_counts = Counter(DRG.values())
    total_patients = sum(drg_counts.values())
    
    actual_percentages = {
        'E65A': drg_counts.get('E65A', 0) / total_patients,
        'E65B': drg_counts.get('E65B', 0) / total_patients,
        'E65C': drg_counts.get('E65C', 0) / total_patients
    }
    
    print(f"\n  Actual distribution:")
    print(f"    E65A: {actual_percentages['E65A']:.1%} ({drg_counts.get('E65A', 0)} patients)")
    print(f"    E65B: {actual_percentages['E65B']:.1%} ({drg_counts.get('E65B', 0)} patients)")
    print(f"    E65C: {actual_percentages['E65C']:.1%} ({drg_counts.get('E65C', 0)} patients)")
    print(f"    Total: {total_patients} patients")
    
    if severity_mix is None:
        # Baseline distribution
        expected = {'E65A': 0.048, 'E65B': 0.278, 'E65C': 0.674}
    else:
        expected = {'E65A': severity_mix[0], 'E65B': severity_mix[1], 'E65C': severity_mix[2]}
    
    print(f"\n  Expected distribution:")
    print(f"    E65A: {expected['E65A']:.1%}")
    print(f"    E65B: {expected['E65B']:.1%}")
    print(f"    E65C: {expected['E65C']:.1%}")
    
    # Check if within tolerance
    within_tolerance = True
    print(f"\n  Deviations (tolerance = {tolerance:.1%}):")
    for drg in ['E65A', 'E65B', 'E65C']:
        deviation = abs(actual_percentages[drg] - expected[drg])
        status = "✓" if deviation <= tolerance else "✗"
        print(f"    {drg}: {deviation:.1%} {status}")
        if deviation > tolerance:
            within_tolerance = False
    
    return within_tolerance

def calculate_total_demand(Req, P):
    """
    Calculate total treatment demand (sum of all sessions).
    
    Args:
        Req: Dictionary mapping (patient, day) to required sessions
        P: List of patient IDs
        
    Returns:
        int: Total sessions required
    """
    total_sessions = sum(Req.get(p, 0) for p in P)
    return total_sessions

def verify_constant_utilization(results, tolerance=0.025):
    """
    Verify that utilization is constant across all severity mixes.
    
    Args:
        results: Dictionary mapping mix_name -> (total_demand, total_capacity, utilization)
        tolerance: Maximum allowed relative deviation (default 2.5% to account for stochastic variance)
        
    Returns:
        bool: True if utilization is constant within tolerance
    """
    utilizations = [data['utilization'] for data in results.values()]
    
    print("\n" + "=" * 80)
    print("UTILIZATION VERIFICATION".center(80))
    print("=" * 80)
    
    for mix_name, data in results.items():
        print(f"\n{mix_name}:")
        print(f"  Total demand:    {data['total_demand']:>6} sessions")
        print(f"  Total capacity:  {data['total_capacity']:>6} sessions")
        print(f"  Utilization:     {data['utilization']:>6.2%}")
        print(f"  Patient count:   {data['patient_count']:>6}")
        print(f"  Avg LOS:         {data['avg_los']:>6.2f} days")
    
    mean_util = np.mean(utilizations)
    std_util = np.std(utilizations)
    rel_std = std_util / mean_util if mean_util > 0 else 0
    
    print(f"\nUtilization statistics:")
    print(f"  Mean:           {mean_util:.4%}")
    print(f"  Std deviation:  {std_util:.6%}")
    print(f"  Relative std:   {rel_std:.6%}")
    print(f"  Tolerance:      {tolerance:.4%}")
    
    within_tolerance = rel_std <= tolerance
    status = "✓ PASS" if within_tolerance else "✗ FAIL"
    print(f"\nResult: {status}")
    
    return within_tolerance

def main():
    print("\n" + "=" * 80)
    print(" PATIENT MIX SENSITIVITY STUDY - VERIFICATION ".center(80, "="))
    print("=" * 80)
    
    # Test parameters
    seed = 42
    T = 10
    D_focus = 30
    pttr_scenario = 'medium'
    
    print(f"\nTest configuration:")
    print(f"  Seed:          {seed}")
    print(f"  Therapists:    {T}")
    print(f"  Focus days:    {D_focus}")
    print(f"  PTTR scenario: {pttr_scenario}")
    
    # Generate instances for each severity mix
    results = {}
    all_passed = True
    
    for mix_name, severity_mix in SEVERITY_MIXES.items():
        print("\n" + "=" * 80)
        print(f" {mix_name} ".center(80, "="))
        print("=" * 80)
        
        try:
            # Generate patient data
            Req, Entry, Max_t, P, D, D_Ext, D_Full, T_list, M_p, W_coeff, DRG = \
                generate_patient_data_log(
                    T=T,
                    D_focus=D_focus,
                    pttr_scenario=pttr_scenario,
                    seed=seed,
                    verbose=True,
                    severity_mix=severity_mix
                )
            
            # Verify DRG distribution
            print("\n" + "-" * 80)
            print("DRG DISTRIBUTION VERIFICATION")
            print("-" * 80)
            dist_passed = verify_drg_distribution(DRG, severity_mix, tolerance=0.02)
            
            if not dist_passed:
                all_passed = False
                print("\n  ✗ DRG distribution FAILED tolerance check")
            else:
                print("\n  ✓ DRG distribution passed")
            
            # Calculate metrics
            total_demand = sum(Req.values())
            
            # Calculate total capacity
            import math
            total_days = len(D_Full)
            working_days = math.ceil(total_days * W_coeff)
            daily_capacity = 4
            total_capacity = working_days * daily_capacity * T
            
            utilization = total_demand / total_capacity if total_capacity > 0 else 0
            
            # Calculate average LOS
            drg_data = {
                'E65A': {'los_mean': 17.9},
                'E65B': {'los_mean': 8.0},
                'E65C': {'los_mean': 6.1}
            }
            
            from collections import Counter
            drg_counts = Counter(DRG.values())
            total_patients = sum(drg_counts.values())
            
            avg_los = sum(
                (drg_counts.get(drg, 0) / total_patients) * drg_data[drg]['los_mean']
                for drg in ['E65A', 'E65B', 'E65C']
            )
            
            results[mix_name] = {
                'total_demand': total_demand,
                'total_capacity': total_capacity,
                'utilization': utilization,
                'patient_count': len(P),
                'avg_los': avg_los
            }
            
        except Exception as e:
            print(f"\n✗ ERROR generating instance for {mix_name}:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # Verify constant utilization
    if len(results) == len(SEVERITY_MIXES):
        util_passed =verify_constant_utilization(results, tolerance=0.025)
        if not util_passed:
            all_passed = False
    else:
        print("\n✗ Not all instances generated successfully, skipping utilization verification")
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 80)
    print(" VERIFICATION SUMMARY ".center(80, "="))
    print("=" * 80)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nThe severity_mix implementation is working correctly:")
        print("  - DRG distributions match target percentages")
        print("  - Utilization is constant across severity mixes")
        print("  - Volume normalization is functioning properly")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease review the output above for details.")
    
    print("\n" + "=" * 80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
