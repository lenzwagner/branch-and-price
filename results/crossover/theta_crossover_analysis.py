"""
Theta-Base Crossover Analysis
==============================
Determines the threshold of theta_base where the model with
T-1 Therapeuten + App is better than T therapists without App.

Procedure:
  1. Baseline: T Therapeuten, kein App (learn_type=0) → LOS_baseline
  2. Sweep:    T-1 Therapeuten + App, theta_base von 0 bis 1 in steps
               → LOS_challenger(theta)
  3. Crossover: first theta value where LOS_challenger <= LOS_baseline

Instance parameters are read from the newest Excel file in results/instances/.
"""

import argparse
import glob
import os
import sys

# Add root directory to path to allow importing from root modules
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pickle
from datetime import datetime

import pandas as pd

from logging_config import setup_multi_level_logging, get_logger
from loop_main_learning import solve_instance
from Utils.Generell.instance_setup import generate_patient_data_log

logger = get_logger(__name__)


# ============================================================
# Helper function: build pre_generated_data for T-1 therapists
# ============================================================

def build_reduced_pre_generated_data(base_data: dict, n_remove: int = 1) -> dict:
    """
    Builds a pre_generated_data dict for T-n therapists from the base dict.

    The last n therapists (highest IDs) are removed:
      - Max_t: only entries for remaining therapists
      - T:     List without removed therapists
      - All other fields (Req, Entry, P, D, D_Ext, D_Full, M_p, W_coeff, DRG)
        remain identical - the patient lists do NOT change.

    Pre-patients of the removed therapist are ignored during initialization
    since pre_processing_schedule() only uses the available
    therapists from Max_t.

    Args:
        base_data:  pre_generated_data of the baseline model (T therapists)
        n_remove:   Number of therapists to remove (Default: 1)

    Returns:
        New pre_generated_data dict with T-n therapists
    """
    full_T = base_data['T']                       # z.B. [1, 2, ..., 10]
    removed_T = set(full_T[-n_remove:])           # letzte n entfernen
    reduced_T = [t for t in full_T if t not in removed_T]

    # Max_t: only entries for remaining therapists behalten
    reduced_Max_t = {
        (t, d): v
        for (t, d), v in base_data['Max_t'].items()
        if t not in removed_T
    }

    reduced_data = {
        'Req':    base_data['Req'],
        'Entry':  base_data['Entry'],
        'Max_t':  reduced_Max_t,
        'P':      base_data['P'],
        'D':      base_data['D'],
        'D_Ext':  base_data['D_Ext'],
        'D_Full': base_data['D_Full'],
        'T':      reduced_T,
        'M_p':    base_data['M_p'],
        'W_coeff': base_data['W_coeff'],
        'DRG':    base_data['DRG'],
    }

    return reduced_data, removed_T


# ============================================================
# Main function
# ============================================================

def run_crossover_analysis(
    seed: int,
    D_focus: int,
    pttr: str,
    T: int,
    reduction: int,
    steps: int,
    learn_type: str,
    k_learn: float,
    infl_point: float,
    lin_increase: float,
    enable_grid: bool = False,
    k_learn_list: list = None
):
    """
    Executes the crossover analysis.

    Args:
        seed:         Random seed
        D_focus:      Number of focus days
        pttr:         Patient-to-Therapist-Ratio Szenario
        T:            Number of therapists (Baseline)
        reduction:    Number of therapists to remove
        steps:        Number of theta steps (0..1 in 1/steps increments)
        learn_type:   Learning curve type for Challenger ('sigmoid', 'exp', 'lin')
        k_learn:      k_learn parameter for Challenger
        infl_point:   infl_point parameter for Challenger
        lin_increase: lin_increase parameter for Challenger
        enable_grid:  If True, iterates over a list of k_learn
        k_learn_list: List of k_learn values for Grid-Search
    """
    if enable_grid and not k_learn_list:
        k_learn_list = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    elif not enable_grid:
        k_learn_list = [k_learn]

    T_challenger = T - reduction

    print("\n" + "=" * 100)
    print(" THETA-BASE CROSSOVER ANALYSIS ".center(100, "="))
    print("=" * 100)
    print(f"  Baseline:   T={T}, learn_type=0 (no App)")
    print(f"  Challenger: T={T_challenger}, learn_type={learn_type}")
    print(f"  Seed={seed}, D_focus={D_focus}, pttr={pttr}")
    if enable_grid:
        print(f"  Grid-Search : Active for k_learn={k_learn_list}")
    print(f"  Theta-Search: Binary search with 0.5% (0.005) granularity")
    print("=" * 100 + "\n")

    # ----------------------------------------------------------
    # 1. Instanzdaten einmalig generieren (T Therapeuten)
    # ----------------------------------------------------------
    print("[1/3] Generating instance data (T Therapeuten)...")
    Req, Entry, Max_t, P, D, D_planning, D_full, T_list, M_p, W_coeff, DRG = \
        generate_patient_data_log(
            T=T,
            D_focus=D_focus,
            pttr_scenario=pttr,
            seed=seed,
            verbose=False,
        )

    base_pre_generated_data = {
        'Req':    Req,
        'Entry':  Entry,
        'Max_t':  Max_t,
        'P':      P,
        'D':      D,
        'D_Ext':  D_planning,   # position 5: planning horizon (D_focus + D_max days)
        'D_Full': D_full,        # position 6: extended horizon (D_ext, includes pre-period)
        'T':      T_list,
        'M_p':    M_p,
        'W_coeff': W_coeff,
        'DRG':    DRG,
    }

    # ----------------------------------------------------------
    # 2. Solve Baseline (T therapists, no app)
    # ----------------------------------------------------------
    print("\n" + "─" * 100)
    print(" SOLVE: BASELINE ".center(100, "─"))
    print("─" * 100)
    print(f"  Model       : Baseline (no app)")
    print(f"  Therapists  : T = {T}")
    print(f"  App         : NO   (learn_type = 0)")
    print(f"  theta_base  : –")
    print(f"  D_focus     : {D_focus}")
    print(f"  PTTR        : {pttr}")
    print("─" * 100)

    # Base directory for LP files
    script_dir_lp = os.path.dirname(os.path.abspath(__file__))
    lp_base_dir = os.path.join(script_dir_lp, 'lp')
    os.makedirs(lp_base_dir, exist_ok=True)

    # Baseline LP path
    lp_path_baseline = os.path.join(lp_base_dir, f"baseline_T{T}")

    baseline_result = solve_instance(
        seed=seed,
        D_focus=D_focus,
        pttr=pttr,
        T=T,
        learn_type=0,
        app_data_overrides={
            'learn_type': 0,
            'theta_base': 0.0,
            'k_learn': 0.0,
            'lin_increase': 0.0,
            'infl_point': 0.0,
        },
        pre_generated_data=base_pre_generated_data,
        lp_output_path=lp_path_baseline,
    )

    LOS_baseline = baseline_result.get('final_ub')
    b_P_F   = baseline_result.get('P_F', [])
    b_P_Post = baseline_result.get('P_Post', [])
    b_P_Pre = baseline_result.get('P_Pre', [])
    b_pre_x = baseline_result.get('pre_x', {})
    
    # Filter pre_x to only include days >= 1
    b_pre_x_filtered = {
        p: {td: v for td, v in p_dict.items() if td[1] >= 1}
        for p, p_dict in b_pre_x.items()
        if any(td[1] >= 1 for td in p_dict.keys())
    }

    print(f"\n  ✓ Baseline solved:")
    print(f"    LOS          : {LOS_baseline}")
    print(f"    Pre   patients  ({len(b_P_Pre):>3}): {sorted(b_P_Pre)}")
    print(f"    Focus patients  ({len(b_P_F):>3}): {sorted(b_P_F)}")
    print(f"    Post  patients  ({len(b_P_Post):>3}): {sorted(b_P_Post)}")
    print(f"    pre_x (d >= 1)  : {b_pre_x_filtered}")

    if LOS_baseline is None:
        print("  ✗ Baseline could not be solved. Aborting.")
        return None

    # ----------------------------------------------------------
    # 3. Build reduced pre_generated_data for T-1 therapists
    # ----------------------------------------------------------
    print(f"\n[3/3] Building reduced model (T={T_challenger} therapists)...")
    reduced_pre_generated_data, removed_T = build_reduced_pre_generated_data(
        base_pre_generated_data, n_remove=reduction
    )
    print(f"  Removed therapists   : {sorted(removed_T)}")
    print(f"  Remaining therapists : {reduced_pre_generated_data['T']}")

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # 4. Theta Binary Search (Divide and Conquer)
    # ----------------------------------------------------------
    # We want to find the lowest theta in [0, 1] where LOS_challenger <= LOS_baseline.
    step_size = 0.005 # 0.5% steps
    max_idx = int(1.0 / step_size)
    precision = step_size

    sweep_results = []
    grid_results = []

    for current_k in k_learn_list:
        print(f"\n{'=' * 100}")
        print(f" THETA BINARY SEARCH FOR k_learn={current_k} (precision: {precision}) ".center(100, "="))
        print(f"{'=' * 100}")
        print(f"  {'theta_base':<15} {'LOS_challenger':<20} {'LOS_baseline':<20} {'Better?':<10}")
        print(f"  {'-'*15} {'-'*20} {'-'*20} {'-'*10}")

        crossover_theta = None
        left_idx = 0
        right_idx = max_idx
        step_nr = 0

        while left_idx <= right_idx:
            step_nr += 1
            mid_idx = (left_idx + right_idx) // 2
            theta = round(mid_idx * step_size, 3)
            
            left_theta_bound = round(left_idx * step_size, 3)
            right_theta_bound = round(right_idx * step_size, 3)
            
            print("\n" + "─" * 100)
            print(f" SOLVE: CHALLENGER STEP {step_nr} ".center(100, "─"))
            print("─" * 100)
            print(f"  Model       : Challenger (with app)")
            print(f"  Therapists  : T = {T_challenger}  (baseline T={T}, reduction={reduction})")
            print(f"  App         : YES  (learn_type = {learn_type})")
            print(f"  theta_base  : {theta:.3f}  (Search range: [{left_theta_bound:.3f}, {right_theta_bound:.3f}])")
            print(f"  k_learn     : {current_k}")
            print(f"  infl_point  : {infl_point}")
            print(f"  lin_increase: {lin_increase}")
            print(f"  D_focus     : {D_focus}")
            print(f"  PTTR        : {pttr}")
            print(f"  LOS_baseline: {LOS_baseline}  (target: LOS_challenger ≤ {LOS_baseline})")
            print("─" * 100)

            # Challenger LP path
            lp_path_challenger = os.path.join(lp_base_dir, f"challenger_T{T_challenger}_theta{theta:.3f}_k{current_k}")

            try:
                challenger_result = solve_instance(
                    seed=seed,
                    D_focus=D_focus,
                    pttr=pttr,
                    T=T_challenger,
                    learn_type=learn_type,
                    app_data_overrides={
                        'learn_type': learn_type,
                        'theta_base': theta,
                        'k_learn': current_k,
                        'infl_point': infl_point,
                        'lin_increase': lin_increase,
                    },
                    pre_generated_data=reduced_pre_generated_data,
                    lp_output_path=lp_path_challenger,
                    cutoff=LOS_baseline,
                )

                if challenger_result.get('cutoff_exceeded', False):
                    LOS_challenger = None
                    is_better = False
                    print(f"\n  Result:")
                    print(f"    LOS_challenger  : Cutoff (>{LOS_baseline})")
                    print(f"    LOS_baseline    : {LOS_baseline}")
                    print(f"    Better?         : NO ✗ (Early Cutoff)")
                else:
                    LOS_challenger = challenger_result.get('final_ub')
                    c_P_F    = challenger_result.get('P_F', [])
                    c_P_Post = challenger_result.get('P_Post', [])
                    c_P_Pre  = challenger_result.get('P_Pre', [])
                    c_pre_x  = challenger_result.get('pre_x', {})
                    is_better = (LOS_challenger is not None) and (LOS_challenger <= LOS_baseline)
                    
                    # Filter pre_x to only include days >= 1
                    c_pre_x_filtered = {
                        p: {td: v for td, v in p_dict.items() if td[1] >= 1}
                        for p, p_dict in c_pre_x.items()
                        if any(td[1] >= 1 for td in p_dict.keys())
                    }

                    print(f"\n  Result:")
                    print(f"    LOS_challenger  : {LOS_challenger}")
                    print(f"    LOS_baseline    : {LOS_baseline}")
                    print(f"    Better?         : {'YES ✓' if is_better else 'NO ✗'}")
                    print(f"    Pre   patients  ({len(c_P_Pre):>3}): {sorted(c_P_Pre)}")
                    print(f"    Focus patients  ({len(c_P_F):>3}): {sorted(c_P_F)}")
                    print(f"    Post  patients  ({len(c_P_Post):>3}): {sorted(c_P_Post)}")
                    print(f"    pre_x (d >= 1)  : {c_pre_x_filtered}")

                print(f"\n  {'theta_base':<15} {'LOS_challenger':<20} {'LOS_baseline':<20} {'Better?':<10}")
                print(f"  {theta:<15.3f} {str(LOS_challenger) if not challenger_result.get('cutoff_exceeded') else '> '+str(LOS_baseline):<20} {str(LOS_baseline):<20} {'YES ✓' if is_better else 'NO ✗':<10}")

                sweep_results.append({
                    'theta_base': theta,
                    'k_learn': current_k,
                    'LOS_baseline': LOS_baseline,
                    'LOS_challenger': LOS_challenger,
                    'cutoff_exceeded': challenger_result.get('cutoff_exceeded', False),
                    'is_better': is_better,
                    'T_baseline': T,
                    'T_challenger': T_challenger,
                    'learn_type': learn_type,
                    'seed': seed,
                    'D_focus': D_focus,
                    'pttr': pttr,
                    'is_optimal_challenger': challenger_result.get('is_optimal'),
                    'total_time_challenger': challenger_result.get('total_time'),
                    'is_optimal_baseline': baseline_result.get('is_optimal'),
                    'total_time_baseline': baseline_result.get('total_time'),
                })

                if is_better:
                    crossover_theta = theta
                    right_idx = mid_idx - 1
                    new_right_bound = round(right_idx * step_size, 3)
                    print(f"  🎯 App is better! Crossover is <= {theta:.3f}. Narrowing search to [{left_theta_bound:.3f}, {new_right_bound:.3f}].")
                else:
                    left_idx = mid_idx + 1
                    new_left_bound = round(left_idx * step_size, 3)
                    print(f"  ✗ App is worse. Crossover is > {theta:.3f}. Narrowing search to [{new_left_bound:.3f}, {right_theta_bound:.3f}].")

            except Exception as e:
                print(f"  ✗ Error at theta={theta:.3f}: {e}")
                logger.error(f"Challenger theta={theta} failed: {e}", exc_info=True)
                sweep_results.append({
                    'theta_base': theta,
                    'k_learn': current_k,
                    'LOS_baseline': LOS_baseline,
                    'LOS_challenger': None,
                    'is_better': False,
                    'T_baseline': T,
                    'T_challenger': T_challenger,
                    'learn_type': learn_type,
                    'seed': seed,
                    'D_focus': D_focus,
                    'pttr': pttr,
                    'error': str(e),
                })
                # Handle solver failure by assuming it's unfeasible/worse and needs higher theta
                left_idx = mid_idx + 1
                new_left_bound = round(left_idx * step_size, 3)
                print(f"  ⚠️ Error encountered, assuming we need higher theta. Narrowing search to [{new_left_bound:.3f}, {right_theta_bound:.3f}].")

        # ----------------------------------------------------------
        # 4b. Solve Baseline + App (T therapists, with crossover theta)
        # ----------------------------------------------------------
        LOS_baseline_app = None

        if crossover_theta is not None:
            print("\n" + "─" * 100)
            print(" SOLVE: BASELINE + APP (Verification) ".center(100, "─"))
            print("─" * 100)
            print(f"  Model       : Baseline + App")
            print(f"  Therapists  : T = {T}")
            print(f"  App         : YES  (learn_type = {learn_type})")
            print(f"  theta_base  : {crossover_theta:.3f} (Crossover Point)")
            print(f"  k_learn     : {current_k}")
            print(f"  infl_point  : {infl_point}")
            print(f"  lin_increase: {lin_increase}")
            print("─" * 100)

            # Baseline App LP path
            lp_path_baseline_app = os.path.join(lp_base_dir, f"baseline_app_T{T}_theta{crossover_theta:.3f}_k{current_k}")

            try:
                baseline_app_result = solve_instance(
                    seed=seed,
                    D_focus=D_focus,
                    pttr=pttr,
                    T=T,
                    learn_type=learn_type,
                    app_data_overrides={
                        'learn_type': learn_type,
                        'theta_base': crossover_theta,
                        'k_learn': current_k,
                        'infl_point': infl_point,
                        'lin_increase': lin_increase,
                    },
                    pre_generated_data=base_pre_generated_data,
                    lp_output_path=lp_path_baseline_app,
                    print_pre_x=True,
                )

                LOS_baseline_app = baseline_app_result.get('final_ub')
                ba_P_F    = baseline_app_result.get('P_F', [])
                ba_P_Post = baseline_app_result.get('P_Post', [])
                ba_P_Pre  = baseline_app_result.get('P_Pre', [])
                ba_pre_x  = baseline_app_result.get('pre_x', {})
                
                # Filter pre_x to only include days >= 1
                ba_pre_x_filtered = {
                    p: {td: v for td, v in p_dict.items() if td[1] >= 1}
                    for p, p_dict in ba_pre_x.items()
                    if any(td[1] >= 1 for td in p_dict.keys())
                }

                print(f"\n  Result:")
                print(f"    LOS_baseline_app: {LOS_baseline_app}")
                print(f"    LOS_baseline    : {LOS_baseline}") # T, no app
                print(f"    Pre   patients  ({len(ba_P_Pre):>3}): {sorted(ba_P_Pre)}")
                print(f"    Focus patients  ({len(ba_P_F):>3}): {sorted(ba_P_F)}")
                print(f"    Post  patients  ({len(ba_P_Post):>3}): {sorted(ba_P_Post)}")
                print(f"    pre_x (d >= 1)  : {ba_pre_x_filtered}")

            except Exception as e:
                print(f"  ✗ Error at Baseline + App run: {e}")
                logger.error(f"Baseline + App theta={crossover_theta} failed: {e}", exc_info=True)

        grid_results.append({
            'k_learn': current_k,
            'crossover_theta': crossover_theta,
            'LOS_baseline_app': LOS_baseline_app
        })

    # ----------------------------------------------------------
    # 5. Save results
    # ----------------------------------------------------------
    import math

    print("\n" + "=" * 100)
    print(" RESULT ".center(100, "="))
    print("=" * 100)

    # Analytical Threshold Prediction
    rho = T_challenger / T
    analytical_theta_req = 1 - rho
    print(f"  📊 Analytical Prediction:")
    print(f"     rho = T_challenger / T = {T_challenger} / {T} = {rho:.3f}")
    print(f"     Calculated Break-Even (theta_req) = 1 - rho = {analytical_theta_req:.3f}")
    print("-" * 100)

    # Calculation for Average Horizons (N_k)
    # Average across all patients in join set
    if len(base_pre_generated_data['P']) > 0:
        sum_N_k = 0
        total_patients = 0
        for p in base_pre_generated_data['P']:
            d_entry = base_pre_generated_data['Entry'].get(p, 0)
            d_max = max(base_pre_generated_data['D'])
            N_k = d_max - d_entry + 1
            if N_k > 0:
                sum_N_k += N_k
                total_patients += 1
        avg_N_k = sum_N_k / total_patients if total_patients > 0 else 0
    else:
        avg_N_k = 0

    print(f"  📈 Theoretical Equivalent Starting Proficiencies (for avg N_k = {avg_N_k:.2f} sessions):")

    for current_k in k_learn_list:

        # Linear
        theta_req_lin = analytical_theta_req - lin_increase * (avg_N_k - 1) / 2
        print(f"     * Linear     (rate={lin_increase:.3f}) : theta_req_0 = {theta_req_lin:.3f}")

        # Exponential
        # H(N, k) = (1 - e^(-k N)) / (1 - e^(-k))
        if current_k > 0:
            H_N_k = (1 - math.exp(-current_k * avg_N_k)) / (1 - math.exp(-current_k))
            theta_req_exp = 1 - (avg_N_k * (1 - analytical_theta_req)) / H_N_k
        else:
            theta_req_exp = analytical_theta_req
        print(f"     * Exponent.  (rate={current_k:.3f}) : theta_req_0 = {theta_req_exp:.3f}")

        # Sigmoidal
        # S(N, k, infl) = (1/N) * sum_{n=0}^{N-1} 1 / (1 + exp(-k * (n - infl)))
        S_N_k = 0
        n_limit = int(avg_N_k)
        for n in range(n_limit):
            S_N_k += 1 / (1 + math.exp(-current_k * (n - infl_point)))
        if avg_N_k > 0:
            S_N_k /= avg_N_k
        
        if (1 - S_N_k) != 0:
            theta_req_sig = (analytical_theta_req - S_N_k) / (1 - S_N_k)
        else:
            theta_req_sig = float('inf')
        print(f"     * Sigmoidal  (rate={current_k:.3f}, infl={infl_point}) : theta_req_0 = {theta_req_sig:.3f}")
        print("  " + "-" * 98)

    for res in grid_results:
        k_val = res['k_learn']
        ctheta = res['crossover_theta']
        lba = res['LOS_baseline_app']
        print(f"\n  [k_learn = {k_val}]")
        if ctheta is not None:
            print(f"  ✅ Crossover threshold: theta_base = {ctheta:.3f}")
            print(f"     From this value onwards, T={T_challenger} + App outperforms T={T} without App.")
        else:
            print(f"  ❌ No crossover found in range [0.0, 1.0].")

        print(f"  LOS_baseline (T={T}, no app): {LOS_baseline}")
        if lba is not None:
            print(f"  LOS_baseline_app (T={T}, app, theta={ctheta:.3f}): {lba}")
            if LOS_baseline is not None and LOS_baseline > 0:
                imp = (LOS_baseline - lba) / LOS_baseline * 100
                print(f"  → Improvement with App: {imp:.2f}%")
    
    print("=" * 100 + "\n")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Results go into crossover/results/ (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)

    results_df = pd.DataFrame(sweep_results)

    excel_path = os.path.join(out_dir, f"crossover_seed{seed}_{timestamp}.xlsx")
    pickle_path = os.path.join(out_dir, f"crossover_seed{seed}_{timestamp}.pkl")

    try:
        results_df.to_excel(excel_path, index=False)
        print(f"  ✓ Excel saved: {excel_path}")
    except Exception as e:
        print(f"  ✗ Excel error: {e}")

    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'sweep_results': sweep_results,
                'grid_results': grid_results,
                'LOS_baseline': LOS_baseline,
                'T': T,
                'T_challenger': T_challenger,
                'seed': seed,
                'D_focus': D_focus,
                'pttr': pttr,
                'learn_type': learn_type,
            }, f)
        print(f"  ✓ Pickle saved: {pickle_path}")
    except Exception as e:
        print(f"  ✗ Pickle error: {e}")

    return {
        'grid_results': grid_results,
        'LOS_baseline': LOS_baseline,
        'sweep_results': sweep_results,
    }


# ============================================================
# CLI
# ============================================================

def main():
    setup_multi_level_logging(base_log_dir='logs', enable_console=True, print_all_logs=False)

    parser = argparse.ArgumentParser(
        description="Theta-Base Crossover Analysis: Determines the threshold of theta_base, "
                    "ab dem T-1 Therapeuten + App is better than T therapists without App."
    )

    # Instanzparameter
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed (overrides Excel value)')
    parser.add_argument('--D_focus', type=int, default=None,
                        help='Number of focus days (overrides Excel value)')
    parser.add_argument('--pttr', type=str, default=None,
                        choices=['light', 'medium', 'heavy'],
                        help='PTTR-Szenario (overrides Excel value)')
    parser.add_argument('--T', type=int, default=None,
                        help='Number of baseline therapists (overrides Excel value)')

    # Sweep parameters
    parser.add_argument('--reduction', type=int, default=1,
                        help='Number of therapists to remove (default: 1)')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of theta steps from 0 to 1 (default: 20 → Δ=0.05)')

    # Challenger learning curve
    parser.add_argument('--learn_type', type=str, default='sigmoid',
                        choices=['sigmoid', 'exp', 'lin'],
                        help='Learning curve type for challenger (default: sigmoid)')
    parser.add_argument('--k_learn', type=float, default=1.5,
                        help='k_learn parameter (default: 1.5)')
    parser.add_argument('--infl_point', type=float, default=4.0,
                        help='infl_point parameter (default: 4.0)')
    parser.add_argument('--lin_increase', type=float, default=0.0,
                        help='lin_increase parameter (default: 0.0)')
    
    # ----------------------------------------------------
    # GRID SEARCH TOGGLE (Einfach hier True/False setzen!)
    # ----------------------------------------------------
    ENABLE_GRID_DEFAULT = True 
    K_LEARN_LIST_DEFAULT = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    
    parser.add_argument('--grid', action='store_true', default=ENABLE_GRID_DEFAULT,
                        help='Enable 2D Grid-Search over multiple k_learn values')
    parser.add_argument('--k_learn_list', nargs='+', type=float, default=K_LEARN_LIST_DEFAULT,
                        help='List of k_learn values for Grid-Search')

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Read instance parameters from Excel (if not provided via CLI)
    # ----------------------------------------------------------
    seed = args.seed
    D_focus = args.D_focus
    pttr = args.pttr
    T = args.T

    # Paths are always relative to the project root (parent of the crossover/ folder)
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)   # bnp-hybrid-scheduling/

    if any(v is None for v in [seed, D_focus, pttr, T]):
        instances_dir = os.path.join(script_dir, 'instances')
        excel_files = glob.glob(os.path.join(instances_dir, '*.xlsx'))

        if not excel_files:
            raise FileNotFoundError(
                f"No Excel files found in {instances_dir}.\n"
                f"Either place an instances Excel there, or pass "
                f"--seed, --D_focus, --pttr and --T as arguments."
            )

        newest_excel = max(excel_files, key=os.path.getmtime)
        print(f"Reading instance parameters from: {newest_excel}")
        scenarios_df = pd.read_excel(newest_excel)

        if scenarios_df.empty:
            raise ValueError("Excel file is empty.")

        # Use first row
        row = scenarios_df.iloc[0]

        if seed is None:
            seed = int(row.get('seed', 42))
        if D_focus is None:
            D_focus = int(row.get('D_focus', row.get('D_focus_count', 14)))
        if pttr is None:
            pttr = str(row.get('pttr', 'medium'))
        if T is None:
            if 'T_count' in row:
                T = int(row['T_count'])
            else:
                T_val = row.get('T', 10)
                try:
                    T = int(T_val)
                except ValueError:
                    import ast
                    T = len(ast.literal_eval(T_val))

        print(f"  Parameters read: seed={seed}, D_focus={D_focus}, pttr={pttr}, T={T}")

    # ----------------------------------------------------------
    # Run analysis
    # ----------------------------------------------------------
    # Change to project root so all internal imports and relative paths work
    os.chdir(project_root)

    run_crossover_analysis(
        seed=seed,
        D_focus=D_focus,
        pttr=pttr,
        T=T,
        reduction=args.reduction,
        steps=args.steps,
        learn_type=args.learn_type,
        k_learn=args.k_learn,
        infl_point=args.infl_point,
        lin_increase=args.lin_increase,
        enable_grid=args.grid,
        k_learn_list=args.k_learn_list,
    )


if __name__ == "__main__":
    main()
