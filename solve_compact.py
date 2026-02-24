"""
Compact Model Solver - Batch Processing
========================================

This script loads all instances from the most recent instance Excel file
and solves each one using the Compact Model directly (without Column Generation or Branch-and-Price).

Results are saved to an Excel file with one row per instance containing:
- instance_id
- solve_time
- lower_bound
- upper_bound
- gap
- status
"""

import gurobipy as gu
import time
import pandas as pd
import json
import os
import glob
from datetime import datetime
from CG import ColumnGeneration
from logging_config import setup_multi_level_logging, get_logger

# Setup logging
print_all_logs = False
setup_multi_level_logging(base_log_dir='logs/compact', enable_console=True, print_all_logs=print_all_logs)
logger = get_logger(__name__)

# Time limit for each solve (in seconds)
TIME_LIMIT = 1200


def find_latest_instance_file():
    """
    Find the most recent instance Excel file.

    Returns:
        str: Path to the latest instance file
    """
    instance_dir = 'results/instances'
    pattern = os.path.join(instance_dir, 'instances_*.xlsx')
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No instance files found in {instance_dir}")

    # Sort by modification time, get most recent
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def load_instances_from_excel(excel_file):
    """
    Load all instances from Excel file.

    Args:
        excel_file: Path to Excel file

    Returns:
        pd.DataFrame: DataFrame with all instances
    """
    print(f"\nLoading instances from: {excel_file}")
    df = pd.read_excel(excel_file, sheet_name='Instances')
    print(f"  - Found {len(df)} instances")
    return df


def solve_instance_compact(instance_row, verbose=False):
    """
    Solve a single instance using the compact model.

    Args:
        instance_row: Pandas Series with instance data
        verbose: Print detailed output

    Returns:
        dict: Results dictionary with time, bounds, gap, status
    """
    try:
        # Extract configuration from instance row
        seed = int(instance_row['seed'])
        pttr = instance_row['pttr']
        T = int(instance_row['T_count'])
        D_focus = int(instance_row['D_focus_count'])

        # Extract learning parameters
        app_data = {
            'learn_type': [instance_row['learn_type']],
            'theta_base': [float(instance_row['theta_base'])],
            'lin_increase': [float(instance_row['lin_increase'])],
            'k_learn': [float(instance_row['k_learn'])],
            'infl_point': [float(instance_row['infl_point'])],
            'MS': [int(instance_row['MS'])],
            'MS_min': [int(instance_row['MS_min'])],
            'W_on': [int(instance_row['W_on'])],
            'W_off': [int(instance_row['W_off'])],
            'daily': [int(instance_row['daily'])]
        }

        # Create CG solver to generate instance (same as in instance generation)
        cg_solver = ColumnGeneration(
            seed=seed,
            app_data=app_data,
            T=T,
            D_focus=D_focus,
            max_itr=100,
            threshold=1e-5,
            pttr=pttr,
            show_plots=False,
            pricing_filtering=True,
            therapist_agg=False,
            max_stagnation_itr=20,
            stagnation_threshold=1e-5,
            learn_method='pwl',
            save_lps=False,
            verbose=False,
            deterministic=False
        )

        # Setup instance (creates compact model)
        cg_solver.setup()

        # Set Gurobi parameters
        cg_solver.problem.Model.setParam('TimeLimit', TIME_LIMIT)
        cg_solver.problem.Model.setParam('OutputFlag', 1)
        cg_solver.problem.Model.setParam('Threads', 1)  # Use single thread for deterministic results
        cg_solver.problem.Model.setParam('Presolve', 0)
        cg_solver.problem.Model.setParam('NoRelHeurTime', 10)
        cg_solver.problem.Model.setParam('MIPFocus', 1)
        cg_solver.problem.Model.setParam('Cuts', 1)  # No Cuts
        cg_solver.problem.Model.setParam('Heuristics', 0)  # Heuristics off
        cg_solver.problem.Model.setParam('RINS', 10)  # RINS off
        #cg_solver.problem.Model.setParam('NodeMethod', 0)  # Primal Simplex (langsamer)
        cg_solver.problem.Model.setParam('Symmetry', -1)  # Symmetry detection off
        #cg_solver.problem.Model.setParam('Aggregate', 0)  # Aggregation off
        cg_solver.problem.Model.setParam('ImproveStartTime', 1e100)

        # Solve compact model
        start_time = time.time()
        cg_solver.problem.solveModel()
        solve_time = time.time() - start_time

        # Extract results
        status = cg_solver.problem.Model.status

        if status == gu.GRB.OPTIMAL:
            return {
                'instance_id': instance_row['instance_id'],
                'seed': seed,
                'pttr': pttr,
                'T': T,
                'D_focus': D_focus,
                'solve_time': solve_time,
                'lower_bound': cg_solver.problem.Model.objVal,
                'upper_bound': cg_solver.problem.Model.objVal,
                'gap': 0.0,
                'status': 'OPTIMAL',
                'status_code': status
            }
        elif status == gu.GRB.TIME_LIMIT:
            # Best bound and best objective
            obj_val = cg_solver.problem.Model.ObjVal if cg_solver.problem.Model.SolCount > 0 else None
            obj_bound = cg_solver.problem.Model.ObjBound
            gap = cg_solver.problem.Model.MIPGap if cg_solver.problem.Model.SolCount > 0 else None

            return {
                'instance_id': instance_row['instance_id'],
                'seed': seed,
                'pttr': pttr,
                'T': T,
                'D_focus': D_focus,
                'solve_time': solve_time,
                'lower_bound': obj_bound,
                'upper_bound': obj_val,
                'gap': gap,
                'status': 'TIME_LIMIT',
                'status_code': status
            }
        elif status == gu.GRB.INFEASIBLE:
            return {
                'instance_id': instance_row['instance_id'],
                'seed': seed,
                'pttr': pttr,
                'T': T,
                'D_focus': D_focus,
                'solve_time': solve_time,
                'lower_bound': None,
                'upper_bound': None,
                'gap': None,
                'status': 'INFEASIBLE',
                'status_code': status
            }
        else:
            return {
                'instance_id': instance_row['instance_id'],
                'seed': seed,
                'pttr': pttr,
                'T': T,
                'D_focus': D_focus,
                'solve_time': solve_time,
                'lower_bound': None,
                'upper_bound': None,
                'gap': None,
                'status': f'STATUS_{status}',
                'status_code': status
            }

    except Exception as e:
        logger.error(f"Error solving instance {instance_row['instance_id']}: {str(e)}")
        return {
            'instance_id': instance_row['instance_id'],
            'seed': instance_row.get('seed'),
            'pttr': instance_row.get('pttr'),
            'T': instance_row.get('T_count'),
            'D_focus': instance_row.get('D_focus_count'),
            'solve_time': None,
            'lower_bound': None,
            'upper_bound': None,
            'gap': None,
            'status': 'ERROR',
            'status_code': None,
            'error': str(e)
        }


def main():
    """
    Main function to solve all instances using Compact Model.
    """
    logger.info("=" * 100)
    logger.info("STARTING COMPACT MODEL BATCH SOLVER")
    logger.info("=" * 100)

    print("\n" + "=" * 100)
    print(" COMPACT MODEL BATCH SOLVER ".center(100, "="))
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  - Time limit per instance: {TIME_LIMIT} seconds")
    print(f"  - Threads: 1 (single-threaded for deterministic results)")
    print("=" * 100 + "\n")

    # ===========================
    # FIND AND LOAD INSTANCES
    # ===========================

    try:
        # Find latest instance file
        instance_file = find_latest_instance_file()
        print(f"Using instance file: {instance_file}\n")

        # Load instances
        instances_df = load_instances_from_excel(instance_file)
        total_instances = len(instances_df)

    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("Please generate instances first using generate_instances.py")
        return None
    except Exception as e:
        print(f"\n✗ ERROR loading instances: {e}")
        return None

    # ===========================
    # FILTER INSTANCES
    # ===========================
    # Only solve specific parameter combinations
    FILTER_T = [3]
    FILTER_D_FOCUS = [5]
    FILTER_SEEDS = list(range(1,6))  # Seeds 1-10
    
    # Apply filters
    filtered_df = instances_df[
        (instances_df['T_count'].isin(FILTER_T)) &
        (instances_df['D_focus_count'].isin(FILTER_D_FOCUS)) &
        (instances_df['seed'].isin(FILTER_SEEDS))
    ]
    
    print(f"\n  - Filter: T ∈ {FILTER_T}, D_focus ∈ {FILTER_D_FOCUS}, seeds ∈ [1-10]")
    print(f"  - Filtered instances: {len(filtered_df)} (from {total_instances} total)")
    
    total_instances = len(filtered_df)
    instances_df = filtered_df
    
    # ===========================
    # SOLVE ALL INSTANCES
    # ===========================

    print("\n" + "=" * 100)
    print(" SOLVING INSTANCES ".center(100, "="))
    print("=" * 100 + "\n")

    results = []
    start_overall = time.time()

    for idx, instance_row in instances_df.iterrows():
        instance_id = instance_row['instance_id']
        
        print(f"[{idx+1}/{total_instances}] Solving {instance_id}...", end=" ", flush=True)

        # Solve instance
        result = solve_instance_compact(instance_row, verbose=False)
        results.append(result)

        # Print result
        if result['status'] == 'OPTIMAL':
            print(f"✓ OPTIMAL (obj={result['upper_bound']:.2f}, time={result['solve_time']:.2f}s)")
        elif result['status'] == 'TIME_LIMIT':
            gap_str = f"{result['gap']:.2%}" if result['gap'] is not None else "N/A"
            ub_str = f"{result['upper_bound']:.2f}" if result['upper_bound'] is not None else "N/A"
            lb_str = f"{result['lower_bound']:.2f}" if result['lower_bound'] is not None else "N/A"
            print(f"⏱ TIME_LIMIT (UB={ub_str}, LB={lb_str}, gap={gap_str}, time={result['solve_time']:.2f}s)")
        elif result['status'] == 'INFEASIBLE':
            print(f"✗ INFEASIBLE (time={result['solve_time']:.2f}s)")
        elif result['status'] == 'ERROR':
            print(f"✗ ERROR: {result.get('error', 'Unknown error')}")
        else:
            print(f"? {result['status']} (time={result['solve_time']:.2f}s)")

    total_time = time.time() - start_overall

    # ===========================
    # SAVE RESULTS
    # ===========================

    print("\n" + "=" * 100)
    print(" SAVING RESULTS ".center(100, "="))
    print("=" * 100 + "\n")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Create output directory
    output_dir = 'results/compact_solutions'
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'{output_dir}/compact_results_{timestamp}.xlsx'

    # Save to Excel
    results_df.to_excel(output_file, sheet_name='Results', index=False)

    print(f"✓ Results saved to: {output_file}")
    print(f"  - Total instances: {len(results_df)}")
    print(f"  - Total columns: {len(results_df.columns)}")

    # ===========================
    # SUMMARY STATISTICS
    # ===========================

    print("\n" + "=" * 100)
    print(" SUMMARY STATISTICS ".center(100, "="))
    print("=" * 100 + "\n")

    status_counts = results_df['status'].value_counts()
    print("Status Distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count} ({count/len(results_df)*100:.1f}%)")

    # Statistics for solved instances
    solved = results_df[results_df['status'].isin(['OPTIMAL', 'TIME_LIMIT'])]
    if len(solved) > 0:
        print(f"\nSolve Time Statistics (n={len(solved)}):")
        print(f"  Mean: {solved['solve_time'].mean():.2f}s")
        print(f"  Median: {solved['solve_time'].median():.2f}s")
        print(f"  Min: {solved['solve_time'].min():.2f}s")
        print(f"  Max: {solved['solve_time'].max():.2f}s")

    # Gap statistics for TIME_LIMIT instances
    tl_instances = results_df[results_df['status'] == 'TIME_LIMIT']
    if len(tl_instances) > 0:
        valid_gaps = tl_instances[tl_instances['gap'].notna()]
        if len(valid_gaps) > 0:
            print(f"\nGap Statistics for TIME_LIMIT instances (n={len(valid_gaps)}):")
            print(f"  Mean: {valid_gaps['gap'].mean():.2%}")
            print(f"  Median: {valid_gaps['gap'].median():.2%}")
            print(f"  Min: {valid_gaps['gap'].min():.2%}")
            print(f"  Max: {valid_gaps['gap'].max():.2%}")

    print(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("=" * 100 + "\n")

    logger.info("=" * 100)
    logger.info("COMPACT MODEL BATCH SOLVER FINISHED")
    logger.info("=" * 100)

    return results_df


if __name__ == "__main__":
    results = main()
