"""
Comparison Script: Branch-and-Price vs Compact Model
=====================================================

This script compares the solutions of Branch-and-Price and the Compact Model
across 100 different random seeds.

For each seed:
1. Solve the instance using Branch-and-Price (or Column Generation)
2. Solve the same instance using the Compact Model
3. Compare the objective values
4. Skip infeasible instances and continue with the next seed

Results are saved to a CSV file for analysis.
"""

import gurobipy as gu
import pandas as pd
import time
from datetime import datetime
from CG import ColumnGeneration
from branch_and_price import BranchAndPrice
from logging_config import setup_logging, get_logger
import os

# Setup logging
setup_logging(log_level='WARNING', log_to_file=True, log_dir='logs/comparison')
logger = get_logger(__name__)


class SeedComparison:
    """
    Class to compare Branch-and-Price and Compact Model solutions across multiple seeds.
    """

    def __init__(self, num_seeds=100, use_branch_and_price=False):
        """
        Initialize the comparison.

        Args:
            num_seeds: Number of different seeds to test
            use_branch_and_price: If True, use full Branch-and-Price; if False, use simple CG
        """
        self.num_seeds = num_seeds
        self.use_branch_and_price = use_branch_and_price
        self.results = []

        # Default configuration parameters
        self.app_data = {
            'learn_type': ['lin'],
            'theta_base': [0.02],
            'lin_increase': [0.01],
            'k_learn': [0.01],
            'infl_point': [2],
            'MS': [5],
            'MS_min': [2],
            'W_on': [6],
            'W_off': [1],
            'daily': [4]
        }

        self.T = 3  # Number of therapists
        self.D_focus = 5  # Number of focus days
        self.dual_improvement_iter = 20
        self.dual_stagnation_threshold = 1e-5
        self.max_itr = 100
        self.threshold = 1e-5
        self.pttr = 'medium'
        self.show_plots = False
        self.pricing_filtering = True
        self.therapist_agg = False
        self.learn_method = 'pwl'

        # Branch-and-Price settings
        self.branching_strategy = 'sp'
        self.search_strategy = 'bfs'

    def solve_with_cg(self, seed):
        """
        Solve instance using Column Generation.

        Args:
            seed: Random seed for instance generation

        Returns:
            dict: Result dictionary with objective value and status, or None if infeasible
        """
        try:
            # Create CG solver
            cg_solver = ColumnGeneration(
                seed=seed,
                app_data=self.app_data,
                T=self.T,
                D_focus=self.D_focus,
                max_itr=self.max_itr,
                threshold=self.threshold,
                pttr=self.pttr,
                show_plots=self.show_plots,
                pricing_filtering=self.pricing_filtering,
                therapist_agg=self.therapist_agg,
                max_stagnation_itr=self.dual_improvement_iter,
                stagnation_threshold=self.dual_stagnation_threshold,
                learn_method=self.learn_method
            )

            # Setup instance
            cg_solver.setup()

            # Solve Column Generation
            cg_solver.solve_cg()

            # Finalize: solve IP
            cg_solver.master.finSol()

            # Check if optimal solution was found
            if cg_solver.master.Model.status == gu.GRB.OPTIMAL:
                return {
                    'objective': cg_solver.master.Model.objVal,
                    'status': 'optimal',
                    'iterations': cg_solver.num_iterations,
                    'time': time.time() - cg_solver.start_time,
                    'is_integral': cg_solver.master.check_fractionality()[0]
                }
            elif cg_solver.master.Model.status == gu.GRB.INFEASIBLE:
                return None  # Skip infeasible instances
            else:
                return {
                    'objective': None,
                    'status': f'status_{cg_solver.master.Model.status}',
                    'iterations': cg_solver.num_iterations,
                    'time': time.time() - cg_solver.start_time,
                    'is_integral': False
                }

        except Exception as e:
            logger.error(f"Error solving with CG for seed {seed}: {str(e)}")
            return None

    def solve_with_bnp(self, seed):
        """
        Solve instance using Branch-and-Price.

        Args:
            seed: Random seed for instance generation

        Returns:
            dict: Result dictionary with objective value and status, or None if infeasible
        """
        try:
            # Create CG solver
            cg_solver = ColumnGeneration(
                seed=seed,
                app_data=self.app_data,
                T=self.T,
                D_focus=self.D_focus,
                max_itr=self.max_itr,
                threshold=self.threshold,
                pttr=self.pttr,
                show_plots=self.show_plots,
                pricing_filtering=self.pricing_filtering,
                therapist_agg=self.therapist_agg,
                max_stagnation_itr=self.dual_improvement_iter,
                stagnation_threshold=self.dual_stagnation_threshold,
                learn_method=self.learn_method
            )

            # Setup instance
            cg_solver.setup()

            # Initialize Branch-and-Price
            bnp_solver = BranchAndPrice(
                cg_solver,
                branching_strategy=self.branching_strategy,
                search_strategy=self.search_strategy,
                verbose=False,
                ip_heuristic_frequency=6,
                early_incumbent_iteration=1
            )

            # Solve
            results = bnp_solver.solve(time_limit=1800, max_nodes=100)

            # Check if solution was found
            if results['incumbent'] is not None:
                return {
                    'objective': results['incumbent'],
                    'status': 'optimal',
                    'nodes_explored': results['nodes_explored'],
                    'time': results['total_time'],
                    'is_integral': results['is_integral'],
                    'gap': results.get('gap', 0.0)
                }
            else:
                # Check if infeasible
                return None

        except Exception as e:
            logger.error(f"Error solving with B&P for seed {seed}: {str(e)}")
            return None

    def solve_with_compact(self, seed):
        """
        Solve instance using Compact Model.

        Args:
            seed: Random seed for instance generation (must use same instance as CG/BnP)

        Returns:
            dict: Result dictionary with objective value and status, or None if infeasible
        """
        try:
            # We need to create a CG solver just to get the instance data
            # but we won't solve the CG part
            cg_solver = ColumnGeneration(
                seed=seed,
                app_data=self.app_data,
                T=self.T,
                D_focus=self.D_focus,
                max_itr=self.max_itr,
                threshold=self.threshold,
                pttr=self.pttr,
                show_plots=False,
                pricing_filtering=self.pricing_filtering,
                therapist_agg=self.therapist_agg,
                max_stagnation_itr=self.dual_improvement_iter,
                stagnation_threshold=self.dual_stagnation_threshold,
                learn_method=self.learn_method
            )

            # Setup instance (this creates the compact model in cg_solver.problem)
            cg_solver.setup()

            # Solve compact model directly
            start_time = time.time()
            cg_solver.problem.solveModel()
            solve_time = time.time() - start_time

            # Check model status
            if cg_solver.problem.Model.status == gu.GRB.OPTIMAL:
                return {
                    'objective': cg_solver.problem.Model.objVal,
                    'status': 'optimal',
                    'time': solve_time
                }
            elif cg_solver.problem.Model.status == gu.GRB.INFEASIBLE:
                return None  # Skip infeasible instances
            else:
                return {
                    'objective': None,
                    'status': f'status_{cg_solver.problem.Model.status}',
                    'time': solve_time
                }

        except Exception as e:
            logger.error(f"Error solving with Compact for seed {seed}: {str(e)}")
            return None

    def run_comparison(self, start_seed=1):
        """
        Run comparison across all seeds.

        Args:
            start_seed: Starting seed value
        """
        print("\n" + "=" * 100)
        print(" SEED COMPARISON: Branch-and-Price vs Compact Model ".center(100, "="))
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  - Number of seeds: {self.num_seeds}")
        print(f"  - Starting seed: {start_seed}")
        print(f"  - Mode: {'Branch-and-Price' if self.use_branch_and_price else 'Column Generation'}")
        print(f"  - Therapists: {self.T}")
        print(f"  - Focus days: {self.D_focus}")
        print(f"  - Learning method: {self.learn_method}")
        print(f"  - PTTR scenario: {self.pttr}")
        print("\n" + "=" * 100 + "\n")

        total_start_time = time.time()
        seeds_processed = 0
        seeds_skipped = 0

        for i in range(self.num_seeds):
            seed = start_seed + i

            print(f"\n[{i+1}/{self.num_seeds}] Processing seed {seed}...")

            try:
                # Solve with CG or BnP
                if self.use_branch_and_price:
                    bnp_result = self.solve_with_bnp(seed)
                else:
                    bnp_result = self.solve_with_cg(seed)

                # Skip if infeasible
                if bnp_result is None:
                    print(f"  Seed {seed}: INFEASIBLE - Skipping")
                    seeds_skipped += 1
                    continue

                # Solve with Compact Model
                compact_result = self.solve_with_compact(seed)

                # Skip if compact is infeasible
                if compact_result is None:
                    print(f"  Seed {seed}: Compact INFEASIBLE - Skipping")
                    seeds_skipped += 1
                    continue

                # Compare results
                bnp_obj = bnp_result['objective']
                compact_obj = compact_result['objective']

                if bnp_obj is not None and compact_obj is not None:
                    abs_diff = abs(bnp_obj - compact_obj)
                    rel_diff = abs_diff / max(abs(compact_obj), 1e-6) if compact_obj != 0 else abs_diff
                    match = abs_diff < 1e-3  # Consider equal if difference < 0.001

                    result_entry = {
                        'seed': seed,
                        'bnp_objective': bnp_obj,
                        'compact_objective': compact_obj,
                        'absolute_difference': abs_diff,
                        'relative_difference': rel_diff,
                        'match': match,
                        'bnp_time': bnp_result['time'],
                        'compact_time': compact_result['time'],
                        'bnp_status': bnp_result['status'],
                        'compact_status': compact_result['status']
                    }

                    if self.use_branch_and_price:
                        result_entry['bnp_nodes'] = bnp_result.get('nodes_explored', 0)
                        result_entry['bnp_gap'] = bnp_result.get('gap', 0.0)
                    else:
                        result_entry['cg_iterations'] = bnp_result.get('iterations', 0)
                        result_entry['cg_integral'] = bnp_result.get('is_integral', False)

                    self.results.append(result_entry)
                    seeds_processed += 1

                    match_str = "✓ MATCH" if match else "✗ DIFFER"
                    print(f"  Seed {seed}: {match_str}")
                    print(f"    BnP/CG: {bnp_obj:.4f} | Compact: {compact_obj:.4f} | Diff: {abs_diff:.6f}")
                else:
                    print(f"  Seed {seed}: One or both solves failed")
                    seeds_skipped += 1

            except Exception as e:
                logger.error(f"Error processing seed {seed}: {str(e)}")
                print(f"  Seed {seed}: ERROR - {str(e)}")
                seeds_skipped += 1
                continue

        total_time = time.time() - total_start_time

        # Print summary
        print("\n" + "=" * 100)
        print(" COMPARISON SUMMARY ".center(100, "="))
        print("=" * 100)

        if self.results:
            df = pd.DataFrame(self.results)

            num_matches = df['match'].sum()
            num_total = len(df)
            match_rate = num_matches / num_total * 100 if num_total > 0 else 0

            print(f"\nTotal seeds attempted: {self.num_seeds}")
            print(f"Seeds successfully processed: {seeds_processed}")
            print(f"Seeds skipped (infeasible/error): {seeds_skipped}")
            print(f"\nMatching solutions: {num_matches} / {num_total} ({match_rate:.1f}%)")
            print(f"Non-matching solutions: {num_total - num_matches} / {num_total}")

            if num_total > num_matches:
                print(f"\nNon-matching seeds:")
                non_matching = df[~df['match']]
                for _, row in non_matching.iterrows():
                    print(f"  Seed {row['seed']}: BnP={row['bnp_objective']:.4f}, "
                          f"Compact={row['compact_objective']:.4f}, "
                          f"Diff={row['absolute_difference']:.6f}")

            print(f"\nAverage solve times:")
            print(f"  BnP/CG: {df['bnp_time'].mean():.2f}s")
            print(f"  Compact: {df['compact_time'].mean():.2f}s")

            print(f"\nTotal comparison time: {total_time:.2f}s")

            # Print detailed results table
            print("\n" + "=" * 100)
            print(" DETAILED RESULTS TABLE ".center(100, "="))
            print("=" * 100)
            print(f"\n{'Seed':<8} {'BnP/CG Obj':<15} {'Compact Obj':<15} {'Abs Diff':<15} {'Rel Diff (%)':<15} {'Match':<8}")
            print("-" * 100)
            for _, row in df.iterrows():
                match_symbol = "✓" if row['match'] else "✗"
                print(f"{row['seed']:<8} {row['bnp_objective']:<15.6f} {row['compact_objective']:<15.6f} "
                      f"{row['absolute_difference']:<15.6e} {row['relative_difference']*100:<15.6f} {match_symbol:<8}")
            print("=" * 100)

            # Save results to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.join(script_dir, '..', 'plots', 'results')
            os.makedirs(results_dir, exist_ok=True)

            method_name = 'bnp' if self.use_branch_and_price else 'cg'
            filename = os.path.join(results_dir, f'comparison_{method_name}_{timestamp}.csv')
            df.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")

        else:
            print("\nNo valid results to report (all instances were infeasible or failed).")

        print("=" * 100 + "\n")

        return self.results


def main():
    """
    Main function to run the comparison.
    """
    # Configuration
    NUM_SEEDS = 100
    START_SEED = 1
    USE_BRANCH_AND_PRICE = False  # Set to True to use full Branch-and-Price

    print("\n" + "=" * 100)
    print(" STARTING COMPARISON ".center(100, "="))
    print("=" * 100)

    # Create comparison object
    comparison = SeedComparison(
        num_seeds=NUM_SEEDS,
        use_branch_and_price=USE_BRANCH_AND_PRICE
    )

    # Run comparison
    results = comparison.run_comparison(start_seed=START_SEED)

    print("\n" + "=" * 100)
    print(" COMPARISON COMPLETE ".center(100, "="))
    print("=" * 100 + "\n")

    return results


if __name__ == "__main__":
    results = main()
