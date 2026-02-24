"""
Instance Generation and Export to Excel
=========================================

This script generates instances for multiple seeds and configuration combinations,
then exports all instance data to Excel files.

The user can specify:
- Seeds (default: 1-25)
- pttr variants (e.g., ['low', 'medium', 'high'])
- T variants (number of therapists, e.g., [2, 4, 6])
- D_focus variants (number of focus days, e.g., [4, 8, 12])

Features:
- Automatic refilling: If instances fail, new seeds are tried until target is reached
- Scenario numbering: Each (pttr, T, D_focus) combination gets scenario_nr from 1 to N
  Example: If you want 10 instances per combination, scenario_nr will be 1-10
           even if seeds are [1,2,3,4,5,6,7,8,10,11] (seed 9 failed)

All combinations are generated and saved to Excel.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from itertools import product
from CG import ColumnGeneration
from logging_config import setup_multi_level_logging, get_logger

# Setup logging
setup_multi_level_logging(base_log_dir='logs/instance_generation', enable_console=False, print_all_logs=False)
logger = get_logger(__name__)


class InstanceGenerator:
    """
    Class to generate and export instances to Excel.
    """

    def __init__(self, seeds, pttr_variants, T_variants, D_focus_variants, severity_mix_variants=None):
        """
        Initialize the instance generator.

        Args:
            seeds: List of seeds to test
            pttr_variants: List of pttr scenarios (e.g., ['low', 'medium', 'high'])
            T_variants: List of therapist counts (e.g., [2, 4, 6])
            D_focus_variants: List of focus day counts (e.g., [4, 8, 12])
            severity_mix_variants: List of tuples [(name, (E65A%, E65B%, E65C%))], 
                                  e.g., [('neuro', (0.7, 0.2, 0.1)), ('biasfree', (0.33, 0.34, 0.33))]
                                  If None, uses default distribution.
        """
        self.seeds = seeds
        self.pttr_variants = pttr_variants
        self.T_variants = T_variants
        self.D_focus_variants = D_focus_variants
        self.severity_mix_variants = severity_mix_variants if severity_mix_variants is not None else [(None, None)]

        # Default learning parameters (same as main.py)
        self.app_data = {
            'learn_type': ['sigmoid'],
            'theta_base': [0.3],
            'lin_increase': [0.05],
            'k_learn': [0.5],
            'infl_point': [5],
            'MS': [5],
            'MS_min': [2],
            'W_on': [5],
            'W_off': [2],
            'daily': [4]
        }

        self.dual_improvement_iter = 40
        self.dual_stagnation_threshold = 1e-6
        self.max_itr = 100
        self.threshold = 5e-6
        self.show_plots = False
        self.pricing_filtering = True
        self.therapist_agg = False
        self.learn_method = 'pwl'
        self.deterministic = False

        # Storage for all instances
        self.instances = []
        self.instance_details = {}
        self.failed_instances = []  # Track failed instance combinations

    def record_failed_instance(self, seed, pttr, T, D_focus, reason):
        """Record a failed instance generation."""
        self.failed_instances.append({
            'seed': seed,
            'pttr': pttr,
            'T': T,
            'D_focus': D_focus,
            'reason': reason
        })

    def generate_instance(self, seed, pttr, T, D_focus, severity_mix=None, severity_mix_name=None):
        """
        Generate a single instance with given parameters.

        Args:
            seed: Random seeds
            pttr: Patient-to-therapist ratio scenario
            T: Number of therapists
            D_focus: Number of focus days
            severity_mix: Tuple (E65A%, E65B%, E65C%) for custom distribution
            severity_mix_name: Name of severity mix (e.g., 'neuro', 'ortho')

        Returns:
            dict: Instance data
        """
        try:
            # Create CG solver to generate instance
            cg_solver = ColumnGeneration(
                seed=seed,
                app_data=self.app_data,
                T=T,
                D_focus=D_focus,
                max_itr=self.max_itr,
                threshold=self.threshold,
                pttr=pttr,
                show_plots=self.show_plots,
                pricing_filtering=self.pricing_filtering,
                therapist_agg=self.therapist_agg,
                max_stagnation_itr=self.dual_improvement_iter,
                stagnation_threshold=self.dual_stagnation_threshold,
                learn_method=self.learn_method,
                save_lps=False,
                verbose=False,
                deterministic=self.deterministic,
                severity_mix=severity_mix
            )

            # Setup instance (this generates all data)
            cg_solver.setup()

            # Check if instance is valid (has patients in at least one category)
            if len(cg_solver.P_F) == 0 and len(cg_solver.P_Post) == 0:
                logger.warning(f"Instance has no Focus or Post patients (seed={seed}, pttr={pttr}, T={T}, D={D_focus})")
                print(f"✗ NO PATIENTS (empty instance)")
                self.record_failed_instance(seed, pttr, T, D_focus, "No Focus or Post patients")
                return None

            # Check if pre_processing returned valid results
            if not isinstance(cg_solver.pre_x, dict):
                logger.warning(f"Pre-processing failed (seed={seed}, pttr={pttr}, T={T}, D={D_focus})")
                print(f"✗ PRE-PROCESSING FAILED")
                self.record_failed_instance(seed, pttr, T, D_focus, "Pre-processing failed")
                return None

            # Normalize severity_mix_name: None -> 'baseline'
            config_name = severity_mix_name if severity_mix_name is not None else 'baseline'

            # Create instance identifier
            instance_id = f"seed{seed}_pttr{pttr}_T{T}_D{D_focus}_{config_name}"

            # Extract all relevant instance data
            instance_data = {
                # ===== Configuration =====
                'instance_id': instance_id,
                'seed': seed,
                'pttr': pttr,
                'T_count': T,
                'D_focus_count': D_focus,
                'severity_mix': str(severity_mix) if severity_mix else None,
                'severity_mix_name': severity_mix_name,
                'config': config_name,  # 'baseline', 'neuro', or 'ortho'

                # ===== Learning Parameters =====
                'learn_type': self.app_data['learn_type'][0],
                'theta_base': self.app_data['theta_base'][0],
                'lin_increase': self.app_data['lin_increase'][0],
                'k_learn': self.app_data['k_learn'][0],
                'infl_point': self.app_data['infl_point'][0],
                'MS': self.app_data['MS'][0],
                'MS_min': self.app_data['MS_min'][0],
                'W_on': self.app_data['W_on'][0],
                'W_off': self.app_data['W_off'][0],
                'daily': self.app_data['daily'][0],

                # ===== Patient Counts =====
                'num_patients_total': len(cg_solver.P),
                'num_patients_full': len(cg_solver.Nr),
                'num_patients_pre': len(cg_solver.P_Pre),
                'num_patients_focus': len(cg_solver.P_F),
                'num_patients_post': len(cg_solver.P_Post),
                'num_patients_join': len(cg_solver.P_Join),

                # ===== Therapist Info =====
                'num_therapists': len(cg_solver.T),
                'num_therapist_groups': len(cg_solver.G_C),

                # ===== Horizon Info =====
                'D_length': len(cg_solver.D),
                'D_Ext_length': len(cg_solver.D_Ext),
                'D_Full_length': len(cg_solver.D_Full),

                # ===== Other =====
                'W_coeff': cg_solver.W_coeff,
                'M_p': cg_solver.M_p,
                'avg_req': sum(cg_solver.Req.values()) / len(cg_solver.Req) if cg_solver.Req else 0,
            }

            # Store detailed data separately (for Excel sheets)
            detail_key = instance_id
            self.instance_details[detail_key] = {
                # Lists
                'P': cg_solver.P,
                'Nr': cg_solver.Nr,
                'P_Pre': cg_solver.P_Pre,
                'P_F': cg_solver.P_F,
                'P_Post': cg_solver.P_Post,
                'P_Join': cg_solver.P_Join,
                'T': cg_solver.T,
                'G_C': cg_solver.G_C,
                'D': cg_solver.D,
                'D_Ext': cg_solver.D_Ext,
                'D_Full': cg_solver.D_Full,

                # Dictionaries (need to be converted for Excel)
                'Req': cg_solver.Req,
                'Req_agg': cg_solver.Req_agg,
                'Entry': cg_solver.Entry,
                'Entry_agg': cg_solver.Entry_agg,
                'Nr_agg': cg_solver.Nr_agg,
                'E_dict': cg_solver.E_dict,
                'Max_t': cg_solver.Max_t,
                'S_Bound': cg_solver.S_Bound,
                'therapist_to_type': cg_solver.therapist_to_type,
                'pre_x': cg_solver.pre_x,
                'pre_los': cg_solver.pre_los,
                'agg_to_patient': cg_solver.agg_to_patient,
            }

            return instance_data

        except ValueError as e:
            if "too many values to unpack" in str(e) or "not enough values to unpack" in str(e):
                logger.error(f"Unpacking error for seed={seed}, pttr={pttr}, T={T}, D={D_focus}: {str(e)}")
                print(f"✗ UNPACK ERROR (likely infeasible)")
                self.record_failed_instance(seed, pttr, T, D_focus, "Unpacking error (infeasible)")
                return None
            else:
                logger.error(f"ValueError generating instance (seed={seed}, pttr={pttr}, T={T}, D={D_focus}): {str(e)}")
                print(f"✗ ERROR: {str(e)}")
                self.record_failed_instance(seed, pttr, T, D_focus, f"ValueError: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error generating instance (seed={seed}, pttr={pttr}, T={T}, D={D_focus}): {str(e)}")
            print(f"✗ ERROR: {str(e)}")
            import traceback
            print(f"   Details: {traceback.format_exc()}")
            self.record_failed_instance(seed, pttr, T, D_focus, f"Exception: {str(e)}")
            return None

    def generate_all_instances(self):
        """
        Generate all instances for all combinations of parameters.
        Automatically refills failed combinations with new seeds.
        """
        print("\n" + "=" * 100)
        print(" INSTANCE GENERATION ".center(100, "="))
        print("=" * 100)

        target_seeds_per_combination = len(self.seeds)
        total_combinations = len(self.seeds) * len(self.pttr_variants) * len(self.T_variants) * len(self.D_focus_variants) * len(self.severity_mix_variants)

        print(f"\nConfiguration:")
        print(f"  - Seeds: {min(self.seeds)} to {max(self.seeds)} ({len(self.seeds)} seeds)")
        print(f"  - PTTR variants: {self.pttr_variants}")
        print(f"  - T variants: {self.T_variants}")
        print(f"  - D_focus variants: {self.D_focus_variants}")
        print(f"  - Severity mix variants: {[name if name else 'default' for name, _ in self.severity_mix_variants]}")
        print(f"  - Target instances per (T, D_focus, pttr, severity_mix): {target_seeds_per_combination}")
        print(f"  - Total combinations to attempt: {total_combinations}")
        print("\n" + "=" * 100 + "\n")

        # Track successful instances per combination
        from collections import defaultdict
        successful_per_combo = defaultdict(int)
        scenario_counter = defaultdict(int)  # Track scenario number per combination
        used_seeds = set(self.seeds)

        counter = 0
        successful = 0
        failed = 0

        # Generate all combinations (first round)
        print("Phase 1: Initial generation with specified seeds\n")
        for seed, pttr, T, D_focus, (severity_mix_name, severity_mix) in product(
            self.seeds, self.pttr_variants, self.T_variants, self.D_focus_variants, self.severity_mix_variants
        ):
            counter += 1
            combo_key = (pttr, T, D_focus, severity_mix_name)
            
            mix_str = f", mix={severity_mix_name if severity_mix_name else 'default'}"
            print(f"[{counter}/{total_combinations}] Generating: seed={seed}, pttr={pttr}, T={T}, D_focus={D_focus}{mix_str}...", end=" ")

            instance_data = self.generate_instance(seed, pttr, T, D_focus, severity_mix, severity_mix_name)

            if instance_data is not None:
                # Add scenario number
                scenario_counter[combo_key] += 1
                instance_data['scenario_nr'] = scenario_counter[combo_key]

                self.instances.append(instance_data)
                successful_per_combo[combo_key] += 1
                successful += 1
                print("✓")
            else:
                failed += 1
                print("✗")

        # Check which combinations need refilling
        print("\n" + "=" * 100)
        print(" PHASE 1 COMPLETE - CHECKING FOR REFILLS ".center(100, "="))
        print("=" * 100)
        print(f"\nPhase 1 Results:")
        print(f"  - Successfully generated: {successful}/{total_combinations}")
        print(f"  - Failed: {failed}/{total_combinations}")

        # Identify combinations that need more instances
        combos_to_refill = []
        for pttr, T, D_focus, (severity_mix_name, severity_mix) in product(
            self.pttr_variants, self.T_variants, self.D_focus_variants, self.severity_mix_variants
        ):
            combo_key = (pttr, T, D_focus, severity_mix_name)
            current_count = successful_per_combo[combo_key]
            if current_count < target_seeds_per_combination:
                missing = target_seeds_per_combination - current_count
                combos_to_refill.append((combo_key, severity_mix, missing))

        if combos_to_refill:
            print(f"\n{len(combos_to_refill)} combination(s) need refilling:")
            for (pttr, T, D_focus, severity_mix_name), severity_mix, missing in combos_to_refill:
                mix_str = severity_mix_name if severity_mix_name else 'default'
                print(f"  - pttr={pttr}, T={T}, D_focus={D_focus}, mix={mix_str}: needs {missing} more instance(s)")

            # Phase 2: Refill with new seeds
            print("\n" + "=" * 100)
            print(" PHASE 2: REFILLING FAILED COMBINATIONS ".center(100, "="))
            print("=" * 100 + "\n")

            next_seed = max(used_seeds) + 1
            refill_counter = 0
            refill_successful = 0
            refill_failed = 0

            for (pttr, T, D_focus, severity_mix_name), severity_mix, missing in combos_to_refill:
                max_attempts_per_combo = missing * 10  # Safety limit: try up to 10x the needed amount
                combo_key = (pttr, T, D_focus, severity_mix_name)
                mix_str = severity_mix_name if severity_mix_name else 'default'
                print(f"\nRefilling pttr={pttr}, T={T}, D_focus={D_focus}, mix={mix_str} (need {missing} more):")

                attempts = 0
                while successful_per_combo[combo_key] < target_seeds_per_combination and attempts < max_attempts_per_combo:
                    refill_counter += 1
                    attempts += 1
                    current_seed = next_seed
                    next_seed += 1
                    used_seeds.add(current_seed)

                    print(f"  [{attempts}] Trying seed={current_seed}...", end=" ")

                    instance_data = self.generate_instance(current_seed, pttr, T, D_focus, severity_mix, severity_mix_name)

                    if instance_data is not None:
                        # Add scenario number
                        scenario_counter[combo_key] += 1
                        instance_data['scenario_nr'] = scenario_counter[combo_key]

                        self.instances.append(instance_data)
                        successful_per_combo[combo_key] += 1
                        refill_successful += 1
                        successful += 1
                        print("✓")
                    else:
                        refill_failed += 1
                        failed += 1
                        print("✗")

                if successful_per_combo[combo_key] >= target_seeds_per_combination:
                    print(f"  ✓ Target reached ({successful_per_combo[combo_key]} instances)")
                else:
                    print(f"  ⚠ Warning: Could not reach target after {attempts} attempts ({successful_per_combo[combo_key]}/{target_seeds_per_combination})")

            print("\n" + "=" * 100)
            print(" PHASE 2 COMPLETE ".center(100, "="))
            print("=" * 100)
            print(f"\nPhase 2 Results:")
            print(f"  - Refill attempts: {refill_counter}")
            print(f"  - Successfully generated: {refill_successful}")
            print(f"  - Failed: {refill_failed}")
        else:
            print("\n✓ All combinations have sufficient instances - no refilling needed")

        print("\n" + "=" * 100)
        print(" FINAL SUMMARY ".center(100, "="))
        print("=" * 100)
        print(f"\nTotal Results:")
        print(f"  - Successfully generated: {successful}")
        print(f"  - Failed: {failed}")
        print(f"  - Total attempts: {successful + failed}")

        # Show distribution of instances per combination
        print(f"\nInstances per combination (scenario_nr: 1 to {target_seeds_per_combination}):")
        all_combos_balanced = True
        for pttr, T, D_focus in product(self.pttr_variants, self.T_variants, self.D_focus_variants):
            combo_key = (pttr, T, D_focus)
            count = successful_per_combo[combo_key]
            status = "✓" if count >= target_seeds_per_combination else "⚠"
            print(f"  {status} pttr={pttr}, T={T}, D_focus={D_focus}: {count}/{target_seeds_per_combination} instances (scenario_nr 1-{count})")
            if count < target_seeds_per_combination:
                all_combos_balanced = False

        if all_combos_balanced:
            print(f"\n✓ All combinations have {target_seeds_per_combination} instances - fully balanced!")
            print(f"   Each combination has scenario_nr from 1 to {target_seeds_per_combination}")
        else:
            print(f"\n⚠ Warning: Not all combinations reached the target of {target_seeds_per_combination} instances")

        # Print failed instances details
        if self.failed_instances:
            print("\n" + "-" * 100)
            print(" FAILED INSTANCES ".center(100, "-"))
            print("-" * 100)
            print(f"\nThe following {len(self.failed_instances)} instance combination(s) failed:\n")

            # Group by reason
            from collections import defaultdict
            failures_by_reason = defaultdict(list)
            for failure in self.failed_instances:
                failures_by_reason[failure['reason']].append(failure)

            for reason, failures in failures_by_reason.items():
                print(f"\n{reason} ({len(failures)} instances):")
                for f in failures:
                    print(f"  - seed={f['seed']}, pttr={f['pttr']}, T={f['T']}, D_focus={f['D_focus']}")

            print("\n" + "-" * 100)

        print("=" * 100 + "\n")

    def export_to_excel(self, filename=None):
        """
        Export all instances to Excel file (single sheet with all data).

        Args:
            filename: Output filename. If it starts with '/' or contains '/', 
                     it's treated as a full/relative path. Otherwise, defaults 
                     to 'results/instances/<filename>'.
        """
        if not self.instances:
            print("No instances to export!")
            return

        # Handle filename
        if filename is None:
            # Generate default filename with timestamp
            output_dir = 'results/instances'
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'{output_dir}/instances_{timestamp}.xlsx'
        elif '/' in filename:
            # User provided a path (relative or absolute)
            # Create parent directory if needed
            parent_dir = os.path.dirname(filename)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        else:
            # Just a filename - use default directory
            output_dir = 'results/instances'
            os.makedirs(output_dir, exist_ok=True)
            filename = f'{output_dir}/{filename}'

        print(f"\nExporting to Excel: {filename}")
        print("  - Preparing data...")

        # Create comprehensive data for each instance
        import json

        all_data = []
        for inst_data in self.instances:
            inst_id = inst_data['instance_id']
            details = self.instance_details[inst_id]

            # Combine basic config with detailed data
            row = {
                # ===== Configuration =====
                'instance_id': inst_id,
                'seed': inst_data['seed'],
                'scenario_nr': inst_data['scenario_nr'],
                'pttr': inst_data['pttr'],
                'T_count': inst_data['T_count'],
                'D_focus_count': inst_data['D_focus_count'],
                'config': inst_data.get('config', 'baseline'),  # 'baseline', 'neuro', or 'ortho'
                'severity_mix_name': inst_data.get('severity_mix_name'),

                # ===== Learning Parameters =====
                'learn_type': inst_data['learn_type'],
                'theta_base': inst_data['theta_base'],
                'lin_increase': inst_data['lin_increase'],
                'k_learn': inst_data['k_learn'],
                'infl_point': inst_data['infl_point'],
                'MS': inst_data['MS'],
                'MS_min': inst_data['MS_min'],
                'W_on': inst_data['W_on'],
                'W_off': inst_data['W_off'],
                'daily': inst_data['daily'],

                # ===== Patient Counts =====
                'num_patients_total': inst_data['num_patients_total'],
                'num_patients_full': inst_data['num_patients_full'],
                'num_patients_pre': inst_data['num_patients_pre'],
                'num_patients_focus': inst_data['num_patients_focus'],
                'num_patients_post': inst_data['num_patients_post'],
                'num_patients_join': inst_data['num_patients_join'],

                # ===== Therapist Info =====
                'num_therapists': inst_data['num_therapists'],
                'num_therapist_groups': inst_data['num_therapist_groups'],

                # ===== Horizon Info =====
                'D_length': inst_data['D_length'],
                'D_Ext_length': inst_data['D_Ext_length'],
                'D_Full_length': inst_data['D_Full_length'],

                # ===== Other =====
                'W_coeff': inst_data['W_coeff'],
                'M_p': inst_data['M_p'],
                'avg_req': inst_data['avg_req'],

                # ===== Lists (as JSON strings) =====
                'P': json.dumps(details['P']),
                'Nr': json.dumps(details['Nr']),
                'P_Pre': json.dumps(details['P_Pre']),
                'P_F': json.dumps(details['P_F']),
                'P_Post': json.dumps(details['P_Post']),
                'P_Join': json.dumps(details['P_Join']),
                'T': json.dumps(details['T']),
                'G_C': json.dumps(details['G_C']),
                'D': json.dumps(details['D']),
                'D_Ext': json.dumps(details['D_Ext']),
                'D_Full': json.dumps(details['D_Full']),

                # ===== Dictionaries (as JSON strings) =====
                'Req': json.dumps(details['Req']),
                'Req_agg': json.dumps(details['Req_agg']),
                'Entry': json.dumps(details['Entry']),
                'Entry_agg': json.dumps(details['Entry_agg']),
                'Nr_agg': json.dumps(details['Nr_agg']),
                'E_dict': json.dumps(details['E_dict']),
                'S_Bound': json.dumps(details['S_Bound']),
                'therapist_to_type': json.dumps(details['therapist_to_type']),
                'pre_los': json.dumps(details['pre_los']),
                'agg_to_patient': json.dumps(details['agg_to_patient']),

                # ===== Complex Dicts (convert tuples to strings for JSON) =====
                'Max_t': json.dumps({str(k): v for k, v in details['Max_t'].items()}),
                'pre_x': json.dumps({str(k): v for k, v in details['pre_x'].items()}),
            }

            all_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Export to Excel (single sheet)
        print("  - Writing to Excel...")
        df.to_excel(filename, sheet_name='Instances', index=False, engine='openpyxl')

        print(f"\n✓ Excel file saved: {filename}")
        print(f"  - Total instances: {len(df)}")
        print(f"  - Total columns: {len(df.columns)}")

        # Also save as pickle for full data preservation
        pickle_file = filename.replace('.xlsx', '.pkl')
        print(f"\nSaving full instance data to pickle: {pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'instances': self.instances,
                'instance_details': self.instance_details
            }, f)
        print(f"✓ Pickle file saved: {pickle_file}")

        return filename


def main():
    """
    Main function with interactive input.
    """
    print("\n" + "=" * 100)
    print(" INSTANCE GENERATION TOOL ".center(100, "="))
    print("=" * 100)
    print("\nThis tool generates instances for multiple seeds and configuration variants.")
    print("All instance data will be saved to Excel files.")
    print("\n" + "=" * 100)

    # ===== Get Seeds =====
    print("\n--- SEEDS ---")
    use_default_seeds = input("Use default seeds 1-25? (y/n): ").strip().lower()
    if use_default_seeds == 'y':
        seeds = list(range(1, 26))
    else:
        seed_input = input("Enter seeds (comma-separated, e.g., '1,2,3' or range '1-10'): ").strip()
        if '-' in seed_input and ',' not in seed_input:
            # Range input
            start, end = seed_input.split('-')
            seeds = list(range(int(start), int(end) + 1))
        else:
            # Comma-separated
            seeds = [int(s.strip()) for s in seed_input.split(',')]

    print(f"Seeds selected: {seeds}")

    # ===== Get PTTR Variants =====
    print("\n--- PTTR VARIANTS ---")
    print("Available: light, medium, heavy")
    pttr_input = input("Enter pttr variants (comma-separated, default='medium'): ").strip()
    if not pttr_input:
        pttr_variants = ['medium']
    else:
        pttr_variants = [p.strip() for p in pttr_input.split(',')]

    print(f"PTTR variants selected: {pttr_variants}")

    # ===== Get T Variants =====
    print("\n--- NUMBER OF THERAPISTS (T) ---")
    T_input = input("Enter T variants (comma-separated, e.g., '2,4,6', default='6'): ").strip()
    if not T_input:
        T_variants = [6]
    else:
        T_variants = [int(t.strip()) for t in T_input.split(',')]

    print(f"T variants selected: {T_variants}")

    # ===== Get D_focus Variants =====
    print("\n--- FOCUS DAYS (D_focus) ---")
    D_input = input("Enter D_focus variants (comma-separated, e.g., '4,8,12', default='30'): ").strip()
    if not D_input:
        D_focus_variants = [30]
    else:
        D_focus_variants = [int(d.strip()) for d in D_input.split(',')]

    print(f"D_focus variants selected: {D_focus_variants}")

    # ===== Confirm =====
    total = len(seeds) * len(pttr_variants) * len(T_variants) * len(D_focus_variants)
    print("\n" + "=" * 100)
    print(" CONFIGURATION SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"\nSeeds: {len(seeds)} ({min(seeds)} to {max(seeds)})")
    print(f"PTTR variants: {pttr_variants}")
    print(f"T variants: {T_variants}")
    print(f"D_focus variants: {D_focus_variants}")
    print(f"\nTotal instances to generate: {total}")
    print("=" * 100)

    confirm = input("\nProceed with generation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    # ===== Generate Instances =====
    generator = InstanceGenerator(seeds, pttr_variants, T_variants, D_focus_variants)
    generator.generate_all_instances()

    # ===== Export to Excel =====
    generator.export_to_excel()

    print("\n" + "=" * 100)
    print(" DONE ".center(100, "="))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
