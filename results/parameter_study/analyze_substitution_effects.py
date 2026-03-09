import pandas as pd
import numpy as np
import pickle
import glob
import os
import sys
import ast
from collections import defaultdict
import math

# Add root directory to sys.path to import Utils
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))  # .../bnp-hybrid-scheduling
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from Utils.Generell.instance_setup import generate_patient_data
except ImportError:
    # Fallback if running from root
    try:
        from Utils.Generell.instance_setup import generate_patient_data
    except Exception as e:
        print(f"Warning: Could not import generate_patient_data: {e}")
        generate_patient_data = None


def get_default_results_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming results are in 'results' subdirectory of the script's directory
    return os.path.join(current_dir, 'results')


def parse_dict_value(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return {}
    return {}


def load_results(results_dir=None):
    """
    Load results from pickle or Excel.
    """
    if results_dir is None:
        results_dir = get_default_results_dir()

    excel_files = glob.glob(os.path.join(results_dir, '*.xlsx'))
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~$')]

    pickle_files = glob.glob(os.path.join(results_dir, '*.pkl'))

    all_files = excel_files + pickle_files

    if all_files:
        newest_file = max(all_files, key=os.path.getmtime)
        print(f"Loading results from: {newest_file}")

        if newest_file.endswith('.pkl'):
            with open(newest_file, 'rb') as f:
                data_dict = pickle.load(f)
            return data_dict
        else:
            print(f"Loading from Excel: {newest_file}")
            df = pd.read_excel(newest_file)
            data_dict = {}
            for idx, row in df.iterrows():
                instance_id = row.get('instance_id', f'instance_{idx}')
                data_dict[instance_id] = row.to_dict()
            return data_dict

    raise FileNotFoundError(f"No data files found in {results_dir}")


def pair_instances(data_dict):
    """
    Pairs Human Only (Baseline) and Hybrid instances.
    Returns: dict mapping (seed, pttr, T, D_focus) -> {'human': instance, 'hybrid': instance}
    """
    paired_data = defaultdict(lambda: {'human': None, 'hybrid': None})

    for instance_id, instance in data_dict.items():
        if instance.get('status') == 'FAILED':
            continue

        seed = instance.get('seed')
        pttr = instance.get('pttr', 'medium').lower()
        if pttr == 'mp': pttr = 'medium'
        t_count = instance.get('T')
        d_focus = instance.get('D_focus')

        if seed is None or t_count is None or d_focus is None:
            continue

        key = (seed, pttr, t_count, d_focus)

        # Debug pairing: print sample key
        # if instance_id == 'seed11_pttrmedium_T10_D28_sigmoid':
        #    print(f"DEBUG: Key for {instance_id}: {key}")

        only_human = instance.get('OnlyHuman')
        if only_human == 1:
            paired_data[key]['human'] = instance
        else:
            paired_data[key]['hybrid'] = instance

    # Filter out incomplete pairs
    complete_pairs = {k: v for k, v in paired_data.items() if v['human'] and v['hybrid']}
    print(f"Found {len(complete_pairs)} complete pairs out of {len(paired_data)} potential combinations.")

    if len(complete_pairs) == 0:
        print("\nDEBUG: Pairing Failure Analysis")
        # Show first 5 keys with what they have
        for k, v in list(paired_data.items())[:5]:
            has_human = "YES" if v['human'] else "NO"
            has_hybrid = "YES" if v['hybrid'] else "NO"
            print(f"Key {k}: Human={has_human}, Hybrid={has_hybrid}")

    return complete_pairs


# --- Metric 1: Efficiency of Substitution ---
def calc_efficiency_of_substitution(complete_pairs):
    """
    Calculates Average Realized AI Effectiveness (theta_bar).
    """
    print("\n--- Calculating Efficiency of Substitution (AI Effectiveness) ---")

    # Store realized thetas by PTTR
    theta_by_pttr = defaultdict(list)

    # Learning parameters (hardcoded from instance_setup/app_data or extracted)
    # Ideally should read from instance, but standard is:
    # learn_type='sigmoid', theta_base=0.2, k_learn=0.25, infl_point=12
    # We will try to extract from instance if available, else default.

    for key, pair in complete_pairs.items():
        seed, pttr, t_count, d_focus = key
        hybrid = pair['hybrid']

        # Extract learning parameters if possible, else default
        # Assuming defaults for now as they are usually constant in the study
        theta_base = float(hybrid.get('theta_base', 0.3))
        k_learn = float(hybrid.get('k_learn', 1.52))
        infl_point = float(hybrid.get('infl_point', 4))  # Often 12 or 15

        def compute_theta(y_cum):
            return theta_base + (1 - theta_base) / (1 + math.exp(-k_learn * (y_cum - infl_point)))

        focus_y = parse_dict_value(hybrid.get('focus_y', {}))

        if not focus_y:
            continue

        # Group y by patient to calculate cumulative
        patient_y = defaultdict(list)
        for (pid, day), val in focus_y.items():
            if val > 0.5:
                patient_y[pid].append(day)

        for pid, days in patient_y.items():
            days.sort()
            y_cum = 0
            for day in days:
                y_cum += 1  # Pre-increment? or Post?
                # In loop_main, theta is calculated based on CURRENT Y_cumulative (which includes current session if we consider 'realized' effectiveness?)
                # Actually derived_vars.py says:
                # if y_it > 0: Y_cumulative += y_it
                # Y_it = Y_cumulative
                # theta_it = compute_theta(Y_it)
                # So yes, it uses the count INCLUDING the current one.

                theta = compute_theta(y_cum)
                theta_by_pttr[pttr].append(theta)

    # Aggregate
    results = {}
    for pttr, thetas in theta_by_pttr.items():
        if thetas:
            results[pttr] = {
                'mean_theta': np.mean(thetas),
                'count': len(thetas),
                'std': np.std(thetas)
            }

    return results


# --- Metric 2: Net Substitution Rate ---
def calc_net_substitution_rate(complete_pairs):
    """
    Calculates (Human_Base - Human_Hybrid) / AI_Hybrid
    """
    print("\n--- Calculating Net Substitution Rate ---")

    stats_by_pttr = defaultdict(list)

    for key, pair in complete_pairs.items():
        seed, pttr, t_count, d_focus = key
        human_inst = pair['human']
        hybrid_inst = pair['hybrid']

        # Get Human Sessions (Sum of focus_x)
        # focus_x might be stored as aggregated sum or detailed dict
        # In calc_statistics, 'focus_x' key usually has the count if it was aggregated in loop_main
        # Let's check keys. 'focus_x' in instance_data from loop_main is `agg_focus_x` (sum).

        # Get Human Sessions (Sum of focus_x)
        h_base_val = human_inst.get('focus_x')
        h_hybrid_val = hybrid_inst.get('focus_x')

        # Parse and sum if it's a dict/string dict
        def get_total_sessions(val):
            if isinstance(val, (int, float)):
                return float(val)
            parsed = parse_dict_value(val)
            # Sum values > 0.5 (assuming binary or fractional assignment)
            return sum(v for v in parsed.values() if v > 0.001)

        h_base_total = get_total_sessions(h_base_val)
        h_hybrid_total = get_total_sessions(h_hybrid_val)

        # Get AI Sessions from focus_y
        ai_hybrid = 0
        focus_y = parse_dict_value(hybrid_inst.get('focus_y', {}))
        for val in focus_y.values():
            if val > 0.5:
                ai_hybrid += 1

        params_delta_h = h_base_total - h_hybrid_total

        if ai_hybrid > 0:
            ratio = params_delta_h / ai_hybrid
            stats_by_pttr[pttr].append(ratio)
        elif ai_hybrid == 0 and params_delta_h == 0:
            # No AI used, no difference -> Neutral? Or skip?
            pass

    results = {}
    for pttr, ratios in stats_by_pttr.items():
        if ratios:
            results[pttr] = {
                'mean_ratio': np.mean(ratios),
                'count': len(ratios),
                'std': np.std(ratios)
            }
    return results


# --- Metric 3: Heterogeneous Impact ---
def calc_heterogeneous_impact(complete_pairs):
    """
    Calculates AI Share and Days Saved per DRG.
    """
    print("\n--- Calculating Heterogeneous Impact (Learning Dividend) ---")

    if generate_patient_data is None:
        print("Cannot generate patient data, skipping DRG analysis.")
        return {}

    drg_means = {17.9: 'E65A', 8.0: 'E65B', 6.1: 'E65C'}

    # Store per DRG: (baseline_los, hybrid_los, ai_share)
    # We want to aggregate across all instances (or per PTTR?)
    # Request implies general analysis, but maybe per PTTR is better.
    # Let's do Overall first as in the prompt example.

    drg_stats = defaultdict(lambda: {'base_los': [], 'hybrid_los': [], 'ai_share': []})

    for key, pair in complete_pairs.items():
        seed, pttr, t_count, d_focus = key  # Unpack key

        # Generate patient data to get DRGs
        try:
            # Suppress prints
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, _, _, _, _, _, _, M_p, _ = generate_patient_data(
                    T=t_count, D_focus=d_focus, pttr_scenario=pttr, seed=seed, plot_show=False
                )
        except Exception:
            continue

        hybrid = pair['hybrid']
        human = pair['human']

        focus_los_hyb = parse_dict_value(hybrid.get('focus_los', {}))
        focus_los_hum = parse_dict_value(human.get('focus_los', {}))
        focus_y_hyb = parse_dict_value(hybrid.get('focus_y', {}))

        for pid in focus_los_hyb.keys():  # Iterate hybrid patients (should be same as human)
            if pid not in M_p: continue

            mean_los = M_p[pid]
            drg = drg_means.get(mean_los)
            if not drg: continue

            los_hyb = focus_los_hyb.get(pid, 0)
            los_hum = focus_los_hum.get(pid, 0)

            if los_hyb == 0: continue  # Should not happen for valid patients

            # AI Share
            ai_count = 0
            for k, v in focus_y_hyb.items():
                if isinstance(k, tuple) and k[0] == pid and v > 0.5:
                    ai_count += 1

            share = ai_count / los_hyb

            drg_stats[drg]['base_los'].append(los_hum)
            drg_stats[drg]['hybrid_los'].append(los_hyb)
            drg_stats[drg]['ai_share'].append(share)

    results = {}
    for drg, data in drg_stats.items():
        count = len(data['base_los'])
        if count == 0: continue

        mean_base = np.mean(data['base_los'])
        mean_hyb = np.mean(data['hybrid_los'])
        mean_share = np.mean(data['ai_share'])
        days_saved = mean_base - mean_hyb

        results[drg] = {
            'mean_base_los': mean_base,
            'mean_share': mean_share,
            'days_saved': days_saved,
            'count': count
        }
    return results


def main():
    try:
        data = load_results()
        pairs = pair_instances(data)

        eff_stats = calc_efficiency_of_substitution(pairs)
        net_sub_stats = calc_net_substitution_rate(pairs)
        drg_impact = calc_heterogeneous_impact(pairs)

        # Report
        print("\n" + "=" * 80)
        print(" ANALYSIS OF SUBSTITUTION EFFECTS ".center(80, "="))
        print("=" * 80)

        print("\n1. Efficiency of Substitution (Average Realized Theta)")
        print(f"{'PTTR':<10} {'Mean Theta':<15} {'Std Dev':<15} {'Count (Sessions)':<10}")
        print("-" * 60)
        for pttr in sorted(eff_stats.keys()):
            s = eff_stats[pttr]
            print(f"{pttr:<10} {s['mean_theta']:<15.4f} {s['std']:<15.4f} {s['count']:<10}")

        print("\n2. Net Substitution Rate (Human Sessions Saved / AI Session)")
        print(f"{'PTTR':<10} {'Ratio':<15} {'Std Dev':<15} {'Count (Instances)':<10}")
        print("-" * 60)
        for pttr in sorted(net_sub_stats.keys()):
            s = net_sub_stats[pttr]
            print(f"{pttr:<10} {s['mean_ratio']:<15.4f} {s['std']:<15.4f} {s['count']:<10}")

        print("\n3. Heterogeneous Impact Analysis (The Learning Dividend)")
        print(f"{'Group':<10} {'Base LOS':<15} {'AI Share':<15} {'Days Saved':<15} {'N':<10}")
        print("-" * 75)
        for drg in sorted(drg_impact.keys()):
            s = drg_impact[drg]
            print(
                f"{drg:<10} {s['mean_base_los']:<15.4f} {s['mean_share']:<15.4%} {s['days_saved']:<15.4f} {s['count']:<10}")

        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
