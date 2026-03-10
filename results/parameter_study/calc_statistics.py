import pandas as pd
import numpy as np
import pickle
import glob
import os
from typing import Dict, List, Tuple
import ast


def load_latest_results(results_dir=None):
    """
    Load the latest results from either pickle or Excel file.
    Returns a dictionary with instance_id as keys.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    # Try pickle first (preferred as it preserves data types)
    pkl_files = glob.glob(os.path.join(results_dir, '*.pkl'))
    pkl_files = [f for f in pkl_files if not os.path.basename(f).startswith('~$')]

    if pkl_files:
        newest_pkl = max(pkl_files, key=os.path.getmtime)
        print(f"Loading from pickle: {newest_pkl}")
        with open(newest_pkl, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    # Fallback to Excel
    excel_files = glob.glob(os.path.join(results_dir, '*.xlsx'))
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~$')]

    if excel_files:
        newest_excel = max(excel_files, key=os.path.getmtime)
        print(f"Loading from Excel: {newest_excel}")
        df = pd.read_excel(newest_excel)
        # Convert to dict format
        data_dict = {}
        for idx, row in df.iterrows():
            instance_id = row.get('instance_id', f'instance_{idx}')
            data_dict[instance_id] = row.to_dict()
        return data_dict

    raise FileNotFoundError(f"No data files found in {results_dir}")


def parse_dict_value(value):
    """
    Parse a dictionary value that might be stored as a string.
    """
    if isinstance(value, dict):
        return value
    elif isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except:
            return {}
    else:
        return {}


def calc_ai_share(data_dict, verbose=True):
    """
    Calculate AI share statistics.

    For each seed:
        - For each focus patient, calculate: (count of theta > 0 where y_it = 1) / LOS
        - Average over all patients to get one value per seed

    Then calculate median, min, max, std over all seeds, separately for:
        - OnlyHuman = 1 (Human only)
        - OnlyHuman = 0 (Hybrid with AI)
        - Split by pttr (light, medium, heavy)

    Returns:
        dict with keys for each combination of OnlyHuman and pttr
    """
    # Organize data by seed, OnlyHuman, and pttr
    seed_data = {}

    for instance_id, instance in data_dict.items():
        # Skip failed instances
        if instance.get('status') == 'FAILED':
            continue

        seed = instance.get('seed')
        only_human = instance.get('OnlyHuman')
        pttr = instance.get('pttr', 'unknown')

        if seed is None or only_human is None:
            continue

        # Get focus_theta, focus_los, and focus_y
        focus_theta = parse_dict_value(instance.get('focus_theta', {}))
        focus_los = parse_dict_value(instance.get('focus_los', {}))
        focus_y = parse_dict_value(instance.get('focus_y', {}))

        if not focus_theta or not focus_los:
            continue

        # Calculate AI share for each patient in this instance
        patient_ai_shares = []

        for patient_id in focus_los.keys():
            los = focus_los.get(patient_id, 0)

            if los == 0:
                continue

            # Count theta > 0 for this patient ONLY when y_it = 1
            # focus_theta is a dict with keys like (patient, day)
            # focus_y is a dict with keys like (patient, day)
            theta_count = 0
            for key, theta_value in focus_theta.items():
                # Key format: (patient, day)
                if isinstance(key, tuple) and len(key) >= 2:
                    key_patient = key[0]
                    key_day = key[1]  # day is at index 1

                    if key_patient == patient_id and theta_value > 0:
                        # Check if y_it = 1 for this patient and day
                        y_key = (patient_id, key_day)
                        y_value = focus_y.get(y_key, 0)

                        # Only count if y = 1 (patient is present on that day)
                        if abs(y_value - 1.0) < 1e-6:
                            theta_count += 1

            # AI share for this patient
            ai_share_patient = theta_count / los
            patient_ai_shares.append(ai_share_patient)

        if not patient_ai_shares:
            continue

        # Average AI share over all patients for this seed
        avg_ai_share = np.mean(patient_ai_shares)

        # Store by (seed, only_human, pttr)
        key = (seed, only_human, pttr)
        if key not in seed_data:
            seed_data[key] = []
        seed_data[key].append(avg_ai_share)

    # Now group by OnlyHuman and pttr and calculate statistics
    grouped_values = {}

    for (seed, only_human, pttr), ai_shares in seed_data.items():
        # Take the mean if there are multiple values for the same seed/only_human/pttr
        avg_value = np.mean(ai_shares)

        # Normalize pttr names
        pttr_normalized = pttr.lower() if pttr else 'unknown'
        if pttr_normalized == 'mp':
            pttr_normalized = 'medium'

        # Create key for grouping
        group_key = (only_human, pttr_normalized)

        if group_key not in grouped_values:
            grouped_values[group_key] = []
        grouped_values[group_key].append(avg_value)

    # Calculate statistics for each group
    results = {}

    # Also calculate overall statistics (across all pttr)
    overall_human = []
    overall_hybrid = []

    for (only_human, pttr), values in grouped_values.items():
        if only_human == 1:
            overall_human.extend(values)
        elif only_human == 0:
            overall_hybrid.extend(values)

        group_name = f"{'human_only' if only_human == 1 else 'hybrid'}_{pttr}"

        if values:
            results[group_name] = {
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values),
                'mean': np.mean(values),
                'count': len(values),
                'values': values
            }

    # Add overall statistics
    if overall_human:
        results['human_only_overall'] = {
            'median': np.median(overall_human),
            'min': np.min(overall_human),
            'max': np.max(overall_human),
            'std': np.std(overall_human),
            'mean': np.mean(overall_human),
            'count': len(overall_human),
            'values': overall_human
        }

    if overall_hybrid:
        results['hybrid_overall'] = {
            'median': np.median(overall_hybrid),
            'min': np.min(overall_hybrid),
            'max': np.max(overall_hybrid),
            'std': np.std(overall_hybrid),
            'mean': np.mean(overall_hybrid),
            'count': len(overall_hybrid),
            'values': overall_hybrid
        }

    if verbose:
        print("\n" + "=" * 80)
        print(" AI SHARE STATISTICS ".center(80, "="))
        print("=" * 80)

        # Print overall statistics first
        if 'human_only_overall' in results:
            print("\n📊 HUMAN ONLY (OnlyHuman = 1) - OVERALL:")
            print(f"  Median:  {results['human_only_overall']['median']:.4f}")
            print(f"  Mean:    {results['human_only_overall']['mean']:.4f}")
            print(f"  Min:     {results['human_only_overall']['min']:.4f}")
            print(f"  Max:     {results['human_only_overall']['max']:.4f}")
            print(f"  Std Dev: {results['human_only_overall']['std']:.4f}")
            print(f"  Count:   {results['human_only_overall']['count']} instances")

        if 'hybrid_overall' in results:
            print("\n📊 HYBRID (OnlyHuman = 0) - OVERALL:")
            print(f"  Median:  {results['hybrid_overall']['median']:.4f}")
            print(f"  Mean:    {results['hybrid_overall']['mean']:.4f}")
            print(f"  Min:     {results['hybrid_overall']['min']:.4f}")
            print(f"  Max:     {results['hybrid_overall']['max']:.4f}")
            print(f"  Std Dev: {results['hybrid_overall']['std']:.4f}")
            print(f"  Count:   {results['hybrid_overall']['count']} instances")

        # Print by pttr
        print("\n" + "-" * 80)
        print(" BY PTTR (Patient Treatment Intensity) ".center(80, "-"))
        print("-" * 80)

        for pttr in ['light', 'medium', 'heavy']:
            print(f"\n--- {pttr.upper()} ---")

            human_key = f'human_only_{pttr}'
            if human_key in results:
                print(f"\n  HUMAN ONLY:")
                print(f"    Median:  {results[human_key]['median']:.4f}")
                print(f"    Mean:    {results[human_key]['mean']:.4f}")
                print(f"    Min:     {results[human_key]['min']:.4f}")
                print(f"    Max:     {results[human_key]['max']:.4f}")
                print(f"    Std Dev: {results[human_key]['std']:.4f}")
                print(f"    Count:   {results[human_key]['count']} instances")
            else:
                print(f"\n  HUMAN ONLY: No data")

            hybrid_key = f'hybrid_{pttr}'
            if hybrid_key in results:
                print(f"\n  HYBRID:")
                print(f"    Median:  {results[hybrid_key]['median']:.4f}")
                print(f"    Mean:    {results[hybrid_key]['mean']:.4f}")
                print(f"    Min:     {results[hybrid_key]['min']:.4f}")
                print(f"    Max:     {results[hybrid_key]['max']:.4f}")
                print(f"    Std Dev: {results[hybrid_key]['std']:.4f}")
                print(f"    Count:   {results[hybrid_key]['count']} instances")
            else:
                print(f"\n  HYBRID: No data")

        print("\n" + "=" * 80 + "\n")

    return results


def calc_therapist_utilization(data_dict, verbose=True):
    """
    Calculate therapist utilization statistics from focus_therapist_daily_util_dict.

    For each therapist:
        - Calculate average utilization over days where therapist is present (util > 0)
        - Average over all seeds

    Then calculate median, min, max, std over all seeds, separately for:
        - OnlyHuman = 1 (Human only)
        - OnlyHuman = 0 (Hybrid with AI)
        - Split by pttr (light, medium, heavy)

    Also calculate global utilization (across all therapists).

    Returns:
        dict with keys for each combination of OnlyHuman and pttr
    """
    # Organize data by seed, OnlyHuman, and pttr
    seed_data = {}

    for instance_id, instance in data_dict.items():
        # Skip failed instances
        if instance.get('status') == 'FAILED':
            continue

        seed = instance.get('seed')
        only_human = instance.get('OnlyHuman')
        pttr = instance.get('pttr', 'unknown')

        if seed is None or only_human is None:
            continue

        # Get focus_therapist_daily_util_dict
        therapist_daily_util = parse_dict_value(instance.get('focus_therapist_daily_util_dict', {}))

        if not therapist_daily_util:
            continue

        # Calculate average utilization per therapist (only days > 0)
        therapist_avg_utils = []

        for therapist_id, daily_utils in therapist_daily_util.items():
            # Filter days where therapist is present (util > 0)
            present_days = {day: util for day, util in daily_utils.items() if util > 0}

            if present_days:
                avg_util = np.mean(list(present_days.values()))
                therapist_avg_utils.append(avg_util)

        if not therapist_avg_utils:
            continue

        # Average utilization across all therapists in this instance
        instance_avg_util = np.mean(therapist_avg_utils)

        # Store by (seed, only_human, pttr)
        key = (seed, only_human, pttr)
        if key not in seed_data:
            seed_data[key] = []
        seed_data[key].append(instance_avg_util)

    # Now group by OnlyHuman and pttr and calculate statistics
    grouped_values = {}

    for (seed, only_human, pttr), utils in seed_data.items():
        # Take the mean if there are multiple values for the same seed/only_human/pttr
        avg_value = np.mean(utils)

        # Normalize pttr names
        pttr_normalized = pttr.lower() if pttr else 'unknown'
        if pttr_normalized == 'mp':
            pttr_normalized = 'medium'

        # Create key for grouping
        group_key = (only_human, pttr_normalized)

        if group_key not in grouped_values:
            grouped_values[group_key] = []
        grouped_values[group_key].append(avg_value)

    # Calculate statistics for each group
    results = {}

    # Also calculate overall statistics (across all pttr)
    overall_human = []
    overall_hybrid = []

    for (only_human, pttr), values in grouped_values.items():
        if only_human == 1:
            overall_human.extend(values)
        elif only_human == 0:
            overall_hybrid.extend(values)

        group_name = f"{'human_only' if only_human == 1 else 'hybrid'}_{pttr}"

        if values:
            results[group_name] = {
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values),
                'mean': np.mean(values),
                'count': len(values),
                'values': values
            }

    # Add overall statistics
    if overall_human:
        results['human_only_overall'] = {
            'median': np.median(overall_human),
            'min': np.min(overall_human),
            'max': np.max(overall_human),
            'std': np.std(overall_human),
            'mean': np.mean(overall_human),
            'count': len(overall_human),
            'values': overall_human
        }

    if overall_hybrid:
        results['hybrid_overall'] = {
            'median': np.median(overall_hybrid),
            'min': np.min(overall_hybrid),
            'max': np.max(overall_hybrid),
            'std': np.std(overall_hybrid),
            'mean': np.mean(overall_hybrid),
            'count': len(overall_hybrid),
            'values': overall_hybrid
        }

    if verbose:
        print("\n" + "=" * 80)
        print(" THERAPIST UTILIZATION STATISTICS ".center(80, "="))
        print("=" * 80)

        # Print overall statistics first
        if 'human_only_overall' in results:
            print("\n📊 HUMAN ONLY (OnlyHuman = 1) - OVERALL:")
            print(f"  Median:  {results['human_only_overall']['median']:.2f}%")
            print(f"  Mean:    {results['human_only_overall']['mean']:.2f}%")
            print(f"  Min:     {results['human_only_overall']['min']:.2f}%")
            print(f"  Max:     {results['human_only_overall']['max']:.2f}%")
            print(f"  Std Dev: {results['human_only_overall']['std']:.2f}%")
            print(f"  Count:   {results['human_only_overall']['count']} instances")

        if 'hybrid_overall' in results:
            print("\n📊 HYBRID (OnlyHuman = 0) - OVERALL:")
            print(f"  Median:  {results['hybrid_overall']['median']:.2f}%")
            print(f"  Mean:    {results['hybrid_overall']['mean']:.2f}%")
            print(f"  Min:     {results['hybrid_overall']['min']:.2f}%")
            print(f"  Max:     {results['hybrid_overall']['max']:.2f}%")
            print(f"  Std Dev: {results['hybrid_overall']['std']:.2f}%")
            print(f"  Count:   {results['hybrid_overall']['count']} instances")

        # Print by pttr
        print("\n" + "-" * 80)
        print(" BY PTTR (Patient Treatment Intensity) ".center(80, "-"))
        print("-" * 80)

        for pttr in ['light', 'medium', 'heavy']:
            print(f"\n--- {pttr.upper()} ---")

            human_key = f'human_only_{pttr}'
            if human_key in results:
                print(f"\n  HUMAN ONLY:")
                print(f"    Median:  {results[human_key]['median']:.2f}%")
                print(f"    Mean:    {results[human_key]['mean']:.2f}%")
                print(f"    Min:     {results[human_key]['min']:.2f}%")
                print(f"    Max:     {results[human_key]['max']:.2f}%")
                print(f"    Std Dev: {results[human_key]['std']:.2f}%")
                print(f"    Count:   {results[human_key]['count']} instances")
            else:
                print(f"\n  HUMAN ONLY: No data")

            hybrid_key = f'hybrid_{pttr}'
            if hybrid_key in results:
                print(f"\n  HYBRID:")
                print(f"    Median:  {results[hybrid_key]['median']:.2f}%")
                print(f"    Mean:    {results[hybrid_key]['mean']:.2f}%")
                print(f"    Min:     {results[hybrid_key]['min']:.2f}%")
                print(f"    Max:     {results[hybrid_key]['max']:.2f}%")
                print(f"    Std Dev: {results[hybrid_key]['std']:.2f}%")
                print(f"    Count:   {results[hybrid_key]['count']} instances")
            else:
                print(f"\n  HYBRID: No data")

        print("\n" + "=" * 80 + "\n")

    return results


def calc_therapist_max_utilization(data_dict, verbose=True):
    """
    Calculate MAXIMUM therapist utilization statistics.

    For each instance:
        - Find the maximum utilization across all therapists and all days

    Then calculate median, min, max, std over all seeds, separately for:
        - OnlyHuman = 1 (Human only)
        - OnlyHuman = 0 (Hybrid with AI)
        - Split by pttr (light, medium, heavy)

    Returns:
        dict with max_util statistics for each combination
    """
    # Organize data by seed, OnlyHuman, and pttr
    seed_data = {}

    for instance_id, instance in data_dict.items():
        # Skip failed instances
        if instance.get('status') == 'FAILED':
            continue

        seed = instance.get('seed')
        only_human = instance.get('OnlyHuman')
        pttr = instance.get('pttr', 'unknown')

        if seed is None or only_human is None:
            continue

        # Get focus_therapist_daily_util_dict
        therapist_daily_util = parse_dict_value(instance.get('focus_therapist_daily_util_dict', {}))

        if not therapist_daily_util:
            continue

        # Find maximum utilization across ALL therapists and ALL days
        all_utils = []

        for therapist_id, daily_utils in therapist_daily_util.items():
            # Get all utilization values > 0
            present_days_utils = [util for util in daily_utils.values() if util > 0]
            all_utils.extend(present_days_utils)

        if not all_utils:
            continue

        # Maximum utilization in this instance
        instance_max_util = np.max(all_utils)

        # Store by (seed, only_human, pttr)
        key = (seed, only_human, pttr)
        if key not in seed_data:
            seed_data[key] = []
        seed_data[key].append(instance_max_util)

    # Now group by OnlyHuman and pttr and calculate statistics
    grouped_values = {}

    for (seed, only_human, pttr), utils in seed_data.items():
        # Take the max if there are multiple values for the same seed/only_human/pttr
        max_value = np.max(utils)

        # Normalize pttr names
        pttr_normalized = pttr.lower() if pttr else 'unknown'
        if pttr_normalized == 'mp':
            pttr_normalized = 'medium'

        # Create key for grouping
        group_key = (only_human, pttr_normalized)

        if group_key not in grouped_values:
            grouped_values[group_key] = []
        grouped_values[group_key].append(max_value)

    # Calculate statistics for each group
    results = {}

    # Also calculate overall statistics (across all pttr)
    overall_human = []
    overall_hybrid = []

    for (only_human, pttr), values in grouped_values.items():
        if only_human == 1:
            overall_human.extend(values)
        elif only_human == 0:
            overall_hybrid.extend(values)

        group_name = f"{'human_only' if only_human == 1 else 'hybrid'}_{pttr}"

        if values:
            results[group_name] = {
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values),
                'mean': np.mean(values),
                'count': len(values),
                'values': values
            }

    # Add overall statistics
    if overall_human:
        results['human_only_overall'] = {
            'median': np.median(overall_human),
            'min': np.min(overall_human),
            'max': np.max(overall_human),
            'std': np.std(overall_human),
            'mean': np.mean(overall_human),
            'count': len(overall_human),
            'values': overall_human
        }

    if overall_hybrid:
        results['hybrid_overall'] = {
            'median': np.median(overall_hybrid),
            'min': np.min(overall_hybrid),
            'max': np.max(overall_hybrid),
            'std': np.std(overall_hybrid),
            'mean': np.mean(overall_hybrid),
            'count': len(overall_hybrid),
            'values': overall_hybrid
        }

    if verbose:
        print("\n" + "=" * 80)
        print(" MAXIMUM THERAPIST UTILIZATION STATISTICS ".center(80, "="))
        print("=" * 80)

        # Print overall statistics first
        if 'human_only_overall' in results:
            print("\n📊 HUMAN ONLY (OnlyHuman = 1) - OVERALL:")
            print(f"  Median max_util:  {results['human_only_overall']['median']:.2f}%")
            print(f"  Mean max_util:    {results['human_only_overall']['mean']:.2f}%")
            print(f"  Min max_util:     {results['human_only_overall']['min']:.2f}%")
            print(f"  Max max_util:     {results['human_only_overall']['max']:.2f}%")
            print(f"  Std Dev:          {results['human_only_overall']['std']:.2f}%")
            print(f"  Count:            {results['human_only_overall']['count']} instances")

        if 'hybrid_overall' in results:
            print("\n📊 HYBRID (OnlyHuman = 0) - OVERALL:")
            print(f"  Median max_util:  {results['hybrid_overall']['median']:.2f}%")
            print(f"  Mean max_util:    {results['hybrid_overall']['mean']:.2f}%")
            print(f"  Min max_util:     {results['hybrid_overall']['min']:.2f}%")
            print(f"  Max max_util:     {results['hybrid_overall']['max']:.2f}%")
            print(f"  Std Dev:          {results['hybrid_overall']['std']:.2f}%")
            print(f"  Count:            {results['hybrid_overall']['count']} instances")

        # Print by pttr
        print("\n" + "-" * 80)
        print(" BY PTTR (Patient Treatment Intensity) ".center(80, "-"))
        print("-" * 80)

        for pttr in ['light', 'medium', 'heavy']:
            print(f"\n--- {pttr.upper()} ---")

            human_key = f'human_only_{pttr}'
            if human_key in results:
                print(f"\n  HUMAN ONLY:")
                print(f"    Median max_util:  {results[human_key]['median']:.2f}%")
                print(f"    Mean max_util:    {results[human_key]['mean']:.2f}%")
                print(f"    Min max_util:     {results[human_key]['min']:.2f}%")
                print(f"    Max max_util:     {results[human_key]['max']:.2f}%")
                print(f"    Std Dev:          {results[human_key]['std']:.2f}%")
                print(f"    Count:            {results[human_key]['count']} instances")
            else:
                print(f"\n  HUMAN ONLY: No data")

            hybrid_key = f'hybrid_{pttr}'
            if hybrid_key in results:
                print(f"\n  HYBRID:")
                print(f"    Median max_util:  {results[hybrid_key]['median']:.2f}%")
                print(f"    Mean max_util:    {results[hybrid_key]['mean']:.2f}%")
                print(f"    Min max_util:     {results[hybrid_key]['min']:.2f}%")
                print(f"    Max max_util:     {results[hybrid_key]['max']:.2f}%")
                print(f"    Std Dev:          {results[hybrid_key]['std']:.2f}%")
                print(f"    Count:            {results[hybrid_key]['count']} instances")
            else:
                print(f"\n  HYBRID: No data")

        print("\n" + "=" * 80 + "\n")

    return results


def generate_statistics_report(data_dict=None, results_dir=None, save_to_file=True):
    """
    generate a comprehensive statistics report.

    args:
        data_dict: optional pre-loaded data dictionary
        results_dir: directory to load results from if data_dict not provided
        save_to_file: if true, save report to a text file

    returns:
        dictionary with all statistics
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if data_dict is None:
        data_dict = load_latest_results(results_dir)

    print(f"\n{'=' * 80}")
    print(f" generating statistics report ".center(80, '='))
    print(f"{'=' * 80}\n")
    print(f"total instances loaded: {len(data_dict)}\n")

    # calculate all statistics
    ai_share_stats = calc_ai_share(data_dict, verbose=True)
    therapist_util_stats = calc_therapist_utilization(data_dict, verbose=True)
    therapist_max_util_stats = calc_therapist_max_utilization(data_dict, verbose=True)

    all_stats = {
        'ai_share': ai_share_stats,
        'therapist_utilization': therapist_util_stats,
        'therapist_max_utilization': therapist_max_util_stats
    }

    if save_to_file:
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d_%h%m%s")
        report_filename = os.path.join(results_dir, f"statistics_report_{timestamp}.txt")

        os.makedirs(results_dir, exist_ok=True)

        with open(report_filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(" statistics report ".center(80, "=") + "\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"generated: {datetime.now().strftime('%y-%m-%d %h:%m:%s')}\n")
            f.write(f"total instances: {len(data_dict)}\n\n")

            # ai share
            f.write("\n" + "=" * 80 + "\n")
            f.write(" ai share statistics ".center(80, "=") + "\n")
            f.write("=" * 80 + "\n\n")

            # write overall statistics
            if 'human_only_overall' in ai_share_stats:
                f.write("human only (onlyhuman = 1) - overall:\n")
                f.write(f"  median:  {ai_share_stats['human_only_overall']['median']:.4f}\n")
                f.write(f"  mean:    {ai_share_stats['human_only_overall']['mean']:.4f}\n")
                f.write(f"  min:     {ai_share_stats['human_only_overall']['min']:.4f}\n")
                f.write(f"  max:     {ai_share_stats['human_only_overall']['max']:.4f}\n")
                f.write(f"  std dev: {ai_share_stats['human_only_overall']['std']:.4f}\n")
                f.write(f"  count:   {ai_share_stats['human_only_overall']['count']} instances\n\n")

            if 'hybrid_overall' in ai_share_stats:
                f.write("hybrid (onlyhuman = 0) - overall:\n")
                f.write(f"  median:  {ai_share_stats['hybrid_overall']['median']:.4f}\n")
                f.write(f"  mean:    {ai_share_stats['hybrid_overall']['mean']:.4f}\n")
                f.write(f"  min:     {ai_share_stats['hybrid_overall']['min']:.4f}\n")
                f.write(f"  max:     {ai_share_stats['hybrid_overall']['max']:.4f}\n")
                f.write(f"  std dev: {ai_share_stats['hybrid_overall']['std']:.4f}\n")
                f.write(f"  count:   {ai_share_stats['hybrid_overall']['count']} instances\n\n")

            # write by pttr
            f.write("\n" + "-" * 80 + "\n")
            f.write(" by pttr (patient treatment intensity) ".center(80, "-") + "\n")
            f.write("-" * 80 + "\n")

            for pttr in ['light', 'medium', 'heavy']:
                f.write(f"\n--- {pttr.upper()} ---\n")

                human_key = f'human_only_{pttr}'
                if human_key in ai_share_stats:
                    f.write(f"  human only:\n")
                    f.write(f"    median:  {ai_share_stats[human_key]['median']:.4f}\n")
                    f.write(f"    mean:    {ai_share_stats[human_key]['mean']:.4f}\n")
                    f.write(f"    min:     {ai_share_stats[human_key]['min']:.4f}\n")
                    f.write(f"    max:     {ai_share_stats[human_key]['max']:.4f}\n")
                    f.write(f"    std dev: {ai_share_stats[human_key]['std']:.4f}\n")
                    f.write(f"    count:   {ai_share_stats[human_key]['count']}\n")

                hybrid_key = f'hybrid_{pttr}'
                if hybrid_key in ai_share_stats:
                    f.write(f"  hybrid:\n")
                    f.write(f"    median:  {ai_share_stats[hybrid_key]['median']:.4f}\n")
                    f.write(f"    mean:    {ai_share_stats[hybrid_key]['mean']:.4f}\n")
                    f.write(f"    min:     {ai_share_stats[hybrid_key]['min']:.4f}\n")
                    f.write(f"    max:     {ai_share_stats[hybrid_key]['max']:.4f}\n")
                    f.write(f"    std dev: {ai_share_stats[hybrid_key]['std']:.4f}\n")
                    f.write(f"    count:   {ai_share_stats[hybrid_key]['count']}\n")

            # therapist utilization
            f.write("\n" + "=" * 80 + "\n")
            f.write(" therapist utilization statistics ".center(80, "=") + "\n")
            f.write("=" * 80 + "\n\n")

            # write overall statistics
            if 'human_only_overall' in therapist_util_stats:
                f.write("human only (onlyhuman = 1) - overall:\n")
                f.write(f"  median:  {therapist_util_stats['human_only_overall']['median']:.2f}%\n")
                f.write(f"  mean:    {therapist_util_stats['human_only_overall']['mean']:.2f}%\n")
                f.write(f"  min:     {therapist_util_stats['human_only_overall']['min']:.2f}%\n")
                f.write(f"  max:     {therapist_util_stats['human_only_overall']['max']:.2f}%\n")
                f.write(f"  std dev: {therapist_util_stats['human_only_overall']['std']:.2f}%\n")
                f.write(f"  count:   {therapist_util_stats['human_only_overall']['count']} instances\n\n")

            if 'hybrid_overall' in therapist_util_stats:
                f.write("hybrid (onlyhuman = 0) - overall:\n")
                f.write(f"  median:  {therapist_util_stats['hybrid_overall']['median']:.2f}%\n")
                f.write(f"  mean:    {therapist_util_stats['hybrid_overall']['mean']:.2f}%\n")
                f.write(f"  min:     {therapist_util_stats['hybrid_overall']['min']:.2f}%\n")
                f.write(f"  max:     {therapist_util_stats['hybrid_overall']['max']:.2f}%\n")
                f.write(f"  std dev: {therapist_util_stats['hybrid_overall']['std']:.2f}%\n")
                f.write(f"  count:   {therapist_util_stats['hybrid_overall']['count']} instances\n\n")

            # write by pttr
            f.write("\n" + "-" * 80 + "\n")
            f.write(" by pttr (patient treatment intensity) ".center(80, "-") + "\n")
            f.write("-" * 80 + "\n")

            for pttr in ['light', 'medium', 'heavy']:
                f.write(f"\n--- {pttr.upper()} ---\n")

                human_key = f'human_only_{pttr}'
                if human_key in therapist_util_stats:
                    f.write(f"  human only:\n")
                    f.write(f"    median:  {therapist_util_stats[human_key]['median']:.2f}%\n")
                    f.write(f"    mean:    {therapist_util_stats[human_key]['mean']:.2f}%\n")
                    f.write(f"    min:     {therapist_util_stats[human_key]['min']:.2f}%\n")
                    f.write(f"    max:     {therapist_util_stats[human_key]['max']:.2f}%\n")
                    f.write(f"    std dev: {therapist_util_stats[human_key]['std']:.2f}%\n")
                    f.write(f"    count:   {therapist_util_stats[human_key]['count']}\n")

                hybrid_key = f'hybrid_{pttr}'
                if hybrid_key in therapist_util_stats:
                    f.write(f"  hybrid:\n")
                    f.write(f"    median:  {therapist_util_stats[hybrid_key]['median']:.2f}%\n")
                    f.write(f"    mean:    {therapist_util_stats[hybrid_key]['mean']:.2f}%\n")
                    f.write(f"    min:     {therapist_util_stats[hybrid_key]['min']:.2f}%\n")
                    f.write(f"    max:     {therapist_util_stats[hybrid_key]['max']:.2f}%\n")
                    f.write(f"    std dev: {therapist_util_stats[hybrid_key]['std']:.2f}%\n")
                    f.write(f"    count:   {therapist_util_stats[hybrid_key]['count']}\n")

        print(f"\n✓ report saved to: {report_filename}\n")

    return all_stats


if __name__ == "__main__":
    # Generate comprehensive statistics report
    stats = generate_statistics_report()