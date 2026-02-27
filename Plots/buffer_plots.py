"""
Stacked Bar Chart: AI vs Human Session Demand (Absolute Values) - Stratified by PTR

Visualizes the daily distribution of Human sessions (x-assignments) and 
AI sessions (y-assignments) over the 28-day planning horizon.
Combines pre_x/pre_y (for days >= 1) with focus_session_dict.
Shows 3 separate panels for Light/Medium/Heavy PTR categories.
Filtered to learn_type='sigmoid' scenarios only.

Story: AI buffers the demand - shows how many sessions per day are handled by AI vs Human.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Configuration
STRATIFY_BY_PTR = False  # Set to False to average over all PTR categories
INPUT_FILE = "../results/parameter_study/results/results_los_study.xlsx"
OUTPUT_DIR = ""
OUTPUT_FILE = "demand_stacked_absolute.png"
D_FOCUS = 28  # Planning horizon

# Colors (matching los_initial_plot color scheme)
COLOR_HUMAN = "#FFC20A"  # Yellow/Orange for Human (HOM Baseline)
COLOR_AI = "#0C7BDC"     # Blue for AI (HBM AI-Hybrid)

def parse_session_dict(session_dict_str):
    """Parse the session dictionary string to Python dict."""
    try:
        return json.loads(session_dict_str.replace("'", '"'))
    except:
        return eval(session_dict_str)

def parse_pre_x(pre_x_str):
    """Parse pre_x string representation."""
    try:
        return eval(pre_x_str)
    except:
        return {}

def derive_pre_sessions(pre_x_dict):
    """
    Derive Human and AI sessions per day from pre_x.

    pre_x format: {"(patient_id, therapist_id, day)": 1, ...}
    - If (patient, therapist, day) exists -> Human session (x-assignment)
    - If day is between first/last day for patient but no x -> AI session (pre_y)

    Returns:
        dict: {day: {'H': count, 'A': count}}
    """
    import ast

    # Group by patient and day
    patient_days = {}  # {patient_id: {day: has_therapist}}

    for key_str, value in pre_x_dict.items():
        # Parse tuple from string
        key_tuple = ast.literal_eval(key_str) if isinstance(key_str, str) else key_str

        if not isinstance(key_tuple, (tuple, list)) or len(key_tuple) != 3:
            # Skip unexpected formats like single integers instead of (patient, therapist, day)
            continue

        patient_id, therapist_id, day = key_tuple

        if patient_id not in patient_days:
            patient_days[patient_id] = {}

        # Mark this day as having a therapist assignment (Human session)
        patient_days[patient_id][day] = True

    # Count sessions per day
    sessions_per_day = {}  # {day: {'H': count, 'A': count}}

    for patient_id, days_dict in patient_days.items():
        if not days_dict:
            continue

        # Find range
        all_days = sorted(days_dict.keys())
        first_day = min(all_days)
        last_day = max(all_days)

        # Count sessions in range
        for day in range(first_day, last_day + 1):
            if day not in sessions_per_day:
                sessions_per_day[day] = {'H': 0, 'A': 0}

            if day in days_dict:
                # Has therapist assignment -> Human
                sessions_per_day[day]['H'] += 1
            else:
                # No assignment but in range -> AI (pre_y)
                sessions_per_day[day]['A'] += 1

    return sessions_per_day

def count_sessions_per_day(session_dict, d_focus=28):
    """
    Count Human and AI sessions per day from session dictionary.

    Returns:
        human_sessions: dict {day: count}
        ai_sessions: dict {day: count}
    """
    human_sessions = {day: 0 for day in range(1, d_focus + 1)}
    ai_sessions = {day: 0 for day in range(1, d_focus + 1)}

    for patient_id, schedule in session_dict.items():
        for day, session_type in schedule.items():
            day = int(day)
            if day <= d_focus:  # Only count days within focus period
                if session_type == 'H':
                    human_sessions[day] += 1
                elif session_type == 'A':
                    ai_sessions[day] += 1

    return human_sessions, ai_sessions

def aggregate_scenarios(df, d_focus=28):
    """
    Aggregate session counts across all scenarios (seeds) for days 1-28.
    Combines pre-period data (for days >= 1) with focus-period data.

    Returns:
        days: array of days 1-28
        human_mean: mean human sessions per day
        human_std: std deviation of human sessions
        ai_mean: mean AI sessions per day
        ai_std: std deviation of AI sessions
    """
    days = list(range(1, d_focus + 1))
    
    all_human_sessions = []
    all_ai_sessions = []
    
    import ast
    
    def parse_dict_col(val):
        if pd.isna(val) or val == '':
            return {}
        if isinstance(val, dict):
            return val
        try:
            return ast.literal_eval(str(val))
        except:
            return {}
            
    def extract_days_from_dict(d, target_days, type_col='X'):
        """
        Recursively find tuples containing day integers and increment the count.
        Handles both flat {(p, t, d): 1} and nested {t: {(p, d): 1}} structures.
        """
        daily_counts = {day: 0 for day in target_days}
        if not isinstance(d, dict):
            return daily_counts
            
        for k, v in d.items():
            if isinstance(v, dict):
                # Nested structure like {therapist: {(patient, day): 1}}
                nested_counts = extract_days_from_dict(v, target_days, type_col)
                for day in target_days:
                    daily_counts[day] += nested_counts[day]
            else:
                # Base structure like {(patient, day): 1} or {(patient, therapist, day): 1}
                try:
                    key_tuple = ast.literal_eval(k) if isinstance(k, str) else k
                    if isinstance(key_tuple, (tuple, list)) and len(key_tuple) >= 2 and v > 0:
                        # Extract day. Usually day is the last element
                        day = int(key_tuple[-1])
                        if day in daily_counts:
                            daily_counts[day] += 1
                except:
                    pass
        return daily_counts

    for idx, row in df.iterrows():
        human_daily = {day: 0 for day in days}
        ai_daily = {day: 0 for day in days}
        
        # We need to collect from: pre_x, focus_x, post_x (Human) and focus_y (AI)
        # Note: 'pre_y' and 'post_y' don't exist in the data schema, AI sessions there are counted through focus_session_dict if at all.
        x_cols = ['pre_x', 'focus_x', 'post_x']
        y_cols = ['focus_y']
        
        # Process X assignments (Human)
        for col in x_cols:
            if col in row:
                x_dict = parse_dict_col(row[col])
                counts = extract_days_from_dict(x_dict, days, 'X')
                for day in days:
                    human_daily[day] += counts[day]
                        
        # Process Y assignments (AI)
        for col in y_cols:
            if col in row:
                y_dict = parse_dict_col(row[col])
                counts = extract_days_from_dict(y_dict, days, 'Y')
                for day in days:
                    ai_daily[day] += counts[day]

        # Also fallback strictly parsing focus_session_dict if focus_x/y is missing but we're in the loop
        if 'focus_session_dict' in row and pd.notna(row['focus_session_dict']):
            session_dict = parse_dict_col(row['focus_session_dict'])
            focus_x_empty = ('focus_x' not in row or pd.isna(row['focus_x']) or str(row['focus_x']) == '{}')
            focus_y_empty = ('focus_y' not in row or pd.isna(row['focus_y']) or str(row['focus_y']) == '{}')
            
            if focus_x_empty and focus_y_empty:
                # Use focus_session_dict alone
                for patient_id, schedule in session_dict.items():
                    for d_str, session_type in schedule.items():
                        day = int(d_str)
                        if 1 <= day <= d_focus:
                            if session_type == 'H':
                                human_daily[day] += 1
                            elif session_type == 'A':
                                ai_daily[day] += 1

        # Convert to arrays in day order
        human_counts = [human_daily[day] for day in days]
        ai_counts = [ai_daily[day] for day in days]

        all_human_sessions.append(human_counts)
        all_ai_sessions.append(ai_counts)

    # (Replaced by single iteration loop above)

    # Calculate mean and std
    human_mean = np.mean(all_human_sessions, axis=0) if all_human_sessions else []
    human_std = np.std(all_human_sessions, axis=0) if all_human_sessions else []
    ai_mean = np.mean(all_ai_sessions, axis=0) if all_ai_sessions else []
    ai_std = np.std(all_ai_sessions, axis=0) if all_ai_sessions else []

    return days, human_mean, human_std, ai_mean, ai_std, all_human_sessions, all_ai_sessions

def create_stacked_bar_plot_by_ptr(days, ptr_data, output_path):
    """
    Create 3-panel stacked bar chart stratified by PTR category.

    Args:
        days: array of days (1-28)
        ptr_data: dict mapping PTR category to (human_mean, human_std, ai_mean, ai_std)
        output_path: path to save the plot
    """
    # PTR category order and labels
    ptr_order = ['Light', 'Medium', 'Heavy']

    # Create figure with 3 vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    #fig.suptitle('AI vs Human Session Demand by Patient-to-Therapist Ratio',
                 #fontsize=16, fontweight='bold', y=0.995)

    # Bar positions
    x = np.arange(len(days))
    width = 0.8

    for idx, (ptr_cat, ax) in enumerate(zip(ptr_order, axes)):
        if ptr_cat not in ptr_data:
            continue

        human_mean, human_std, ai_mean, ai_std = ptr_data[ptr_cat]

        # Create stacked bars
        ax.bar(x, human_mean, width, label='Therapist Sessions',
               color=COLOR_HUMAN, edgecolor='white', linewidth=0.5)
        ax.bar(x, ai_mean, width, bottom=human_mean,
               label='AI Sessions',
               color=COLOR_AI, edgecolor='white', linewidth=0.5)

        # Styling
        ax.set_ylabel('Sessions', fontsize=11, fontweight='bold')
        ax.set_title(f'{ptr_cat} PTR', fontsize=13, fontweight='bold', loc='left', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Highlight weekends (days 6-7, 13-14, 20-21, 27-28)
        # x-axis: day D is at position D-1, so days 6-7 are at positions 5-6
        # To cover BOTH days, span from 4.5 to 6.5 (covers positions 5 and 6)
        weekend_ranges_x = [(4.5, 6.5), (11.5, 13.5), (18.5, 20.5), (25.5, 27.5)]
        for start_x, end_x in weekend_ranges_x:
            ax.axvspan(start_x, end_x, color='gray', alpha=0.1, zorder=0)
            ax.axvline(x=start_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(x=end_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Add legend only to top subplot
        if idx == 0:
            ax.legend(loc='upper right', ncol=2, framealpha=0.95, fontsize=10)

    # X-axis label and ticks on bottom subplot only
    axes[-1].set_xlabel('Day', fontsize=12, fontweight='bold')
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(d) for d in days])

    plt.tight_layout()

    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved (PNG): {output_path}")

    # Save as SVG
    svg_path = str(output_path).replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"✓ Plot saved (SVG): {svg_path}")

    plt.close()

def create_single_stacked_bar_plot(days, human_mean, human_std, ai_mean, ai_std, output_path):
    """
    Create a single stacked bar chart for aggregated data.

    Args:
        days: array of days (1-28)
        human_mean: mean human sessions per day
        human_std: std deviation of human sessions
        ai_mean: mean AI sessions per day
        ai_std: std deviation of AI sessions
        output_path: path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    #fig.suptitle('AI vs Human Session Demand (Sigmoid Scenarios)',
                 #fontsize=16, fontweight='bold', y=0.995)

    x = np.arange(len(days))
    width = 0.8

    ax.bar(x, human_mean, width, label='Therapist Sessions',
           color=COLOR_HUMAN, edgecolor='white', linewidth=0.5)
    ax.bar(x, ai_mean, width, bottom=human_mean,
           label='AI Sessions',
           color=COLOR_AI, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Sessions', fontsize=11, fontweight='bold')
    ax.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in days])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.95, fontsize=10)

    # Highlight weekends (days 6-7, 13-14, 20-21, 27-28)
    # x-axis: day D is at position D-1, so days 6-7 are at positions 5-6
    weekend_ranges_x = [(4.5, 6.5), (11.5, 13.5), (18.5, 20.5), (25.5, 27.5)]
    for start_x, end_x in weekend_ranges_x:
        ax.axvspan(start_x, end_x, color='gray', alpha=0.1, zorder=0)
        ax.axvline(x=start_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=end_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved (PNG): {output_path}")

    # Save as SVG
    svg_path = str(output_path).replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"✓ Plot saved (SVG): {svg_path}")

    plt.close()

def main():
    """Main execution function."""

    print("=" * 60)
    print("Stacked Bar Chart: AI vs Human Demand (Absolute) - by PTR")
    print("=" * 60)

    import glob

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '..', 'results', 'parameter_study', 'results', 'results_los_study.xlsx')

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find any input file. Last checked: {input_file}")

    output_dir = os.path.join(script_dir, '..', 'plots', 'results')

    # Load data
    print(f"\n1. Loading data from: {input_file}")
    df = pd.read_excel(input_file)

    # Filter for sigmoid scenarios only
    df = df[df['learn_type'] == 'sigmoid'].copy()
    print(f"   → {len(df)} sigmoid scenarios found")

    # Check if stratifying by PTR
    if STRATIFY_BY_PTR:
        # Map PTTR to categories (pttr values are strings: 'light'/'medium'/'heavy')
        df['ptr_category'] = df['pttr'].str.capitalize()  # Capitalize: light->Light

        print(f"\n2. Scenarios by PTR category:")
        for cat in ['Light', 'Medium', 'Heavy']:
            count = (df['ptr_category'] == cat).sum()
            print(f"   → {cat}: {count} scenarios")

        # Aggregate by PTR category
        print(f"\n3. Aggregating by PTR category...")
        ptr_data = {}
        days = list(range(1, D_FOCUS + 1))

        for ptr_cat in ['Light', 'Medium', 'Heavy']:
            df_cat = df[df['ptr_category'] == ptr_cat]
            if len(df_cat) == 0:
                print(f"   → No scenarios found for {ptr_cat} PTR category. Skipping.")
                continue

            days_cat, human_mean, human_std, ai_mean, ai_std, all_human_sessions, all_ai_sessions = aggregate_scenarios(df_cat, D_FOCUS)
            ptr_data[ptr_cat] = (human_mean, human_std, ai_mean, ai_std)

            # Summary for this category
            total_human = human_mean.sum()
            total_ai = ai_mean.sum()
            total = total_human + total_ai
            if total > 0:
                print(f"   {ptr_cat}: {100*total_human/total:.1f}% Human, {100*total_ai/total:.1f}% AI")
            else:
                print(f"   {ptr_cat}: No sessions found.")

        Path(output_dir).mkdir(exist_ok=True, parents=True)

        # Create output directory and plot
        output_path = Path(output_dir) / OUTPUT_FILE.replace('.png', '_by_ptr.png')

        print(f"\n4. Creating 3-panel plot...")
        create_stacked_bar_plot_by_ptr(
            days=days,
            ptr_data=ptr_data,
            output_path=output_path
        )
    else:
        # Aggregate over all PTR categories
        print(f"\n2. Aggregating across all scenarios...")
        days, human_mean, human_std, ai_mean, ai_std, all_human_sessions, all_ai_sessions = aggregate_scenarios(df, D_FOCUS)

        print("\n--- Daily Session Means (All Scenarios) ---")
        print(f"{'Day':>4} | {'Therapist (Human)':>17} | {'AI':>10}")
        print("-" * 38)
        for d in range(D_FOCUS):
            print(f"{d+1:>4} | {human_mean[d]:>17.4f} | {ai_mean[d]:>10.4f}")
        print("-" * 38)
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        total_human = human_mean.sum()
        total_ai = ai_mean.sum()
        total = total_human + total_ai
        print(f"   → Total: {100*total_human/total:.1f}% Human, {100*total_ai/total:.1f}% AI")

        # Export CSV
        try:
            csv_path = Path(output_dir) / "buffer_boxplot_data.csv"
            stats_list = []
            for d in range(D_FOCUS):
                day_num = d + 1
                is_weekend = 1 if day_num in [6, 7, 13, 14, 20, 21, 27, 28] else 0
                stats_list.append({
                    'period': day_num,
                    'hom_sessions': round(human_mean[d], 4),
                    'ai_sessions': round(ai_mean[d], 4),
                    'buffer_period': is_weekend
                })
            
            if stats_list:
                stats_df = pd.DataFrame(stats_list)
                stats_df.to_csv(csv_path, index=False)
                print(f"✓ Basic CSV data saved to: {csv_path}")
        except Exception as e:
            print(f"[Error] Could not save Boxplot CSV: {e}")

        # Create output directory and plot
        Path(output_dir).mkdir(exist_ok=True)
        output_path = Path(output_dir) / OUTPUT_FILE
        
        print(f"\n3. Creating single plot...")
        create_single_stacked_bar_plot(
            days=days,
            human_mean=human_mean,
            human_std=human_std,
            ai_mean=ai_mean,
            ai_std=ai_std,
            output_path=output_path
        )
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
