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

# Period-split colors (light → main → dark for Pre / Focus / Post)
COLOR_PRE_HUMAN   = "#FFE8A0"   # light yellow  – Pre  / Human
COLOR_FOCUS_HUMAN = "#FFC20A"   # yellow        – Focus / Human
COLOR_POST_HUMAN  = "#A07800"   # dark gold     – Post  / Human
COLOR_PRE_AI      = "#99CCFF"   # light blue    – Pre  / AI
COLOR_FOCUS_AI    = "#0C7BDC"   # blue          – Focus / AI
COLOR_POST_AI     = "#084A86"   # dark blue     – Post  / AI

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
    Returns 6 daily time-series split by period (pre / focus / post) and
    session type (Human / AI).

    Returns:
        days             : list of ints 1..d_focus
        period_means     : dict with keys 'pre_human', 'focus_human', 'post_human',
                           'pre_ai', 'focus_ai', 'post_ai'  →  np.array shape (d_focus,)
        period_stds      : same structure, standard deviations
        # legacy flat arrays (kept for backwards compatibility)
        human_mean, human_std, ai_mean, ai_std
        all_human_sessions, all_ai_sessions
    """
    days = list(range(1, d_focus + 1))

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

    def extract_days_from_dict(d, target_days):
        """Return {day: count} for all keys whose last tuple element is a day."""
        daily_counts = {day: 0 for day in target_days}
        if not isinstance(d, dict):
            return daily_counts
        for k, v in d.items():
            if isinstance(v, dict):
                nested = extract_days_from_dict(v, target_days)
                for day in target_days:
                    daily_counts[day] += nested[day]
            else:
                try:
                    key_tuple = ast.literal_eval(k) if isinstance(k, str) else k
                    if isinstance(key_tuple, (tuple, list)) and len(key_tuple) >= 2 and v > 0:
                        day = int(key_tuple[-1])
                        if day in daily_counts:
                            daily_counts[day] += 1
                except:
                    pass
        return daily_counts

    def extract_pre_y_days(d, target_days):
        """pre_y: {(patient, day): 0=Human / 1=AI}"""
        human_counts = {day: 0 for day in target_days}
        ai_counts    = {day: 0 for day in target_days}
        if not isinstance(d, dict):
            return human_counts, ai_counts
        for k, v in d.items():
            try:
                key_tuple = ast.literal_eval(k) if isinstance(k, str) else k
                if isinstance(key_tuple, (tuple, list)) and len(key_tuple) == 2:
                    day = int(key_tuple[1])
                    if day in target_days:
                        if v == 0:
                            human_counts[day] += 1
                        elif v == 1:
                            ai_counts[day] += 1
            except:
                pass
        return human_counts, ai_counts

    # Per-scenario lists for each of the 6 series
    period_keys = ['pre_human', 'focus_human', 'post_human',
                   'pre_ai',    'focus_ai',    'post_ai']
    all_series = {k: [] for k in period_keys}

    for idx, row in df.iterrows():
        # Initialise per-day counters for this scenario
        sc = {k: {day: 0 for day in days} for k in period_keys}

        # ── PRE period ────────────────────────────────────────────────
        if 'pre_x' in row:
            for day, cnt in extract_days_from_dict(parse_dict_col(row['pre_x']), days).items():
                sc['pre_human'][day] += cnt

        if 'pre_y' in row:
            h, a = extract_pre_y_days(parse_dict_col(row['pre_y']), days)
            for day in days:
                sc['pre_human'][day] += h[day]
                sc['pre_ai'][day]    += a[day]

        # ── FOCUS period ───────────────────────────────────────────────
        if 'focus_x' in row:
            for day, cnt in extract_days_from_dict(parse_dict_col(row['focus_x']), days).items():
                sc['focus_human'][day] += cnt

        if 'focus_y' in row:
            for day, cnt in extract_days_from_dict(parse_dict_col(row['focus_y']), days).items():
                sc['focus_ai'][day] += cnt

        # Fallback: use focus_session_dict if focus_x / focus_y are both empty
        if 'focus_session_dict' in row and pd.notna(row['focus_session_dict']):
            fx_empty = 'focus_x' not in row or pd.isna(row['focus_x']) or str(row['focus_x']) == '{}'
            fy_empty = 'focus_y' not in row or pd.isna(row['focus_y']) or str(row['focus_y']) == '{}'
            if fx_empty and fy_empty:
                sd = parse_dict_col(row['focus_session_dict'])
                for patient_id, schedule in sd.items():
                    for d_str, session_type in schedule.items():
                        day = int(d_str)
                        if 1 <= day <= d_focus:
                            if session_type == 'H':
                                sc['focus_human'][day] += 1
                            elif session_type == 'A':
                                sc['focus_ai'][day]    += 1

        # ── POST period ────────────────────────────────────────────────
        if 'post_x' in row:
            for day, cnt in extract_days_from_dict(parse_dict_col(row['post_x']), days).items():
                sc['post_human'][day] += cnt

        if 'post_y' in row:
            for day, cnt in extract_days_from_dict(parse_dict_col(row['post_y']), days).items():
                sc['post_ai'][day] += cnt

        # Append per-day arrays to scenario lists
        for k in period_keys:
            all_series[k].append([sc[k][day] for day in days])

    # ── Compute means / stds  ─────────────────────────────────────────
    period_means = {}
    period_stds  = {}
    for k in period_keys:
        arr = np.array(all_series[k]) if all_series[k] else np.zeros((1, len(days)))
        period_means[k] = np.mean(arr, axis=0)
        period_stds[k]  = np.std(arr,  axis=0)

    # Legacy flat arrays (sum of all periods)
    human_mean = period_means['pre_human'] + period_means['focus_human'] + period_means['post_human']
    human_std  = np.sqrt(period_stds['pre_human']**2 + period_stds['focus_human']**2 + period_stds['post_human']**2)
    ai_mean    = period_means['pre_ai']    + period_means['focus_ai']    + period_means['post_ai']
    ai_std     = np.sqrt(period_stds['pre_ai']**2    + period_stds['focus_ai']**2    + period_stds['post_ai']**2)

    all_human_sessions = [np.array(all_series['pre_human'][i]) +
                          np.array(all_series['focus_human'][i]) +
                          np.array(all_series['post_human'][i])
                          for i in range(len(all_series['pre_human']))]
    all_ai_sessions    = [np.array(all_series['pre_ai'][i]) +
                          np.array(all_series['focus_ai'][i]) +
                          np.array(all_series['post_ai'][i])
                          for i in range(len(all_series['pre_ai']))]

    return (days, period_means, period_stds,
            human_mean, human_std, ai_mean, ai_std,
            all_human_sessions, all_ai_sessions)

def _add_period_bars(ax, x, width, period_means, weekend=True):
    """
    Draw 6-segment stacked bars (Pre/Focus/Post × Human/AI) on *ax*.
    Stacking order (bottom→top): Pre-H · Focus-H · Post-H · Pre-AI · Focus-AI · Post-AI
    Returns the list of bar-container objects (for legend handles).
    """
    ph  = period_means['pre_human']
    fh  = period_means['focus_human']
    poh = period_means['post_human']
    pa  = period_means['pre_ai']
    fa  = period_means['focus_ai']
    poa = period_means['post_ai']

    kw = dict(width=width, edgecolor='white', linewidth=0.4)
    b0 = ax.bar(x, ph,  **kw, color=COLOR_PRE_HUMAN,   label='Pre – Therapist')
    b1 = ax.bar(x, fh,  **kw, bottom=ph,               color=COLOR_FOCUS_HUMAN, label='Focus – Therapist')
    b2 = ax.bar(x, poh, **kw, bottom=ph+fh,            color=COLOR_POST_HUMAN,  label='Post – Therapist')
    b3 = ax.bar(x, pa,  **kw, bottom=ph+fh+poh,        color=COLOR_PRE_AI,      label='Pre – AI')
    b4 = ax.bar(x, fa,  **kw, bottom=ph+fh+poh+pa,     color=COLOR_FOCUS_AI,    label='Focus – AI')
    b5 = ax.bar(x, poa, **kw, bottom=ph+fh+poh+pa+fa,  color=COLOR_POST_AI,     label='Post – AI')

    if weekend:
        for start_x, end_x in [(4.5, 6.5), (11.5, 13.5), (18.5, 20.5), (25.5, 27.5)]:
            ax.axvspan(start_x, end_x, color='gray', alpha=0.10, zorder=0)
            ax.axvline(x=start_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(x=end_x,   color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    return [b0, b1, b2, b3, b4, b5]


def create_stacked_bar_plot_by_ptr(days, ptr_data, output_path):
    """
    Create 3-panel stacked bar chart stratified by PTR category.
    Each bar is split into 6 segments: Pre/Focus/Post × Human/AI.

    Args:
        days: array of days (1-28)
        ptr_data: dict mapping PTR category to (period_means, period_stds)
        output_path: path to save the plot
    """
    ptr_order = ['Light', 'Medium', 'Heavy']
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

    x = np.arange(len(days))
    width = 0.8

    for idx, (ptr_cat, ax) in enumerate(zip(ptr_order, axes)):
        if ptr_cat not in ptr_data:
            continue

        period_means, period_stds = ptr_data[ptr_cat]
        bar_handles = _add_period_bars(ax, x, width, period_means)

        ax.set_ylabel('Sessions', fontsize=11, fontweight='bold')
        ax.set_title(f'{ptr_cat} PTR', fontsize=13, fontweight='bold', loc='left', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        if idx == 0:
            ax.legend(handles=bar_handles,
                      loc='upper right', ncol=3, framealpha=0.95, fontsize=9)

    axes[-1].set_xlabel('Day', fontsize=12, fontweight='bold')
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(d) for d in days])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved (PNG): {output_path}")
    svg_path = str(output_path).replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"✓ Plot saved (SVG): {svg_path}")
    plt.close()

def create_single_stacked_bar_plot(days, period_means, period_stds, output_path,
                                   # legacy positional args (ignored if period_means is a dict)
                                   _human_std=None, _ai_mean=None, _ai_std=None):
    """
    Create a single stacked bar chart split by Pre/Focus/Post × Human/AI.

    Args:
        days         : array of days (1-28)
        period_means : dict with keys pre_human / focus_human / post_human /
                       pre_ai / focus_ai / post_ai  →  np.array
        period_stds  : same structure (currently unused in plot, kept for API)
        output_path  : path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x     = np.arange(len(days))
    width = 0.8

    bar_handles = _add_period_bars(ax, x, width, period_means)

    ax.set_ylabel('Sessions', fontsize=11, fontweight='bold')
    ax.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in days])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(handles=bar_handles,
              loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, framealpha=0.95, fontsize=10)

    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved (PNG): {output_path}")
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

            (days_cat, period_means, period_stds,
             human_mean, human_std, ai_mean, ai_std,
             all_human_sessions, all_ai_sessions) = aggregate_scenarios(df_cat, D_FOCUS)
            ptr_data[ptr_cat] = (period_means, period_stds)

            # Summary for this category
            total_human = human_mean.sum()
            total_ai    = ai_mean.sum()
            total       = total_human + total_ai
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
        (days, period_means, period_stds,
         human_mean, human_std, ai_mean, ai_std,
         all_human_sessions, all_ai_sessions) = aggregate_scenarios(df, D_FOCUS)

        print("\n--- Daily Session Means by Period (All Scenarios) ---")
        header = f"{'Day':>4} | {'Pre-H':>8} | {'Foc-H':>8} | {'Post-H':>8} | {'Pre-AI':>8} | {'Foc-AI':>8} | {'Post-AI':>8}"
        print(header)
        print("-" * len(header))
        for d in range(D_FOCUS):
            print(f"{d+1:>4} | "
                  f"{period_means['pre_human'][d]:>8.2f} | "
                  f"{period_means['focus_human'][d]:>8.2f} | "
                  f"{period_means['post_human'][d]:>8.2f} | "
                  f"{period_means['pre_ai'][d]:>8.2f} | "
                  f"{period_means['focus_ai'][d]:>8.2f} | "
                  f"{period_means['post_ai'][d]:>8.2f}")
        print("-" * len(header))

        Path(output_dir).mkdir(exist_ok=True, parents=True)
        total_human = human_mean.sum()
        total_ai    = ai_mean.sum()
        total       = total_human + total_ai
        print(f"   → Total: {100*total_human/total:.1f}% Human, {100*total_ai/total:.1f}% AI")

        # Export CSV (extended with per-period breakdown)
        try:
            csv_path  = Path(output_dir) / "buffer_boxplot_data.csv"
            stats_list = []
            for d in range(D_FOCUS):
                day_num = d + 1
                stats_list.append({
                    'period':          day_num,
                    'pre_human':       round(period_means['pre_human'][d],   4),
                    'focus_human':     round(period_means['focus_human'][d], 4),
                    'post_human':      round(period_means['post_human'][d],  4),
                    'pre_ai':          round(period_means['pre_ai'][d],      4),
                    'focus_ai':        round(period_means['focus_ai'][d],    4),
                    'post_ai':         round(period_means['post_ai'][d],     4),
                    'hom_sessions':    round(human_mean[d], 4),
                    'ai_sessions':     round(ai_mean[d],    4),
                    'buffer_period':   1 if day_num in [6, 7, 13, 14, 20, 21, 27, 28] else 0,
                })
            if stats_list:
                pd.DataFrame(stats_list).to_csv(csv_path, index=False)
                print(f"✓ CSV data saved to: {csv_path}")
        except Exception as e:
            print(f"[Error] Could not save CSV: {e}")

        Path(output_dir).mkdir(exist_ok=True)
        output_path = Path(output_dir) / OUTPUT_FILE

        print(f"\n3. Creating single plot...")
        create_single_stacked_bar_plot(
            days=days,
            period_means=period_means,
            period_stds=period_stds,
            output_path=output_path,
        )
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
