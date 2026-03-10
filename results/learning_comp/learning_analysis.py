import os
import glob
import ast
import pandas as pd
import numpy as np
import scipy.stats as stats
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

def main():
    # Find all .xlsx files, excluding temp files
    excel_files = glob.glob(os.path.join(RESULTS_DIR, '*.xlsx'))
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~$')]
    
    if not excel_files:
        print(f"Keine .xlsx Dateien gefunden in {RESULTS_DIR}")
        return
        
    input_file = max(excel_files, key=os.path.getmtime)
    print(f"Lade Daten aus: {os.path.basename(input_file)}...\n")
    
    df = pd.read_excel(input_file)
    
    # Print available columns if requested ones are missing
    req_cols = ['pttr', 'learn_type', 'sum_focus_los', 'focus_los']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"Fehler: Fehlende Spalten in der Datei: {missing}")
        print(f"Verfügbare Spalten: {list(df.columns)}")
        return
        
    # Standardize string columns just in case
    df['pttr'] = df['pttr'].astype(str).str.lower().str.strip()
    # Normalize 'mp' to 'medium' if present (similar to other scripts)
    df.loc[df['pttr'] == 'mp', 'pttr'] = 'medium'
    
    df['learn_type'] = df['learn_type'].astype(str).str.strip()
    
    # Filter out anything that's completely empty in sum_focus_los before mean
    df_clean = df.dropna(subset=['sum_focus_los']).copy()
    
    # Calculate metrics directly from focus_los values
    def parse_focus_los(val):
        try:
            if isinstance(val, str):
                parsed = ast.literal_eval(val)
            else:
                parsed = val
                
            if isinstance(parsed, dict):
                values = list(parsed.values())
                if len(values) > 0:
                    return sum(values) / len(values)
        except:
            pass
        return None
        
    df_clean['los_avg'] = df_clean['focus_los'].apply(parse_focus_los)
    df_clean = df_clean.dropna(subset=['los_avg'])
    
    def q1(x): return x.quantile(0.25)
    def q3(x): return x.quantile(0.75)
    def whisker_low(x): 
        q1_val = x.quantile(0.25)
        iqr = x.quantile(0.75) - q1_val
        return max(x.min(), q1_val - 1.5 * iqr)
    def whisker_high(x): 
        q3_val = x.quantile(0.75)
        iqr = q3_val - x.quantile(0.25)
        return min(x.max(), q3_val + 1.5 * iqr)
        
    # Count instances per group as well
    analysis = df_clean.groupby(['pttr', 'learn_type']).agg(
        mean=('los_avg', 'mean'),
        median=('los_avg', 'median'),
        q1=('los_avg', q1),
        q3=('los_avg', q3),
        whisker_low=('los_avg', whisker_low),
        whisker_high=('los_avg', whisker_high),
        min_val=('los_avg', 'min'),
        max_val=('los_avg', 'max'),
        count=('los_avg', 'count')
    ).reset_index()
    
    print("=" * 110)
    print(" BOXPLOT METRIKEN: DURCHSCHNITTLICHE FOCUS LOS PRO PATIENT ")
    print("=" * 110)
    
    pttr_order = ["light", "medium", "heavy"]
    available_pttrs = analysis['pttr'].unique()
    
    # Sort pttr based on the logical order if they exist, else append the rest
    sorted_pttrs = [p for p in pttr_order if p in available_pttrs] + \
                   [p for p in available_pttrs if p not in pttr_order]
    
    for pttr in sorted_pttrs:
        print(f"\n--- PTTR: {pttr.upper()} ---")
        subset = analysis[analysis['pttr'] == pttr].sort_values('learn_type')
        for _, row in subset.iterrows():
            mean_val = row['mean']
            med_val = row['median']
            q1_val = row['q1']
            q3_val = row['q3']
            wl_val = row['whisker_low']
            wh_val = row['whisker_high']
            count_val = row['count']
            
            print(f"  {row['learn_type']:<10} | Mean: {mean_val:>6.3f} | Med: {med_val:>6.3f} | Q1: {q1_val:>6.3f} | Q3: {q3_val:>6.3f} | W_Low: {wl_val:>6.3f} | W_High: {wh_val:>6.3f} | (n={count_val})")
            
    print("\n" + "=" * 110)
    print(" STATISTISCHE SIGNIFIKANZTESTS (vs. Baseline '0') ")
    print("=" * 110)
    
    for pttr in sorted_pttrs:
        print(f"\n--- PTTR: {pttr.upper()} ---")
        subset = df_clean[df_clean['pttr'] == pttr]
        
        base_data = subset[subset['learn_type'] == '0']['los_avg'].dropna().values
        
        if len(base_data) < 2:
            print("  Zu wenige Baseline ('0') Daten für einen Test vorhanden.")
            continue
            
        for lt in ['lin', 'exp', 'sigmoid']:
            comp_data = subset[subset['learn_type'] == lt]['los_avg'].dropna().values
            
            if len(comp_data) < 2:
                continue
                
            # Welch's t-test (assumes normal distribution, unequal variances)
            try:
                t_stat, p_val_t = stats.ttest_ind(base_data, comp_data, equal_var=False)
            except Exception:
                p_val_t = float('nan')
                
            # Mann-Whitney U test (non-parametric)
            try:
                u_stat, p_val_u = stats.mannwhitneyu(base_data, comp_data, alternative='two-sided')
            except Exception:
                p_val_u = float('nan')
            
            sig_t = "*" if not np.isnan(p_val_t) and p_val_t < 0.05 else " "
            sig_u = "*" if not np.isnan(p_val_u) and p_val_u < 0.05 else " "
            
            print(f"  Vergleich Baseline vs. {lt:>7}:")
            print(f"    Welch's t-Test: p-Wert = {p_val_t:.4f} {sig_t} (Signifikanz bei < 0.05)")
            print(f"    Mann-Whitney-U: p-Wert = {p_val_u:.4f} {sig_u} (Signifikanz bei < 0.05)")

        # Zusatz-Analyse: Vergleich Linear vs Sigmoidal
        lin_mean = subset[subset['learn_type'] == 'lin']['los_avg'].mean()
        sig_mean = subset[subset['learn_type'] == 'sigmoid']['los_avg'].mean()
        if not np.isnan(lin_mean) and not np.isnan(sig_mean):
            gap = abs(lin_mean - sig_mean)
            print(f"  -> Interestingly, the performance gap between linear and sigmoidal learning is marginal across all ptr-scenarios (Gap: {gap:.4f} days).")
            
    print("\n" + "=" * 110 + "\n")
    
    

    # --- DECONSTRUCTION OF AGGREGATE PERFORMANCE GAP (DRG-LEVEL) ---
    print("\n" + "=" * 110)
    print(" DECONSTRUCTION OF AGGREGATE PERFORMANCE GAP (DRG-LEVEL) ")
    print("=" * 110)

    def safe_parse_dict(val):
        try:
            if isinstance(val, str): return ast.literal_eval(val)
            return val if isinstance(val, dict) else {}
        except: return {}

    # Extrahiere reale LOS-Werte pro DRG pro Zeile
    drg_metrics = []
    drg_cols = ['E65A', 'E65B', 'E65C']
    
    for _, row in df_clean.iterrows():
        los_dict = safe_parse_dict(row['focus_los'])
        # In diesem File scheint 'focus_avg_los' die Patienten-Mapping-Listen zu enthalten
        drg_mapping = safe_parse_dict(row.get('focus_avg_los', {}))
        
        for drg in drg_cols:
            p_list = drg_mapping.get(drg, [])
            if not isinstance(p_list, list):
                # Fallback auf die anderen Spalten, falls nötig
                p_list = safe_parse_dict(row.get(f'drg_patients_{drg}', []))
            
            # Filtere Patienten, die im focus_los Dict vorhanden sind
            los_vals = [los_dict[p] for p in p_list if p in los_dict]
            
            if los_vals:
                drg_metrics.append({
                    'seed': row['seed'],
                    'pttr': row['pttr'],
                    'learn_type': str(row['learn_type']),
                    'drg': drg,
                    'avg_los': sum(los_vals) / len(los_vals)
                })

    if drg_metrics:
        dm_df = pd.DataFrame(drg_metrics)
        
        for pttr in sorted_pttrs:
            print(f"\n--- PTTR: {pttr.upper()} ---")
            # Filtere nach PTTR
            p_subset = dm_df[dm_df['pttr'] == pttr]
            available_drgs = sorted(p_subset['drg'].unique()) if not p_subset.empty else []
            
            for drg in available_drgs:
                subset = p_subset[p_subset['drg'] == drg]
                
                # Baseline Daten (lt '0')
                base_subset = subset[subset['learn_type'] == '0']
                if base_subset.empty: continue
                
                base_mean = base_subset['avg_los'].mean()
                print(f"  DRG {drg:4} | Baseline Mean: {base_mean:.3f}")
                
                for lt in ['lin', 'exp', 'sigmoid']:
                    comp_subset = subset[subset['learn_type'] == lt]
                    if comp_subset.empty: continue
                    
                    comp_mean = comp_subset['avg_los'].mean()
                    reduction = ((base_mean - comp_mean) / base_mean) * 100
                    
                    # Statistik (Welch's t-test zwischen den Gruppen von Scenarios)
                    sig_char = " "
                    if len(base_subset) > 1 and len(comp_subset) > 1:
                        _, p_val = stats.ttest_ind(base_subset['avg_los'].values, 
                                                   comp_subset['avg_los'].values, 
                                                   equal_var=False)
                        if p_val < 0.05: sig_char = "*"
                    
                    print(f"    vs {lt:<7}: Mean: {comp_mean:.3f} | Reduction: {reduction:5.1f}% {sig_char}")

        # --- OVERALL ANALYSIS (All PTTRs combined) ---
        print(f"\n--- OVERALL (ALL PTTR SCENARIOS) ---")
        available_drgs = sorted(dm_df['drg'].unique())
        
        summary_values = {}
        
        # Overall Gap lin vs sigmoid (across all PTTRs)
        lin_overall = df_clean[df_clean['learn_type'] == 'lin']['los_avg'].mean()
        sig_overall = df_clean[df_clean['learn_type'] == 'sigmoid']['los_avg'].mean()
        overall_gap = abs(lin_overall - sig_overall)
        summary_values['lin_sig'] = f"{overall_gap:.4f}"

        for drg in available_drgs:
            subset = dm_df[dm_df['drg'] == drg]
            
            # Baseline Daten (lt '0')
            base_subset = subset[subset['learn_type'] == '0']
            if base_subset.empty: continue
            
            base_mean = base_subset['avg_los'].mean()
            print(f"  DRG {drg:4} | Baseline Mean: {base_mean:.3f}")
            
            for lt in ['lin', 'exp', 'sigmoid']:
                comp_subset = subset[subset['learn_type'] == lt]
                if comp_subset.empty: continue
                
                comp_mean = comp_subset['avg_los'].mean()
                reduction = ((base_mean - comp_mean) / base_mean) * 100
                
                # Store for final summary
                lt_short = 'sigmoid' if lt == 'sigmoid' else lt # keep original name for mapping
                # Mapping: lin_a, lin_b, lin_c, exp_a, etc.
                lt_key = 'sig' if lt == 'sigmoid' else lt
                drg_key = drg[-1].lower() # 'a', 'b', 'c'
                summary_values[f"{lt_key}_{drg_key}"] = f"{reduction:.2f}"

                # Statistik (Welch's t-test zwischen den Gruppen von Scenarios)
                sig_char = " "
                if len(base_subset) > 1 and len(comp_subset) > 1:
                    _, p_val = stats.ttest_ind(base_subset['avg_los'].values, 
                                               comp_subset['avg_los'].values, 
                                               equal_var=False)
                    if p_val < 0.05: sig_char = "*"
                
                print(f"    vs {lt:<7}: Mean: {comp_mean:.3f} | Reduction: {reduction:5.1f}% {sig_char}")

        # --- COPYABLE SUMMARY OUTPUT ---
        print("\n" + "=" * 30)
        print(" FINAL SUMMARY (COPYABLE) ")
        print("=" * 30)
        print("Scenario,value")
        # Define exact order requested
        order = ['lin_sig', 'lin_a', 'lin_b', 'lin_c', 'exp_a', 'exp_b', 'exp_c', 'sig_a', 'sig_b', 'sig_c']
        for key in order:
            val = summary_values.get(key, "N/A")
            print(f"{key},{val}")
        print("=" * 30 + "\n")

    print("\n" + "=" * 110 + "\n")
    print("series,category,cat_id,q1,median,q3,whisker_low,whisker_high,mean")
    
    cat_id_map = {"light": 1, "medium": 2, "heavy": 3}
    series_map = {"0": "base", "exp": "exp", "lin": "lin", "sigmoid": "sig"}
    
    # Sort for CSV output primarily by learn_type (series), then by pttr (category)
    csv_order_series = ["sig", "exp", "lin", "base"]
    
    csv_rows = []
    for _, row in analysis.iterrows():
        series_raw = row['learn_type']
        series = series_map.get(series_raw, series_raw)
        
        category_raw = row['pttr']
        category = category_raw.capitalize()
        cat_id = cat_id_map.get(category_raw, 0)
        
        mean_val = row['mean']
        med_val = row['median']
        q1_val = row['q1']
        q3_val = row['q3']
        wl_val = row['whisker_low']
        wh_val = row['whisker_high']
        
        csv_rows.append({
            'series': series,
            'category': category,
            'cat_id': cat_id,
            'q1': round(q1_val, 4),
            'median': round(med_val, 4),
            'q3': round(q3_val, 4),
            'whisker_low': round(wl_val, 4),
            'whisker_high': round(wh_val, 4),
            'mean': round(mean_val, 4)
        })
        
    df_csv = pd.DataFrame(csv_rows)
    
    # Sort the dataframe to match the requested output format (series then category id)
    df_csv['series_order'] = pd.Categorical(df_csv['series'], categories=csv_order_series, ordered=True)
    df_csv = df_csv.sort_values(['series_order', 'cat_id']).drop(columns=['series_order'])
    
    for _, row in df_csv.iterrows():
        print(f"{row['series']},{row['category']},{row['cat_id']},{row['q1']},{row['median']},{row['q3']},{row['whisker_low']},{row['whisker_high']},{row['mean']}")

if __name__ == "__main__":
    main()
