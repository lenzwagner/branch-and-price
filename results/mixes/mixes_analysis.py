import os
import glob
import ast
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Current file is in results/mixes/mixes_analysis.py
# Data is in results/mixes/results/results_mixes.xlsx
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results", "results_mixes.xlsx")
# Ensure plot directory exists
PLOT_DIR = os.path.join(SCRIPT_DIR, "../../plots/results/mixes")
os.makedirs(PLOT_DIR, exist_ok=True)

def safe_parse_dict(val):
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val if isinstance(val, dict) else {}
    except:
        return {}

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Fehler: Datei nicht gefunden: {RESULTS_FILE}")
        return

    print(f"Lade Daten aus: {RESULTS_FILE}...\n")
    df = pd.read_excel(RESULTS_FILE)

    # 1. Mapping der Mix-Szenarien
    # config_name.1: NaN/0 -> 'Base', 'biasfree' -> 'Uniform', 'neuro' -> 'Complex'
    def map_mix(row):
        cfg = str(row.get('config_name.1', '')).lower()
        if 'biasfree' in cfg: return 'Uniform'
        if 'neuro' in cfg: return 'Complex'
        return 'Base'

    df['mix_scenario'] = df.apply(map_mix, axis=1)
    
    # Standardize string columns
    df['pttr'] = df['pttr'].astype(str).str.lower().str.strip()
    df['learn_type'] = df['learn_type'].astype(str).str.strip()
    # Normalize '0' to 'humanonly' if it's there as '0'
    df.loc[df['learn_type'] == '0', 'learn_type'] = 'humanonly'

    # Filter out anything that's completely empty in sum_focus_los
    df_clean = df.dropna(subset=['sum_focus_los']).copy()
    
    # Calculate average LOS per patient from focus_los dict
    def parse_focus_los_avg(val):
        d = safe_parse_dict(val)
        if d:
            values = list(d.values())
            return sum(values) / len(values)
        return None

    df_clean['los_avg'] = df_clean['focus_los'].apply(parse_focus_los_avg)
    df_clean = df_clean.dropna(subset=['los_avg'])

    # =========================================================================
    # PART B: SYSTEM RESULTS (Relative Savings)
    # =========================================================================
    print("=" * 80)
    print(" B: SYSTEM-ERGEBNISSE (RELATIVE EINSPARUNGEN) ")
    print("=" * 80)
    
    # Filter for Medium/Heavy Workload
    workloads = ['medium', 'heavy']
    mix_order = ['Base', 'Uniform', 'Complex']
    
    for ptr in workloads:
        print(f"\n--- WORKLOAD: {ptr.upper()} ---")
        ptr_df = df_clean[df_clean['pttr'] == ptr]
        
        results = []
        for mix in mix_order:
            mix_df = ptr_df[ptr_df['mix_scenario'] == mix]
            
            # Human-Only
            human_los = mix_df[mix_df['learn_type'] == 'humanonly']['los_avg'].mean()
            # Hybrid - Exponential
            exp_los = mix_df[mix_df['learn_type'] == 'exp']['los_avg'].mean()
            # Hybrid - Sigmoidal
            sig_los = mix_df[mix_df['learn_type'] == 'sigmoid']['los_avg'].mean()
            
            delta_exp = ((human_los - exp_los) / human_los * 100) if human_los and not np.isnan(human_los) else np.nan
            delta_sig = ((human_los - sig_los) / human_los * 100) if human_los and not np.isnan(human_los) else np.nan
            
            results.append({
                'Mix': mix,
                'Human': human_los,
                'Exp': exp_los,
                'Sig': sig_los,
                'Delta_Exp%': delta_exp,
                'Delta_Sig%': delta_sig
            })
            
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False, float_format=lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A"))

    # =========================================================================
    # PART C: SUB-GRUPPEN ERGEBNISSE (Spillover Effect E65C)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" C: SUB-GRUPPEN ERGEBNISSE (PER-PATIENT SPILLOVER E65C) ")
    print("=" * 80)
    
    drg_metrics = []
    # Relevant for spillover is usually a higher workload where AI helps offload therapists
    for ptr in workloads:
        print(f"\n--- ANALYSIS FOR E65C (Short-stayers) in {ptr.upper()} workload ---")
        ptr_df = df_clean[df_clean['pttr'] == ptr]
        
        for mix in ['Base', 'Complex']:
            mix_df = ptr_df[ptr_df['mix_scenario'] == mix]
            
            # For each learn_type
            for lt in ['humanonly', 'exp']:
                lt_df = mix_df[mix_df['learn_type'] == lt]
                
                e65c_los_vals = []
                for _, row in lt_df.iterrows():
                    los_dict = safe_parse_dict(row['focus_los'])
                    # Patient list for E65C
                    p_list = safe_parse_dict(row.get('drg_patients_E65C', []))
                    if not p_list:
                        mapping = safe_parse_dict(row.get('focus_avg_los', {}))
                        p_list = mapping.get('E65C', [])
                    
                    p_los = [los_dict[p] for p in p_list if p in los_dict]
                    e65c_los_vals.extend(p_los)
                
                if e65c_los_vals:
                    avg_e65c = sum(e65c_los_vals) / len(e65c_los_vals)
                    drg_metrics.append({
                        'Workload': ptr,
                        'Mix': mix,
                        'LearnType': lt,
                        'E65C_Avg_LOS': avg_e65c,
                        'N': len(e65c_los_vals)
                    })
    
    if drg_metrics:
        dm_df = pd.DataFrame(drg_metrics)
        for ptr in workloads:
            print(f"\nComparison for {ptr}:")
            sub = dm_df[dm_df['Workload'] == ptr]
            print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            
            # Check Spillover
            try:
                base_exp = sub[(sub['Mix'] == 'Base') & (sub['LearnType'] == 'exp')]['E65C_Avg_LOS'].values[0]
                comp_exp = sub[(sub['Mix'] == 'Complex') & (sub['LearnType'] == 'exp')]['E65C_Avg_LOS'].values[0]
                diff = base_exp - comp_exp
                print(f"\nResulting Spillover effect for E65C in {ptr}: {diff:.3f} days reduction in Complex vs Base mix.")
                if diff > 0:
                    print("-> Positive Spillover confirmed!")
                else:
                    print("-> No positive spillover observed for E65C in this configuration.")
            except:
                pass


    # =========================================================================
    # PART D: HEATMAPS (Efficiency Gains by DRG and Mix)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" D: DRG-STRATIFIZIERTE HEATMAPS ZU EFFIZIENZGEWINNEN")
    print("=" * 80)
    
    drgs = ['E65A', 'E65B', 'E65C']
    learn_types = ['humanonly', 'exp', 'sigmoid', 'lin']
    drg_data = []

    # Aggregiere über alle pttr (Workloads)
    for mix in mix_order:
        mix_df = df_clean[df_clean['mix_scenario'] == mix]
        
        for lt in learn_types:
            lt_df = mix_df[mix_df['learn_type'] == lt]
            
            # Sammle alle Patienten-LOS für jedes DRG
            for drg in drgs:
                all_drg_los = []
                for _, row in lt_df.iterrows():
                    los_dict = safe_parse_dict(row['focus_los'])
                    p_list = safe_parse_dict(row.get(f'drg_patients_{drg}', []))
                    if not p_list:
                        mapping = safe_parse_dict(row.get('focus_avg_los', {}))
                        p_list = mapping.get(drg, [])
                        
                    p_los = [los_dict[p] for p in p_list if p in los_dict]
                    all_drg_los.extend(p_los)
                
                if all_drg_los:
                    avg_los = sum(all_drg_los) / len(all_drg_los)
                    drg_data.append({
                        'Mix': mix,
                        'LearnType': lt,
                        'DRG': drg,
                        'AvgLOS': avg_los
                    })

    if not drg_data:
        print("Keine DRG-Daten für Heatmap gefunden.")
        return

    hm_df = pd.DataFrame(drg_data)
    
    # Berechne relative Effizienzgewinne für exp, sig, lin im Vergleich zu humanonly
    heatmaps_data = {
        'exp': pd.DataFrame(index=drgs, columns=mix_order, dtype=float),
        'sigmoid': pd.DataFrame(index=drgs, columns=mix_order, dtype=float),
        'lin': pd.DataFrame(index=drgs, columns=mix_order, dtype=float)
    }

    for mix in mix_order:
        for drg in drgs:
            # Baseline = humanonly im GLEICHEN Mix
            base_rows = hm_df[(hm_df['Mix'] == mix) & (hm_df['DRG'] == drg) & (hm_df['LearnType'] == 'humanonly')]
            if base_rows.empty:
                continue
            base_los = base_rows['AvgLOS'].values[0]
            
            for lt in ['exp', 'sigmoid', 'lin']:
                comp_rows = hm_df[(hm_df['Mix'] == mix) & (hm_df['DRG'] == drg) & (hm_df['LearnType'] == lt)]
                if not comp_rows.empty:
                    comp_los = comp_rows['AvgLOS'].values[0]
                    # Gain in %: (Human - AI) / Human * 100
                    gain = ((base_los - comp_los) / base_los) * 100
                    heatmaps_data[lt].loc[drg, mix] = gain

    # Plotte die Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = {'exp': 'Exponential Learning', 'sigmoid': 'Sigmoidal Learning', 'lin': 'Linear Learning'}
    col_idx = 0
    
    # Custom colormap (white-to-green mapped, could use YlGn or similar)
    cmap = sns.color_palette("Greens", as_cmap=True)
    
    # Optional: Find global vmin/vmax for consistent color scaling across plots
    all_vals = []
    for lt in ['exp', 'sigmoid', 'lin']:
        all_vals.extend(heatmaps_data[lt].values.flatten())
    all_vals = [v for v in all_vals if not np.isnan(v)]
    vmin = min(0, min(all_vals) if all_vals else 0) # Base 0 or negative
    vmax = max(all_vals) if all_vals else 10

    for lt in ['exp', 'sigmoid', 'lin']:
        ax = axes[col_idx]
        data = heatmaps_data[lt]
        
        # Annotate with formatting handling NaNs
        annot = data.map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        
        sns.heatmap(data.astype(float), annot=annot, fmt="", cmap=cmap, 
                    vmin=vmin, vmax=vmax, ax=ax, cbar=(col_idx==2),
                    cbar_kws={'label': 'Efficiency Gain (%)'} if col_idx==2 else None,
                    linewidths=1, linecolor='white')
        
        ax.set_title(titles[lt], fontsize=14)
        ax.set_ylabel("Patient DRG Severity" if col_idx == 0 else "")
        ax.set_xlabel("Patient Mix Scenario")
        
        # Adjust Y ticks to specify descriptions if wanted
        # ax.set_yticklabels(['E65A (Long)', 'E65B (Medium)', 'E65C (Short)'], rotation=0)
        
        col_idx += 1

    plt.suptitle("AI Efficiency Gain (%) Stratified by DRG and Patient Mix", fontsize=16, y=1.05)
    plt.tight_layout()
    
    save_path_png = os.path.join(PLOT_DIR, "drg_heatmap.png")
    save_path_svg = os.path.join(PLOT_DIR, "drg_heatmap.svg")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Heatmaps erfolgreich erstellt und gespeichert unter:\n- {save_path_png}\n- {save_path_svg}")

    # =========================================================================
    # PART E: CSV EXPORT FOR HEATMAP VALUES
    # =========================================================================
    print("\n" + "=" * 80)
    print(" E: CSV EXPORT (HEATMAP WERTE) ")
    print("=" * 80)

    csv_rows = []
    
    # Mapping for the new 1-9 numbering scheme
    # Order in heatmap (rows: E65A, E65B, E65C) (cols: Base, Uniform, Complex)
    # Cell 1: E65A, Base      Cell 2: E65A, Uniform      Cell 3: E65A, Complex
    # Cell 4: E65B, Base      Cell 5: E65B, Uniform      Cell 6: E65B, Complex
    # Cell 7: E65C, Base      Cell 8: E65C, Uniform      Cell 9: E65C, Complex
    
    for lt in ['exp', 'sigmoid', 'lin']:
        data = heatmaps_data[lt]
        cell_num = 1
        lt_prefix = 'sig' if lt == 'sigmoid' else lt
        for drg in drgs:
            for mix in mix_order:
                val = data.loc[drg, mix]
                csv_rows.append({
                    'Workload': f"{lt_prefix}_{cell_num}",
                    'value': f"{val:.4f}" if pd.notna(val) else "N/A"
                })
                cell_num += 1
                
    csv_df = pd.DataFrame(csv_rows)
    # Format required: Workload,value
    csv_filename = os.path.join(PLOT_DIR, "heatmap_values.csv")
    csv_df.to_csv(csv_filename, index=False)
    
    print(f"CSV Export erstellt: {csv_filename}")
    print("\nAuszug (copyable):")
    print("Workload,value")
    for _, row in csv_df.iterrows():
        print(f"{row['Workload']},{row['value']}")

    print("\n" + "=" * 80)
    print(" ANALYSE ABGESCHLOSSEN ")
    print("=" * 80)

if __name__ == "__main__":
    main()


