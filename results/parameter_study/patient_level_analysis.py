"""
patient_level_analysis.py
==========================
Schritt 1: Durchschnittlicher AI-Share (mit Std. Dev.) je DRG-Gruppe,
           aufgeschlüsselt nach PTTR-Szenario.

Berechnung je Patient P_F (Hybrid-Instanzen, OnlyHuman=0):
    AI-Share_p = Σ focus_y[(p, t)]  / focus_los[p]

Dann: je DRG-Gruppe × PTTR → Mean und Std. Dev. über alle Seeds.
Output: 3 PTTR × 3 DRG = 9 Mittelwerte + 9 Std. Dev.
"""

import ast
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
INPUT_FILE  = os.path.join(RESULTS_DIR, "results_los_study.xlsx")

DRG_COLS    = ["drg_patients_E65A", "drg_patients_E65B", "drg_patients_E65C"]
DRG_LABELS  = ["E65A", "E65B", "E65C"]
PTTR_ORDER  = ["light", "medium", "heavy"]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def parse_dict(value) -> dict:
    """Liest einen Wert als Python-dict (ggf. aus String-Repr.)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def parse_list(value) -> list:
    """Liest einen Wert als Python-list (ggf. aus String-Repr.)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Kern-Berechnung
# ---------------------------------------------------------------------------

def calc_ai_share_per_patient(row) -> dict:
    """
    Berechnet den AI-Share für jeden Fokus-Patienten einer Instanz.

    AI-Sessions je Patient = Summe aller focus_y[(p, t)] > 0.5

    Returns: {patient_id: ai_share}   (ai_share = AI-Sessions / LOS)
    """
    focus_y   = parse_dict(row.get("focus_y", {}))
    focus_los = parse_dict(row.get("focus_los", {}))

    if not focus_y or not focus_los:
        return {}

    ai_sessions: dict = {}

    for (pid, _day), val in focus_y.items():
        if val > 0.5:
            ai_sessions[pid] = ai_sessions.get(pid, 0) + 1

    result = {}
    for pid, los in focus_los.items():
        if los > 0:
            result[pid] = ai_sessions.get(pid, 0) / los

    return result


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def print_results(summary: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  AI-SHARE JE DRG-GRUPPE UND PTTR  (nur Hybrid-Instanzen)")
    print("=" * 65)
    print(f"  {'PTTR':<10} {'DRG':<8} {'Mean AI-Share':>14} {'Std Dev':>12} {'N Pat.':>8}")
    print("  " + "-" * 55)

    for pttr in PTTR_ORDER:
        for drg in DRG_LABELS:
            row = summary[(summary["pttr"] == pttr) & (summary["drg"] == drg)]
            if row.empty:
                print(f"  {pttr:<10} {drg:<8} {'n/a':>14} {'n/a':>12} {'0':>8}")
            else:
                r = row.iloc[0]
                print(f"  {pttr:<10} {drg:<8} "
                      f"{r['ai_share_mean']:>13.2%} "
                      f"{r['ai_share_std']:>12.2%} "
                      f"{int(r['n_patients']):>8}")
        if pttr != PTTR_ORDER[-1]:
            print("  " + "-" * 55)

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import glob
    from collections import defaultdict
    
    # Finde die neueste .pkl Datei, sonst .xlsx
    pkl_files = glob.glob(os.path.join(RESULTS_DIR, '*.pkl'))
    pkl_files = [f for f in pkl_files if not os.path.basename(f).startswith('~$')]
    
    if pkl_files:
        input_file = max(pkl_files, key=os.path.getmtime)
        print(f"Lade Daten aus Pickle: {input_file}")
        import pickle
        with open(input_file, 'rb') as f:
            data_dict = pickle.load(f)
            
        rows = []
        for instance_id, instance in data_dict.items():
            if instance.get('status') != 'FAILED':
                row = instance.copy()
                row['instance_id'] = instance_id
                rows.append(row)
        df_all = pd.DataFrame(rows)
    else:
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"Keine Dateien gefunden in {RESULTS_DIR}")
        print(f"Lade Daten aus Excel: {INPUT_FILE}")
        df_all = pd.read_excel(INPUT_FILE)

    # 1. Pairing Human & Hybrid instances
    paired_data = defaultdict(lambda: {"human": None, "hybrid": None})
    for _, row in df_all.iterrows():
        seed = row.get("seed")
        pttr = str(row.get("pttr", "medium")).lower().strip()
        if pttr == "mp": pttr = "medium"
        t_count = row.get("T")
        d_focus = row.get("D_focus")
        only_human = row.get("OnlyHuman")
        
        if pd.isna(seed) or pd.isna(t_count) or pd.isna(only_human):
            continue
            
        key = (seed, pttr, t_count, d_focus)
        if only_human == 1:
            paired_data[key]["human"] = row
        else:
            paired_data[key]["hybrid"] = row

    print(f"Gesamt: {len(df_all)} Instanzen | Paare gefunden: {len([p for p in paired_data.values() if p['hybrid'] is not None])}\n")

    # 2. Patient Level Data Extraction
    patient_records = []
    
    for key, pair in paired_data.items():
        hybrid_row = pair["hybrid"]
        human_row = pair["human"]
        
        if hybrid_row is None:
            continue
            
        pttr = key[1]
        
        # DRG mapping
        drg_map = {}
        for drg_col, drg_label in zip(DRG_COLS, DRG_LABELS):
            if drg_col in hybrid_row:
                p_list = parse_list(hybrid_row[drg_col]) # Use parse_list here
                for pid in p_list:
                    drg_map[pid] = drg_label
                        
        # LOS
        los_hybrid = parse_dict(hybrid_row.get("focus_los", {}))
        los_human = parse_dict(human_row.get("focus_los", {})) if human_row is not None else {}
        
        # AI Share
        ai_shares = calc_ai_share_per_patient(hybrid_row)
        
        for pid, l_hyb in los_hybrid.items():
            drg = drg_map.get(pid, "unknown")
            share = ai_shares.get(pid, np.nan)
            
            op_gain = np.nan
            l_hum = np.nan
            if pid in los_human:
                l_hum = los_human[pid]
                op_gain = l_hum - l_hyb
                
            patient_records.append({
                "pttr": pttr,
                "drg": drg,
                "ai_share": share,
                "op_gain": op_gain,
                "los_human": l_hum
            })
            
    df_pat = pd.DataFrame(patient_records)
    print(f"Extrahierte Patientendatensätze: {len(df_pat):,}\n")
    
    # Calculate Relative Gain at the patient level
    df_pat["rel_gain"] = df_pat["op_gain"] / df_pat["los_human"]

    # 3. Aggregation (Rest of code remains unmodified...)
    summary = (
        df_pat[df_pat["drg"] != "unknown"]
        .groupby(["pttr", "drg"])
        .agg(
            ai_share_mean=("ai_share", "mean"),
            ai_share_std=("ai_share", "std"),
            op_gain_mean=("op_gain", "mean"),
            op_gain_std=("op_gain", "std"),
            n_patients=("ai_share", "count")
        )
        .reset_index()
    )

    # 1. Standard Output Format
    print("\n" + "=" * 80)
    print("  PATIENT-LEVEL AI-SHARE & OPERATIONAL GAIN JE DRG-GRUPPE")
    print("=" * 80)
    print(f"  {'PTTR':<10} {'DRG':<8} {'Mean AI-Share':>14} {'Std':>8} | {'Mean Gain (d)':>14} {'Std':>8} | {'N Pat.':>8}")
    print("  " + "-" * 75)

    for pttr in PTTR_ORDER:
        for drg in DRG_LABELS:
            row = summary[(summary["pttr"] == pttr) & (summary["drg"] == drg)]
            if row.empty:
                print(f"  {pttr:<10} {drg:<8} {'n/a':>14} {'n/a':>8} | {'n/a':>14} {'n/a':>8} | {'0':>8}")
            else:
                r = row.iloc[0]
                ai_m = f"{r['ai_share_mean']:>13.2%}" if pd.notna(r['ai_share_mean']) else "n/a"
                ai_s = f"{r['ai_share_std']:>7.2%}" if pd.notna(r['ai_share_std']) else "n/a"
                og_m = f"{r['op_gain_mean']:>14.2f}" if pd.notna(r['op_gain_mean']) else "n/a"
                og_s = f"{r['op_gain_std']:>8.2f}" if pd.notna(r['op_gain_std']) else "n/a"
                
                print(f"  {pttr:<10} {drg:<8} {ai_m} {ai_s} | {og_m} {og_s} | {int(r['n_patients']):>8}")
        if pttr != PTTR_ORDER[-1]:
            print("  " + "-" * 75)

    print("=" * 80 + "\n")
    
    out_path_standard = os.path.join(RESULTS_DIR, "patient_ai_share_by_drg_pttr.csv")
    summary.to_csv(out_path_standard, index=False, float_format="%.6f")

    # 2. Custom Output Format: DRG_PTTR, AI Std, AI Mean, Gain Std, Gain Mean
    print("\n" + "=" * 80)
    print("  RESULTS (DRG_PTTR, AI_Std.Dev, AI_Mean, Gain_Std.Dev, Gain_Mean)")
    print("=" * 80)

    pttr_map = {"light": "L", "medium": "M", "heavy": "H"}
    
    output_rows = []
    
    for drg in DRG_LABELS:
        for pttr in PTTR_ORDER:
            row_data = summary[(summary["pttr"] == pttr) & (summary["drg"] == drg)]
            
            label = f"{drg}_{pttr_map[pttr]}"
            
            if row_data.empty:
                output_rows.append(f"{label},n/a,n/a,n/a,n/a")
                print(f"{label}, n/a, n/a, n/a, n/a")
            else:
                r = row_data.iloc[0]
                
                # AI-Share
                ai_std   = r['ai_share_std'] * 100 if pd.notna(r['ai_share_std']) else float('nan')
                ai_mean  = r['ai_share_mean'] * 100 if pd.notna(r['ai_share_mean']) else float('nan')
                
                # Gain
                og_std   = r['op_gain_std'] if pd.notna(r['op_gain_std']) else float('nan')
                og_mean  = r['op_gain_mean'] if pd.notna(r['op_gain_mean']) else float('nan')
                
                output_rows.append(f"{label},{ai_std:.2f},{ai_mean:.2f},{og_std:.2f},{og_mean:.2f}")
                print(f"{label}, {ai_std:.2f}, {ai_mean:.2f}, {og_std:.2f}, {og_mean:.2f}")

    print("=" * 80 + "\n")

    out_path_formatted = os.path.join(RESULTS_DIR, "patient_ai_share_drg_pttr_formatted.csv")
    with open(out_path_formatted, 'w') as f:
        f.write("Group,AI_StdDev,AI_Mean,Gain_StdDev,Gain_Mean\n")
        for row in output_rows:
            f.write(row + "\n")
            
    print(f"✓ Formatiertes Ergebnis gespeichert in:\n  - {out_path_standard}\n  - {out_path_formatted}\n")

    # =======================================================================
    # ZWEITE ANALYSE: Operational Gain je DRG über ALLE PTTR/Szenarien hinweg
    # =======================================================================
    
    summary_drg = (
        df_pat[df_pat["drg"] != "unknown"]
        .groupby(["drg"])
        .agg(
            op_gain_mean=("op_gain", "mean"),
            op_gain_std=("op_gain", "std"),
            rel_gain_mean=("rel_gain", "mean"),
            rel_gain_std=("rel_gain", "std"),
            n_patients=("op_gain", "count")
        )
        .reset_index()
    )

    print("=" * 85)
    print("  OVERALL OPERATIONAL GAIN & EFFICIENCY JE DRG (Alle Szenarien & PTTR)")
    print("=" * 85)
    print(f"  {'DRG':<8} {'Mean Gain (d)':>14} {'Std':>8} | {'Relative Gain':>16} {'Std':>8} | {'N Pat.':>8}")
    print("  " + "-" * 80)

    for drg in DRG_LABELS:
        row = summary_drg[summary_drg["drg"] == drg]
        if row.empty:
            print(f"  {drg:<8} {'n/a':>14} {'n/a':>8} | {'n/a':>16} {'n/a':>8} | {'0':>8}")
        else:
            r = row.iloc[0]
            og_m = f"{r['op_gain_mean']:>14.2f}" if pd.notna(r['op_gain_mean']) else "n/a"
            og_s = f"{r['op_gain_std']:>8.2f}" if pd.notna(r['op_gain_std']) else "n/a"
            rg_m = f"{r['rel_gain_mean']:>15.2%}" if pd.notna(r['rel_gain_mean']) else "n/a"
            rg_s = f"{r['rel_gain_std']:>7.2%}" if pd.notna(r['rel_gain_std']) else "n/a"
            
            print(f"  {drg:<8} {og_m} {og_s} | {rg_m} {rg_s} | {int(r['n_patients']):>8}")
            
    print("=" * 85 + "\n")

    # =======================================================================
    # DRITTE ANALYSE: STATISTISCHE ROBUSTHEIT & EFFEKTSTÄRKE (Cohen's d)
    # Target: Gain_E65A > Gain_E65B > Gain_E65C, p < 0.001, large effect.
    # =======================================================================
    import scipy.stats as stats
    
    def cohen_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

    # Filtern der gültigen op_gains
    gain_A = df_pat[(df_pat["drg"] == "E65A") & df_pat["op_gain"].notna()]["op_gain"].values
    gain_B = df_pat[(df_pat["drg"] == "E65B") & df_pat["op_gain"].notna()]["op_gain"].values
    gain_C = df_pat[(df_pat["drg"] == "E65C") & df_pat["op_gain"].notna()]["op_gain"].values
    
    print("=" * 70)
    print("  STATISTICAL SIGNIFICANCE (OPERATIONAL GAIN)")
    print("=" * 70)
    
    comparisons = [
        ("E65A vs. E65C", gain_A, gain_C),
        ("E65A vs. E65B", gain_A, gain_B),
        ("E65B vs. E65C", gain_B, gain_C)
    ]
    
    for name, g1, g2 in comparisons:
        if len(g1) > 1 and len(g2) > 1:
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False) # Welch's t-test
            d_val = cohen_d(g1, g2)
            
            p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
            print(f"  {name:<15} | t-stat = {t_stat:>6.2f} | p {p_str:<8} | Cohen's d = {d_val:>5.2f}")
        else:
            print(f"  {name:<15} | Not enough data elements.")
            
    print("=" * 70 + "\n")

    # Custom Output Format #2: Final Consolidated CSV
    print("\n" + "=" * 65)
    print("  FINAL CONSOLIDATED OUTPUT")
    print("=" * 65)
    
    final_csv_rows = ["Workload,Red,Delta"]
    
    # 1. E65A1, E65A2, etc. (Relative Gain %, Mean Gain, / Std.Dev of Gain)
    for drg in DRG_LABELS:
        row = summary_drg[summary_drg["drg"] == drg]
        lbl_1 = f"{drg}1"
        lbl_2 = f"{drg}2"
        
        if row.empty:
            str_1 = f"{lbl_1},n/a,n/a"
            str_2 = f"{lbl_2},n/a"
        else:
            r = row.iloc[0]
            rg_m_val = r['rel_gain_mean'] * 100 if pd.notna(r['rel_gain_mean']) else float('nan')
            og_m_val = r['op_gain_mean'] if pd.notna(r['op_gain_mean']) else float('nan')
            og_s_val = r['op_gain_std'] if pd.notna(r['op_gain_std']) else float('nan')
            
            str_1 = f"{lbl_1},{rg_m_val:.2f},{og_m_val:.2f}"
            str_2 = f"{lbl_2},{og_s_val:.2f}"
            
        final_csv_rows.extend([str_1, str_2])
        
    # 2. Cohen row
    if len(gain_A) > 1 and len(gain_C) > 1:
        t_stat, _ = stats.ttest_ind(gain_A, gain_C, equal_var=False)
        d_val = cohen_d(gain_A, gain_C)
        cohen_str = f"Cohen,{t_stat:.2f},{d_val:.2f}"
    else:
        cohen_str = "Cohen,n/a,n/a"
        
    final_csv_rows.append(cohen_str)
    
    # 3. AI Share rows (E65A_L, E65A_M, etc.)
    pttr_map = {"light": "L", "medium": "M", "heavy": "H"}
    for drg in DRG_LABELS:
        for pttr in PTTR_ORDER:
            row_data = summary[(summary["pttr"] == pttr) & (summary["drg"] == drg)]
            label = f"{drg}_{pttr_map[pttr]}"
            if row_data.empty:
                ai_str = f"{label},n/a,n/a"
            else:
                r = row_data.iloc[0]
                ai_std = r['ai_share_std'] * 100 if pd.notna(r['ai_share_std']) else float('nan')
                ai_mean = r['ai_share_mean'] * 100 if pd.notna(r['ai_share_mean']) else float('nan')
                ai_str = f"{label},{ai_std:.2f},{ai_mean:.2f}"
            
            final_csv_rows.append(ai_str)

    # Print and Save
    for r in final_csv_rows:
        print(r)

    print("=" * 65 + "\n")

    out_path_final = os.path.join(RESULTS_DIR, "final_consolidated_output.csv")
    with open(out_path_final, 'w') as f:
        for line in final_csv_rows:
            f.write(line + "\n")

    return summary

if __name__ == "__main__":
    main()
