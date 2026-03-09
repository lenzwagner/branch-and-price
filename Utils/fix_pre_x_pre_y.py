"""
fix_pre_x_pre_y.py
==================
Liest eine instances.xlsx Datei ein, erstellt korrekt formatierte
pre_x_edit und pre_y_edit Spalten und speichert die Datei zurück.

Aufruf:
    python3 fix_pre_x_pre_y.py <pfad_zur_instance.xlsx>
    
Falls kein Pfad angegeben wird, wird die Standard-Instanzdatei verwendet.
"""

import sys
import ast
import json
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# Konfiguration
# ═══════════════════════════════════════════════════════════════════
DEFAULT_FILE = "results/parameter_study/instances/instances_los_study.xlsx"


# ═══════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════════

def parse_pre_x(pre_x_str) -> dict:
    """
    Parst pre_x aus der gespeicherten Excel-Form in ein echtes Python-Dict.

    Unterstützt beide Formate:
      - Neues Format:  '{"(1, 2, 3)": 1.0, ...}'  (Keys als Strings)
      - Altes Format:  '{(1, 2, 3): 1.0, ...}'    (Keys als Tupel)

    Rückgabe: {(patient_id, therapist_id, day): value, ...}
    """
    if not isinstance(pre_x_str, str) or pre_x_str.strip() in ('{}', ''):
        return {}
    try:
        parsed = ast.literal_eval(pre_x_str)
        result = {}
        for key, value in parsed.items():
            real_key = ast.literal_eval(key) if isinstance(key, str) else key
            result[real_key] = value
        return result
    except Exception as e:
        print(f"  Warnung: Konnte pre_x nicht parsen: {e}")
        return {}


def build_pre_x_edit(pre_x_str) -> str:
    """Erstellt eine normalisierte String-Repr. mit echten Tupel-Keys."""
    return str(parse_pre_x(pre_x_str))


def build_pre_y_edit(row) -> str:
    """
    Berechnet pre_y_edit:
      - 1 wenn an diesem Tag KEIN Therapeut-Termin in pre_x vorhanden ist
      - 0 wenn an diesem Tag EIN Therapeut-Termin vorhanden ist
    Nur für Tage im Bereich [entry_day, entry_day + los - 1].
    """
    try:
        P_Pre = ast.literal_eval(row['P_Pre']) if isinstance(row['P_Pre'], str) else row['P_Pre']
        Entry = json.loads(row['Entry'])
        pre_los = json.loads(row['pre_los'])
        pre_x_dict = parse_pre_x(row['pre_x'])
    except Exception as e:
        print(f"  Warnung: Konnte Zeile nicht verarbeiten: {e}")
        return '{}'

    # Therapie-Tage pro Patient (aus pre_x)
    x_days: dict[int, set] = {p: set() for p in P_Pre}
    for (p, t, d), value in pre_x_dict.items():
        if p in x_days and value > 0:
            x_days[p].add(d)

    pre_y: dict = {}
    for p in P_Pre:
        entry_d = int(Entry.get(str(p), 0))
        los = int(pre_los.get(str(p), 0))
        if los == 0:
            continue
        for d in range(entry_d, entry_d + los):
            pre_y[(p, d)] = 0 if d in x_days[p] else 1

    return str(pre_y)


# ═══════════════════════════════════════════════════════════════════
# Hauptprogramm
# ═══════════════════════════════════════════════════════════════════

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

    print(f"Lese Datei:      {file_path}")
    df = pd.read_excel(file_path)

    required_cols = ['P_Pre', 'pre_x', 'Entry', 'pre_los']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"FEHLER: Fehlende Spalten: {missing}")
        sys.exit(1)

    print("Erstelle pre_x_edit ...")
    df['pre_x_edit'] = df['pre_x'].apply(build_pre_x_edit)

    print("Erstelle pre_y_edit ...")
    df['pre_y_edit'] = df.apply(build_pre_y_edit, axis=1)

    # ── Schnellvalidierung für die erste Zeile ─────────────────────
    print("\nValidierung (Zeile 0, erster Patient):")
    row0 = df.iloc[0]
    P_Pre0 = ast.literal_eval(row0['P_Pre'])
    p0 = P_Pre0[0]
    pre_x0 = parse_pre_x(row0['pre_x'])
    pre_y0 = ast.literal_eval(row0['pre_y_edit'])
    x_days_p0 = sorted(d for (p, t, d) in pre_x0 if p == p0)
    y_vals_p0 = {d: v for (p, d), v in pre_y0.items() if p == p0}
    print(f"  Patient {p0} – Therapie-Tage (pre_x):  {x_days_p0}")
    print(f"  Patient {p0} – pre_y == 0 (Therapie):  {sorted(d for d, v in y_vals_p0.items() if v == 0)}")
    print(f"  Patient {p0} – pre_y == 1 (App):        {sorted(d for d, v in y_vals_p0.items() if v == 1)}")
    assert sorted(d for d, v in y_vals_p0.items() if v == 0) == x_days_p0, "Mismatch zwischen pre_x und pre_y!"
    print("  ✅ Validierung erfolgreich")

    print(f"\nSpeichere Datei: {file_path}")
    try:
        df.to_excel(file_path, index=False)
        print("Fertig!")
    except PermissionError:
        print("FEHLER: Datei ist in Excel geöffnet. Bitte schließen und erneut starten.")
        sys.exit(1)


if __name__ == "__main__":
    main()
