import pandas as pd
import ast
import json

file_path = 'results/parameter_study/instances/instances_los_study.xlsx'
print(f"Lese Datei ein: {file_path}")
df = pd.read_excel(file_path)

def reconstruct_pre_y(row):
    if pd.isna(row['P_Pre']) or pd.isna(row['pre_x_edit']) or pd.isna(row['Entry']) or pd.isna(row['pre_los']):
        return '{}'

    try:
        P_Pre = ast.literal_eval(row['P_Pre'])
        Entry = json.loads(row['Entry'])
        pre_los = json.loads(row['pre_los'])
        pre_x_edit = ast.literal_eval(row['pre_x_edit'])

        # Find therapy days per patient
        x_days = {p: set() for p in P_Pre}
        for (p, t, d), value in pre_x_edit.items():
            if p in x_days:
                x_days[p].add(d)

        pre_y_edit = {}
        for p in P_Pre:
            entry_d = int(Entry.get(str(p), 0))
            los = int(pre_los.get(str(p), 0))
            
            # If the patient has no entry or los recorded accurately, skip
            if los == 0:
                continue
                
            for d in range(entry_d, entry_d + los):
                if d not in x_days[p]:
                    pre_y_edit[(p, d)] = 1

        return str(pre_y_edit)
    except Exception as e:
        print(f"Fehler bei Zeile: {e}")
        return '{}'

print("Erstelle neue Spalte 'pre_y_edit'...")
df['pre_y_edit'] = df.apply(reconstruct_pre_y, axis=1)

print("Speichere Excel-Datei...")
df.to_excel(file_path, index=False)
print("Fertig! 'pre_y_edit' wurde erfolgreich hinzugefügt.")
