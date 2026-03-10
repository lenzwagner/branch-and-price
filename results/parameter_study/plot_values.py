import os
import glob
import ast
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

def main():
    input_file = os.path.join(RESULTS_DIR, 'results_los_study.xlsx')
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    df = pd.read_excel(input_file)
    
    # Pre-process columns
    df['pttr'] = df['pttr'].astype(str).str.lower().str.strip()
    df.loc[df['pttr'] == 'mp', 'pttr'] = 'medium'
    df['learn_type'] = df['learn_type'].astype(str).str.strip()
    
    # Keep only 0 and sigmoid
    df = df[df['learn_type'].isin(['0', 'sigmoid', 'sig'])]

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
        
    df['los_avg'] = df['focus_los'].apply(parse_focus_los)
    df_clean = df.dropna(subset=['los_avg']).copy()
    
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
    
    # Generate CSV Output
    print("series,category,cat_id,q1,median,q3,whisker_low,whisker_high,mean")
    
    cat_id_map = {"light": 1, "medium": 2, "heavy": 3}
    series_map = {"0": "HOM", "sigmoid": "HYB", "sig": "HYB"}
    
    csv_order_series = ["HOM", "HYB"]
    
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
    df_csv['series_order'] = pd.Categorical(df_csv['series'], categories=csv_order_series, ordered=True)
    df_csv = df_csv.sort_values(['series_order', 'cat_id']).drop(columns=['series_order'])
    
    for _, row in df_csv.iterrows():
        print(f"{row['series']},{row['category']},{row['cat_id']},{row['q1']},{row['median']},{row['q3']},{row['whisker_low']},{row['whisker_high']},{row['mean']}")

if __name__ == "__main__":
    main()