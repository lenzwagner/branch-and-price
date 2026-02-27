import pandas as pd
import os
import ast

def get_focus_patient_count(val):
    if pd.isna(val) or val == '':
        return 0
    if isinstance(val, dict):
        return sum(val.values())
    try:
        d = ast.literal_eval(str(val))
        if isinstance(d, dict):
            return sum(d.values())
        return 0
    except:
        return 0

def calculate_stats():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'results', 'results_los_study.xlsx')
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Reading data from {input_file}...")
    df = pd.read_excel(input_file)
    
    # Identify target column
    target_col = 'sum_focus_los'
    if target_col not in df.columns:
        print(f"Error: Column '{target_col}' not found in the dataset.")
        return
            
    # Check if required grouping columns exist
    group_cols = ['pttr', 'OnlyHuman']
    missing_cols = [c for c in group_cols + ['focus_drg_patient_count', 'seed'] if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return

    # Calculate number of focus patients
    df['num_focus_patients'] = df['focus_drg_patient_count'].apply(get_focus_patient_count)
    
    # Calculate normalized LOS
    # Avoid division by zero
    mask = df['num_focus_patients'] > 0
    df.loc[mask, 'normalized_focus_los'] = df.loc[mask, target_col] / df.loc[mask, 'num_focus_patients']
    df.loc[~mask, 'normalized_focus_los'] = 0

    print(f"\n--- 1. Absolute & Normalized LOS grouped by 'pttr' and 'OnlyHuman' ---\n")
    
    # Calculate stats for both absolute and normalized
    stats = df.groupby(group_cols).agg(
        sum_focus_los_mean=(target_col, 'mean'),
        sum_focus_los_sigma=(target_col, 'std'),
        normalized_focus_los_mean=('normalized_focus_los', 'mean'),
        normalized_focus_los_sigma=('normalized_focus_los', 'std')
    ).reset_index()
    
    # Display the results
    print(stats.to_string(index=False))
    
    # Save the results to CSV
    output_file = os.path.join(script_dir, 'results', 'focus_stats.csv')
    stats.to_csv(output_file, index=False)
    print(f"\nStats successfully saved to {output_file}")
    
    # Calculate difference between OnlyHuman 1 and 0 per seed and pttr
    print(f"\n--- 2. Difference (Savings) between OnlyHuman=1 and OnlyHuman=0 per seed and pttr ---\n")
    
    # We can use pivot_table, but to be robust if duplicates exist, we take mean first
    pivot_df = df.pivot_table(
        index=['seed', 'pttr'], 
        columns='OnlyHuman', 
        values=[target_col, 'normalized_focus_los'],
        aggfunc='mean'
    )
    
    if 0 not in pivot_df.columns.levels[1] or 1 not in pivot_df.columns.levels[1]:
        print("Error: Could not find both OnlyHuman=0 and OnlyHuman=1 in the dataset.")
        return
        
    # Calculate differences (Human(1) - Hybrid(0) = Savings)
    diff_abs = pivot_df[(target_col, 1)] - pivot_df[(target_col, 0)]
    diff_norm = pivot_df[('normalized_focus_los', 1)] - pivot_df[('normalized_focus_los', 0)]
    
    # Create a new dataframe with differences
    diff_df = pd.DataFrame({
        'diff_abs_savings': diff_abs,
        'diff_norm_savings': diff_norm
    }).reset_index()
    
    # Group by pttr to get mean and sigma of differences
    diff_stats = diff_df.groupby('pttr').agg(
        Delta=('diff_abs_savings', 'mean'),
        Sigma=('diff_abs_savings', 'std'),
        Normalized_Delta=('diff_norm_savings', 'mean'),
        Normalized_Sigma=('diff_norm_savings', 'std')
    ).reset_index()
    
    # Rename pttr to Workload and map values to properly capitalized ones
    diff_stats.rename(columns={'pttr': 'Workload'}, inplace=True)
    diff_stats['Workload'] = diff_stats['Workload'].str.capitalize()
    
    # Calculate the 'All' row (overall average over all seeds and pttr)
    all_stats = pd.DataFrame({
        'Workload': ['All'],
        'Delta': [diff_df['diff_abs_savings'].mean()],
        'Sigma': [diff_df['diff_abs_savings'].std()],
        'Normalized_Delta': [diff_df['diff_norm_savings'].mean()],
        'Normalized_Sigma': [diff_df['diff_norm_savings'].std()]
    })
    
    # Append 'All' to diff_stats
    final_stats = pd.concat([diff_stats, all_stats], ignore_index=True)
    
    # Ensure correct order: Light, Medium, Heavy, All
    workload_order = {'Light': 0, 'Medium': 1, 'Heavy': 2, 'All': 3}
    final_stats['sort_order'] = final_stats['Workload'].map(workload_order)
    final_stats = final_stats.sort_values('sort_order').drop(columns=['sort_order'])
    
    # Select final columns to match requirements
    final_cols = ['Workload', 'Delta', 'Sigma', 'Normalized_Delta', 'Normalized_Sigma']
    final_stats = final_stats[final_cols]
    
    # Round numeric columns to 2 decimal places
    numeric_cols = ['Delta', 'Sigma', 'Normalized_Delta', 'Normalized_Sigma']
    final_stats[numeric_cols] = final_stats[numeric_cols].round(2)
    
    print(final_stats.to_string(index=False))
    
    diff_output_file = os.path.join(script_dir, 'results', 'data_los.csv')
    final_stats.to_csv(diff_output_file, index=False)
    print(f"\nDifference stats successfully saved to {diff_output_file}")

if __name__ == "__main__":
    calculate_stats()
