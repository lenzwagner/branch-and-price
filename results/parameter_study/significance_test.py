"""
Significance test: Hybrid vs. Human Only (LOS differences)
=======================================================

This script tests whether the observed differences in
length of stay (LOS) between Hybrid and Human Only
are statistically significant.

TEST PROCEDURE:
-----------------

1. PREPARE DATA (Paired Observations)
   For each seed, the results of Hybrid and Human Only are paired:

   | Seed | Human LOS | Hybrid LOS | Difference |
   |------|-----------|------------|---------- -|
   | 42   | 720       | 699        | +21       |
   | 54   | 588       | 571        | +17       |

   This results in paired observations – the same instance once with
   and once without AI support.


2. PAIR T-TEST (parametric)
   Question: Is the mean of the differences significantly ≠ 0?

   - H₀: μ_diff = 0 (no difference between human and hybrid)
   - H₁: μ_diff ≠ 0 (there is a difference)

   Calculation: t = (mean of the differences) / (standard error)

   If p < 0.05 → difference is statistically significant.


3. WILCOXON SIGNED-RANK TEST (non-parametric)
   Alternative to the t-test:
   - Makes no assumption about normal distribution
   - Tests whether the rank sums of the positive/negative differences are equal
   - More robust with outliers


4. COHEN'S D (effect size)
   Shows how large the difference is in practical terms:

   d = mean of the differences / standard deviation

   | d       | Interpretation   |
   |---------|----------------- -|
   | < 0.2   | negligible |
   | 0.2-0.5 | small            |
   | 0.5-0.8 | medium           |
   | > 0.8   | large             |


INTERPRETATION:
---------------
The test answers: "Does hybrid really save significantly LOS compared to
human only, or could the difference be random?"

Typical result: Hybrid saves ~24 days LOS, this difference is
highly significant (p < 0.0001) with large effect size (d ≈ 2.0).
"""


import pandas as pd
import numpy as np
from scipy import stats
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare data from Excel file."""
    df = pd.read_excel(filepath)
    
    # Check for required columns
    required = ['seed', 'OnlyHuman', 'sum_focus_los', 'pttr']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def calculate_paired_differences(df: pd.DataFrame, 
                                  group_col: str = 'pttr',
                                  value_col: str = 'sum_focus_los') -> pd.DataFrame:
    """
    Calculate paired differences between Hybrid and Human Only for each seed.
    
    Returns DataFrame with columns: seed, group, human_los, hybrid_los, difference
    """
    results = []
    
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        
        # Get all seeds that have both Human and Hybrid results
        human_df = group_df[group_df['OnlyHuman'] == 1].set_index('seed')
        hybrid_df = group_df[group_df['OnlyHuman'] == 0].set_index('seed')
        
        common_seeds = human_df.index.intersection(hybrid_df.index)
        
        for seed in common_seeds:
            human_los = human_df.loc[seed, value_col]
            hybrid_los = hybrid_df.loc[seed, value_col]
            
            # Handle potential Series (if multiple rows per seed)
            if isinstance(human_los, pd.Series):
                human_los = human_los.iloc[0]
            if isinstance(hybrid_los, pd.Series):
                hybrid_los = hybrid_los.iloc[0]
            
            results.append({
                'seed': seed,
                'group': group,
                'human_los': human_los,
                'hybrid_los': hybrid_los,
                'difference': human_los - hybrid_los  # Positive = Human takes longer
            })
    
    return pd.DataFrame(results)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for paired samples."""
    diff = group1 - group2
    return np.mean(diff) / np.std(diff, ddof=1)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def run_significance_tests(paired_df: pd.DataFrame, 
                           alpha: float = 0.05) -> dict:
    """
    Run significance tests on paired differences.
    
    Args:
        paired_df: DataFrame with paired differences
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Overall test (all groups combined)
    all_human = paired_df['human_los'].values
    all_hybrid = paired_df['hybrid_los'].values
    all_diff = paired_df['difference'].values
    
    # Remove NaN values
    mask = ~(np.isnan(all_human) | np.isnan(all_hybrid))
    all_human = all_human[mask]
    all_hybrid = all_hybrid[mask]
    all_diff = all_diff[mask]
    
    if len(all_diff) > 1:
        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(all_human, all_hybrid)
        
        # Wilcoxon signed-rank test
        try:
            w_stat, w_pvalue = stats.wilcoxon(all_diff, alternative='two-sided')
        except ValueError:
            w_stat, w_pvalue = np.nan, np.nan
        
        # Effect size
        d = cohens_d(all_human, all_hybrid)
        
        results['overall'] = {
            'n_pairs': len(all_diff),
            'mean_human': np.mean(all_human),
            'mean_hybrid': np.mean(all_hybrid),
            'mean_difference': np.mean(all_diff),
            'std_difference': np.std(all_diff, ddof=1),
            'variance_difference': np.var(all_diff, ddof=1),
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            't_significant': t_pvalue < alpha,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pvalue,
            'wilcoxon_significant': w_pvalue < alpha if not np.isnan(w_pvalue) else False,
            'cohens_d': d,
            'effect_size': interpret_cohens_d(d)
        }
    
    # Tests by group
    for group in paired_df['group'].unique():
        group_data = paired_df[paired_df['group'] == group]
        
        human = group_data['human_los'].values
        hybrid = group_data['hybrid_los'].values
        diff = group_data['difference'].values
        
        # Remove NaN values
        mask = ~(np.isnan(human) | np.isnan(hybrid))
        human = human[mask]
        hybrid = hybrid[mask]
        diff = diff[mask]
        
        if len(diff) > 1:
            # Paired t-test
            t_stat, t_pvalue = stats.ttest_rel(human, hybrid)
            
            # Wilcoxon signed-rank test
            try:
                w_stat, w_pvalue = stats.wilcoxon(diff, alternative='two-sided')
            except ValueError:
                w_stat, w_pvalue = np.nan, np.nan
            
            # Effect size
            d = cohens_d(human, hybrid)
            
            results[group] = {
                'n_pairs': len(diff),
                'mean_human': np.mean(human),
                'mean_hybrid': np.mean(hybrid),
                'mean_difference': np.mean(diff),
                'std_difference': np.std(diff, ddof=1),
                'variance_difference': np.var(diff, ddof=1),
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                't_significant': t_pvalue < alpha,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_pvalue': w_pvalue,
                'wilcoxon_significant': w_pvalue < alpha if not np.isnan(w_pvalue) else False,
                'cohens_d': d,
                'effect_size': interpret_cohens_d(d)
            }
    
    return results


def print_results(results: dict, alpha: float = 0.05):
    """Print formatted test results."""
    print("=" * 80)
    print("SIGNIFICANCE TESTING: Hybrid vs Human Only (LOS)")
    print("=" * 80)
    print(f"\nSignificance level: α = {alpha}")
    print(f"H0: No difference in LOS between Hybrid and Human Only")
    print(f"H1: Significant difference in LOS")
    print()
    
    for group, res in results.items():
        print("-" * 80)
        print(f"Group: {group.upper()}")
        print("-" * 80)
        print(f"  Sample size (pairs): {res['n_pairs']}")
        print(f"  Mean Human LOS:      {res['mean_human']:.4f}")
        print(f"  Mean Hybrid LOS:     {res['mean_hybrid']:.4f}")
        print(f"  Mean Difference:     {res['mean_difference']:.4f}")
        print(f"  Std. Dev. Difference:{res['std_difference']:.4f}")
        print(f"  Variance Difference: {res['variance_difference']:.4f}")
        print()
        
        # Paired t-test
        sig_t = "✓ SIGNIFICANT" if res['t_significant'] else "✗ Not significant"
        print(f"  Paired t-test:")
        print(f"    t-statistic: {res['t_statistic']:.4f}")
        print(f"    p-value:     {res['t_pvalue']:.6f} {sig_t}")
        print()
        
        # Wilcoxon test
        if not np.isnan(res['wilcoxon_pvalue']):
            sig_w = "✓ SIGNIFICANT" if res['wilcoxon_significant'] else "✗ Not significant"
            print(f"  Wilcoxon signed-rank test:")
            print(f"    W-statistic: {res['wilcoxon_statistic']:.4f}")
            print(f"    p-value:     {res['wilcoxon_pvalue']:.6f} {sig_w}")
        else:
            print(f"  Wilcoxon test: Not applicable (insufficient data)")
        print()
        
        # Effect size
        print(f"  Effect size (Cohen's d): {res['cohens_d']:.4f} ({res['effect_size']})")
        print()


def save_results_to_excel(results: dict, paired_df: pd.DataFrame, output_path: str):
    """Save results to Excel file."""
    # Convert results dict to DataFrame
    results_list = []
    for group, res in results.items():
        row = {'group': group}
        row.update(res)
        results_list.append(row)
    
    results_df = pd.DataFrame(results_list)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Summary', index=False)
        paired_df.to_excel(writer, sheet_name='Paired_Data', index=False)
    
    print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main function to run significance tests."""
    # Path to data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'parameter_study', 'results', 'results_main.xlsx')
    output_path = os.path.join(script_dir, 'significance_results.xlsx')
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Trying alternative path...")
        data_path = os.path.join(script_dir, '..', 'cg', 'results_main.xlsx')
        if not os.path.exists(data_path):
            print(f"Error: Data file not found: {data_path}")
            return
    
    print(f"Loading data from: {data_path}")
    
    # Load data
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Calculate paired differences
    paired_df = calculate_paired_differences(df)
    print(f"Created {len(paired_df)} paired observations")
    
    # Filter out problematic data (e.g., seed 66 with 0 LOS)
    initial_count = len(paired_df)
    paired_df = paired_df[(paired_df['human_los'] > 1) & (paired_df['hybrid_los'] > 1)]
    filtered_count = initial_count - len(paired_df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} pairs with anomalous LOS values")
    
    # Run significance tests
    alpha = 0.05
    results = run_significance_tests(paired_df, alpha=alpha)
    
    # Print results
    print_results(results, alpha=alpha)
    
    # Save to Excel
    save_results_to_excel(results, paired_df, output_path)


if __name__ == "__main__":
    main()
