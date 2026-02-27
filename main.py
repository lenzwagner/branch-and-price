import pandas as pd
import math
from CG import ColumnGeneration
from branch_and_price import BranchAndPrice
from logging_config import setup_multi_level_logging, get_logger
from Utils.derived_vars import compute_derived_variables
from Utils.extra_values import calculate_extra_metrics
from collections import defaultdict

logger = get_logger(__name__)


def disaggregate_solution(inc_sol, profile_to_all_patients, global_solutions):
    """
    Convert profile-based solution to original patient IDs.
    
    For each profile with Nr_agg > 1:
    1. Find active columns (lambda=1) for this profile from inc_sol
    2. Get schedules from global_solutions for each column
    3. Assign one schedule to each original patient (round-robin)
    
    Args:
        inc_sol: Incumbent solution dict with 'x', 'y', 'LOS', etc.
        profile_to_all_patients: {profile_id: [original_patient_ids]}
        global_solutions: Global solutions dict containing all column data
        
    Returns:
        Tuple of (disagg_x, disagg_y, disagg_los) with original patient IDs
    """
    disagg_x = {}
    disagg_y = {}
    disagg_los = {}
    
    # Get active solution dicts
    x_dict = inc_sol.get('x', {})
    y_dict = inc_sol.get('y', {})
    los_dict = inc_sol.get('LOS', {})
    
    # Group active columns by profile
    # x_dict keys are (profile, therapist, day, col_id)
    active_cols_by_profile = defaultdict(list)
    
    for key in x_dict.keys():
        if len(key) == 4:
            profile_id, t, d, col_id = key
            if col_id not in active_cols_by_profile[profile_id]:
                active_cols_by_profile[profile_id].append(col_id)
    
    # For each profile, assign columns to original patients
    for profile_id, original_patients in profile_to_all_patients.items():
        active_cols = active_cols_by_profile.get(profile_id, [])
        
        if not active_cols:
            # No active columns for this profile, skip
            continue
        
        # Assign columns to patients round-robin
        for i, patient_id in enumerate(original_patients):
            col_idx = i % len(active_cols)
            assigned_col = active_cols[col_idx]
            
            # Copy x values for this patient from the assigned column
            for (p, t, d, c), val in x_dict.items():
                if p == profile_id and c == assigned_col and val > 0.5:
                    disagg_x[(patient_id, t, d)] = val
            
            # Copy y values
            for (p, d, c), val in y_dict.items():
                if p == profile_id and c == assigned_col and val > 0.5:
                    disagg_y[(patient_id, d)] = val
            
            # Copy LOS value
            for (p, c), val in los_dict.items():
                if p == profile_id and c == assigned_col:
                    disagg_los[patient_id] = val

    return disagg_x, disagg_y, disagg_los

def print_derived_variables_for_samples(f_e, f_Y, f_theta, f_omega, f_g_comp, f_g_gap, f_z, 
                                         p_e, p_Y, p_theta, p_omega, p_g_comp, p_g_gap, p_z,
                                         original_P_F, original_P_Post, disagg_x, disagg_y, disagg_los):
    """
    Print all derived variables for 2 random Focus patients and 2 random Post patients.
    Each patient's variables are shown as dictionaries.
    """
    
    print("\n" + "=" * 100)
    print(" DERIVED VARIABLES - SAMPLE PATIENTS (Prioritizing Gaps, then LOS) ".center(100, "="))
    print("=" * 100)
    
    # Helper to count gaps per patient
    def count_gaps_per_patient(patient_list, gap_dict):
        gap_counts = {p: 0 for p in patient_list}
        for (p, d), val in gap_dict.items():
            if p in gap_counts and val > 0.5:
                gap_counts[p] += 1
        return gap_counts
        
    f_gap_counts = count_gaps_per_patient(original_P_F, f_g_gap)
    p_gap_counts = count_gaps_per_patient(original_P_Post, p_g_gap)
    
    # Select patients: Priority 1 = Gap Count (Desc), Priority 2 = LOS (Desc)
    focus_metrics = [(p, f_gap_counts.get(p, 0), disagg_los.get(p, 0)) for p in original_P_F]
    post_metrics = [(p, p_gap_counts.get(p, 0), disagg_los.get(p, 0)) for p in original_P_Post]
    
    # Sort by (gaps, los) descending
    focus_metrics.sort(key=lambda x: (x[1], x[2]), reverse=True)
    post_metrics.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    sample_focus = [p for p, gaps, los in focus_metrics[:2]]  # Top 2
    sample_post = [p for p, gaps, los in post_metrics[:2]]
    
    # Helper to get all time periods for a patient from the dicts
    def get_patient_data(patient_id, e_dict, Y_dict, theta_dict, omega_dict, g_comp_dict, g_gap_dict, z_dict, x_dict, y_dict, los_dict):
        # Get all days where this patient has data
        days = sorted(set(d for (p, d) in e_dict.keys() if p == patient_id))
        
        result = {
            'patient_id': patient_id,
            'LOS': los_dict.get(patient_id, 0),
            'e': {d: e_dict.get((patient_id, d), 0) for d in days},
            'Y': {d: Y_dict.get((patient_id, d), 0) for d in days},
            'theta': {d: theta_dict.get((patient_id, d), 0.0) for d in days},
            'omega': {d: omega_dict.get((patient_id, d), 0.0) for d in days},
            'g_comp': {d: g_comp_dict.get((patient_id, d), 0) for d in days},
            'g_gap': {d: g_gap_dict.get((patient_id, d), 0) for d in days},
            # Unfiltered x and y dictionaries
            'x': {(t, d): v for (p, t, d), v in x_dict.items() if p == patient_id},
            # Explicitly show 0 for y if missing for a day
            'y': {d: y_dict.get((patient_id, d), 0) for d in days},
            'z': {j: v for (p, j), v in z_dict.items() if p == patient_id and v == 1}
        }
        return result
    
    # Print Focus patients
    print("\n" + "-" * 100)
    print(" FOCUS PATIENTS ".center(100, "-"))
    print("-" * 100)
    for patient_id in sample_focus:
        data = get_patient_data(patient_id, f_e, f_Y, f_theta, f_omega, f_g_comp, f_g_gap, f_z, disagg_x, disagg_y, disagg_los)
        print(f"\n📊 Patient {patient_id}:")
        for key, value in data.items():
            if key != 'patient_id':
                print(f"  {key:12s}: {value}")
    
    # Print Post patients
    print("\n" + "-" * 100)
    print(" POST PATIENTS ".center(100, "-"))
    print("-" * 100)
    for patient_id in sample_post:
        data = get_patient_data(patient_id, p_e, p_Y, p_theta, p_omega, p_g_comp, p_g_gap, p_z, disagg_x, disagg_y, disagg_los)
        print(f"\n📊 Patient {patient_id}:")
        for key, value in data.items():
            if key != 'patient_id':
                print(f"  {key:12s}: {value}")
    
    print("\n" + "=" * 100 + "\n")

def main(allow_gaps=False, use_warmstart=True, dual_smoothing_alpha=None):
    """
    Main function to run Column Generation or Branch-and-Price algorithm.
    Labeling Algorithm Performance Optimizations:
    """

    # ===========================
    # LOGGING CONFIGURATION
    # ===========================
    # Setup multi-level logging: separate files for DEBUG, INFO, WARNING, ERROR
    # Set print_all_logs=True to also print all log levels to console (not just PRINT level)

    print_all_logs = False # Set to True to see all logger output on console
    setup_multi_level_logging(base_log_dir='logs', enable_console=True, print_all_logs=print_all_logs)

    # ===========================
    # RESULTS DATAFRAME (initialize empty)
    # ===========================
    results_df = pd.DataFrame(columns=[
        'seed', 'branching_strategy', 'search_strategy', 'learn_type', 'theta_base', 'lin_increase',
        'k_learn', 'infl_point', 'MS', 'MS_min', 'W_on', 'W_off', 'daily',
        'T', 'D', 'final_ub', 'final_lb', 'final_gap', 'root_lp', 'root_gap',
        'total_nodes', 'total_cg_iterations', 'iterations_per_node',
        'root_integral', 'is_optimal', 'incumbent_node_id',
        'total_time', 'time_in_mp', 'time_in_sp', 'time_in_ip_heuristic',
        'time_in_root', 'time_in_branching', 'time_to_first_incumbent', 'time_overhead',
        'pattern_size_counts', 'total_columns',
        'P_Pre', 'P_F', 'P_Post', 'P_Join', 'Nr_agg', 'E_dict', 'Q_jt', 'Req', 'Entry',
        'pre_x', 'pre_los', 'focus_x', 'focus_los', 'post_x', 'post_los',
        'focus_e', 'focus_Y', 'focus_theta', 'focus_omega', 'focus_g', 'focus_z',
        'post_e', 'post_Y', 'post_theta', 'post_omega', 'post_g', 'post_z'
    ])





    logger.info("=" * 100)
    logger.info("STARTING BRANCH-AND-PRICE SOLVER")
    logger.info("=" * 100)

    # ===========================
    # CONFIGURATION PARAMETERS
    # ===========================

    # Random seed
    seed = 12

    # Learning parameters
    app_data = {
        'learn_type': ['sigmoid'],  # Learning curve type: 'exp', 'sigmoid', 'lin', or numeric value
        'theta_base': [0.3],  # Base effectiveness
        'lin_increase': [0.01],  # Linear increase rate (for 'lin' type)
        'k_learn': [0.5],  # Learning rate (for 'exp' and 'sigmoid')
        'infl_point': [5],  # Inflection point (for 'sigmoid')
        'MS': [5],  # Maximum session window
        'MS_min': [2],  # Minimum sessions in window
        'W_on': [6],  # Work days per week
        'W_off': [1],  # Days off per week
        'daily': [4]  # Daily capacity per therapist
    }

    # Instance parameters
    T = 4  # Number of therapists
    D_focus = 10  # Number of focus days

    # Algorithm parameters
    dual_improvement_iter = 20  # Max Iterations without dual improvement
    dual_stagnation_threshold = 1e-5
    max_itr = 100  # Maximum CG iterations
    threshold = 1e-5  # Convergence threshold

    # Additional settings
    pttr = 'medium'  # Patient-to-therapist ratio: 'low', 'medium', 'high'
    show_plots = False  # Show visualization plots
    pricing_filtering = True  # Enable pricing filter
    therapist_agg = False  # Enable therapist aggregation
    learn_method = 'pwl'

    # Logger info
    logger.info(f"Configuration: seed={seed}, T={T}, D_focus={D_focus}, pttr={pttr}")

    # Branch-and-Price settings
    use_branch_and_price = True  # Set to False for standard CG
    branching_strategy = 'mp'  # 'mp' for MP variable branching, 'sp' for SP pattern variable branching
    search_strategy = 'bfs' # 'dfs' for Depth-First, 'bfs' for Best-Fit-Search
    
    # Parallelization settings
    use_parallel_pricing = True  # Enable parallel pricing (requires use_labeling=True)
    use_parallel_tree = True  # Enable parallel tree exploration
    import os
    n_pricing_workers = min(os.cpu_count(), 4) if use_parallel_pricing else 1  # Auto-detect CPUs, max 4
    # Use fewer tree workers to avoid oversubscription (pricing workers are more important)
    n_tree_workers = min(os.cpu_count() // 2, 4) if use_parallel_tree else 1

    # Output settings
    save_lps = True # Set to True to save LP and SOL files
    verbose_output = False # Set to False to suppress all non-final output
    print_solutions = True # Set to True to print coefficients of relevant lambdas in P_Focus

    # Solver settings
    deterministic = False  # Set to True for deterministic solver behavior (single-threaded, barrier method)
    
    # Treatment Gaps setting
    allow_gaps = False  # Set to True to allow treatment gaps (relaxed x+y constraint)

    # Visualization settings
    visualize_tree = False  # Enable tree visualization

    labeling_spec = {'use_labeling': True, 'max_columns_per_iter': 50, 
                     # Pricing parallelization
                     'use_parallel_pricing': use_parallel_pricing,
                     'n_pricing_workers': n_pricing_workers,
                     # Tree exploration parallelization
                     'use_parallel_tree': use_parallel_tree,
                     'n_tree_workers': n_tree_workers,
                     # Other settings
                     'debug_mode': True, 'use_apriori_pruning': False, 'use_pure_dp_optimization': True,
                     'use_persistent_pool': True,
                     'use_heuristic_pricing': False, 'heuristic_max_labels': 20, 'use_relaxed_history': False,
                     'use_numba_labeling': True,
                     'allow_gaps': allow_gaps, 'use_label_recycling': False}

    # ===========================
    # CONFIGURATION SUMMARY
    # ===========================

    if verbose_output:
        print("\n" + "=" * 100)
        print(" STARTING SETUP ".center(100, "="))
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  - Mode: {'Branch-and-Price' if use_branch_and_price else 'Column Generation'}")
        if use_branch_and_price:
            print(f"  - Branching Strategy: {branching_strategy.upper()}")
            print(f"  - Search Strategy: {'Depth-First (DFS)' if search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
            print(f"  - Parallel Pricing: {'Enabled' if use_parallel_pricing else 'Disabled'}")
            if use_parallel_pricing:
                print(f"  - Pricing Workers: {n_pricing_workers}")
            print(f"  - Parallel Tree: {'Enabled' if use_parallel_tree else 'Disabled'}")
            if use_parallel_tree:
                print(f"  - Tree Workers: {n_tree_workers}")
        print(f"  - Seed: {seed}")
        print(f"  - Learning type: {app_data['learn_type'][0]}")
        print(f"  - Learning method: {learn_method}")
        print(f"  - Therapists: {T}")
        print(f"  - Focus days: {D_focus}")
        print(f"  - Max CG iterations: {max_itr}")
        print(f"  - Threshold: {threshold}")
        print(f"  - PTTR scenario: {pttr}")
        print(f"  - Pricing filtering: {pricing_filtering}")
        print(f"  - Save LPs: {save_lps}")
        print()

    # ===========================
    # SETUP CG SOLVER
    # ===========================

    # Create CG solver
    cg_solver = ColumnGeneration(
        seed=seed,
        app_data=app_data,
        T=T,
        D_focus=D_focus,
        max_itr=max_itr,
        threshold=threshold,
        pttr=pttr,
        show_plots=show_plots,
        pricing_filtering=pricing_filtering,
        therapist_agg=therapist_agg,
        max_stagnation_itr=dual_improvement_iter,
        stagnation_threshold=dual_stagnation_threshold,
        learn_method=learn_method,
        save_lps=save_lps,
        verbose=verbose_output,
        deterministic=deterministic,
        use_warmstart=use_warmstart,
        dual_smoothing_alpha=dual_smoothing_alpha
    )

    # Setup
    cg_solver.setup()

    # ===========================
    # SOLVE
    # ===========================

    if use_branch_and_price:
        # Branch-and-Price
        if verbose_output:
            print("\n" + "=" * 100)
            print(" INITIALIZING BRANCH-AND-PRICE ".center(100, "="))
            print("=" * 100 + "\n")

        bnp_solver = BranchAndPrice(cg_solver,
                                    branching_strategy=branching_strategy,
                                    search_strategy=search_strategy,
                                    verbose=verbose_output,
                                    ip_heuristic_frequency=5,
                                    early_incumbent_iteration=1,
                                    save_lps=save_lps,
                                    label_dict=labeling_spec)
        results = bnp_solver.solve(time_limit=3600, max_nodes=300)

        # Extract optimal schedules
        if results['incumbent'] is not None:
            if verbose_output:
                print("\n" + "=" * 100)
                print(" EXTRACTING OPTIMAL SCHEDULES ".center(100, "="))
                print("=" * 100)

            optimal_schedules = bnp_solver.extract_optimal_schedules()

            # Print example schedules
            if optimal_schedules and verbose_output:
                p_focus_patients = {
                    patient_id: info
                    for patient_id, info in optimal_schedules['patient_schedules'].items()
                    if info['profile'] in cg_solver.P_F
                }

                # Print first 3 patient schedules as examples
                patient_ids = list(p_focus_patients.keys())[:3]
                for patient_id in patient_ids:
                    bnp_solver.print_detailed_schedule(
                        patient_id,
                        p_focus_patients[patient_id]
                    )

            # Export to CSV
            if verbose_output:
                bnp_solver.export_schedules_to_csv('plots/results/optimal_schedules.csv')

            if verbose_output:
                print("\n" + "=" * 100)
                print(" SCHEDULE EXTRACTION COMPLETE ".center(100, "="))
                print("=" * 100)

        # Print CG statistics (from root node)
        if verbose_output:
            print("\n" + "=" * 100)
            print(" COLUMN GENERATION STATISTICS (ROOT NODE) ".center(100, "="))
            print("=" * 100 + "\n")
            cg_solver.print_statistics()

        # Visualize tree
        if visualize_tree:
            if verbose_output:
                print("\n" + "=" * 100)
                print(" GENERATING TREE VISUALIZATION ".center(100, "="))
                print("=" * 100 + "\n")

            import os
            os.makedirs("Pictures/Tree", exist_ok=True)

            # Academic/Thesis style visualization (publication-ready)
            bnp_solver.visualize_tree(
                academic=True,
                save_path='plots/results/tree_academic.png',
                dpi=600  # High resolution for papers
            )

    else:
        # Standard Column Generation
        results = cg_solver.solve()

    # Debug: Print timing values from results
    print("\\n[DEBUG] Timing values in results dict:")
    for key in ['total_time', 'time_in_mp', 'time_in_sp', 'time_in_ip_heuristic', 'time_in_root', 'time_in_branching', 'time_to_first_incumbent', 'time_overhead']:
        print(f"  {key}: {results.get(key)}")
    print()

    # ===========================
    # DERIVED VARIABLES COMPUTATION
    # ===========================
    # Initialize variables with None
    f_e, f_Y, f_theta, f_omega, f_g, f_z = None, None, None, None, None, None
    p_e, p_Y, p_theta, p_omega, p_g, p_z = None, None, None, None, None, None
    agg_focus_x, agg_focus_los = None, None
    agg_post_x, agg_post_los = None, None
    
    # Default to profile lists
    save_P_F = cg_solver.P_F
    save_P_Post = cg_solver.P_Post
    save_P_Pre = cg_solver.P_Pre
    save_P_Join = cg_solver.P_Join
    save_pre_x = cg_solver.pre_x
    save_pre_los = cg_solver.pre_los
    save_Entry = cg_solver.Entry_agg
    save_Req = cg_solver.Req_agg
    save_E_dict = cg_solver.E_dict


    if use_branch_and_price and results.get('incumbent_solution'):
        inc_sol = results['incumbent_solution']
        
        # ==============================================================================
        # DISAGGREGATION: Convert profile-based solution to original patient IDs
        # ==============================================================================
        print("\n" + "=" * 100)
        print(" DISAGGREGATING SOLUTION TO ORIGINAL PATIENT IDs ".center(100, "="))
        print("=" * 100)
        
        # Prepare Disaggregated Dicts for Entry and Req (needed globally)
        disagg_Entry = {}
        disagg_Req = {}
        # Iterate over ALL profiles (P_Pre + P_F + P_Post + P_Join) - actually just use profile_to_all_patients
        for profile_id, patients in cg_solver.profile_to_all_patients.items():
            p_entry = cg_solver.Entry_agg.get(profile_id, 1)
            p_req = cg_solver.Req_agg.get(profile_id, 0)
            for p_orig in patients:
                disagg_Entry[p_orig] = p_entry
                disagg_Req[p_orig] = p_req
        
        save_Entry = disagg_Entry
        save_Req = disagg_Req
        

        
        disagg_x, disagg_y, disagg_los = disaggregate_solution(
            inc_sol, 
            cg_solver.profile_to_all_patients,
            cg_solver.global_solutions
        )
        

        
        # Create a "virtual" solution dict with disaggregated values
        # This will be passed to compute_derived_variables instead of inc_sol
        disagg_sol = {
            'x': disagg_x,
            'y': disagg_y,
            'LOS': disagg_los,
        }
        
        # We also need to define the lists of ORIGINAL patients for Focus and Post groups
        original_P_F = []
        for profile_id in cg_solver.P_F:
            if profile_id in cg_solver.profile_to_all_patients:
                original_P_F.extend(cg_solver.profile_to_all_patients[profile_id])
                

        original_P_Post = []
        for profile_id in cg_solver.P_Post:
            if profile_id in cg_solver.profile_to_all_patients:
                original_P_Post.extend(cg_solver.profile_to_all_patients[profile_id])
                
        original_P_Pre = []
        disagg_pre_x_dict = {}
        disagg_pre_los_dict = {}

        for profile_id in cg_solver.P_Pre:
            if profile_id in cg_solver.profile_to_all_patients:
                patients = cg_solver.profile_to_all_patients[profile_id]
                original_P_Pre.extend(patients)
                
                # Disaggregate history
                # Note: pre_x uses profile_ID as key in cg_solver.pre_x
                p_hist_x = cg_solver.pre_x.get(profile_id, {})
                p_hist_los = cg_solver.pre_los.get(profile_id, 0)
                
                for p_orig in patients:
                    disagg_pre_x_dict[p_orig] = p_hist_x
                    disagg_pre_los_dict[p_orig] = p_hist_los

        original_P_Join = []
        disagg_E_dict = {}

        for profile_id in cg_solver.P_Join:
            if profile_id in cg_solver.profile_to_all_patients:
                patients = cg_solver.profile_to_all_patients[profile_id]
                original_P_Join.extend(patients)
                
                # Disaggregate E_dict
                p_e_val = cg_solver.E_dict.get(profile_id, 0)
                for p_orig in patients:
                    disagg_E_dict[p_orig] = p_e_val

        # Use original patient IDs for output
        save_P_F = original_P_F
        save_P_Post = original_P_Post
        save_P_Pre = original_P_Pre
        save_P_Join = original_P_Join
        save_pre_x = disagg_pre_x_dict
        save_pre_los = disagg_pre_los_dict
        save_E_dict = disagg_E_dict

        # Calculate derived variables for Focus Patients using DISAGGREGATED data
        print("\n" + "-" * 100)
        print(" COMPUTING DERIVED VARIABLES FOR FOCUS PATIENTS (DISAGGREGATED) ".center(100, "-"))
        print("-" * 100)
        f_e, f_Y, f_theta, f_omega, f_g_comp, f_z, f_g_gap = compute_derived_variables(cg_solver, disagg_sol, app_data, patients_list=original_P_F)
        
        
        # Calculate derived variables for Post Patients using DISAGGREGATED data
        print("\n" + "-" * 100)
        print(" COMPUTING DERIVED VARIABLES FOR POST PATIENTS (DISAGGREGATED) ".center(100, "-"))
        print("-" * 100)
        p_e, p_Y, p_theta, p_omega, p_g_comp, p_z, p_g_gap = compute_derived_variables(cg_solver, disagg_sol, app_data, patients_list=original_P_Post)
        
        # Print derived variables for sample patients (2 Focus + 2 Post)
        print_derived_variables_for_samples(
            f_e, f_Y, f_theta, f_omega, f_g_comp, f_g_gap, f_z,
            p_e, p_Y, p_theta, p_omega, p_g_comp, p_g_gap, p_z,
            original_P_F, original_P_Post, disagg_x, disagg_y, disagg_los
        )
        
        # Calculate EXTRA METRICS (E.2 - E.8)
        # Pack data for Focus (use g_comp for compatibility)
        f_derived_data = {'e': f_e, 'Y': f_Y, 'theta': f_theta, 'omega': f_omega, 'g': f_g_comp, 'z': f_z, 'g_gap': f_g_gap}
        print("\n" + "*" * 100)
        print(" CALCULATING EXTRA METRICS FOR FOCUS PATIENTS ".center(100, "*"))
        print("*" * 100)
        # Prepare Focus Horizon: Days 1 to D_focus
        f_start, f_end = 1, D_focus
        f_metrics = calculate_extra_metrics(cg_solver, disagg_sol, original_P_F, f_derived_data, cg_solver.T, start_day=f_start, end_day=f_end)
        
        # Pack data for Post (use g_comp for compatibility)
        p_derived_data = {'e': p_e, 'Y': p_Y, 'theta': p_theta, 'omega': p_omega, 'g': p_g_comp, 'z': p_z, 'g_gap': p_g_gap}
        print("\n" + "*" * 100)
        print(" CALCULATING EXTRA METRICS FOR POST PATIENTS ".center(100, "*"))
        print("*" * 100)
        p_start = D_focus + 1
        p_end = max(cg_solver.D_Ext)

        p_metrics = calculate_extra_metrics(cg_solver, disagg_sol, original_P_Post, p_derived_data, cg_solver.T, start_day=p_start, end_day=p_end)
        
        # Store disaggregated x and LOS for Excel output
        # Filter x and LOS for Focus and Post patients
        focus_x = {k: v for k, v in disagg_x.items() if k[0] in original_P_F}
        focus_los = {k: v for k, v in disagg_los.items() if k in original_P_F}
        
        post_x = {k: v for k, v in disagg_x.items() if k[0] in original_P_Post}
        post_los = {k: v for k, v in disagg_los.items() if k in original_P_Post}
        
        # We need aggregator variables (just to match variable names expected below)
        agg_focus_x = focus_x 
        agg_post_x = post_x
        agg_focus_los = focus_los
        agg_post_los = post_los
        
        # Calculate Total Columns Count
        total_columns = 0
        if cg_solver.master and hasattr(cg_solver.master, 'lmbda'):
            total_columns = len(cg_solver.master.lmbda)

        # Initialize dummy dicts to be overwritten below
        agg_focus_los = focus_los
        agg_post_los = post_los

        # Build focus_y and post_y from disagg_y
        focus_y = {k: v for k, v in disagg_y.items() if k[0] in original_P_F}
        post_y = {k: v for k, v in disagg_y.items() if k[0] in original_P_Post}
        
        # ========================================


    # ==============================================================================
    # COMBINED METRICS (Focus + Post together)
    # ==============================================================================
    
    # DRG patient lists combined (Focus + Post)
    drg_patients_E65A = f_metrics.get('drg_patients', {}).get('E65A', []) + p_metrics.get('drg_patients', {}).get('E65A', [])
    drg_patients_E65B = f_metrics.get('drg_patients', {}).get('E65B', []) + p_metrics.get('drg_patients', {}).get('E65B', [])
    drg_patients_E65C = f_metrics.get('drg_patients', {}).get('E65C', []) + p_metrics.get('drg_patients', {}).get('E65C', [])
    
    # Continuity violations combined (Focus + Post)
    combined_continuity_violations = f_metrics.get('continuity_violations', []) + p_metrics.get('continuity_violations', [])
    
    # Patients per therapist combined (Focus + Post)
    # {therapist_id: count_of_patients}
    from collections import Counter
    f_ppt = f_metrics.get('patients_per_therapist', {})
    p_ppt = p_metrics.get('patients_per_therapist', {})
    combined_patients_per_therapist = dict(Counter(f_ppt) + Counter(p_ppt))

    # Add derived variables to DataFrame row
    new_row_data = {
         # Focus
        'focus_x': agg_focus_x, 'focus_y': focus_y, 'focus_los': agg_focus_los,
        'focus_e': f_e, 'focus_Y': f_Y, 'focus_theta': f_theta, 
        'focus_omega': f_omega, 'focus_g_comp': f_g_comp, 'focus_g_gap': f_g_gap, 'focus_z': f_z,
        # Post
        'post_x': agg_post_x, 'post_y': post_y, 'post_los': agg_post_los,
        'post_e': p_e, 'post_Y': p_Y, 'post_theta': p_theta,
        'post_omega': p_omega, 'post_g_comp': p_g_comp, 'post_g_gap': p_g_gap, 'post_z': p_z,
        # Add Extra Metrics with prefixes
        **{f"focus_{k}": v for k, v in f_metrics.items()},
        **{f"post_{k}": v for k, v in p_metrics.items()},
        # Combined metrics (not split by Focus/Post)
        'drg_patients_E65A': drg_patients_E65A,
        'drg_patients_E65B': drg_patients_E65B,
        'drg_patients_E65C': drg_patients_E65C,
        'combined_continuity_violations': combined_continuity_violations,
        'combined_num_continuity_violations': len(combined_continuity_violations),
        'combined_patients_per_therapist': combined_patients_per_therapist,
    }
    
    # ===========================
    # POPULATE RESULTS DATAFRAME
    # ===========================
    # Combine basic stats and derived vars
    full_row_data = {
        'seed': seed,
        'branching_strategy': branching_strategy,
        'search_strategy': search_strategy,
        'learn_type': app_data['learn_type'][0],
        'theta_base': app_data['theta_base'][0],
        'lin_increase': app_data['lin_increase'][0],
        'k_learn': app_data['k_learn'][0],
        'infl_point': app_data['infl_point'][0],
        'MS': app_data['MS'][0],
        'MS_min': app_data['MS_min'][0],
        'W_on': app_data['W_on'][0],
        'W_off': app_data['W_off'][0],
        'daily': app_data['daily'][0],
        'T': len(cg_solver.T),
        'D': len(cg_solver.D),
        'final_ub': results.get('incumbent'),
        'final_lb': results.get('lp_bound'),
        'final_gap': results.get('gap'),
        'root_lp': results['root_node'].lp_bound if results.get('root_node') else None,
        'root_gap': (results.get('incumbent') - results['root_node'].lp_bound) / abs(results.get('incumbent')) if results.get('incumbent') and results.get('root_node') and results.get('incumbent') != 0 else None,
        'total_nodes': results.get('total_nodes'),
        'total_cg_iterations': results.get('cg_iterations'),
        'iterations_per_node': results.get('iterations_per_node'),
        'root_integral': results.get('root_integral'),
        'is_optimal': results.get('is_optimal'),
        'incumbent_node_id': results.get('incumbent_node_id'),
        'total_time': results.get('total_time'),
        'time_in_mp': results.get('time_in_mp'),
        'time_in_sp': results.get('time_in_sp'),
        'time_in_ip_heuristic': results.get('time_in_ip_heuristic'),
        'time_in_root': results.get('time_in_root'),
        'time_in_branching': results.get('time_in_branching'),
        'time_to_first_incumbent': results.get('time_to_first_incumbent'),
        'time_overhead': results.get('time_overhead'),
        'pattern_size_counts': results.get('pattern_size_counts'),
        'max_tree_depth': results.get('max_tree_depth'),
        'nodes_pruned': results.get('nodes_pruned'),
        'integer_solutions_found': results.get('integer_solutions_found'),
        'total_columns': total_columns,
        'P_Pre': save_P_Pre,
        'P_F': save_P_F,
        'P_Post': save_P_Post,
        'P_Join': save_P_Join,
        'Nr_agg': cg_solver.Nr_agg,
        'Nr_agg': cg_solver.Nr_agg,
        'E_dict': save_E_dict,
        'Q_jt': cg_solver.Max_t,
        'Req': save_Req,
        'Entry': save_Entry,
        'pre_x': save_pre_x,
        'pre_los': save_pre_los,
        **new_row_data
    }

    new_row = pd.DataFrame([full_row_data])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # ===========================
    # SUMMARY
    # ===========================

    print("\n" + "=" * 100)
    print(" EXECUTION SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"Completed successfully!")
    print(f"  - Mode: {'Branch-and-Price' if use_branch_and_price else 'Column Generation'}")
    print(f"  - Total time: {results['total_time']:.2f}s")

    if use_branch_and_price:
        print(f"\nBranch-and-Price Results:")
        print(f"  - Branching strategy: {branching_strategy.upper()}")
        print(f"  - Search strategy: {'Depth-First (DFS)' if search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
        print(f"  - Nodes explored: {results['nodes_explored']}")
        print(f"  - Nodes fathomed: {results['nodes_fathomed']}")
        print(f"  - Nodes branched: {results.get('nodes_branched', 0)}")
        print(f"  - LP bound (LB): {results['lp_bound']:.5f}")
        if results['incumbent']:
            print(f"  - Incumbent (UB): {results['incumbent']:.5f}")
            print(f"  - Gap: {results['gap']:.5%}")
        else:
            print(f"  - Incumbent (UB): None")
        print(f"  - Integral: {results['is_integral']}")
        print(f"  - Total CG iterations (all nodes): {results['cg_iterations']}")
        print(f"  - IP solves: {results['ip_solves']}")

    # Save to Excel
    try:
        results_df.to_excel("results_test.xlsx", index=False)
        print(f"\nSaved results to results_test.xlsx")
    except Exception as e:
        print(f"\nCould not save Excel: {e}")

        # Convergence and optimal solution status
        print(f"\nAlgorithm Status:")
        if results['incumbent'] is not None:
            # Check if tree is complete (all nodes fathomed)
            if results.get('tree_complete', False):
                print(f"  ✓ TREE COMPLETE - All nodes explored")
                print(f"  ✓ OPTIMAL SOLUTION FOUND: {results['incumbent']:.5f}")
                if results['gap'] > 0:
                    print(f"  (Numerical gap: {results['gap']:.5%}, due to floating-point precision)")
            else:
                gap_threshold = 1e-4  # 0.01% gap threshold
                if results['gap'] < gap_threshold:
                    print(f"  ✓ Algorithm CONVERGED (Gap < {gap_threshold:.2%})")
                    print(f"  ✓ OPTIMAL SOLUTION FOUND: {results['incumbent']:.5f}")
                else:
                    print(f"  ! Algorithm terminated with gap: {results['gap']:.5%}")
                    print(f"  ! Best solution found: {results['incumbent']:.5f}")
                    print(f"  ! Lower bound: {results['lp_bound']:.5f}")
        else:
            print(f"  ✗ No feasible solution found")
    else:
        print(f"\nColumn Generation Results:")
        print(f"  - Iterations: {results.get('num_iterations', results.get('cg_iterations', 'N/A'))}")
        print(f"  - LP objective: {results.get('lp_obj', 'N/A')}")
        print(f"  - IP objective: {results.get('ip_obj', 'N/A')}")
        print(f"  - Compact model: {results.get('comp_obj', 'N/A')}")
        print(f"  - Gap: {results.get('gap', 'N/A')}")
        print(f"  - Integral?: {results.get('is_integral', 'N/A')}")

        # Convergence and optimal solution status
        print(f"\nAlgorithm Status:")
        gap_threshold = 1e-4  # 0.01% gap threshold
        gap = results.get('gap', 1.0)
        gap = gap if gap is not None else 1.0
        
        if gap < gap_threshold:
            print(f"  ✓ Algorithm CONVERGED (Gap < {gap_threshold:.2%})")
            print(f"  ✓ OPTIMAL SOLUTION FOUND: {results.get('ip_obj', 'N/A')}")
        else:
            print(f"  ! Algorithm terminated with gap: {gap:.5%}")
            print(f"  ! Best IP solution: {results.get('ip_obj', 'N/A')}")
            print(f"  ! LP relaxation: {results.get('lp_obj', 'N/A')}")

    print("=" * 100 + "\n")

    # Print focus patient lambdas if requested
    if print_solutions:
        print("\n" + "=" * 100)
        print(" FOCUS PATIENT LAMBDA COEFFICIENTS (>0) ".center(100, "="))
        print("=" * 100)
        
        lambdas = results.get('incumbent_lambdas') if use_branch_and_price else results.get('lambdas')
        
        if lambdas:
            # Filter for focus patients and print
            focus_lambdas = {k: v for k, v in lambdas.items() if k[0] in cg_solver.P_F}
            if focus_lambdas:
                print(f"{'Patient Profile':<20} {'Column ID':<15} {'Obj Coeff':<15} {'LOS':<10} {'Schedule':<30}")
                print("-" * 95)
                for (p, a), val in sorted(focus_lambdas.items()):
                    # Extract (t, d) pairs from global_solutions['x']
                    x_sol = cg_solver.global_solutions['x'].get((p, a), {})
                    # Key format: (p, t, d, a)
                    active_days = sorted([(t, d) for (p_val, t, d, a_val), v in x_sol.items() if v > 0.5])
                    schedule_str = ", ".join([f"({t},{d})" for t, d in active_days])
                    los_data = cg_solver.global_solutions['LOS'].get((p, a), {})
                    los_val = list(los_data.values())[0] if isinstance(los_data, dict) and los_data else (los_data if isinstance(los_data, (int, float)) else 0)
                    print(f"{p:<20} {a:<15} {int(round(val['obj'])):<15} {int(round(los_val)):<10} {schedule_str}")
            else:
                print("No active lambdas found for focus patients.")
        else:
            print("No lambda solution available.")
        print("=" * 100 + "\n")

        print("\n" + "=" * 100)
        print(" POST PATIENT LAMBDA COEFFICIENTS (>0) ".center(100, "="))
        print("=" * 100)

        if lambdas:
            # Filter for post patients and print
            post_lambdas = {k: v for k, v in lambdas.items() if k[0] in cg_solver.P_Post}
            if post_lambdas:
                print(f"{'Patient Profile':<20} {'Column ID':<15} {'Obj Coeff':<15} {'LOS':<10} {'Schedule':<30}")
                print("-" * 95)
                for (p, a), val in sorted(post_lambdas.items()):
                    # Extract (t, d) pairs from global_solutions['x']
                    x_sol = cg_solver.global_solutions['x'].get((p, a), {})
                    # Key format: (p, t, d, a)
                    active_days = sorted([(t, d) for (p_val, t, d, a_val), v in x_sol.items() if v > 0.5])
                    schedule_str = ", ".join([f"({t},{d})" for t, d in active_days])
                    los_data = cg_solver.global_solutions['LOS'].get((p, a), {})
                    los_val = list(los_data.values())[0] if isinstance(los_data, dict) and los_data else (los_data if isinstance(los_data, (int, float)) else 0)
                    print(f"{p:<20} {a:<15} {int(round(val['obj'])):<15} {int(round(los_val)):<10} {schedule_str}")
            else:
                print("No active lambdas found for post patients.")
        else:
            print("No lambda solution available.")
        print("=" * 100 + "\n")

    # ===========================
    # PRINT RESULTS DATAFRAME
    # ===========================
    print("\n" + "=" * 100)
    print(" RESULTS DATAFRAME ".center(100, "="))
    print("=" * 100)
    #print(results_df.to_string())
    print("=" * 100 + "\n")

    # ===========================
    # DERIVED VARIABLES FOR FOCUS PATIENTS
    # ===========================
    # Old derived variable block removed as it is now integrated above
    pass
    #print(cg_solver.Nr_agg, cg_solver.agg_to_patient, sep="\n")
    return results


if __name__ == "__main__":
    results = main(allow_gaps=True, use_warmstart=True, dual_smoothing_alpha=None)  # Set alpha=None to disable