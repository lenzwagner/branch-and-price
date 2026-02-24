import pandas as pd
import math
from CG import ColumnGeneration
from branch_and_price import BranchAndPrice
from logging_config import setup_multi_level_logging, get_logger
from Utils.derived_vars import compute_derived_variables
from Utils.extra_values import calculate_extra_metrics
import pickle
from datetime import datetime
import glob
import os
from collections import defaultdict
from main import disaggregate_solution

logger = get_logger(__name__)

def solve_instance(seed, D_focus, pttr='medium', T=2, allow_gaps=False, use_warmstart=True, dual_smoothing_alpha=None, learn_type=0, app_data_overrides=None, T_demand=None, pre_generated_data=None, lp_output_path=None, cutoff=None):
    """
    Solve a single instance with given seed, D_focus, pttr, and T.
    Returns a dictionary with instance parameters and results.
    """
    
    logger.info("=" * 100)
    logger.info(f"SOLVING INSTANCE: seed={seed}, D_focus={D_focus}, learn_type={learn_type}")
    if cutoff is not None:
        logger.info(f"CUTOFF LIMIT: {cutoff}")
    logger.info("=" * 100)
    
    # ===========================
    # CONFIGURATION PARAMETERS
    # ===========================

    # Learning parameters
    app_data = {
        'learn_type': [learn_type],
        'theta_base': [0.3],
        'lin_increase': [0.05],
        'k_learn': [1.5],
        'infl_point': [4],
        'MS': [5],
        'MS_min': [2],
        'W_on': [5],
        'W_off': [2],
        'daily': [4]
    }

    # Apply overrides if provided
    if app_data_overrides:
        for key, value in app_data_overrides.items():
            if key in app_data:
                # Ensure value is a list as expected by the rest of the code
                app_data[key] = [value] if not isinstance(value, list) else value
                logger.info(f"Overriding {key} with {value}")
            else:
                logger.warning(f"Override key {key} not found in app_data")

    # Algorithm parameters
    dual_improvement_iter = 20
    dual_stagnation_threshold = 1e-5
    max_itr = 100
    threshold = 1e-5

    # Additional settings
    show_plots = False
    pricing_filtering = True
    therapist_agg = False
    learn_method = 'pwl'

    logger.info(f"Configuration: seed={seed}, T={T}, D_focus={D_focus}, pttr={pttr}")

    # Branch-and-Price settings
    use_branch_and_price = True
    branching_strategy = 'mp'
    search_strategy = 'bfs'
    
    # Parallelization settings
    use_parallel_pricing = True
    use_parallel_tree = False  # Enable parallel tree exploration
    n_pricing_workers = min(os.cpu_count(), 4) if use_parallel_pricing else 1
    # Use fewer tree workers to avoid oversubscription (pricing workers are more important)
    n_tree_workers = min(os.cpu_count() // 2, 4) if use_parallel_tree else 1

    # Output settings
    save_lps = lp_output_path is not None  # Enable LP saving only when an output path is given
    verbose_output = False
    print_solutions = False
    save_transition_matrix = False

    # Solver settings
    deterministic = False

    # Visualization settings
    visualize_tree = False
    
    # Treatment Gaps setting
    allow_gaps = False  # Set to True to allow treatment gaps (relaxed x+y constraint)

    # Define labeling specs
    labeling_spec = {
        'use_labeling': True, 
        'max_columns_per_iter': 100,
        # Pricing parallelization
        'use_parallel_pricing': use_parallel_pricing,
        'n_pricing_workers': n_pricing_workers,
        # Tree exploration parallelization (NEW)
        'use_parallel_tree': use_parallel_tree,
        'n_tree_workers': n_tree_workers,
        # Other settings
        'debug_mode': False,  # Disable debug for batch runs
        'use_apriori_pruning': False, 
        'use_pure_dp_optimization': True,
        'use_persistent_pool': True,
        'use_heuristic_pricing': False, 
        'heuristic_max_labels': 20, 
        'use_relaxed_history': False,
        'use_numba_labeling': True,
        'allow_gaps': allow_gaps, 
        'use_label_recycling': False
    }

    # ===========================
    # SETUP CG SOLVER
    # ===========================

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
        dual_smoothing_alpha=dual_smoothing_alpha,
        T_demand=T_demand,
        pre_generated_data=pre_generated_data
    )

    cg_solver.setup()

    # ===========================
    # SOLVE
    # ===========================

    bnp_solver = BranchAndPrice(
        cg_solver,
        branching_strategy=branching_strategy,
        search_strategy=search_strategy,
        verbose=verbose_output,
        ip_heuristic_frequency=5,
        early_incumbent_iteration=None,
        save_lps=save_lps,
        label_dict=labeling_spec
    )
    
    # Solve with 20-minute timeout per instance
    # If timeout occurs, solver returns current incumbent and best LP bound
    results = bnp_solver.solve(time_limit=2400, max_nodes=500, cutoff=cutoff)  # 1200s = 20 minutes

    # Export LP files if requested
    if lp_output_path is not None:
        try:
            import os as _os
            _os.makedirs(lp_output_path, exist_ok=True)
            final_lp = _os.path.join(lp_output_path, 'final_incumbent.lp')
            
            # Export ONLY the incumbent MIP (LP format with Integer vars)
            cg_solver.export_models(
                master_filename=None, 
                compact_filename=None, 
                incumbent_lp_filename=final_lp
            )
            logger.info(f"Incumbent LP written to {final_lp}")
        except Exception as _e:
            logger.warning(f"LP export failed: {_e}")

    # ===========================
    # DERIVED VARIABLES COMPUTATION
    # ===========================
    f_e, f_Y, f_theta, f_omega, f_g_comp, f_z, f_g_gap = None, None, None, None, None, None, None
    p_e, p_Y, p_theta, p_omega, p_g_comp, p_z, p_g_gap = None, None, None, None, None, None, None
    agg_focus_x, agg_focus_los = None, None
    agg_post_x, agg_post_los = None, None
    total_columns = 0

    # Default to profile lists (for safety/fallback)
    save_P_F = cg_solver.P_F
    save_P_Post = cg_solver.P_Post
    save_P_Pre = cg_solver.P_Pre
    save_P_Join = cg_solver.P_Join
    save_pre_x = cg_solver.pre_x
    save_pre_los = cg_solver.pre_los
    save_Entry = cg_solver.Entry_agg
    save_Req = cg_solver.Req_agg
    save_E_dict = cg_solver.E_dict

    if results.get('incumbent_solution'):
        inc_sol = results['incumbent_solution']
        
        # ==============================================================================
        # DISAGGREGATION: Convert profile-based solution to original patient IDs
        # ==============================================================================
        
        # Prepare Disaggregated Dicts for Entry and Req (needed globally)
        disagg_Entry = {}
        disagg_Req = {}
        # Iterate over ALL profiles (P_Pre + P_F + P_Post + P_Join)
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
        disagg_sol = {
            'x': disagg_x,
            'y': disagg_y,
            'LOS': disagg_los,
        }
        
        # Get original patient lists
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
                # Extract all pre_x entries for this profile_id: (profile_id, t, d) -> {(t, d): value}
                p_hist_x = {(t, d): v for (p, t, d), v in cg_solver.pre_x.items() if p == profile_id}
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

        # Use original patient IDs for calculation
        save_P_F = original_P_F
        save_P_Post = original_P_Post
        save_P_Pre = original_P_Pre
        save_P_Join = original_P_Join
        save_pre_x = disagg_pre_x_dict
        save_pre_los = disagg_pre_los_dict
        save_E_dict = disagg_E_dict
        
        # Calculate derived variables for Focus Patients using DISAGGREGATED data
        f_e, f_Y, f_theta, f_omega, f_g_comp, f_z, f_g_gap = compute_derived_variables(
            cg_solver, disagg_sol, app_data, patients_list=original_P_F
        )
        
        # Calculate derived variables for Post Patients using DISAGGREGATED data
        p_e, p_Y, p_theta, p_omega, p_g_comp, p_z, p_g_gap = compute_derived_variables(
            cg_solver, disagg_sol, app_data, patients_list=original_P_Post
        )
        
        # Calculate EXTRA METRICS (use g_comp for compatibility) with DISAGGREGATED data
        f_derived_data = {'e': f_e, 'Y': f_Y, 'theta': f_theta, 'omega': f_omega, 'g': f_g_comp, 'z': f_z, 'g_gap': f_g_gap}
        f_start, f_end = 1, D_focus
        f_metrics = calculate_extra_metrics(
            cg_solver, disagg_sol, original_P_F, f_derived_data, cg_solver.T, 
            start_day=f_start, end_day=f_end
        )
        
        p_derived_data = {'e': p_e, 'Y': p_Y, 'theta': p_theta, 'omega': p_omega, 'g': p_g_comp, 'z': p_z, 'g_gap': p_g_gap}
        p_start = D_focus + 1
        p_end = max(cg_solver.D_Ext)
        p_metrics = calculate_extra_metrics(
            cg_solver, disagg_sol, original_P_Post, p_derived_data, cg_solver.T, 
            start_day=p_start, end_day=p_end
        )
        
        # Filter x and LOS for Focus and Post patients (Disaggregated)
        focus_x = {k: v for k, v in disagg_x.items() if k[0] in original_P_F}
        focus_los = {k: v for k, v in disagg_los.items() if k in original_P_F}
        
        post_x = {k: v for k, v in disagg_x.items() if k[0] in original_P_Post}
        post_los = {k: v for k, v in disagg_los.items() if k in original_P_Post}
        
        # Build focus_y and post_y from disagg_y
        focus_y = {k: v for k, v in disagg_y.items() if k[0] in original_P_F}
        post_y = {k: v for k, v in disagg_y.items() if k[0] in original_P_Post}
        
        # Rename for output compatibility
        agg_focus_x = focus_x
        agg_post_x = post_x
        agg_focus_los = focus_los
        agg_post_los = post_los
        
        if cg_solver.master and hasattr(cg_solver.master, 'lmbda'):
            total_columns = len(cg_solver.master.lmbda)
            
    else:
        f_metrics = {}
        p_metrics = {}
        focus_y = {}
        post_y = {}
        # Ensure lists are at least available if solver fails, they default to initialized lists above

    # Combined metrics
    from collections import Counter
    drg_patients_E65A = f_metrics.get('drg_patients', {}).get('E65A', []) + p_metrics.get('drg_patients', {}).get('E65A', [])
    drg_patients_E65B = f_metrics.get('drg_patients', {}).get('E65B', []) + p_metrics.get('drg_patients', {}).get('E65B', [])
    drg_patients_E65C = f_metrics.get('drg_patients', {}).get('E65C', []) + p_metrics.get('drg_patients', {}).get('E65C', [])
    
    combined_continuity_violations = f_metrics.get('continuity_violations', []) + p_metrics.get('continuity_violations', [])
    
    f_ppt = f_metrics.get('patients_per_therapist', {})
    p_ppt = p_metrics.get('patients_per_therapist', {})
    combined_patients_per_therapist = dict(Counter(f_ppt) + Counter(p_ppt))
    
    # Check if final_ub equals sum of focus_los (Consistency Check)
    ub_equals_focus_los = 0
    sum_focus_los = 0
    if results.get('incumbent') is not None and agg_focus_los:
        # Since agg_focus_los IS NOW DISAGGREGATED, we just sum values directly (no Nr_agg needed)
        sum_focus_los = int(round(sum(agg_focus_los.values())))
        if abs(results.get('incumbent') - sum_focus_los) < 1e-6:
            ub_equals_focus_los = 1

    # Filter P_Pre to only include patients with Entry + LOS >= 0
    filtered_P_Pre = []
    for p in save_P_Pre:
        entry = save_Entry.get(p, 0)
        los = save_pre_los.get(p, 0)
        if entry + los >= 0:
            filtered_P_Pre.append(p)
    save_P_Pre = filtered_P_Pre

    # ===========================
    # BUILD INSTANCE DATA DICTIONARY
    # ===========================
    instance_data = {
        # Instance parameters
        'seed': seed,
        'D_focus': D_focus,
        'branching_strategy': branching_strategy,
        'pttr': pttr,
        'search_strategy': search_strategy,
        'learn_type': app_data['learn_type'][0],
        'OnlyHuman': 1 if app_data['learn_type'][0] == 0 else 0,
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
        
        # Results
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
        'max_tree_depth': results.get('max_tree_depth'),
        
        # Timing
        'total_time': results.get('total_time'),
        'time_in_mp': results.get('time_in_mp'),
        'time_in_sp': results.get('time_in_sp'),
        'time_in_ip_heuristic': results.get('time_in_ip_heuristic'),
        'time_in_root': results.get('time_in_root'),
        'time_in_branching': results.get('time_in_branching'),
        'time_to_first_incumbent': results.get('time_to_first_incumbent'),
        'time_overhead': results.get('time_overhead'),
        
        # Column statistics
        'pattern_size_counts': results.get('pattern_size_counts'),
        'total_columns': total_columns,
        
        # Instance data
        'P_Pre': save_P_Pre,
        'P_F': save_P_F,
        'P_Post': save_P_Post,
        'P_Join': save_P_Join,
        'Nr_agg': cg_solver.Nr_agg,
        'E_dict': save_E_dict,
        'Q_jt': cg_solver.Max_t,
        'Req': save_Req,
        'Entry': save_Entry,
        'pre_x': save_pre_x,
        'pre_los': save_pre_los,
        
        # Derived variables - Focus
        'focus_x': agg_focus_x, 
        'focus_y': focus_y, 
        'focus_los': agg_focus_los,
        'focus_e': f_e, 
        'focus_Y': f_Y, 
        'focus_theta': f_theta,
        'focus_omega': f_omega, 
        'focus_g_comp': f_g_comp,
        'focus_g_gap': f_g_gap,
        'focus_z': f_z,
        
        # Derived variables - Post
        'post_x': agg_post_x, 
        'post_y': post_y, 
        'post_los': agg_post_los,
        'post_e': p_e, 
        'post_Y': p_Y, 
        'post_theta': p_theta,
        'post_omega': p_omega, 
        'post_g_comp': p_g_comp,
        'post_g_gap': p_g_gap,
        'post_z': p_z,
        
        # Extra metrics with prefixes (excluding large trigram data)
        **{f"focus_{k}": v for k, v in f_metrics.items() if k != 'trigrams_per_patient'},
        **{f"post_{k}": v for k, v in p_metrics.items() if k != 'trigrams_per_patient'},
        
        # Combined metrics
        'drg_patients_E65A': drg_patients_E65A,
        'drg_patients_E65B': drg_patients_E65B,
        'drg_patients_E65C': drg_patients_E65C,
        'combined_continuity_violations': combined_continuity_violations,
        'combined_num_continuity_violations': len(combined_continuity_violations),
        'combined_patients_per_therapist': combined_patients_per_therapist,
        'ub_equals_focus_los': ub_equals_focus_los,
        'sum_focus_los': sum_focus_los,
        'focus_patient_count': len(save_P_F),  # Count of focus patients
    }
    
    return instance_data


def main_loop():
    """
    Main loop to solve instances from the newest Excel file in results/instances.
    Stores results in a dictionary and saves to Excel and pickle.
    """
    
    # ===========================
    # LOGGING CONFIGURATION
    # ===========================
    print_all_logs = False
    setup_multi_level_logging(base_log_dir='logs', enable_console=True, print_all_logs=print_all_logs)
    
    # ===========================
    # LOAD SCENARIOS FROM EXCEL
    # ===========================
    
    # Find the newest Excel file in results/instances
    instances_dir = 'results/instances'
    excel_files = glob.glob(os.path.join(instances_dir, '*.xlsx'))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {instances_dir}")
    
    # Get the newest file by modification time
    newest_excel = max(excel_files, key=os.path.getmtime)
    
    print("\n" + "=" * 100)
    print(f" LOADING SCENARIOS FROM: {newest_excel} ".center(100, "="))
    print("=" * 100 + "\n")
    
    # Load scenarios
    scenarios_df = pd.read_excel(newest_excel)
    
    print(f"Loaded {len(scenarios_df)} scenarios from {os.path.basename(newest_excel)}")
    print(f"Columns: {scenarios_df.columns.tolist()}\n")
    
    # Dictionary to store all results
    # Key format: instance_id from the Excel file
    results_dict = {}
    
    # DataFrame to collect all results
    results_df = pd.DataFrame()
    
    # ===========================
    # MAIN LOOP
    # ===========================
    total_instances = len(scenarios_df)
    
    # Define Configurations
    configurations = [
        {
            'name': 'linear',
            'overrides': {
                'learn_type': 'lin',
                'theta_base': 0.3,
                'lin_increase': 0.088,
                'k_learn': 0,
                'infl_point': 0
            }
        },
        {
            'name': 'sigmoid',
            'overrides': {
                'learn_type': 'sigmoid',
                'theta_base': 0.3,
                'k_learn': 1.5,
                'infl_point': 4,
                'lin_increase': 0
            }
        },
        {
            'name': 'exponential',
            'overrides': {
                'learn_type': 'exp',
                'theta_base': 0.3,
                'k_learn': 0.732,
                'lin_increase': 0,
                'infl_point': 0
            }
        },
        {
            'name': 'humanonly',
            'overrides': {
                'learn_type': 0,
                'theta_base': 0,
                'k_learn': 0,
                'lin_increase': 0,
                'infl_point': 0
            }
        }
    ]

    print("\n" + "=" * 100)
    print(f" BATCH RUN: {total_instances} instances x {len(configurations)} configs ".center(100, "="))
    print("=" * 100 + "\n")
    
    for idx, row in scenarios_df.iterrows():
        current_instance = idx + 1
        instance_id = row.get('instance_id', f'instance_{idx}')
        seed = int(row['seed'])
        D_focus = int(row['D_focus_count'])
        pttr = row.get('pttr', 'medium')
        T = int(row.get('T_count', 2))
        
        print("\n" + "=" * 100)
        print(f" Instance {current_instance}/{total_instances}: {instance_id} ".center(100, "="))
        print(f" seed={seed}, D_focus={D_focus}, pttr={pttr}, T={T} ".center(100, "="))
        print("=" * 100 + "\n")
        
        # Flag to track if we should skip remaining configs for this seed
        skip_remaining_configs = False
        
        for config in configurations:
            lt_name = config['name']
            overrides = config['overrides']
            
            # Create a unique instance_id for this learn_type
            current_instance_id = f"{instance_id}_{lt_name}"
            
            # Skip this configuration if a previous one was not optimal
            if skip_remaining_configs:
                print(f"\n⊘ Skipping learn_type: {lt_name} (ID: {current_instance_id}) - previous config was not optimal")
                
                # Store skipped indication with NOT_OPTIMAL status
                results_dict[current_instance_id] = {
                    'instance_id': current_instance_id,
                    'original_instance_id': instance_id,
                    'seed': seed,
                    'D_focus': D_focus,
                    'pttr': pttr,
                    'T': T,
                    'learn_type': overrides['learn_type'],
                    'config_name': lt_name,
                    'status': 'SKIPPED',
                    'is_optimal': False,
                    'final_ub': None,
                    'final_lb': None,
                    'final_gap': None,
                    'total_time': None,
                    'total_nodes': None
                }
                
                # Add to DataFrame
                results_df = pd.concat([results_df, pd.DataFrame([results_dict[current_instance_id]])], ignore_index=True)
                continue
            
            print(f"\n Solving for learn_type: {lt_name} (ID: {current_instance_id})")
            
            try:
                # Solve instance with parameters from Excel AND overrides
                instance_data = solve_instance(
                    seed=seed,
                    D_focus=D_focus,
                    pttr=pttr,
                    T=T,
                    allow_gaps=False,
                    use_warmstart=True,
                    dual_smoothing_alpha=None,
                    learn_type=overrides['learn_type'], # Pass learn type for logging/initial setup
                    app_data_overrides=overrides
                )
                
                # Add instance_id to the results
                instance_data['instance_id'] = current_instance_id
                instance_data['scenario_nr'] = row.get('scenario_nr', idx)
                instance_data['original_instance_id'] = instance_id
                instance_data['config_name'] = lt_name
                
                # Store in dictionary with instance_id as key
                results_dict[current_instance_id] = instance_data
                
                # Add to DataFrame
                results_df = pd.concat([results_df, pd.DataFrame([instance_data])], ignore_index=True)
                
                # Print summary
                print(f"\n✓ Instance {current_instance}/{total_instances} [{lt_name}] completed:")
                print(f"  - Instance ID: {current_instance_id}")
                print(f"  - Final UB: {instance_data['final_ub']}")
                print(f"  - Final LB: {instance_data['final_lb']}")
                print(f"  - Gap: {instance_data['final_gap']:.5%}" if instance_data['final_gap'] else "  - Gap: N/A")
                print(f"  - Total time: {instance_data['total_time']:.2f}s")
                print(f"  - Total nodes: {instance_data['total_nodes']}")
                print(f"  - Is optimal: {instance_data['is_optimal']}")
                
                # Check if solution is NOT optimal - if so, skip remaining configs for this seed
                if not instance_data.get('is_optimal', False):
                    print(f"\n⚠ Solution is NOT optimal - skipping remaining configurations for seed {seed}")
                    skip_remaining_configs = True
                
            except Exception as e:
                print(f"\n✗ Instance {current_instance}/{total_instances} [{lt_name}] FAILED:")
                print(f"  Error: {str(e)}")
                logger.error(f"Instance {current_instance_id} failed: {str(e)}", exc_info=True)
                
                # Store failure indication
                results_dict[current_instance_id] = {
                    'instance_id': current_instance_id,
                    'original_instance_id': instance_id,
                    'seed': seed,
                    'D_focus': D_focus,
                    'pttr': pttr,
                    'T': T,
                    'learn_type': overrides['learn_type'],
                    'config_name': lt_name,
                    'error': str(e),
                    'status': 'FAILED',
                    'is_optimal': False
                }
                
                # Also skip remaining configs if there's a failure
                print(f"\n⚠ Solver failed - skipping remaining configurations for seed {seed}")
                skip_remaining_configs = True
    
    # ===========================
    # SAVE RESULTS
    # ===========================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to Excel
    excel_filename = f"results/cg/results_loop_learning_{timestamp}.xlsx"
    try:
        os.makedirs('results/cg', exist_ok=True)
        results_df.to_excel(excel_filename, index=False)
        print(f"\n✓ Results saved to {excel_filename}")
    except Exception as e:
        print(f"\n✗ Could not save Excel: {e}")
    
    # Save dictionary to pickle
    pickle_filename = f"results/cg/results_dict_learning_{timestamp}.pkl"
    try:
        with open(pickle_filename, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"✓ Results dictionary saved to {pickle_filename}")
    except Exception as e:
        print(f"✗ Could not save pickle: {e}")
    
    # ===========================
    # SUMMARY
    # ===========================
    print("\n" + "=" * 100)
    print(" BATCH RUN SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"Total scenarios (seeds): {total_instances}")
    print(f"Total configurations: {total_instances * len(configurations)}")
    
    completed_instances = [v for v in results_dict.values() if v.get('status') not in ['FAILED', 'SKIPPED']]
    failed_instances = [v for v in results_dict.values() if v.get('status') == 'FAILED']
    skipped_instances = [v for v in results_dict.values() if v.get('status') == 'SKIPPED']
    
    print(f"Completed successfully: {len(completed_instances)}")
    print(f"Failed: {len(failed_instances)}")
    print(f"Skipped (non-optimal predecessor): {len(skipped_instances)}")
    
    if failed_instances:
        print("\n" + "-" * 50)
        print(" FAILED INSTANCES DETAILS ".center(50, "-"))
        print("-" * 50)
        for fail in failed_instances:
             print(f"• {fail['instance_id']}: {fail.get('error', 'Unknown error')}")
        print("-" * 50)

    print("=" * 100 + "\n")
    
    # Print summary table
    print("\nResults Summary Table:")
    print(f"{'Instance ID':<30} {'Seed':<8} {'D_focus':<10} {'PTTR':<10} {'Config':<12} {'UB':<15} {'LB':<15} {'Gap (%)':<12} {'Time (s)':<12} {'Nodes':<10} {'Optimal':<10}")
    print("-" * 150)
    
    for instance_id, data in results_dict.items():
        if data.get('status') == 'FAILED':
            print(f"{instance_id:<30} {data.get('seed', 'N/A'):<8} {data.get('D_focus', 'N/A'):<10} {data.get('pttr', 'N/A'):<10} {data.get('config_name', 'N/A'):<12} {'FAILED':<15} {'':<15} {'':<12} {'':<12} {'':<10} {'':<10}")
        elif data.get('status') == 'SKIPPED':
            print(f"{instance_id:<30} {data.get('seed', 'N/A'):<8} {data.get('D_focus', 'N/A'):<10} {data.get('pttr', 'N/A'):<10} {data.get('config_name', 'N/A'):<12} {'SKIPPED':<15} {'':<15} {'':<12} {'':<12} {'':<10} {'No':<10}")
        else:
            ub = f"{data.get('final_ub', 'N/A'):.2f}" if data.get('final_ub') else "N/A"
            lb = f"{data.get('final_lb', 'N/A'):.2f}" if data.get('final_lb') else "N/A"
            gap = f"{data.get('final_gap', 0)*100:.3f}" if data.get('final_gap') is not None else "N/A"
            time_val = f"{data.get('total_time', 0):.2f}" if data.get('total_time') else "N/A"
            nodes = data.get('total_nodes', 'N/A')
            optimal = "Yes" if data.get('is_optimal') else "No"
            seed_val = data.get('seed', 'N/A')
            d_focus_val = data.get('D_focus', 'N/A')
            pttr_val = data.get('pttr', 'N/A')
            config_val = data.get('config_name', 'N/A')
            print(f"{instance_id:<30} {seed_val:<8} {d_focus_val:<10} {pttr_val:<10} {config_val:<12} {ub:<15} {lb:<15} {gap:<12} {time_val:<12} {nodes:<10} {optimal:<10}")
    
    print("=" * 100 + "\n")
    
    return results_dict, results_df


if __name__ == "__main__":
    results_dict, results_df = main_loop()
