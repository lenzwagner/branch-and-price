
import math

def compute_derived_variables(cg_solver, inc_sol, app_data, patients_list=None):
    """
    Computes derived variables (e, Y, theta, omega, g_comp, z, g_gap) for a list of patients based on the incumbent solution.
    
    Args:
        cg_solver: The ColumnGeneration instance containing problem data (P_F, Entry_agg, D_Ext, etc.).
        inc_sol: The incumbent solution dictionary from the Branch-and-Price solver.
        app_data: Dictionary containing learning parameters (learn_type, theta_base, etc.).
        patients_list: Optional list of patients to compute variables for. If None, defaults to cg_solver.P_F.
        
    Returns:
        tuple: (all_e, all_Y, all_theta, all_omega, all_g_comp, all_z, all_g_gap) dictionaries.
            - all_g_comp: completion indicator (1 at discharge, 0 otherwise)
            - all_g_gap: gap indicator (1 if eligible but no treatment, 0 otherwise)
    """
    
    x_dict = inc_sol.get('x', {})
    y_dict = inc_sol.get('y', {})
    los_dict = inc_sol.get('LOS', {})
    
    # Get learning parameters
    learn_type = app_data['learn_type'][0]
    theta_base = app_data['theta_base'][0]
    lin_increase = app_data['lin_increase'][0]
    k_learn = app_data['k_learn'][0]
    infl_point = app_data['infl_point'][0]
    
    # Learning function
    def compute_theta(Y_count):
        if learn_type == 'lin':
            return min(1.0, theta_base + lin_increase * Y_count)
        elif learn_type == 'exp':
            return theta_base + (1 - theta_base) * (1 - math.exp(-k_learn * Y_count))
        elif learn_type == 'sigmoid':
            return theta_base + (1 - theta_base) / (1 + math.exp(-k_learn * (Y_count - infl_point)))
        else:
            return theta_base
    
    # Get time horizon
    D_ext = cg_solver.D_Ext
    
    # Initialize derived variable dictionaries
    all_e = {}
    all_Y = {}
    all_theta = {}
    all_omega = {}
    all_g_comp = {}  # Completion indicator (discharge day)
    all_g_gap = {}   # Gap indicator (eligible but no treatment)
    all_z = {}
    
    # Determine patients to iterate over
    target_patients = patients_list if patients_list is not None else cg_solver.P_F

    for p in target_patients:
        # Try to get entry from original patient dict first (if available), then aggregated
        entry = None
        if hasattr(cg_solver, 'Entry'):
            entry = cg_solver.Entry.get(p)
        
        if entry is None:
            entry = cg_solver.Entry_agg.get(p, min(D_ext))
        
        # Get LOS for this patient
        p_los = None
        for k, v in los_dict.items():
            if (k[0] if isinstance(k, tuple) else k) == p:
                p_los = v
                break
        
        if p_los is None:
            continue
        
        # z_{ij}: therapist assignment - DERIVED FROM x
        # Since Labeling algorithm doesn't compute z explicitly, we derive it from x:
        # z_{i,j} = 1 if patient i has ANY session with therapist j (i.e., ∃d: x_{i,j,d} > 0)
        
        # Find all therapists from x_dict assignments for this patient
        assigned_therapists = set()
        for k, v in x_dict.items():
            if k[0] == p and v > 0.5:
                # k is (p, t, d, col_id) - extract therapist t
                therapist = k[1]
                assigned_therapists.add(therapist)
        
        # Get all available therapists from Max_t or T
        all_J = set()
        if hasattr(cg_solver, 'Max_t') and cg_solver.Max_t:
            all_J = set(k[0] for k in cg_solver.Max_t.keys())
        elif hasattr(cg_solver, 'T') and cg_solver.T:
            all_J = set(cg_solver.T)
        
        # Store z derived for all therapists (default 0)
        for j in all_J:
            all_z[(p, j)] = 0
        
        # Set assigned ones to 1
        for j in assigned_therapists:
            all_z[(p, j)] = 1
        
        # print(f"\n📊 Patient {p} (Entry: {entry}, LOS: {p_los})")
        # print(f"   Assigned Therapist(s) z_ij: {assigned_therapists}")
        # print("-" * 75)
        # print(f"{'t':>4} | {'e_it':>4} | {'x_it':>4} | {'y_it':>4} | {'Y_it':>4} | {'θ_it':>6} | {'ω_it':>6} | {'g_it':>4}")
        # print("-" * 75)
        
        Y_cumulative = 0
        omega_cumulative = 0.0
        discharge_t = entry + p_los - 1  # Discharge period
        
        for t in D_ext:
            # e_{it}: eligibility (1 if in system)
            e_it = 1 if entry <= t <= entry + p_los - 1 else 0
            
            # x_{it}: therapy session (sum over all therapists)
            x_it = sum(v for k, v in x_dict.items() if k[0] == p and k[2] == t)
            
            # y_{it}: AI session - sum over all columns (col_ids)
            # y_dict keys can be (p, d, col_id) format
            y_it = 0
            for k, v in y_dict.items():
                if k[0] == p and k[1] == t:
                    y_it += v  # Sum across all columns
            
            # Y_{it}: cumulative AI sessions
            if y_it > 0:
                Y_cumulative += y_it
            Y_it = Y_cumulative
            
            # θ_{it}: effectiveness (0 if outside system)
            if e_it or t == entry: # Allow theta at entry
                 theta_it = compute_theta(Y_it)
            else:
                 theta_it = 0.0
            
            # Force theta to 0 if after discharge or before entry
            if t > discharge_t or t < entry:
                theta_it = 0.0
            
            # ω_{it}: cumulative progress
            if e_it:
                omega_cumulative += x_it + theta_it * y_it
            omega_it = omega_cumulative
            
            # g_{it}^Comp: completion indicator (1 at discharge, 0 otherwise)
            g_comp_it = 1 if t == discharge_t else 0
            
            # g_{it}^Gap: gap indicator (1 if eligible but no treatment, 0 otherwise)
            # Gap occurs when e=1 but x=0 and y=0 (eligible but idle)
            g_gap_it = 1 if (e_it == 1 and x_it == 0 and y_it == 0) else 0
            
            # Store in dicts
            all_e[(p, t)] = e_it
            all_Y[(p, t)] = Y_it
            all_theta[(p, t)] = theta_it
            all_omega[(p, t)] = omega_it
            all_g_comp[(p, t)] = g_comp_it
            all_g_gap[(p, t)] = g_gap_it

            # Only print if in system
            # if e_it or t == entry - 1:
            #     print(f"{t:>4} | {e_it:>4} | {x_it:>4} | {y_it:>4} | {Y_it:>4} | {theta_it:>6.3f} | {omega_it:>6.2f} | {g_it:>4}")
        
        # print(f"\nFinal: ω = {omega_cumulative:.2f}, Req = {cg_solver.Req_agg.get(p, '?')}")

    return all_e, all_Y, all_theta, all_omega, all_g_comp, all_z, all_g_gap
