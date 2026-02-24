import numpy as np
from collections import defaultdict

def calculate_extra_metrics(cg_solver, inc_sol, patients_list, derived_data, T, start_day=None, end_day=None):
    """
    Calculates detailed metrics (E.2 - E.8) based on the solution and derived variables.

    Metrics are calculated in two scopes:
    - period_*: Over the defined time horizon (start_day to end_day)
    - patient_*: Over all days where the specified patients have sessions

    Args:
        cg_solver: The ColumnGeneration object (contains static data, M_p, Max_t, etc.)
        inc_sol: The incumbent solution dict (contains 'x', 'y' raw keys)
        patients_list: List of patients to consider (e.g. cg_solver.P_F)
        derived_data: Dict containing 'e', 'Y', 'theta', 'omega', 'g', 'z', etc.
        T: Set of therapists
        start_day: Start of the period horizon (for period_* metrics)
        end_day: End of the period horizon (for period_* metrics)

    Returns:
        dict: A dictionary containing all the calculated metrics.
    """

    # Unpack derived data
    e_dict = derived_data.get('e', {})
    Y_dict = derived_data.get('Y', {})
    theta_dict = derived_data.get('theta', {})
    z_dict = derived_data.get('z', {})

    # Raw solution data
    raw_x = inc_sol.get('x', {})
    raw_y = inc_sol.get('y', {})

    # ==============================================================================
    # AGGREGATE x AND y FOR PATIENTS IN patients_list
    # ==============================================================================

    # x_agg: (p, t, d) -> value (aggregated over col_ids)
    x_agg = defaultdict(float)
    for k, v in raw_x.items():
        if v > 1e-6:
            p = k[0]
            if p in patients_list:
                t, d = k[1], k[2]
                x_agg[(p, t, d)] += v

    # y_agg: (p, d) -> value (aggregated over col_ids)
    y_agg = defaultdict(float)
    for k, v in raw_y.items():
        if v > 1e-6:
            p = k[0]
            if p in patients_list:
                d = k[1]
                y_agg[(p, d)] += v

    # ==============================================================================
    # DETERMINE DAY RANGES
    # ==============================================================================

    # Period range: from start_day to end_day (passed as parameters)
    if x_agg:
        actual_days = set(d for (p, t, d) in x_agg.keys())
        patient_start = min(actual_days)
        patient_end = max(actual_days)
    else:
        patient_start = start_day if start_day is not None else 1
        patient_end = end_day if end_day is not None else 1

    # Use passed parameters for period range, with fallback
    period_start = start_day if start_day is not None else patient_start
    period_end = end_day if end_day is not None else patient_end

    period_range = range(period_start, period_end + 1)
    patient_range = range(patient_start, patient_end + 1)

    metrics = {}

    # ==============================================================================
    # E.2 RESOURCE UTILIZATION METRICS - PERIOD SCOPE (period_*)
    # ==============================================================================

    # Filter to only include sessions within period_range
    period_x = {k: v for k, v in x_agg.items() if k[2] in period_range}
    period_y = {k: v for k, v in y_agg.items() if k[1] in period_range}

    # N_human (period): Σ x_{ijt} for t ∈ period
    period_N_human = sum(period_x.values())
    period_N_human_total = period_N_human # Alias if we want total shown
    
    # NOTE: User requested Option 2: Subtract Pre-Patients from Capacity (Denominator)
    # So N_human remains just the Focus/Post sessions.


    # N_AI (period): Σ y_{it} for t ∈ period
    period_N_AI = sum(period_y.values())

    # N_total (period)
    period_N_total = period_N_human + period_N_AI

    # AI/Human Session Shares (period)
    period_ai_share = (period_N_AI / period_N_total * 100) if period_N_total > 0 else 0
    period_human_share = (period_N_human / period_N_total * 100) if period_N_total > 0 else 0

    # C_total (period): Σ Q_{jt} for t ∈ period - PRE_X USAGE
    period_C_total = 0
    if hasattr(cg_solver, 'Max_t'):
        for t_id in T:
            for d in period_range:
                if (t_id, d) in cg_solver.Max_t:
                    period_C_total += cg_solver.Max_t[(t_id, d)]
    
    # SUBTRACT PRE-PATIENT LOAD FROM CAPACITY (Option 2)
    period_C_net = period_C_total
    pre_load_period = 0
    if hasattr(cg_solver, 'pre_x'):
        for (p, t, d), val in cg_solver.pre_x.items():
            if d in period_range and val > 0.5:
                pre_load_period += val
        period_C_net = max(0, period_C_total - pre_load_period)

    # Human Utilization (period) = Focus/Post Sessions / (Gross Capacity - Pre Sessions)
    period_human_util = (period_N_human / period_C_net * 100) if period_C_net > 0 else 0


    # Therapist Workload (period): Σ_i x_{ijt} for each j
    period_therapist_workload = defaultdict(float)
    for (p, t, d), val in period_x.items():
        period_therapist_workload[t] += val

    # Average Therapist Workload (period)
    num_therapists = len(T) if T else 1
    period_avg_workload = sum(period_therapist_workload.values()) / num_therapists if period_therapist_workload else 0

    # Peak Therapist Workload (period): max_j Σ_{i,t} x_{ijt}
    period_peak_workload = max(period_therapist_workload.values()) if period_therapist_workload else 0

    # Peak Therapist-Day Workload (period): max_{j,t} Σ_i x_{ijt}
    period_therapist_day = defaultdict(float)
    for (p, t, d), val in period_x.items():
        period_therapist_day[(t, d)] += val
    period_peak_day_workload = max(period_therapist_day.values()) if period_therapist_day else 0

    # Peak Period Utilization (period): max_t (Σ_{i,j} x_{ijt} / Σ_j Q_{jt})
    period_day_utilization = {}
    for d in period_range:
        sessions_on_day = sum(val for (p, t, day), val in period_x.items() if day == d)
        
        # Calculate Net Capacity for the day
        capacity_on_day_gross = sum(cg_solver.Max_t.get((t_id, d), 0) for t_id in T) if hasattr(cg_solver, 'Max_t') else 0
        pre_load_on_day = 0
        if hasattr(cg_solver, 'pre_x'):
             pre_load_on_day = sum(val for (p, t, day), val in cg_solver.pre_x.items() if day == d)
        
        capacity_on_day_net = max(0, capacity_on_day_gross - pre_load_on_day)

        if capacity_on_day_net > 0:
            period_day_utilization[d] = sessions_on_day / capacity_on_day_net * 100
    period_peak_util = max(period_day_utilization.values()) if period_day_utilization else 0

    # NEW: DETAILED PER-THERAPIST DAILY UTILIZATION (Net Capacity)
    # 1. Calculate Util_{j,t} = Sessions_{j,t} / (Cap_{j,t} - Pre_{j,t})
    therapist_daily_util_dict = defaultdict(dict) # {t: {d: util}}
    therapist_daily_util_values = defaultdict(list) # {t: [utils]} for easy aggression
    therapist_max_util = {}
    therapist_avg_util = {}

    for t_id in T:
        for d in period_range:
            # Gross Capacity
            cap_gross = cg_solver.Max_t.get((t_id, d), 0) if hasattr(cg_solver, 'Max_t') else 0
            
            # Skip if therapist not working that day
            if cap_gross <= 0:
                continue

            # Pre-Load
            pre_load = 0
            if hasattr(cg_solver, 'pre_x'):
                pre_load = sum(val for (p, t, day), val in cg_solver.pre_x.items() if t == t_id and day == d)
            
            # Net Capacity
            cap_net = max(0, cap_gross - pre_load)

            # Focus/Post Load
            # Optimized lookup: period_x is {(p,t,d): val}
            # This inner loop is okay for small T/D but could be optimized if needed
            load_focus = sum(val for (p, t, day), val in period_x.items() if t == t_id and day == d)

            if cap_net > 0:
                util = (load_focus / cap_net) * 100
                therapist_daily_util_dict[t_id][d] = util
                therapist_daily_util_values[t_id].append(util)
            else:
                # Capacity was fully consumed by Pre-Patients or 0 net capacity
                if load_focus > 0:
                     therapist_daily_util_dict[t_id][d] = float('inf')
                     therapist_daily_util_values[t_id].append(float('inf'))
                # else: ignore undefined utilization
    
    # Aggregating per therapist
    for t_id in T:
        utils = therapist_daily_util_values[t_id]
        if utils:
            therapist_max_util[t_id] = max(utils)
            therapist_avg_util[t_id] = sum(utils) / len(utils)
        else:
            therapist_max_util[t_id] = 0.0
            therapist_avg_util[t_id] = 0.0

    # Calculate global average of Max Daily Util per Therapist (e.g., "Avg Peak Load per Therapist")
    avg_peak_daily_load_per_therapist = sum(therapist_max_util.values()) / len(T) if T else 0

    metrics['therapist_daily_util_dict'] = dict(therapist_daily_util_dict)
    metrics['therapist_max_util'] = therapist_max_util
    metrics['therapist_avg_util'] = therapist_avg_util


    # Store period metrics
    metrics['period_start_day'] = period_start
    metrics['period_end_day'] = period_end
    metrics['period_N_human'] = period_N_human
    metrics['period_N_AI'] = period_N_AI
    metrics['period_N_total'] = period_N_total
    metrics['period_ai_share_pct'] = period_ai_share
    metrics['period_human_share_pct'] = period_human_share
    metrics['period_C_total'] = period_C_total
    metrics['period_human_util_pct'] = period_human_util
    metrics['period_avg_workload'] = period_avg_workload
    metrics['period_peak_workload'] = period_peak_workload
    metrics['period_peak_day_workload'] = period_peak_day_workload
    metrics['period_peak_util_pct'] = period_peak_util
    metrics['therapist_daily_util_dict'] = dict(therapist_daily_util_dict)
    metrics['therapist_max_daily_util'] = therapist_max_util
    metrics['therapist_avg_daily_util'] = therapist_avg_util

    # ==============================================================================
    # E.2 RESOURCE UTILIZATION METRICS - PATIENT SCOPE (patient_*)
    # ==============================================================================

    # N_human (patient): Σ x_{ijt} for all sessions of these patients
    patient_N_human = sum(x_agg.values())
    patient_N_human_total = patient_N_human
    
    # NOTE: Option 2 - No Pre-Patients in numerator.


    # N_AI (patient): Σ y_{it} for all sessions of these patients
    patient_N_AI = sum(y_agg.values())

    # N_total (patient)
    patient_N_total = patient_N_human + patient_N_AI

    # AI/Human Session Shares (patient)
    patient_ai_share = (patient_N_AI / patient_N_total * 100) if patient_N_total > 0 else 0
    patient_human_share = (patient_N_human / patient_N_total * 100) if patient_N_total > 0 else 0

    # C_total (patient): Σ Q_{jt} for t ∈ patient_range - PRE_X USAGE
    patient_C_total = 0
    if hasattr(cg_solver, 'Max_t'):
        for t_id in T:
            for d in patient_range:
                if (t_id, d) in cg_solver.Max_t:
                    patient_C_total += cg_solver.Max_t[(t_id, d)]
    
    # SUBTRACT PRE-PATIENT LOAD FROM CAPACITY (Option 2)
    patient_C_net = patient_C_total
    pre_load_patient = 0
    if hasattr(cg_solver, 'pre_x'):
        for (p, t, d), val in cg_solver.pre_x.items():
            if d in patient_range and val > 0.5:
                pre_load_patient += val
        patient_C_net = max(0, patient_C_total - pre_load_patient)

    # Human Utilization (patient)
    patient_human_util = (patient_N_human / patient_C_net * 100) if patient_C_net > 0 else 0


    # Therapist Workload (patient)
    patient_therapist_workload = defaultdict(float)
    for (p, t, d), val in x_agg.items():
        patient_therapist_workload[t] += val

    # Average Therapist Workload (patient)
    patient_avg_workload = sum(patient_therapist_workload.values()) / num_therapists if patient_therapist_workload else 0

    # Peak Therapist Workload (patient)
    patient_peak_workload = max(patient_therapist_workload.values()) if patient_therapist_workload else 0

    # Peak Therapist-Day Workload (patient)
    patient_therapist_day = defaultdict(float)
    for (p, t, d), val in x_agg.items():
        patient_therapist_day[(t, d)] += val
    patient_peak_day_workload = max(patient_therapist_day.values()) if patient_therapist_day else 0

    # Peak Period Utilization (patient)
    patient_day_utilization = {}
    for d in patient_range:
        sessions_on_day = sum(val for (p, t, day), val in x_agg.items() if day == d)
        capacity_on_day = sum(cg_solver.Max_t.get((t_id, d), 0) for t_id in T) if hasattr(cg_solver, 'Max_t') else 0
        if capacity_on_day > 0:
            patient_day_utilization[d] = sessions_on_day / capacity_on_day * 100
    patient_peak_util = max(patient_day_utilization.values()) if patient_day_utilization else 0

    # Store patient metrics
    metrics['patient_start_day'] = patient_start
    metrics['patient_end_day'] = patient_end
    metrics['patient_N_human'] = patient_N_human
    metrics['patient_N_AI'] = patient_N_AI
    metrics['patient_N_total'] = patient_N_total
    metrics['patient_ai_share_pct'] = patient_ai_share
    metrics['patient_human_share_pct'] = patient_human_share
    metrics['patient_C_total'] = patient_C_total
    metrics['patient_human_util_pct'] = patient_human_util
    metrics['patient_avg_workload'] = patient_avg_workload
    metrics['patient_peak_workload'] = patient_peak_workload
    metrics['patient_peak_day_workload'] = patient_peak_day_workload
    metrics['patient_peak_util_pct'] = patient_peak_util

    # Avg Human Sessions per Patient (Outcome View)
    patient_avg_human_sessions = patient_N_human / len(patients_list) if patients_list else 0
    metrics['patient_avg_human_sessions'] = patient_avg_human_sessions

    # ==============================================================================
    # E.3 AI LEARNING DYNAMICS METRICS
    # ==============================================================================

    # Get theta from derived_data
    theta_dict = derived_data.get('theta', {})

    # --- Per-Patient Dicts ---
    initial_theta = {}  # θ_{i,r_i} - effectiveness at admission
    final_theta = {}    # θ_{i,F_i} - effectiveness at discharge (last day with e=1)
    max_theta = {}      # max_t θ_it
    min_theta_at_first_ai = {} # min theta at first AI use
    ai_sessions_per_patient = {}  # Σ_t y_{it} per patient
    time_to_proficiency = {}  # min{t - r_i | θ_{it} > 0.7}
    max_consecutive_ai = {}   # longest streak of consecutive y=1 per patient

    # --- Aggregate Values ---
    theta_when_ai_used_sum = 0.0
    theta_when_ai_used_count = 0
    all_consecutive_streaks = []

    for p in patients_list:
        # Robust Entry lookup
        entry_day = 1
        if hasattr(cg_solver, 'Entry') and p in cg_solver.Entry:
             entry_day = cg_solver.Entry[p]
        else:
             entry_day = cg_solver.Entry_agg.get(p, 1)

        # Find last day with e=1 (discharge day)
        patient_e_days = [(d, e_dict.get((p, d), 0)) for d in sorted(set(d for (pi, d) in e_dict.keys() if pi == p))]
        last_eligible_day = entry_day
        for d, e_val in patient_e_days:
            if e_val == 1:
                last_eligible_day = d

        # Initial Theta: θ_{i,r_i}
        initial_theta[p] = theta_dict.get((p, entry_day), 0.0)

        # Final Theta: θ_{i,F_i} (at discharge/last eligible day)
        final_theta[p] = theta_dict.get((p, last_eligible_day), 0.0)

        # Max Theta per Patient
        patient_thetas = [theta_dict.get((p, d), 0.0) for d in range(entry_day, last_eligible_day + 1)]
        max_theta[p] = max(patient_thetas) if patient_thetas else 0.0

        # Minimum Theta at First AI Use
        # Find first day where y=1
        first_ai_day = None
        for d in range(entry_day, last_eligible_day + 1):
            if y_agg.get((p, d), 0) > 0.5:
                first_ai_day = d
                break
        
        if first_ai_day is not None:
            min_theta_at_first_ai[p] = theta_dict.get((p, first_ai_day), 0.0)
        else:
            min_theta_at_first_ai[p] = None


        # AI Sessions per Patient: Σ_t y_{it}
        patient_ai_count = sum(v for (pi, d), v in y_agg.items() if pi == p)
        ai_sessions_per_patient[p] = patient_ai_count

        # Time to Proficiency: min{t - r_i | θ_{it} > 0.7}
        proficiency_threshold = 0.7
        ttp = None
        patient_theta_days = sorted([(d, theta_dict.get((p, d), 0.0)) for d in range(entry_day, last_eligible_day + 1)])
        for d, theta_val in patient_theta_days:
            if theta_val > proficiency_threshold:
                ttp = d - entry_day
                break
        time_to_proficiency[p] = ttp  # None if never reached

        # Max Consecutive AI Sessions
        # Build sequence of y values for this patient
        y_sequence = []
        for d in range(entry_day, last_eligible_day + 1):
            y_val = y_agg.get((p, d), 0)
            y_sequence.append(1 if y_val > 0.5 else 0)

        # Find max consecutive 1s
        max_streak = 0
        current_streak = 0
        streaks = []
        for y_val in y_sequence:
            if y_val == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    all_consecutive_streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)
            all_consecutive_streaks.append(current_streak)
        max_consecutive_ai[p] = max(streaks) if streaks else 0

        # Average Theta When AI Used: accumulate
        for d in range(entry_day, last_eligible_day + 1):
            y_val = y_agg.get((p, d), 0)
            if y_val > 0.5:
                theta_val = theta_dict.get((p, d), 0.0)
                theta_when_ai_used_sum += theta_val
                theta_when_ai_used_count += 1

    # --- Compute Aggregate Values ---

    # Average Final Theta
    final_theta_values = [v for v in final_theta.values() if v is not None]
    avg_final_theta = sum(final_theta_values) / len(final_theta_values) if final_theta_values else 0.0

    # Average Theta When AI Used
    avg_theta_when_ai_used = theta_when_ai_used_sum / theta_when_ai_used_count if theta_when_ai_used_count > 0 else 0.0

    # Average AI Sessions per Patient
    avg_ai_sessions = sum(ai_sessions_per_patient.values()) / len(ai_sessions_per_patient) if ai_sessions_per_patient else 0.0

    # Average Time to Proficiency (only for patients who reached it)
    ttp_values = [v for v in time_to_proficiency.values() if v is not None]
    avg_time_to_proficiency = sum(ttp_values) / len(ttp_values) if ttp_values else None

    # Max Consecutive AI (global)
    max_consecutive_ai_global = max(max_consecutive_ai.values()) if max_consecutive_ai else 0

    # Average Consecutive AI (across all streaks)
    avg_consecutive_ai = sum(all_consecutive_streaks) / len(all_consecutive_streaks) if all_consecutive_streaks else 0.0

    # Store E.3 metrics
    metrics['initial_theta'] = initial_theta
    metrics['final_theta'] = final_theta
    metrics['avg_final_theta'] = avg_final_theta
    metrics['avg_theta_when_ai_used'] = avg_theta_when_ai_used
    metrics['max_theta'] = max_theta
    metrics['min_theta_at_first_ai'] = min_theta_at_first_ai
    metrics['ai_sessions_per_patient'] = ai_sessions_per_patient
    metrics['avg_ai_sessions'] = avg_ai_sessions
    metrics['time_to_proficiency'] = time_to_proficiency
    metrics['avg_time_to_proficiency'] = avg_time_to_proficiency
    metrics['max_consecutive_ai'] = max_consecutive_ai
    metrics['max_consecutive_ai_global'] = max_consecutive_ai_global
    metrics['avg_consecutive_ai'] = avg_consecutive_ai

    # ==============================================================================
    # CONSOLIDATED OUTPUT - All Metrics Summary (SUPPRESSED FOR STRESS TEST)
    # ==============================================================================
    # print(f"\n{'='*80}")
    # print(f" E.2 RESOURCE UTILIZATION METRICS ({len(patients_list)} patients) ".center(80, '='))
    # ...


    # print(f"\n{'='*80}")
    # print(f" E.3 AI LEARNING DYNAMICS METRICS ".center(80, '='))
    # ...


    # ==============================================================================
    # E.4 THERAPIST CONTINUITY METRICS
    # ==============================================================================

    # z_dict is already available from derived_data
    z_derived = derived_data.get('z', {})

    # Count therapists assigned per patient: Σ_j z_{ij}
    therapists_assigned = {}  # p -> count of therapists with z=1
    primary_therapist = {}    # p -> therapist j where z_{ij}=1 (or list if multiple)

    for p in patients_list:
        # Get all z values for this patient
        patient_z = {k: v for k, v in z_derived.items() if k[0] == p and v == 1}
        assigned_count = len(patient_z)
        therapists_assigned[p] = assigned_count

        # Primary therapist(s)
        if assigned_count == 1:
            primary_therapist[p] = list(patient_z.keys())[0][1]  # Single therapist
        elif assigned_count > 1:
            primary_therapist[p] = [k[1] for k in patient_z.keys()]  # Multiple (continuity violation)
        else:
            primary_therapist[p] = None  # No therapist assigned (all AI?)

    # Continuity violations: patients with more than 1 therapist assigned
    continuity_violations = [p for p, count in therapists_assigned.items() if count > 1]
    num_continuity_violations = len(continuity_violations)

    # Patients per therapist: {therapist_id: count_of_patients}
    patients_per_therapist = defaultdict(int)
    for p, therapist in primary_therapist.items():
        if therapist is not None:
            if isinstance(therapist, list):
                # Multiple therapists (continuity violation)
                for t in therapist:
                    patients_per_therapist[t] += 1
            else:
                patients_per_therapist[therapist] += 1
    patients_per_therapist = dict(patients_per_therapist)

    # Store E.4 metrics
    metrics['therapists_assigned'] = therapists_assigned
    metrics['primary_therapist'] = primary_therapist
    metrics['patients_per_therapist'] = patients_per_therapist
    metrics['continuity_violations'] = continuity_violations
    metrics['num_continuity_violations'] = num_continuity_violations

    # Print E.4
    # print(f"\n{'='*80}")
    # print(f" E.4 THERAPIST CONTINUITY METRICS ".center(80, '='))
    # ...


    # ==============================================================================
    # E.5 TREATMENT GAP METRICS (Relaxed Continuity Only)
    # ==============================================================================

    # Gap Indicator: gap_{it} = e_{it} - Σ_j x_{ijt} - y_{it}
    # Equals 1 if idle day (eligible but no treatment), 0 otherwise

    gap_per_day = {}           # {(p, d): 0 or 1}
    total_gaps_per_patient = {}  # {p: count}
    longest_gap_per_patient = {}  # {p: max_consecutive_gap}
    theta_during_gaps = {}     # {p: [theta values during gaps]}

    for p in patients_list:
        # Robust Entry lookup
        entry_day = 1
        if hasattr(cg_solver, 'Entry') and p in cg_solver.Entry:
             entry_day = cg_solver.Entry[p]
        else:
             entry_day = cg_solver.Entry_agg.get(p, 1)

        # Find all days where e=1 for this patient
        eligible_days = sorted([d for (pi, d) in e_dict.keys() if pi == p and e_dict.get((pi, d), 0) == 1])

        if not eligible_days:
            total_gaps_per_patient[p] = 0
            longest_gap_per_patient[p] = 0
            theta_during_gaps[p] = []
            continue

        patient_gaps = []
        patient_theta_gaps = []
        current_gap_streak = 0
        max_gap_streak = 0

        for d in eligible_days:
            # Check for any treatment on this day
            has_human = any(v > 0.5 for (pi, t, day), v in x_agg.items() if pi == p and day == d)
            has_ai = y_agg.get((p, d), 0) > 0.5

            # Gap indicator: e=1 but no x and no y
            is_gap = 1 if (not has_human and not has_ai) else 0
            gap_per_day[(p, d)] = is_gap
            patient_gaps.append(is_gap)

            if is_gap == 1:
                current_gap_streak += 1
                # Record theta during gap
                theta_val = theta_dict.get((p, d), 0.0)
                patient_theta_gaps.append(theta_val)
            else:
                max_gap_streak = max(max_gap_streak, current_gap_streak)
                current_gap_streak = 0

        # Final streak check
        max_gap_streak = max(max_gap_streak, current_gap_streak)

        total_gaps_per_patient[p] = sum(patient_gaps)
        longest_gap_per_patient[p] = max_gap_streak
        theta_during_gaps[p] = patient_theta_gaps

    # Aggregate metrics
    total_idle_days = sum(total_gaps_per_patient.values())
    avg_gaps_per_patient = total_idle_days / len(patients_list) if patients_list else 0

    # Average theta during all gaps
    all_gap_thetas = [t for thetas in theta_during_gaps.values() for t in thetas]
    avg_theta_during_gaps = sum(all_gap_thetas) / len(all_gap_thetas) if all_gap_thetas else None

    # Store E.5 metrics
    metrics['gap_per_day'] = gap_per_day
    metrics['total_gaps_per_patient'] = total_gaps_per_patient
    metrics['longest_gap_per_patient'] = longest_gap_per_patient
    metrics['theta_during_gaps'] = theta_during_gaps
    metrics['total_idle_days'] = total_idle_days
    metrics['avg_gaps_per_patient'] = avg_gaps_per_patient
    metrics['avg_theta_during_gaps'] = avg_theta_during_gaps

    # Print E.5
    # print(f"\n{'='*80}")
    # print(f" E.5 TREATMENT GAP METRICS ".center(80, '='))
    # ...


    # ==============================================================================
    # E.6 DRG-SPECIFIC METRICS
    # ==============================================================================

    # Get DRG mapping and LOS data
    drg_agg = getattr(cg_solver, 'DRG_agg', {})
    req_agg = getattr(cg_solver, 'Req_agg', {})
    nr_agg = getattr(cg_solver, 'Nr_agg', {})
    los_dict = inc_sol.get('LOS', {})

    # Get LOS per patient (aggregate over col_ids)
    patient_los = {}
    for k, v in los_dict.items():
        p = k[0] if isinstance(k, tuple) else k
        if p in patients_list:
            patient_los[p] = v

    # DRG groups
    drg_groups = ['E65A', 'E65B', 'E65C']

    # Initialize DRG metrics
    drg_patient_count = {g: 0 for g in drg_groups}
    drg_los_sum = {g: 0.0 for g in drg_groups}
    drg_req_sum = {g: 0.0 for g in drg_groups}
    drg_human_sessions = {g: 0.0 for g in drg_groups}
    drg_ai_sessions = {g: 0.0 for g in drg_groups}
    drg_patients = {g: [] for g in drg_groups}  # List of profile IDs per DRG

    for p in patients_list:
        # Robust DRG lookup
        drg = 'Unknown'
        if hasattr(cg_solver, 'DRG') and p in cg_solver.DRG:
             drg = cg_solver.DRG[p]
        else:
             drg = drg_agg.get(p, 'Unknown')
             
        if drg not in drg_groups:
            continue

        # Track which profiles/patients belong to this DRG
        drg_patients[drg].append(p)

        # Patient count (weighted by Nr_agg). 
        # If disaggregated, p is patient ID and not in nr_agg, so returns 1 (correct).
        # If aggregated, p is profile ID, returns count (correct).
        n_patients = nr_agg.get(p, 1)
        drg_patient_count[drg] += n_patients

        # LOS sum
        los_val = patient_los.get(p, 0)
        drg_los_sum[drg] += los_val * n_patients

        # Robust Requirements lookup
        req_val = 0
        if hasattr(cg_solver, 'Req') and p in cg_solver.Req:
            req_val = cg_solver.Req[p]
        else:
            req_val = req_agg.get(p, 0)
            
        drg_req_sum[drg] += req_val * n_patients

        # Human sessions for this patient
        human_sessions = sum(v for (pi, t, d), v in x_agg.items() if pi == p)
        drg_human_sessions[drg] += human_sessions * n_patients

        # AI sessions for this patient
        ai_sessions = sum(v for (pi, d), v in y_agg.items() if pi == p)
        drg_ai_sessions[drg] += ai_sessions * n_patients

    # Calculate DRG metrics
    drg_avg_los = {}
    drg_avg_req = {}
    drg_ai_share = {}

    for g in drg_groups:
        if drg_patient_count[g] > 0:
            drg_avg_los[g] = drg_los_sum[g] / drg_patient_count[g]
            drg_avg_req[g] = drg_req_sum[g] / drg_patient_count[g]
            total_sessions = drg_human_sessions[g] + drg_ai_sessions[g]
            drg_ai_share[g] = (drg_ai_sessions[g] / total_sessions * 100) if total_sessions > 0 else 0.0
        else:
            drg_avg_los[g] = None
            drg_avg_req[g] = None
            drg_ai_share[g] = None

    # Calculate Global Average LoS
    total_los = sum(patient_los.values())
    avg_los = total_los / len(patients_list) if patients_list else 0
    metrics['avg_los'] = avg_los

    # Store E.6 metrics
    metrics['drg_patients'] = drg_patients
    metrics['drg_patient_count'] = drg_patient_count
    metrics['drg_avg_los'] = drg_avg_los
    metrics['drg_avg_req'] = drg_avg_req
    metrics['drg_ai_share'] = drg_ai_share

    # Print E.6
    # print(f"\n{'='*80}")
    # print(f" E.6 DRG-SPECIFIC METRICS ".center(80, '='))
    # ...


    # ==============================================================================
    # E.7 SESSION PATTERN ANALYSIS
    # ==============================================================================

    # Build session sequence for each patient
    # For each day d where e_{p,d} = 1:
    #   - 'H' if any x_{p,j,d} > 0 (Human session)
    #   - 'A' if y_{p,d} > 0 (AI session)
    #   - '_' if neither (gap, only when allow_gaps=True)

    session_dict = {}       # {p: {d1: 'H', d2: 'A', d3: 'H', ...}}
    session_string = {}     # {p: "HAH..."} for easy pattern analysis

    for p in patients_list:
        # Robust Entry lookup
        entry_day = 1
        if hasattr(cg_solver, 'Entry') and p in cg_solver.Entry:
             entry_day = cg_solver.Entry[p]
        else:
             entry_day = cg_solver.Entry_agg.get(p, 1)

        # Find all days where e=1 for this patient
        eligible_days = sorted([d for (pi, d) in e_dict.keys() if pi == p and e_dict.get((pi, d), 0) == 1])

        if not eligible_days:
            session_dict[p] = {}
            session_string[p] = ""
            continue

        patient_sessions = {}
        sequence = []
        for d in eligible_days:
            # Check for human session: any x_{p,j,d} > 0
            has_human = any(v > 0.5 for (pi, t, day), v in x_agg.items() if pi == p and day == d)

            # Check for AI session: y_{p,d} > 0
            has_ai = y_agg.get((p, d), 0) > 0.5

            if has_human:
                session_type = 'H'
            elif has_ai:
                session_type = 'A'
            else:
                session_type = '_'  # Gap

            patient_sessions[d] = session_type
            sequence.append(session_type)

        session_dict[p] = patient_sessions
        session_string[p] = ''.join(sequence)

    # --- Trigram Analysis ---
    # Extract all 3-character substrings from session sequences

    trigrams_per_patient = {}  # {p: ["HHA", "HAA", "AAH", ...]}
    trigram_frequency = defaultdict(int)  # {"HHA": 5, "HAH": 3, ...}

    for p, seq in session_string.items():
        trigrams = []
        for i in range(len(seq) - 2):
            trigram = seq[i:i+3]
            trigrams.append(trigram)
            trigram_frequency[trigram] += 1
        trigrams_per_patient[p] = trigrams

    trigram_frequency = dict(trigram_frequency)

    # Store E.7 metrics
    metrics['session_dict'] = session_dict
    metrics['session_string'] = session_string
    metrics['trigrams_per_patient'] = trigrams_per_patient
    metrics['trigram_frequency'] = trigram_frequency

    # Print E.7
    # print(f"\n{'='*80}")
    # print(f" E.7 SESSION PATTERN ANALYSIS ".center(80, '='))
    # print(f"{'='*80}")

    # print(f"\n--- Session Dict per Patient (day -> type) ---")
    # for p, days_dict in session_dict.items():
    #     print(f"  Patient {p}: {days_dict}")

    # print(f"\n--- Session Strings ---")
    # for p, seq in session_string.items():
    #     print(f"  Patient {p}: {seq}")

    # print(f"\n--- Trigrams per Patient ---")
    # for p, tris in trigrams_per_patient.items():
    #     print(f"  Patient {p}: {tris}")

    # print(f"\n--- Trigram Frequency ---")
    # sorted_trigrams = sorted(trigram_frequency.items(), key=lambda x: -x[1])
    # for trigram, count in sorted_trigrams:
    #     print(f"  {trigram}: {count}")

    # print(f"{'='*80}\n")

    return metrics