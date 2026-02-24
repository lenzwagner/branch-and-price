
import numpy as np
from numba import njit, int64, float64, types, prange
from numba.typed import List, Dict


# =============================================================================
# NUMBA-OPTIMIZED HELPER FUNCTIONS
# =============================================================================

@njit(cache=True)
def validate_column_history_numba(path_mask, duration, MS, MIN_MS):
    """
    Numba-optimized validation that a complete column satisfies rolling window constraints.
    
    Args:
        path_mask: Bitmask representing the path (bit i = 1 means therapist at position i)
        duration: Length of the schedule
        MS: Rolling window size
        MIN_MS: Minimum human services required in window
    
    Returns:
        bool: True if column satisfies all rolling window constraints
    """
    # Check every position in the schedule
    for i in range(duration):
        if i + 1 < MS:
            # Not enough history yet - check if remaining slots can satisfy MIN_MS
            current_sum = 0
            for j in range(i + 1):
                if (path_mask >> j) & 1:
                    current_sum += 1
            remaining_slots = MS - (i + 1)
            max_possible = current_sum + remaining_slots
            if max_possible < MIN_MS:
                return False
        else:
            # Complete window exists
            window_start = i + 1 - MS
            window_sum = 0
            for j in range(window_start, i + 1):
                if (path_mask >> j) & 1:
                    window_sum += 1
            if window_sum < MIN_MS:
                return False
    
    return True


@njit(cache=True, inline='always')
def compute_lower_bound_numba(current_cost, start_time, end_time, gamma_k, obj_mode):
    """
    Numba-optimized Lower Bound calculation for Bound Pruning.
    
    Inlined for maximum performance in hot loops.
    
    Args:
        current_cost: Accumulated cost so far
        start_time: Column start time
        end_time: Column end time
        gamma_k: Gamma dual value
        obj_mode: Objective mode multiplier
    
    Returns:
        float: Minimum achievable final Reduced Cost (optimistic)
    """
    duration = end_time - start_time + 1
    time_cost = duration * obj_mode
    return current_cost + time_cost - gamma_k

@njit(cache=True)
def compute_candidate_workers_numba(workers, r_k, tau_max, pi_matrix):
    """
    Numba-optimized Worker Dominance Pre-Elimination.
    
    Worker j1 dominates j2 if π_{j1,t} >= π_{j2,t} for all t in [r_k, tau_max]
    AND π_{j1,t} > π_{j2,t} for at least one t (strict dominance).
    Since π values are <= 0 (implicit costs), higher π means lower cost.
    
    Args:
        workers: 1D numpy array of worker IDs
        r_k: Release time (int)
        tau_max: Maximum time horizon (int)
        pi_matrix: 2D numpy array [num_workers, max_time+1] of pi values
    
    Returns:
        Tuple of (candidate_array, count) - numpy array with candidates and actual count
    """
    n_workers = len(workers)
    # Pre-allocate result array (worst case: all workers are candidates)
    result = np.empty(n_workers, dtype=np.int64)
    count = 0
    
    for i1 in range(n_workers):
        j1 = workers[i1]
        is_dominated = False
        
        for i2 in range(n_workers):
            if i1 == i2:
                continue
                
            j2 = workers[i2]
            
            # Check if j2 dominates j1
            all_better_or_equal = True
            at_least_one_strictly_better = False
            
            for t in range(r_k, tau_max + 1):
                pi_j1 = pi_matrix[j1, t]
                pi_j2 = pi_matrix[j2, t]
                
                if pi_j2 < pi_j1:  # j2 is worse in this period
                    all_better_or_equal = False
                    break
                elif pi_j2 > pi_j1:  # j2 is strictly better in this period
                    at_least_one_strictly_better = True
            
            # j2 dominates j1 if it's at least as good everywhere and strictly better somewhere
            if all_better_or_equal and at_least_one_strictly_better:
                is_dominated = True
                break
        
        if not is_dominated:
            result[count] = j1
            count += 1
    
    # Return slice of actual candidates
    return result[:count]


@njit(cache=True)
def generate_full_column_vector_numba(worker_id, path_mask, start_time, end_time, max_time, num_workers):
    """
    Numba-optimized generation of the full column vector for a schedule.
    Uses 2-bit path encoding (0=App, 1=Therapist, 2=Gap).
    
    Args:
        worker_id: Worker ID (1-indexed)
        path_mask: Bitmask representing the path (2 bits per step)
        start_time: Start time of schedule
        end_time: End time of schedule
        max_time: Maximum time horizon
        num_workers: Total number of workers
        
    Returns:
        1D numpy array representing the full column vector
    """
    vector_length = num_workers * max_time
    full_vector = np.zeros(vector_length, dtype=np.float64)
    
    worker_offset = (worker_id - 1) * max_time
    duration = end_time - start_time + 1
    
    for t_idx in range(duration):
        # Decode 2-bit value
        val = (path_mask >> (2 * t_idx)) & 3
        
        # Check if val is 1 (Therapist). 0 is App, 2 is Gap.
        if val == 1:
            current_time = start_time + t_idx
            global_idx = worker_offset + (current_time - 1)
            if 0 <= global_idx < vector_length:
                full_vector[global_idx] = 1.0
    
    return full_vector


@njit(cache=True)
def path_mask_to_list(path_mask, duration):
    """
    Convert a 2-bit bitmask path to a list of values (0, 1, 2).
    
    Args:
        path_mask: Bitmask representing the path (2 bits per step)
        duration: Length of the schedule
        
    Returns:
        List of integers (0, 1, 2)
    """
    result = List.empty_list(int64)
    for i in range(duration):
        val = (path_mask >> (2 * i)) & 3
        result.append(int64(val))
    return result


# Type aliases for readability
# State: (cost, progress, path_mask, history_mask, history_len, ai_count)
# We store states in a list.
# Optimisation: We group states by (ai_count, history_mask, history_len) for dominance.

@njit(cache=True)
def check_strict_feasibility_numba(hist_mask, hist_len, next_val, MS, MIN_MS):
    """
    Check if adding next_val to the history satisfies rolling window constraints.
    Using bitwise operations.
    """
    # New history check
    new_len = hist_len + 1
    new_mask = (hist_mask << 1) | next_val
    
    # If we haven't filled the window yet
    if new_len < MS:
        # Check if it's possible to satisfy MIN_MS
        # Current ones + (MS - new_len) ones (optimistic future)
        current_ones = 0
        temp_mask = new_mask
        for _ in range(new_len):
            if temp_mask & 1:
                current_ones += 1
            temp_mask >>= 1
            
        remaining_slots = MS - new_len
        if current_ones + remaining_slots < MIN_MS:
            # Although returning False, we return valid shape placeholders
            return False, new_mask, new_len
        return True, new_mask, new_len
        
    else:
        # Full window check (new_len == MS or greater, but logically we enter with MS-1)
        # We check the window of size MS (which is exactly new_mask if entered with MS-1)
        
        # We assume input hist_len is at most MS-1. So new_len is at most MS.
        # If new_len == MS:
        
        # Count set bits in the window (last MS bits)
        current_ones = 0
        temp_mask = new_mask
        for _ in range(MS):
            if temp_mask & 1:
                current_ones += 1
            temp_mask >>= 1
            
        if current_ones < MIN_MS:
            return False, new_mask, MS 
            
        # Truncate to MS - 1 for state storage
        ms_minus_1 = MS - 1
        trunc_mask = new_mask & ((1 << ms_minus_1) - 1)
        return True, trunc_mask, ms_minus_1

# Type definitions for Dict
# Key: (ai_count, hist_mask, hist_len)
key_type = types.Tuple((types.int64, types.int64, types.int64))
# Value: List of (cost, prog, path_mask)
# Note: We must define the tuple type inside the list
val_tuple_type = types.Tuple((types.float64, types.float64, types.int64))
val_list_type = types.ListType(val_tuple_type)

# Return list type
# (j, rc, start, end, path_mask, prog)
result_tuple_type = types.Tuple((types.float64, types.float64, types.int64, types.int64, types.int64, types.float64))


@njit(cache=True)
def run_fast_path_numba(
    r_k, s_k, gamma_k, obj_mode_float, 
    pi_matrix, # 2D array [worker, time]
    candidate_workers, # Array of worker IDs
    max_time, 
    MS, MIN_MS, 
    theta_lookup, # Array
    epsilon,
    stop_at_first_negative=False,  # Early termination for child nodes
    allow_gaps=False  # Allow treatment gaps (idle days)
):
    """
    Optimized DP loop using Numba.
    """
    best_columns = List.empty_list(result_tuple_type)
    
    # Constants
    obj_mode = obj_mode_float
    
    for j in candidate_workers:
        # For each worker
        time_until_end = max_time - r_k + 1
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1
        
        # BITMASK SAFETY: path_mask is int64 (64 bits)
        # 2-bit encoding: max 32 steps (bits 0-63)
        # Safe limit is 31 steps (bits 0-61), leaving room
        BITMASK_LIMIT = 31
        max_tau = min(max_time, r_k + BITMASK_LIMIT)
        
        # Warning if bitmask limit is active
        #if max_tau < max_time:
            #print(f"  [BITMASK WARNING] Worker {j}: Duration capped at {BITMASK_LIMIT+1} days (r_k={r_k}, max_tau={max_tau} instead of {max_time})")
        
        for tau in range(start_tau, max_tau + 1):
            is_timeout_scenario = (tau == max_time)
            
            # Initial state setup
            start_cost = -pi_matrix[j, r_k]
            
            # Initialize Dict with explicit types
            current_states = Dict.empty(key_type, val_list_type)
            
            init_ai = 0
            init_hist = 1
            init_hlen = 1
            init_path = 1 # Bit 0-1 set to 01 (Therapist)
            
            init_key = (int64(init_ai), int64(init_hist), int64(init_hlen))
            
            # Create list for this bucket
            val_list = List.empty_list(val_tuple_type)
            val_list.append((float64(start_cost), float64(1.0), int64(init_path)))
            current_states[init_key] = val_list
            
            # DP Loop
            for t in range(r_k + 1, tau):
                next_states = Dict.empty(key_type, val_list_type)
                shift = 2 * (t - r_k)
                
                # Iterate over current buckets
                for key, bucket in current_states.items():
                    ai_count, hist_mask, hist_len = key
                    
                    for state in bucket:
                        cost, prog, path_mask = state
                        
                        # Reachability Pruning
                        remaining_steps = tau - t + 1
                        if not is_timeout_scenario:
                            if obj_mode > 0.5:
                                if prog + remaining_steps * 1.0 < s_k - epsilon:
                                    continue
                                    
                        # A: Therapist (1 -> 01)
                        feasible_ther, new_mask_ther, new_len_ther = check_strict_feasibility_numba(
                            hist_mask, hist_len, 1, MS, MIN_MS
                        )
                        
                        if feasible_ther:
                            cost_ther = cost - pi_matrix[j, t]
                            prog_ther = prog + 1.0
                            
                            new_key_ther = (ai_count, new_mask_ther, new_len_ther)
                            new_val_ther = (cost_ther, prog_ther, (path_mask | (1 << shift)))
                            
                            if new_key_ther not in next_states:
                                l = List.empty_list(val_tuple_type)
                                l.append(new_val_ther)
                                next_states[new_key_ther] = l
                            else:
                                bucket_t = next_states[new_key_ther]
                                is_dominated = False
                                for i in range(len(bucket_t)):
                                    c_old, p_old, _ = bucket_t[i]
                                    if c_old <= cost_ther + epsilon and p_old >= prog_ther - epsilon:
                                        is_dominated = True
                                        break
                                
                                if not is_dominated:
                                    clean_bucket = List.empty_list(val_tuple_type)
                                    for i in range(len(bucket_t)):
                                        c_old, p_old, path_old = bucket_t[i]
                                        if cost_ther <= c_old + epsilon and prog_ther >= p_old - epsilon:
                                            pass
                                        else:
                                            clean_bucket.append((c_old, p_old, path_old))
                                    clean_bucket.append(new_val_ther)
                                    next_states[new_key_ther] = clean_bucket

                        # B: AI (0 -> 00)
                        feasible_ai, new_mask_ai, new_len_ai = check_strict_feasibility_numba(
                            hist_mask, hist_len, 0, MS, MIN_MS
                        )
                        
                        if feasible_ai:
                            cost_ai = cost
                            eff = 1.0
                            if ai_count < len(theta_lookup):
                                eff = theta_lookup[ai_count]
                                
                            prog_ai = prog + eff
                            new_ai_count = ai_count + 1
                            
                            new_key_ai = (new_ai_count, new_mask_ai, new_len_ai)
                            # 0 << shift is 0, so just path_mask
                            new_val_ai = (cost_ai, prog_ai, (path_mask))
                            
                            if new_key_ai not in next_states:
                                l = List.empty_list(val_tuple_type)
                                l.append(new_val_ai)
                                next_states[new_key_ai] = l
                            else:
                                bucket_a = next_states[new_key_ai]
                                is_dominated = False
                                for i in range(len(bucket_a)):
                                    c_old, p_old, _ = bucket_a[i]
                                    if c_old <= cost_ai + epsilon and p_old >= prog_ai - epsilon:
                                        is_dominated = True
                                        break
                                
                                if not is_dominated:
                                    clean_bucket = List.empty_list(val_tuple_type)
                                    for i in range(len(bucket_a)):
                                        c_old, p_old, path_old = bucket_a[i]
                                        if cost_ai <= c_old + epsilon and prog_ai >= p_old - epsilon:
                                            pass
                                        else:
                                            clean_bucket.append((c_old, p_old, path_old))
                                    clean_bucket.append(new_val_ai)
                                    next_states[new_key_ai] = clean_bucket

                        # C: GAP (2 -> 10) - New
                        if allow_gaps:
                            # Gaps behave like AI/Idle for feasibility (pass 0)
                            feasible_gap, new_mask_gap, new_len_gap = check_strict_feasibility_numba(
                                hist_mask, hist_len, 0, MS, MIN_MS
                            )
                            
                            if feasible_gap:
                                cost_gap = cost
                                prog_gap = prog # No progress
                                # AI count doesn't increase
                                
                                new_key_gap = (ai_count, new_mask_gap, new_len_gap)
                                new_val_gap = (cost_gap, prog_gap, (path_mask | (2 << shift)))
                                
                                if new_key_gap not in next_states:
                                    l = List.empty_list(val_tuple_type)
                                    l.append(new_val_gap)
                                    next_states[new_key_gap] = l
                                else:
                                    bucket_g = next_states[new_key_gap]
                                    is_dominated = False
                                    for i in range(len(bucket_g)):
                                        c_old, p_old, _ = bucket_g[i]
                                        if c_old <= cost_gap + epsilon and p_old >= prog_gap - epsilon:
                                            is_dominated = True
                                            break
                                    
                                    if not is_dominated:
                                        clean_bucket = List.empty_list(val_tuple_type)
                                        for i in range(len(bucket_g)):
                                            c_old, p_old, path_old = bucket_g[i]
                                            if cost_gap <= c_old + epsilon and prog_gap >= p_old - epsilon:
                                                pass
                                            else:
                                                clean_bucket.append((c_old, p_old, path_old))
                                        clean_bucket.append(new_val_gap)
                                        next_states[new_key_gap] = clean_bucket

                current_states = next_states
                if len(current_states) == 0:
                    break
            
            # Final Step (Transition to Tau)
            shift = 2 * (tau - r_k)
            for key, bucket in current_states.items():
                ai_count, hist_mask, hist_len = key
                
                for state in bucket:
                    cost, prog, path_mask = state
                    
                    # We collect possible end steps
                    # Option 1: End with Therapist (1)
                    feasible_ther, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 1, MS, MIN_MS)
                    
                    if feasible_ther:
                        final_cost = cost - pi_matrix[j, tau]
                        final_prog = prog + 1.0
                        final_path_mask = path_mask | (1 << shift)
                        
                        condition_met = (final_prog >= s_k - epsilon)
                        is_valid = False
                        if obj_mode > 0.5:
                            is_valid = condition_met
                        else:
                            is_valid = condition_met or (tau == max_time)

                        if is_valid:
                            duration_val = (tau - r_k + 1)
                            rc = final_cost + (duration_val * obj_mode) - gamma_k
                            if rc < -1e-6:
                                best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                                if stop_at_first_negative:
                                    return best_columns  # Early termination: found negative RC

                    # Option 2: End with App (0) - ONLY if Timeout
                    if is_timeout_scenario:
                        feasible_ai, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                        if feasible_ai:
                            final_cost = cost
                            eff = 1.0
                            if ai_count < len(theta_lookup):
                                eff = theta_lookup[ai_count]
                            final_prog = prog + eff
                            final_path_mask = path_mask # 00
                            
                            condition_met = (final_prog >= s_k - epsilon)
                            is_valid = False
                            if obj_mode > 0.5:
                                is_valid = condition_met
                            else:
                                is_valid = condition_met or (tau == max_time)
                                
                            if is_valid:
                                duration_val = (tau - r_k + 1)
                                rc = final_cost + (duration_val * obj_mode) - gamma_k
                                if rc < -1e-6:
                                    best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                                    if stop_at_first_negative:
                                        return best_columns  # Early termination: found negative RC

                    # Option 3: End with Gap (2) - ONLY if Timeout (and allow_gaps)
                    if is_timeout_scenario and allow_gaps:
                        feasible_gap, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                        if feasible_gap:
                            final_cost = cost
                            final_prog = prog # No progress
                            final_path_mask = path_mask | (2 << shift)
                            
                            # For Post-Patients, timeout allows ending even if target not met
                            # For Focus-Patients, must meet target (which Gap doesn't help with usually, unless already met)
                            is_focus = (obj_mode > 0.5)
                            condition_met = (final_prog >= s_k - epsilon)
                            is_valid = condition_met if is_focus else (condition_met or is_timeout_scenario)
                                
                            if is_valid:
                                duration_val = (tau - r_k + 1)
                                rc = final_cost + (duration_val * obj_mode) - gamma_k
                                if rc < -1e-6:
                                    best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                                    if stop_at_first_negative:
                                        return best_columns

    return best_columns

@njit(cache=True)
def run_fast_path_single_worker_numba(
    j,  # Single worker ID
    r_k, s_k, gamma_k, obj_mode_float, 
    pi_matrix,
    max_time, 
    MS, MIN_MS, 
    theta_lookup,
    epsilon,
    stop_at_first_negative=False,  # Early termination for child nodes
    allow_gaps=False # Allow treatment gaps
):
    """
    Optimized DP loop for a SINGLE worker.
    This function can be called in parallel from Python using multiprocessing.
    
    Returns:
        List of columns for this worker
    """
    best_columns = List.empty_list(result_tuple_type)
    obj_mode = obj_mode_float
    
    time_until_end = max_time - r_k + 1
    effective_min_duration = min(int(s_k), time_until_end)
    start_tau = r_k + effective_min_duration - 1
    
    # BITMASK SAFETY: path_mask is int64 (64 bits)
    # 2-bit encoding: max 32 steps (bits 0-63)
    BITMASK_LIMIT = 31
    max_tau = min(max_time, r_k + BITMASK_LIMIT)
    
    # Warning if bitmask limit is active
    #if max_tau < max_time:
        #print(f"  [BITMASK WARNING] Worker {j}: Duration capped at {BITMASK_LIMIT+1} days (r_k={r_k}, max_tau={max_tau} instead of {max_time})")
    
    for tau in range(start_tau, max_tau + 1):
        is_timeout_scenario = (tau == max_time)
        start_cost = -pi_matrix[j, r_k]
        
        current_states = Dict.empty(key_type, val_list_type)
        
        init_ai = 0
        init_hist = 1
        init_hlen = 1
        init_path = 1 # 01 (Therapist)
        
        init_key = (int64(init_ai), int64(init_hist), int64(init_hlen))
        val_list = List.empty_list(val_tuple_type)
        val_list.append((float64(start_cost), float64(1.0), int64(init_path)))
        current_states[init_key] = val_list
        
        # DP Loop
        for t in range(r_k + 1, tau):
            next_states = Dict.empty(key_type, val_list_type)
            shift = 2 * (t - r_k)
            
            for key, bucket in current_states.items():
                ai_count, hist_mask, hist_len = key
                
                for state in bucket:
                    cost, prog, path_mask = state
                    
                    # Reachability Pruning
                    remaining_steps = tau - t + 1
                    if not is_timeout_scenario:
                        if obj_mode > 0.5:
                            if prog + remaining_steps * 1.0 < s_k - epsilon:
                                continue
                    
                    # A: Therapist
                    feasible_ther, new_mask_ther, new_len_ther = check_strict_feasibility_numba(
                        hist_mask, hist_len, 1, MS, MIN_MS
                    )
                    
                    if feasible_ther:
                        cost_ther = cost - pi_matrix[j, t]
                        prog_ther = prog + 1.0
                        
                        new_key_ther = (ai_count, new_mask_ther, new_len_ther)
                        new_val_ther = (cost_ther, prog_ther, (path_mask | (1 << shift)))
                        
                        if new_key_ther not in next_states:
                            l = List.empty_list(val_tuple_type)
                            l.append(new_val_ther)
                            next_states[new_key_ther] = l
                        else:
                            bucket_t = next_states[new_key_ther]
                            is_dominated = False
                            for i in range(len(bucket_t)):
                                c_old, p_old, _ = bucket_t[i]
                                if c_old <= cost_ther + epsilon and p_old >= prog_ther - epsilon:
                                    is_dominated = True
                                    break
                            
                            if not is_dominated:
                                clean_bucket = List.empty_list(val_tuple_type)
                                for i in range(len(bucket_t)):
                                    c_old, p_old, path_old = bucket_t[i]
                                    if cost_ther <= c_old + epsilon and prog_ther >= p_old - epsilon:
                                        pass
                                    else:
                                        clean_bucket.append((c_old, p_old, path_old))
                                clean_bucket.append(new_val_ther)
                                next_states[new_key_ther] = clean_bucket
                    
                    # B: AI
                    feasible_ai, new_mask_ai, new_len_ai = check_strict_feasibility_numba(
                        hist_mask, hist_len, 0, MS, MIN_MS
                    )
                    
                    if feasible_ai:
                        cost_ai = cost
                        eff = 1.0
                        if ai_count < len(theta_lookup):
                            eff = theta_lookup[ai_count]
                        prog_ai = prog + eff
                        new_ai_count = ai_count + 1
                        
                        new_key_ai = (new_ai_count, new_mask_ai, new_len_ai)
                        new_val_ai = (cost_ai, prog_ai, path_mask)
                        
                        if new_key_ai not in next_states:
                            l = List.empty_list(val_tuple_type)
                            l.append(new_val_ai)
                            next_states[new_key_ai] = l
                        else:
                            bucket_a = next_states[new_key_ai]
                            is_dominated = False
                            for i in range(len(bucket_a)):
                                c_old, p_old, _ = bucket_a[i]
                                if c_old <= cost_ai + epsilon and p_old >= prog_ai - epsilon:
                                    is_dominated = True
                                    break
                            
                            if not is_dominated:
                                clean_bucket = List.empty_list(val_tuple_type)
                                for i in range(len(bucket_a)):
                                    c_old, p_old, path_old = bucket_a[i]
                                    if cost_ai <= c_old + epsilon and prog_ai >= p_old - epsilon:
                                        pass
                                    else:
                                        clean_bucket.append((c_old, p_old, path_old))
                                clean_bucket.append(new_val_ai)
                                next_states[new_key_ai] = clean_bucket

                    # C: Gap
                    if allow_gaps:
                        feasible_gap, new_mask_gap, new_len_gap = check_strict_feasibility_numba(
                            hist_mask, hist_len, 0, MS, MIN_MS
                        )
                        
                        if feasible_gap:
                            cost_gap = cost
                            prog_gap = prog
                            # ai_count unchanged
                            
                            new_key_gap = (ai_count, new_mask_gap, new_len_gap)
                            new_val_gap = (cost_gap, prog_gap, (path_mask | (2 << shift)))
                            
                            if new_key_gap not in next_states:
                                l = List.empty_list(val_tuple_type)
                                l.append(new_val_gap)
                                next_states[new_key_gap] = l
                            else:
                                bucket_g = next_states[new_key_gap]
                                is_dominated = False
                                for i in range(len(bucket_g)):
                                    c_old, p_old, _ = bucket_g[i]
                                    if c_old <= cost_gap + epsilon and p_old >= prog_gap - epsilon:
                                        is_dominated = True
                                        break
                                
                                if not is_dominated:
                                    clean_bucket = List.empty_list(val_tuple_type)
                                    for i in range(len(bucket_g)):
                                        c_old, p_old, path_old = bucket_g[i]
                                        if cost_gap <= c_old + epsilon and prog_gap >= p_old - epsilon:
                                            pass
                                        else:
                                            clean_bucket.append((c_old, p_old, path_old))
                                    clean_bucket.append(new_val_gap)
                                    next_states[new_key_gap] = clean_bucket

            current_states = next_states
            if len(current_states) == 0:
                break
        
        # Final Step
        shift = 2 * (tau - r_k)
        for key, bucket in current_states.items():
            ai_count, hist_mask, hist_len = key
            
            for state in bucket:
                cost, prog, path_mask = state
                
                # Option 1: End with Therapist
                feasible_ther, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 1, MS, MIN_MS)
                
                if feasible_ther:
                    final_cost = cost - pi_matrix[j, tau]
                    final_prog = prog + 1.0
                    final_path_mask = path_mask | (1 << shift)
                    
                    condition_met = (final_prog >= s_k - epsilon)
                    is_valid = False
                    if obj_mode > 0.5:
                        is_valid = condition_met
                    else:
                        is_valid = condition_met or (tau == max_time)
                    
                    if is_valid:
                        duration_val = tau - r_k + 1
                        rc = final_cost + (duration_val * obj_mode) - gamma_k
                        if rc < -1e-6:
                            best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                            if stop_at_first_negative:
                                return best_columns  # Early termination: found negative RC
                
                # Option 2: End with AI (only on timeout)
                if is_timeout_scenario:
                    feasible_ai, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                    if feasible_ai:
                        final_cost = cost
                        eff = 1.0
                        if ai_count < len(theta_lookup):
                            eff = theta_lookup[ai_count]
                        final_prog = prog + eff
                        final_path_mask = path_mask
                        
                        condition_met = (final_prog >= s_k - epsilon)
                        is_valid = False
                        if obj_mode > 0.5:
                            is_valid = condition_met
                        else:
                            is_valid = condition_met or (tau == max_time)
                        
                        if is_valid:
                            duration_val = tau - r_k + 1
                            rc = final_cost + (duration_val * obj_mode) - gamma_k
                            if rc < -1e-6:
                                best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                                if stop_at_first_negative:
                                    return best_columns  # Early termination: found negative RC

                # Option 3: End with Gap
                if is_timeout_scenario and allow_gaps:
                    feasible_gap, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                    if feasible_gap:
                        final_cost = cost
                        final_prog = prog
                        final_path_mask = path_mask | (2 << shift)
                        
                        is_focus = (obj_mode > 0.5)
                        condition_met = (final_prog >= s_k - epsilon)
                        is_valid = condition_met if is_focus else (condition_met or is_timeout_scenario)
                        
                        if is_valid:
                            duration_val = tau - r_k + 1
                            rc = final_cost + (duration_val * obj_mode) - gamma_k
                            if rc < -1e-6:
                                best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                                if stop_at_first_negative:
                                    return best_columns

    return best_columns


# =============================================================================
# BRANCHING CONSTRAINTS SUPPORT
# =============================================================================
# Extended state tuple types for branching constraints
# State now includes: cost, prog, path_mask, zeta_mask (for MP branching)
# Key now includes: ai_count, hist_mask, hist_len, zeta_mask

# Key type WITH zeta and mu: (ai_count, hist_mask, hist_len, zeta_mask, mu_encoded)
key_with_zeta_type = types.Tuple((types.int64, types.int64, types.int64, types.int64, types.int64))
val_with_zeta_type = types.Tuple((types.float64, types.float64, types.int64))  # cost, prog, path_mask
val_list_with_zeta_type = types.ListType(val_with_zeta_type)


@njit(cache=True)
def run_with_branching_constraints_numba(
    r_k, s_k, gamma_k, obj_mode_float,
    pi_matrix,           # 2D array [worker, time]
    candidate_workers,   # Array of worker IDs
    max_time,
    MS, MIN_MS,
    theta_lookup,
    epsilon,
    # === SP Variable Fixing (B.1) ===
    forbidden_mask,      # 2D bool array [worker, time] - True if fixed to 0
    required_mask,       # 2D bool array [worker, time] - True if fixed to 1
    has_sp_fixing,       # bool - whether any SP fixes are active
    # === MP No-Good Cuts (B.2) - O(1) OPTIMIZATION ===
    mp_forbidden_map,    # 2D bool array [worker, time] - True if (w,t) is forbidden by ANY no-good cut
    has_mp_branching,    # bool - whether any MP branching constraints are active
    # === SP Pattern Branching (B.3) ===
    left_pattern_elements,   # 2D array [pattern_idx, flat_idx] containing encoded (w*1000+t) or -1
    left_pattern_limits,     # 1D array [pattern_idx] - max allowed coverage
    num_left_patterns,       # int
    has_left_patterns,       # bool
    # === SP Right Pattern Branching (B.3.2) ===
    right_pattern_elements,  # 2D array [pat_idx, elem_idx] encoded (w*1M+t)
    right_pattern_starts,    # 1D array [pat_idx] start time of pattern
    right_pattern_duals,     # 1D array [pat_idx] dual reward
    right_pattern_counts,    # 1D array [pat_idx] number of elements in pattern
    num_right_patterns,      # int
    has_right_patterns,      # bool
    stop_at_first_negative=False,  # Early termination for child nodes
    allow_gaps=False # Allow treatment gaps
):
    """
    Extended DP loop with branching constraint support.
    """
    best_columns = List.empty_list(result_tuple_type)
    obj_mode = obj_mode_float
    
    for j in candidate_workers:
        # === SP Variable Fixing Check at r_k ===
        # First time step MUST be therapist (1), check if it's forbidden
        if has_sp_fixing and forbidden_mask[j, r_k]:
            continue
        
        time_until_end = max_time - r_k + 1
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1
        
        # BITMASK SAFETY: path_mask is int64 (64 bits)
        # 2-bit encoding: max 32 steps (bits 0-63)
        BITMASK_LIMIT = 31
        max_tau = min(max_time, r_k + BITMASK_LIMIT)
        
        # Warning if bitmask limit is active
        if max_tau < max_time:
            print(f"  [BITMASK WARNING] Worker {j}: Duration capped at {BITMASK_LIMIT+1} days (r_k={r_k}, max_tau={max_tau} instead of {max_time})")
        
        for tau in range(start_tau, max_tau + 1):
            is_timeout_scenario = (tau == max_time)
            start_cost = -pi_matrix[j, r_k]
            
            # === O(1) MP Branching Check at r_k ===
            # Skip if (j, r_k) is forbidden by any no-good cut
            if has_mp_branching and mp_forbidden_map[j, r_k]:
                continue  # Pruned before even starting
            
            # === Initialize Rho for SP Left Patterns ===
            init_rho = int64(0)
            if has_left_patterns:
                for pat_idx in range(num_left_patterns):
                    # Check if (j, r_k) is in this pattern's elements
                    for elem_idx in range(left_pattern_elements.shape[1]):
                        encoded = left_pattern_elements[pat_idx, elem_idx]
                        if encoded < 0:
                            break
                        w_pat = encoded // 1000000
                        t_pat = encoded % 1000000
                        if w_pat == j and t_pat == r_k:
                            # Increment rho for this pattern
                            current_rho = (init_rho >> (pat_idx * 8)) & 0xFF
                            current_rho += 1
                            # Check limit
                            if current_rho > left_pattern_limits[pat_idx]:
                                init_rho = int64(-1)  # Signal: pruned
                                break
                            # Update rho
                            init_rho = (init_rho & ~(0xFF << (pat_idx * 8))) | (current_rho << (pat_idx * 8))
                    if init_rho == -1:
                        break
            
            if init_rho == -1:
                continue  # This starting state is already infeasible
            
            # === Initialize Mu for SP Right Patterns ===
            init_mu = int64(0)
            if has_right_patterns:
                for pat_idx in range(num_right_patterns):
                    t_start = right_pattern_starts[pat_idx]
                    
                    # Is current (j, r_k) in pattern?
                    in_pattern = False
                    for elem_idx in range(right_pattern_counts[pat_idx]):
                        encoded = right_pattern_elements[pat_idx, elem_idx]
                        w_pat = encoded // 1000000
                        t_pat = encoded % 1000000
                        if w_pat == j and t_pat == r_k:
                            in_pattern = True
                            break
                    
                    if r_k == t_start:
                        if in_pattern:
                            # Enter Cover Mode
                            init_mu = init_mu | (1 << pat_idx)
                        else:
                            # Enter Exclude Mode (bit remains 0)
                            pass
                    elif r_k > t_start:
                         if in_pattern:
                             # We are picking an element but we are in Exclude mode (implicitly)
                             init_mu = int64(-1)
                             break
            
            if init_mu == -1:
                continue

            # Initialize state dict
            current_states = Dict.empty(key_with_zeta_type, val_list_with_zeta_type)
            
            init_ai = int64(0)
            init_hist = int64(1)
            init_hlen = int64(1)
            init_path = int64(1) # 01 (Therapist)
            
            # Zeta unused for MP branching (O(1) pruning instead), but keep for compatibility with combined state
            init_zeta = int64(0)
            combined_zeta_rho = init_zeta | (init_rho << 32)
            
            init_key = (init_ai, init_hist, init_hlen, combined_zeta_rho, init_mu)
            val_list = List.empty_list(val_with_zeta_type)
            val_list.append((float64(start_cost), float64(1.0), init_path))
            current_states[init_key] = val_list
            
            # DP Loop
            for t in range(r_k + 1, tau):
                next_states = Dict.empty(key_with_zeta_type, val_list_with_zeta_type)
                shift = 2 * (t - r_k)
                
                for key, bucket in current_states.items():
                    ai_count, hist_mask, hist_len, combined_zeta_rho, mu_encoded = key
                    zeta_mask = combined_zeta_rho & 0xFFFFFFFF
                    rho_encoded = combined_zeta_rho >> 32
                    
                    for state in bucket:
                        cost, prog, path_mask = state
                        
                        # Reachability Pruning
                        remaining_steps = tau - t + 1
                        if not is_timeout_scenario and obj_mode > 0.5:
                            if prog + remaining_steps * 1.0 < s_k - epsilon:
                                continue
                        
                        # === A: Therapist (action = 1) ===
                        can_take_therapist = True
                        if has_sp_fixing and forbidden_mask[j, t]:
                            can_take_therapist = False
                        
                        if can_take_therapist:
                            feasible_ther, new_mask_ther, new_len_ther = check_strict_feasibility_numba(
                                hist_mask, hist_len, 1, MS, MIN_MS
                            )
                            
                            if feasible_ther:
                                cost_ther = cost - pi_matrix[j, t]
                                prog_ther = prog + 1.0
                                
                                # O(1) MP Branching: Skip if forbidden
                                if has_mp_branching and mp_forbidden_map[j, t]:
                                    continue  # Pruned
                                
                                # Zeta not used for MP branching anymore (O(1) pruning instead)
                                new_zeta = int64(0)
                                
                                # Update rho
                                new_rho = rho_encoded
                                rho_valid = True
                                if has_left_patterns:
                                    for pat_idx in range(num_left_patterns):
                                        for elem_idx in range(left_pattern_elements.shape[1]):
                                            encoded = left_pattern_elements[pat_idx, elem_idx]
                                            if encoded < 0: break
                                            w_pat = encoded // 1000000
                                            t_pat = encoded % 1000000
                                            if w_pat == j and t_pat == t:
                                                current_rho = (new_rho >> (pat_idx * 8)) & 0xFF
                                                current_rho += 1
                                                if current_rho > left_pattern_limits[pat_idx]:
                                                    rho_valid = False; break
                                                new_rho = (new_rho & ~(0xFF << (pat_idx * 8))) | (current_rho << (pat_idx * 8))
                                        if not rho_valid: break
                                
                                # Update mu (Right Pattern)
                                new_mu = mu_encoded
                                mu_valid = True
                                if has_right_patterns and rho_valid:
                                    for pat_idx in range(num_right_patterns):
                                        t_start = right_pattern_starts[pat_idx]
                                        current_mode = (new_mu >> pat_idx) & 1
                                        
                                        in_pattern = False
                                        for elem_idx in range(right_pattern_counts[pat_idx]):
                                            encoded = right_pattern_elements[pat_idx, elem_idx]
                                            w_pat = encoded // 1000000
                                            t_pat = encoded % 1000000
                                            if w_pat == j and t_pat == t:
                                                in_pattern = True
                                                break
                                        
                                        if t == t_start:
                                            if in_pattern: new_mu = new_mu | (1 << pat_idx) # Cover
                                            else: new_mu = new_mu & ~(1 << pat_idx) # Exclude
                                        elif t > t_start:
                                            if current_mode == 0: # Exclude
                                                if in_pattern: mu_valid = False; break
                                            else: # Cover Mode (1)
                                                 pass # Taking therapist covers if in pattern, fine if not

                                if rho_valid and mu_valid:
                                    new_combined = new_zeta | (new_rho << 32)
                                    new_key_ther = (ai_count, new_mask_ther, new_len_ther, new_combined, new_mu)
                                    new_val_ther = (cost_ther, prog_ther, path_mask | (1 << shift))
                                    
                                    if new_key_ther not in next_states:
                                        l = List.empty_list(val_with_zeta_type)
                                        l.append(new_val_ther)
                                        next_states[new_key_ther] = l
                                    else:
                                        bucket_t = next_states[new_key_ther]
                                        is_dominated = False
                                        for i in range(len(bucket_t)):
                                            c_old, p_old, _ = bucket_t[i]
                                            if c_old <= cost_ther + epsilon and p_old >= prog_ther - epsilon:
                                                is_dominated = True; break
                                        if not is_dominated:
                                            clean = List.empty_list(val_with_zeta_type)
                                            for i in range(len(bucket_t)):
                                                c_old, p_old, path_old = bucket_t[i]
                                                if not (cost_ther <= c_old + epsilon and prog_ther >= p_old - epsilon):
                                                    clean.append((c_old, p_old, path_old))
                                            clean.append(new_val_ther)
                                            next_states[new_key_ther] = clean
                        
                        # === B: AI (action = 0) ===
                        can_take_ai = True
                        if has_sp_fixing and required_mask[j, t]:
                            can_take_ai = False  # Required to be 1, can't take 0
                        
                        if can_take_ai:
                            feasible_ai, new_mask_ai, new_len_ai = check_strict_feasibility_numba(
                                hist_mask, hist_len, 0, MS, MIN_MS
                            )
                            
                            if feasible_ai:
                                cost_ai = cost
                                eff = 1.0
                                if ai_count < len(theta_lookup):
                                    eff = theta_lookup[ai_count]
                                prog_ai = prog + eff
                                new_ai_count = ai_count + 1
                                
                                # MP branching doesn't affect AI transitions (only therapist)
                                new_zeta = int64(0)
                                
                                # Rho stays same (AI not in pattern)
                                new_rho = rho_encoded
                                
                                # Update mu (Right Pattern)
                                new_mu = mu_encoded
                                mu_valid = True
                                if has_right_patterns:
                                    for pat_idx in range(num_right_patterns):
                                        t_start = right_pattern_starts[pat_idx]
                                        current_mode = (new_mu >> pat_idx) & 1
                                        
                                        if t == t_start:
                                             # AI (0) implies not in pattern (0 != 1) -> Exclude
                                            new_mu = new_mu & ~(1 << pat_idx)
                                        elif t > t_start:
                                            if current_mode == 0: # Exclude
                                                pass
                                            else: # Cover Mode (1)
                                                # If (j,t) is required, we failed to take it -> Prune
                                                is_required_here = False
                                                for elem_idx in range(right_pattern_counts[pat_idx]):
                                                    encoded = right_pattern_elements[pat_idx, elem_idx]
                                                    w_pat = encoded // 1000000
                                                    t_pat = encoded % 1000000
                                                    if w_pat == j and t_pat == t:
                                                        is_required_here = True; break
                                                if is_required_here:
                                                    mu_valid = False; break
                                
                                if mu_valid:
                                    new_combined = new_zeta | (new_rho << 32)
                                    new_key_ai = (new_ai_count, new_mask_ai, new_len_ai, new_combined, new_mu)
                                    new_val_ai = (cost_ai, prog_ai, path_mask) # 00
                                    
                                    if new_key_ai not in next_states:
                                        l = List.empty_list(val_with_zeta_type)
                                        l.append(new_val_ai)
                                        next_states[new_key_ai] = l
                                    else:
                                        bucket_a = next_states[new_key_ai]
                                        is_dominated = False
                                        for i in range(len(bucket_a)):
                                            c_old, p_old, _ = bucket_a[i]
                                            if c_old <= cost_ai + epsilon and p_old >= prog_ai - epsilon:
                                                is_dominated = True; break
                                        if not is_dominated:
                                            clean = List.empty_list(val_with_zeta_type)
                                            for i in range(len(bucket_a)):
                                                c_old, p_old, path_old = bucket_a[i]
                                                if not (cost_ai <= c_old + epsilon and prog_ai >= p_old - epsilon):
                                                    clean.append((c_old, p_old, path_old))
                                            clean.append(new_val_ai)
                                            next_states[new_key_ai] = clean

                        # === C: GAP (action = 2) ===
                        if allow_gaps:
                            # Gap is like AI (0) regarding constraints
                            can_take_gap = True
                            if has_sp_fixing and required_mask[j, t]:
                                can_take_gap = False # Required to be 1, can't take Gap (0-like)
                            
                            if can_take_gap:
                                feasible_gap, new_mask_gap, new_len_gap = check_strict_feasibility_numba(
                                    hist_mask, hist_len, 0, MS, MIN_MS
                                )
                                
                                if feasible_gap:
                                    cost_gap = cost
                                    prog_gap = prog # No progress
                                    # ai_count unchanged
                                    
                                    # MP branching doesn't affect Gap transitions (only therapist)
                                    new_zeta = int64(0)
                                    
                                    # Rho (same as AI)
                                    new_rho = rho_encoded
                                    
                                    # Mu (same as AI)
                                    new_mu = mu_encoded
                                    mu_valid = True
                                    if has_right_patterns:
                                        for pat_idx in range(num_right_patterns):
                                            t_start = right_pattern_starts[pat_idx]
                                            current_mode = (new_mu >> pat_idx) & 1
                                            
                                            if t == t_start:
                                                new_mu = new_mu & ~(1 << pat_idx)
                                            elif t > t_start:
                                                if current_mode == 0: pass
                                                else:
                                                    is_required_here = False
                                                    for elem_idx in range(right_pattern_counts[pat_idx]):
                                                        encoded = right_pattern_elements[pat_idx, elem_idx]
                                                        w_pat = encoded // 1000000
                                                        t_pat = encoded % 1000000
                                                        if w_pat == j and t_pat == t:
                                                            is_required_here = True; break
                                                    if is_required_here:
                                                        mu_valid = False; break
                                    
                                    if mu_valid:
                                        new_combined = new_zeta | (new_rho << 32)
                                        new_key_gap = (ai_count, new_mask_gap, new_len_gap, new_combined, new_mu)
                                        # Path: 2 << shift
                                        new_val_gap = (cost_gap, prog_gap, (path_mask | (2 << shift)))
                                        
                                        if new_key_gap not in next_states:
                                            l = List.empty_list(val_with_zeta_type)
                                            l.append(new_val_gap)
                                            next_states[new_key_gap] = l
                                        else:
                                            bucket_g = next_states[new_key_gap]
                                            is_dominated = False
                                            for i in range(len(bucket_g)):
                                                c_old, p_old, _ = bucket_g[i]
                                                if c_old <= cost_gap + epsilon and p_old >= prog_gap - epsilon:
                                                    is_dominated = True; break
                                            if not is_dominated:
                                                clean = List.empty_list(val_with_zeta_type)
                                                for i in range(len(bucket_g)):
                                                    c_old, p_old, path_old = bucket_g[i]
                                                    if not (cost_gap <= c_old + epsilon and prog_gap >= p_old - epsilon):
                                                        clean.append((c_old, p_old, path_old))
                                                clean.append(new_val_gap)
                                                next_states[new_key_gap] = clean

                current_states = next_states
                if len(current_states) == 0:
                    break
            
            # === Final Step (Transition to Tau) ===
            shift = 2 * (tau - r_k)
            for key, bucket in current_states.items():
                ai_count, hist_mask, hist_len, combined_zeta_rho, mu_encoded = key
                zeta_mask = combined_zeta_rho & 0xFFFFFFFF
                rho_encoded = combined_zeta_rho >> 32
                
                for state in bucket:
                    cost, prog, path_mask = state
                    
                    # === Option 1: End with Therapist ===
                    can_end_ther = True
                    if has_sp_fixing and forbidden_mask[j, tau]:
                        can_end_ther = False
                    
                    if can_end_ther:
                        feasible_ther, _, _, = check_strict_feasibility_numba(hist_mask, hist_len, 1, MS, MIN_MS)
                        if feasible_ther:
                            # O(1) MP Branching: Skip if final assignment is forbidden
                            if has_mp_branching and mp_forbidden_map[j, tau]:
                                continue  # Pruned
                            
                            final_cost = cost - pi_matrix[j, tau]
                            final_prog = prog + 1.0
                            final_path_mask = path_mask | (1 << shift)
                            
                            # Final Branching Checks (Rho, Mu) - MP branching now done via pruning
                            final_zeta = int64(0)  # Not used for MP branching
                            
                            # Rho
                            final_rho = rho_encoded
                            rho_valid = True
                            if has_left_patterns:
                                for pat_idx in range(num_left_patterns):
                                    for elem_idx in range(left_pattern_elements.shape[1]):
                                        encoded = left_pattern_elements[pat_idx, elem_idx]
                                        if encoded < 0: break
                                        w_pat = encoded // 1000000
                                        t_pat = encoded % 1000000
                                        if w_pat == j and t_pat == tau:
                                            current_rho = (final_rho >> (pat_idx * 8)) & 0xFF
                                            current_rho += 1
                                            if current_rho > left_pattern_limits[pat_idx]:
                                                rho_valid = False; break
                                            final_rho = (final_rho & ~(0xFF << (pat_idx * 8))) | (current_rho << (pat_idx * 8))
                                    if not rho_valid: break
                            if not rho_valid: continue

                            # Mu
                            final_mu = mu_encoded
                            mu_valid = True
                            right_reward = 0.0
                            if has_right_patterns:
                                for pat_idx in range(num_right_patterns):
                                    t_start = right_pattern_starts[pat_idx]
                                    current_mode = (final_mu >> pat_idx) & 1
                                    
                                    in_pattern = False
                                    for elem_idx in range(right_pattern_counts[pat_idx]):
                                        encoded = right_pattern_elements[pat_idx, elem_idx]
                                        w_pat = encoded // 1000000
                                        t_pat = encoded % 1000000
                                        if w_pat == j and t_pat == tau:
                                            in_pattern = True; break
                                    
                                    if tau == t_start:
                                        if in_pattern: final_mu = final_mu | (1 << pat_idx)
                                        else: final_mu = final_mu & ~(1 << pat_idx)
                                    elif tau > t_start:
                                        if current_mode == 0 and in_pattern:
                                            mu_valid = False; break
                                    
                                    if mu_valid and ((final_mu >> pat_idx) & 1):
                                        right_reward += right_pattern_duals[pat_idx]
                            
                            if not mu_valid: continue
                            
                            # Terminal Zeta Check
                            # Terminal Zeta Check
                            # (MP Branching handled via pruning)
                            
                            condition_met = (final_prog >= s_k - epsilon)
                            is_valid = False
                            if obj_mode > 0.5: is_valid = condition_met
                            else: is_valid = condition_met or (tau == max_time)
                            
                            if is_valid:
                                duration_val = tau - r_k + 1
                                rc = final_cost + (duration_val * obj_mode) - gamma_k - right_reward
                                if rc < -1e-6:
                                    best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                                    if stop_at_first_negative: return best_columns
                    
                    # === Option 2: End with AI (only if timeout) ===
                    if is_timeout_scenario:
                        can_end_ai = True
                        if has_sp_fixing and required_mask[j, tau]: can_end_ai = False
                        
                        if can_end_ai:
                            feasible_ai, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                            if feasible_ai:
                                final_cost = cost
                                eff = 1.0
                                if ai_count < len(theta_lookup): eff = theta_lookup[ai_count]
                                final_prog = prog + eff
                                final_path_mask = path_mask
                                
                                # Final Branching Updates
                                final_zeta = zeta_mask
                                # (MP Branching updates handled via pruning, no zeta update needed)
                                final_rho = rho_encoded
                                final_mu = mu_encoded
                                mu_valid = True
                                right_reward = 0.0
                                if has_right_patterns:
                                    for pat_idx in range(num_right_patterns):
                                        t_start = right_pattern_starts[pat_idx]
                                        current_mode = (final_mu >> pat_idx) & 1
                                        if tau == t_start: final_mu = final_mu & ~(1 << pat_idx)
                                        elif tau > t_start:
                                             if current_mode == 1:
                                                 is_req = False
                                                 for elem_idx in range(right_pattern_counts[pat_idx]):
                                                     encoded = right_pattern_elements[pat_idx, elem_idx]
                                                     w_pat = encoded // 1000000
                                                     t_pat = encoded % 1000000
                                                     if w_pat == j and t_pat == tau: is_req = True; break
                                                 if is_req: mu_valid = False; break
                                        if mu_valid and ((final_mu >> pat_idx) & 1):
                                            right_reward += right_pattern_duals[pat_idx]
                                if not mu_valid: continue
                                
                                # (MP Branching check handled via pruning)

                                condition_met = (final_prog >= s_k - epsilon)
                                is_valid = False
                                if obj_mode > 0.5: is_valid = condition_met
                                else: is_valid = condition_met or (tau == max_time)
                                
                                if is_valid:
                                    duration_val = tau - r_k + 1
                                    rc = final_cost + (duration_val * obj_mode) - gamma_k - right_reward
                                    if rc < -1e-6:
                                        best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                                        if stop_at_first_negative: return best_columns

                    # === Option 3: End with Gap ===
                    if is_timeout_scenario and allow_gaps:
                        can_end_gap = True
                        if has_sp_fixing and required_mask[j, tau]: can_end_gap = False
                        
                        if can_end_gap:
                            feasible_gap, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                            if feasible_gap:
                                final_cost = cost
                                final_prog = prog
                                final_path_mask = path_mask | (2 << shift)
                                
                                # Final Branching Updates (Approx same as AI)
                                final_zeta = zeta_mask
                                # (MP Branching updates handled via pruning)
                                final_rho = rho_encoded
                                final_mu = mu_encoded
                                mu_valid = True
                                right_reward = 0.0
                                if has_right_patterns:
                                    for pat_idx in range(num_right_patterns):
                                        t_start = right_pattern_starts[pat_idx]
                                        current_mode = (final_mu >> pat_idx) & 1
                                        if tau == t_start: final_mu = final_mu & ~(1 << pat_idx)
                                        elif tau > t_start:
                                             if current_mode == 1:
                                                 is_req = False
                                                 for elem_idx in range(right_pattern_counts[pat_idx]):
                                                     encoded = right_pattern_elements[pat_idx, elem_idx]
                                                     w_pat = encoded // 1000000
                                                     t_pat = encoded % 1000000
                                                     if w_pat == j and t_pat == tau: is_req = True; break
                                                 if is_req: mu_valid = False; break
                                        if mu_valid and ((final_mu >> pat_idx) & 1):
                                            right_reward += right_pattern_duals[pat_idx]
                                if not mu_valid: continue
                                
                                # (MP Branching check handled via pruning)
                                
                                is_focus = (obj_mode > 0.5)
                                is_valid = (final_prog >= s_k - epsilon) if is_focus else ((final_prog >= s_k - epsilon) or is_timeout_scenario)
                                
                                if is_valid:
                                    duration_val = tau - r_k + 1
                                    rc = final_cost + (duration_val * obj_mode) - gamma_k - right_reward
                                    if rc < -1e-6:
                                        best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                                        if stop_at_first_negative: return best_columns
    
    return best_columns
