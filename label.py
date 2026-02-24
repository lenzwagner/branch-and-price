"""
Core Labeling Algorithm for Column Generation (Pricing Problem Solver)

This module contains the core dynamic programming algorithm for solving 
the pricing problem in column generation, without validation, testing, 
or comparison utilities.
"""
import sys
import heapq
import time
import math
import logging

try:
    import numpy as np
    import label_numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logging.warning("Warning: Numba not found, falling back to Python labeling.")
from logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


# --- Helper Functions ---

def check_strict_feasibility(history, next_val, MS, MIN_MS):
    """
    Check if adding next_val to the history satisfies rolling window constraints.
    
    Args:
        history: Tuple of recent actions (0 or 1 values)
        next_val: The next action to check (0 or 1)
        MS: Rolling window size
        MIN_MS: Minimum human services required in window
        
    Returns:
        bool: True if feasible, False otherwise
    """
    potential_sequence = history + (next_val,)
    seq_len = len(potential_sequence)

    if seq_len < MS:
        current_sum = sum(potential_sequence)
        remaining_slots = MS - seq_len
        max_possible_sum = current_sum + remaining_slots
        if max_possible_sum < MIN_MS:
            return False
        return True
    else:
        current_window = potential_sequence[-MS:]
        if sum(current_window) < MIN_MS:
            return False
        return True


def validate_column_history(path_pattern, MS, MIN_MS):
    """
    Validate that a complete column satisfies rolling window constraints.
    
    Used for post-validation of columns generated with relaxed history tracking.
    
    Args:
        path_pattern: List of 0s and 1s representing the schedule
        MS: Rolling window size
        MIN_MS: Minimum human services required in window
    
    Returns:
        bool: True if column satisfies all rolling window constraints, False otherwise
    """
    path_len = len(path_pattern)
    
    # Check every position in the schedule
    for i in range(path_len):
        # Determine the window to check
        if i + 1 < MS:
            # Not enough history yet - check if remaining slots can satisfy MIN_MS
            current_sum = sum(path_pattern[:i+1])
            remaining_slots = MS - (i + 1)
            max_possible = current_sum + remaining_slots
            if max_possible < MIN_MS:
                return False
        else:
            #Complete window exists
            window_start = i + 1 - MS
            window = path_pattern[window_start:i+1]
            if sum(window) < MIN_MS:
                return False
    
    return True



def _parse_branching_constraints(branch_constraints, recipient_id, branching_variant='mp'):
    """
    Parses branching constraints into efficient lookup structures.
    
    Args:
        branch_constraints: List of constraint objects
        recipient_id: Profile ID
        branching_variant: 'mp' or 'sp'
        
    Returns:
        tuple: (mp_cuts, left_patterns, right_patterns, forbidden_lookup)
    """
    mp_cuts = []
    left_patterns = []
    right_patterns = []
    forbidden_lookup = {}  # NEW: (worker, time) -> True for O(1) lookup

    if not branch_constraints:
        return mp_cuts, left_patterns, right_patterns, forbidden_lookup

    # Handle list of constraint objects (preferred)
    if isinstance(branch_constraints, list):
        for constraint in branch_constraints:
            if not hasattr(constraint, 'profile') or constraint.profile != recipient_id:
                continue

            # SP Branching (Pattern)
            if hasattr(constraint, 'pattern'):
                # SPPatternBranching
                if constraint.direction == 'left':
                    # Limit Usage: sum x <= |P| - 1
                    # Store: (id, elements_set, limit)
                    limit = len(constraint.pattern) - 1 # Prune if count > limit (i.e. count == len)
                    left_patterns.append({
                        'id': id(constraint),
                        'elements': set(constraint.pattern), # Set of (j,t)
                        'limit': limit
                    })
                elif constraint.direction == 'right':
                    # All-or-Nothing: sum x = |P| * w
                    # Store: (id, elements_sorted, first_element, dual)
                    sorted_elements = sorted(list(constraint.pattern), key=lambda x: x[1]) # Sort by time
                    first_element = sorted_elements[0] if sorted_elements else None
                    
                    # Extract dual from master_constraint (like Numba path)
                    dual_val = 0.0
                    if hasattr(constraint, 'master_constraint') and constraint.master_constraint is not None:
                        try:
                            dual_val = constraint.master_constraint.Pi
                        except:
                            dual_val = getattr(constraint, 'dual_var', 0.0)
                    else:
                        dual_val = getattr(constraint, 'dual_var', 0.0)
                    
                    right_patterns.append({
                        'id': id(constraint),
                        'elements': sorted_elements,
                        'elements_set': set(constraint.pattern),
                        'first_element': first_element,
                        'dual': dual_val
                    })

            # MP Branching (No-Good Cuts)
            elif hasattr(constraint, 'original_schedule') and branching_variant == 'mp':
                if constraint.direction == 'left' and constraint.original_schedule:
                    forbidden_schedule = {}
                    for key, val in constraint.original_schedule.items():
                        if len(key) >= 3 and val > 1e-6:
                            j, t = key[1], key[2]
                            forbidden_schedule[(j, t)] = val
                            # NEW: Build transposed lookup for O(1) access
                            forbidden_lookup[(j, t)] = True
                    mp_cuts.append(forbidden_schedule)

    # Debug Printing for SP Constraints
    if left_patterns:
        sorted_elements = sorted(list(p['elements']))
        logger.debug(f"  [SP DEBUG] Recipient {recipient_id}: Found {len(left_patterns)} LEFT patterns")
        for i, p in enumerate(left_patterns):
            logger.debug(f"    Left #{i}: Limit {p['limit']}, Elements: {sorted(list(p['elements']))}")

    if right_patterns:
        logger.debug(f"  [SP DEBUG] Recipient {recipient_id}: Found {len(right_patterns)} RIGHT patterns")
        for i, p in enumerate(right_patterns):
            logger.debug(f"    Right #{i}: Elements: {p['elements']}, Dual: {p['dual']}")

    return mp_cuts, left_patterns, right_patterns, forbidden_lookup


def add_state_to_buckets(buckets, cost, prog, ai_count, hist, path, recipient_id, 
                         pruning_stats, dominance_mode='bucket', 
                         zeta=None, rho=None, mu=None, epsilon=1e-9):
    """
    Adds a state to buckets, applying dominance rules.
    
    Args:
        buckets: State storage structure
        cost: Current accumulated cost
        prog: Current progress towards target
        ai_count: Number of AI sessions used
        hist: History tuple for rolling window
        path: Path pattern (list of 0s and 1s)
        recipient_id: Current recipient ID (for debug output)
        pruning_stats: Statistics dictionary
        dominance_mode: 'bucket' or 'global'
        zeta: MP Branching deviation vector (optional)
        rho: SP Branching Left-Counter vector (optional)
        mu: SP Branching Right-Mode vector (optional)
        epsilon: Tolerance for float comparisons
    """
    # Bucket Key Generation
    # Standard: (ai_count, hist)
    # With MP Branching: + zeta
    # With SP Branching: + mu (Right Modes are incomparable)
    
    key_components = [ai_count, hist]
    if zeta is not None:
        key_components.append(zeta)
    if mu is not None:
        key_components.append(mu)
        
    bucket_key = tuple(key_components)
    
    # Bucket Item Structure:
    # Standard: (cost, prog, path)
    # With SP Left Branching: (cost, prog, rho, path) -- rho is required for dominance
    
    has_rho = (rho is not None)

    # --- BUCKET MANAGEMENT ---
    if bucket_key not in buckets:
        buckets[bucket_key] = []
    
    bucket_list = buckets[bucket_key]

    # --- LOCAL DOMINANCE (Within the Bucket) ---
    is_dominated = False
    dominator = None
    
    for item in bucket_list:
        # Unpack based on structure
        if has_rho:
            c_old, p_old, rho_old, _ = item
        else:
            c_old, p_old, _ = item
            
        # Basic Dominance: Cost' <= Cost AND Prog' >= Prog
        if c_old <= cost + epsilon and p_old >= prog - epsilon:
            # Extended Dominance for rho: rho' <= rho (smaller usage is better/more flexible)
            if has_rho:
                # component-wise check: all rho_old[i] <= rho[i]
                # Wait: IF rho_old <= rho, then old state used less/same of the pattern budget.
                # That means old state is "less restricted" -> BETTER.
                # So if rho_old <= rho, Old dominates New.
                # CORRECT.
                
                # We need to check if rho_old <= rho
                 # Optimization: check length first (should be same)
                if len(rho_old) != len(rho):
                    continue # Should not happen if consistent
                
                rho_better_or_equal = True
                for r1, r2 in zip(rho_old, rho):
                    if r1 > r2: # Old used MORE -> Old is worse
                        rho_better_or_equal = False
                        break
                
                if not rho_better_or_equal:
                    continue
            
            # If we are here, Old dominates New
            is_dominated = True
            dominator = (c_old, p_old)
            break
    
    if is_dominated:
        pruning_stats['dominance'] += 1
        return 
    
    # --- CLEANUP ---
    # Remove existing states that are dominated by the new one
    new_bucket_list = []
    dominated_by_new = 0
    
    for item in bucket_list:
        if has_rho:
            c_old, p_old, rho_old, path_old = item
        else:
            c_old, p_old, path_old = item
            
        # Check if New dominates Old
        # Cost <= Cost' AND Prog >= Prog'
        if cost <= c_old + epsilon and prog >= p_old - epsilon:
            # Check rho: New must have rho <= rho_old
            if has_rho:
                rho_better_or_equal = True
                for r_new, r_old in zip(rho, rho_old):
                    if r_new > r_old: # New used MORE -> New is worse
                        rho_better_or_equal = False
                        break
                
                if not rho_better_or_equal:
                    # New does NOT dominate Old due to rho
                    new_bucket_list.append(item)
                    continue
            
            # New dominates Old
            pruning_stats['dominance'] += 1
            dominated_by_new += 1
            continue
            
        new_bucket_list.append(item)
    
    # Add new state
    if has_rho:
        new_bucket_list.append((cost, prog, rho, path))
    else:
        new_bucket_list.append((cost, prog, path))
        
    buckets[bucket_key] = new_bucket_list


def generate_full_column_vector(worker_id, path_assignments, start_time, end_time, max_time, num_workers):
    """
    Generate the full column vector for a schedule.
    
    Args:
        worker_id: Worker ID (1-indexed)
        path_assignments: List of assignments (0 or 1)
        start_time: Start time of schedule
        end_time: End time of schedule
        max_time: Maximum time horizon
        num_workers: Total number of workers
        
    Returns:
        List of floats representing the full column vector
    """
    vector_length = num_workers * max_time
    full_vector = [0.0] * vector_length
    worker_offset = (worker_id - 1) * max_time
    for t_idx, val in enumerate(path_assignments):
        current_time = start_time + t_idx
        global_idx = worker_offset + (current_time - 1)
        if 0 <= global_idx < vector_length and val == 1:
            full_vector[global_idx] = 1.0
    return full_vector


def compute_lower_bound(current_cost, start_time, end_time, gamma_k, obj_mode,
                         worker=None, pi_dict=None, max_time=None):
    """
    Calculates Enhanced Lower Bound for Bound Pruning.

    Includes:
    1. Naive bound: current_cost + duration * obj_mode - gamma_k
    2. Completion cost: If end_time < max_time, we MUST end with therapist,
       so -π_{j,end_time} is a guaranteed cost component.

    Args:
        current_cost: Accumulated -π values so far
        start_time: r_k (release time)
        end_time: τ (target end time)
        gamma_k: Dual value γ_k
        obj_mode: Objective multiplier (0 or 1)
        worker: Worker ID (for completion cost)
        pi_dict: Dual values {(worker, time): value}
        max_time: Planning horizon |T|

    Returns:
        float: Minimum achievable final Reduced Cost (optimistic but tighter)
    """
    # Time Cost is fixed for the specific column length
    duration = end_time - start_time + 1
    time_cost = duration * obj_mode

    # Naive lower bound
    lower_bound = current_cost + time_cost - gamma_k

    # Enhanced: Add completion cost if we know final therapist visit is required
    if worker is not None and pi_dict is not None and max_time is not None:
        if end_time < max_time:
            # Final action MUST be therapist → this cost is guaranteed
            completion_cost = -pi_dict.get((worker, end_time), 0.0)
            lower_bound += completion_cost

    return lower_bound


def compute_candidate_workers(workers, r_k, tau_max, pi_dict):
    """
    Worker Dominance Pre-Elimination:
    Worker j1 dominates j2 if π_{j1,t} >= π_{j2,t} for all t in [r_k, tau_max]
    AND π_{j1,t} > π_{j2,t} for at least one t (strict dominance).
    Since π values are <= 0 (implicit costs), higher π means lower cost.
    
    Returns:
        List of non-dominated workers
    """
    candidate_workers = []

    for j1 in workers:
        is_dominated = False

        for j2 in workers:
            if j1 == j2:
                continue

            # Check if j2 dominates j1
            all_better_or_equal = True
            at_least_one_strictly_better = False

            for t in range(r_k, tau_max + 1):
                pi_j1 = pi_dict.get((j1, t), 0.0)
                pi_j2 = pi_dict.get((j2, t), 0.0)

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
            candidate_workers.append(j1)



    return candidate_workers


# --- Label Recycling Helper Functions ---

def group_recipients_by_shared_prefix(recipients_r, recipients_s, ms, min_ms):
    """
    Group recipients by (r_k, MS, MIN_MS) for shared prefix computation.
    
    Recipients in the same group can share early DP states since they:
    - Start at the same time (r_k)
    - Have the same rolling window constraints (MS, MIN_MS)
    
    Args:
        recipients_r: Dict {recipient_id: r_k (start time)}
        recipients_s: Dict {recipient_id: s_k (target progress)}
        ms: Rolling window size
        min_ms: Minimum therapist sessions in window
        
    Returns:
        Dict mapping (r_k, ms, min_ms) -> List of (recipient_id, s_k) tuples
    """
    groups = {}
    
    for recipient_id in recipients_r:
        r_k = recipients_r[recipient_id]
        s_k = recipients_s[recipient_id]
        
        # Group key: (start_time, window_size, min_sessions)
        key = (r_k, ms, min_ms)
        
        if key not in groups:
            groups[key] = []
        groups[key].append((recipient_id, s_k))
    
    # Debug: Log groups with multiple recipients (where recycling helps)
    for key, recipients_list in groups.items():
        if len(recipients_list) > 1:
            r_k, ms_val, min_ms_val = key
            recipient_ids = [r[0] for r in recipients_list]
            logger.info(f"  [LABEL RECYCLING] Group r_k={r_k}: {len(recipients_list)} recipients sharing prefix → {recipient_ids}")
    
    return groups


def compute_shared_prefix_states(r_k, worker, max_time, ms, min_ms, pi_dict, 
                                  theta_lookup, max_prefix_length):
    """
    Compute shared prefix DP states up to a certain depth.
    
    These states can be reused for all recipients in a group.
    
    Args:
        r_k: Start time
        worker: Worker ID
        max_time: Planning horizon
        ms: Rolling window size
        min_ms: Min therapist sessions
        pi_dict: Dual values {(worker, time): value}
        theta_lookup: AI efficiency lookup table
        max_prefix_length: How many time steps to compute as shared prefix
        
    Returns:
        Dict mapping bucket_key -> list of (cost, prog, ai_count, hist, path)
    """
    logger.debug(f"  [LABEL RECYCLING] Computing shared prefix for worker {worker}, r_k={r_k}, length={max_prefix_length}")
    epsilon = 1e-9
    
    # Initialize with start state
    start_cost = -pi_dict.get((worker, r_k), 0.0)
    initial_history = (1,)  # First action is always therapist
    
    # State structure: bucket_key = (ai_count, hist)
    # Item: (cost, prog, path_list)
    current_states = {
        (0, initial_history): [(start_cost, 1.0, [1])]
    }
    
    # Expand for prefix_length steps
    for t in range(r_k + 1, r_k + max_prefix_length):
        if t > max_time:
            break
            
        next_states = {}
        
        for bucket_key, bucket_list in current_states.items():
            ai_count, hist = bucket_key
            
            for cost, prog, path in bucket_list:
                # Try therapist action (1)
                if check_strict_feasibility(hist, 1, ms, min_ms):
                    cost_ther = cost - pi_dict.get((worker, t), 0.0)
                    prog_ther = prog + 1.0
                    new_hist_ther = (hist + (1,))
                    if len(new_hist_ther) > ms - 1:
                        new_hist_ther = new_hist_ther[-(ms - 1):]
                    
                    new_key = (ai_count, new_hist_ther)
                    if new_key not in next_states:
                        next_states[new_key] = []
                    
                    # Simple dominance: keep if not dominated
                    dominated = False
                    for existing in next_states[new_key]:
                        if existing[0] <= cost_ther + epsilon and existing[1] >= prog_ther - epsilon:
                            dominated = True
                            break
                    
                    if not dominated:
                        # Remove states dominated by new one
                        next_states[new_key] = [
                            s for s in next_states[new_key]
                            if not (cost_ther <= s[0] + epsilon and prog_ther >= s[1] - epsilon)
                        ]
                        next_states[new_key].append((cost_ther, prog_ther, path + [1]))
                
                # Try AI action (0)
                if check_strict_feasibility(hist, 0, ms, min_ms):
                    cost_ai = cost
                    efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                    prog_ai = prog + efficiency
                    ai_count_new = ai_count + 1
                    new_hist_ai = (hist + (0,))
                    if len(new_hist_ai) > ms - 1:
                        new_hist_ai = new_hist_ai[-(ms - 1):]
                    
                    new_key = (ai_count_new, new_hist_ai)
                    if new_key not in next_states:
                        next_states[new_key] = []
                    
                    dominated = False
                    for existing in next_states[new_key]:
                        if existing[0] <= cost_ai + epsilon and existing[1] >= prog_ai - epsilon:
                            dominated = True
                            break
                    
                    if not dominated:
                        next_states[new_key] = [
                            s for s in next_states[new_key]
                            if not (cost_ai <= s[0] + epsilon and prog_ai >= s[1] - epsilon)
                        ]
                        next_states[new_key].append((cost_ai, prog_ai, path + [0]))
        
        current_states = next_states
        if not current_states:
            break
    
    return current_states


def continue_from_prefix(prefix_states, prefix_end_time, target_end, s_k, 
                         worker, max_time, ms, min_ms, pi_dict, theta_lookup,
                         gamma_k, obj_mode, recipient_id, workers):
    """
    Continue DP from prefix states to find complete columns.
    
    Args:
        prefix_states: Pre-computed states from shared prefix
        prefix_end_time: Time at which prefix states end
        target_end: End time for this column (tau)
        s_k: Target progress
        worker: Worker ID
        max_time: Planning horizon
        ms, min_ms: Rolling window params
        pi_dict: Dual values
        theta_lookup: AI efficiency lookup
        gamma_k: Dual value gamma
        obj_mode: Objective multiplier
        recipient_id: Recipient ID for output
        workers: List of all workers (for column vector generation)
        
    Returns:
        List of column dictionaries with negative reduced cost
    """
    epsilon = 1e-9
    columns = []
    
    # Debug: Log recycling usage
    prefix_state_count = sum(len(b) for b in prefix_states.values()) if prefix_states else 0
    
    # Start from prefix states
    current_states = {}
    for bucket_key, bucket_list in prefix_states.items():
        current_states[bucket_key] = [(c, p, path[:]) for c, p, path in bucket_list]
    
    # Get r_k from path length
    first_bucket_list = list(prefix_states.values())[0]
    if not first_bucket_list:
        return columns
    r_k = prefix_end_time - len(first_bucket_list[0][2]) + 1
    
    # Continue DP from prefix_end_time + 1 to target_end - 1
    for t in range(prefix_end_time + 1, target_end):
        next_states = {}
        
        for bucket_key, bucket_list in current_states.items():
            ai_count, hist = bucket_key
            
            for cost, prog, path in bucket_list:
                # Lower bound pruning
                duration = target_end - r_k + 1
                lb = cost + duration * obj_mode - gamma_k
                if lb >= 0:
                    continue
                
                # Feasibility check
                remaining_steps = target_end - t
                if obj_mode > 0.5:  # Focus patient
                    if prog + remaining_steps * 1.0 < s_k - epsilon:
                        continue
                
                # Try therapist action
                if check_strict_feasibility(hist, 1, ms, min_ms):
                    cost_ther = cost - pi_dict.get((worker, t), 0.0)
                    prog_ther = prog + 1.0
                    new_hist = (hist + (1,))
                    if len(new_hist) > ms - 1:
                        new_hist = new_hist[-(ms - 1):]
                    
                    new_key = (ai_count, new_hist)
                    if new_key not in next_states:
                        next_states[new_key] = []
                    
                    # Dominance check
                    dominated = False
                    for existing in next_states[new_key]:
                        if existing[0] <= cost_ther + epsilon and existing[1] >= prog_ther - epsilon:
                            dominated = True
                            break
                    if not dominated:
                        next_states[new_key] = [
                            s for s in next_states[new_key]
                            if not (cost_ther <= s[0] + epsilon and prog_ther >= s[1] - epsilon)
                        ]
                        next_states[new_key].append((cost_ther, prog_ther, path + [1]))
                
                # Try AI action
                if check_strict_feasibility(hist, 0, ms, min_ms):
                    cost_ai = cost
                    eff = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                    prog_ai = prog + eff
                    ai_new = ai_count + 1
                    new_hist = (hist + (0,))
                    if len(new_hist) > ms - 1:
                        new_hist = new_hist[-(ms - 1):]
                    
                    new_key = (ai_new, new_hist)
                    if new_key not in next_states:
                        next_states[new_key] = []
                    
                    dominated = False
                    for existing in next_states[new_key]:
                        if existing[0] <= cost_ai + epsilon and existing[1] >= prog_ai - epsilon:
                            dominated = True
                            break
                    if not dominated:
                        next_states[new_key] = [
                            s for s in next_states[new_key]
                            if not (cost_ai <= s[0] + epsilon and prog_ai >= s[1] - epsilon)
                        ]
                        next_states[new_key].append((cost_ai, prog_ai, path + [0]))
        
        current_states = next_states
        if not current_states:
            break
    
    # Final step: extract columns
    is_timeout = (target_end == max_time)
    
    for bucket_key, bucket_list in current_states.items():
        ai_count, hist = bucket_key
        
        for cost, prog, path in bucket_list:
            possible_moves = []
            if check_strict_feasibility(hist, 1, ms, min_ms):
                possible_moves.append(1)
            if is_timeout and check_strict_feasibility(hist, 0, ms, min_ms):
                possible_moves.append(0)
            
            for move in possible_moves:
                if move == 1:
                    final_cost = cost - pi_dict.get((worker, target_end), 0.0)
                    final_prog = prog + 1.0
                else:
                    final_cost = cost
                    eff = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                    final_prog = prog + eff
                
                final_path = path + [move]
                
                # Check completion
                is_focus = (obj_mode > 0.5)
                condition_met = (final_prog >= s_k - epsilon)
                is_valid_end = condition_met if is_focus else (condition_met or is_timeout)
                
                if is_valid_end:
                    duration = len(final_path)
                    start = target_end - duration + 1
                    reduced_cost = (obj_mode * duration) + final_cost - gamma_k
                    
                    if reduced_cost < -epsilon:
                        columns.append({
                            'k': recipient_id,
                            'worker': worker,
                            'start': start,
                            'end': target_end,
                            'duration': duration,
                            'reduced_cost': reduced_cost,
                            'final_progress': final_prog,
                            'x_vector': generate_full_column_vector(worker, final_path, start, target_end, max_time, len(workers)),
                            'path_pattern': final_path
                        })
    
    return columns


# --- Core Labeling Algorithm ---

def solve_pricing_for_recipient(recipient_id, r_k, s_k, gamma_k, obj_mode, pi_dict, 
                                workers, max_time, ms, min_ms, theta_lookup,
                                use_bound_pruning=True, dominance_mode='bucket', 
                                branch_constraints=None, branching_variant='mp',
                                max_columns=10, use_pure_dp_optimization=True,
                                use_heuristic_pricing=False, max_labels_per_bucket=None,
                                stop_at_first_negative=False, use_relaxed_history=False,
                                allow_gaps=False):
    """
    Solve the pricing problem for a single recipient.
    
    Args:
        recipient_id: Recipient ID
        r_k: Release time
        s_k: Service target
        gamma_k: Dual value gamma
        obj_mode: Objective mode multiplier
        pi_dict: Dual values pi {(worker_id, time): value}
        workers: List of worker IDs
        max_time: Planning horizon
        ms: Rolling window size
        min_ms: Minimum human services in window
        theta_lookup: AI efficiency lookup table
        use_bound_pruning: Enable lower bound pruning
        dominance_mode: 'bucket' or 'global'
        branch_constraints: Optional branch constraints
        branching_variant: Branching strategy ('mp' or 'sp')
        max_columns: Maximum number of columns to return
        use_pure_dp_optimization: Enable Pure DP fast path when no constraints (Option 4)
        use_heuristic_pricing: Enable heuristic pricing mode (aggressive pruning)
        max_labels_per_bucket: Limit labels per bucket (None = unlimited/exact)
        stop_at_first_negative: Stop DP early if negative RC found
        use_relaxed_history: Skip history tracking in DP (experimental, requires validation)
        
    Returns:
        List of best columns (dictionaries)
    """
    best_columns = []
    epsilon = 1e-9
    
    pruning_stats = {
        'lb': 0,
        'dominance': 0,
        'printed_dominance': {},
        'relaxation_generated': 0,
        'relaxation_rejected': 0
    }

    # --- TIMING INSTRUMENTATION ---
    timers = {
        'init': 0.0,
        'state_expansion': 0.0,
        'final_step': 0.0,
        'total': 0.0
    }
    t_start_total = time.time()
    t_init_start = time.time()

    # Worker Dominance Pre-Elimination
    # Use Numba-optimized version if available
    if HAS_NUMBA:
        # Convert pi_dict to matrix format for Numba
        workers_array = np.array(workers, dtype=np.int64)
        pi_matrix = np.zeros((max(workers) + 1, max_time + 2), dtype=np.float64)
        for (w, t), v in pi_dict.items():
            if w <= max(workers) and t <= max_time + 1:
                pi_matrix[w, t] = v
        candidate_workers_numba = label_numba.compute_candidate_workers_numba(workers_array, r_k, max_time, pi_matrix)
        candidate_workers = candidate_workers_numba.tolist()  # Fast numpy tolist()
    else:
        candidate_workers = compute_candidate_workers(workers, r_k, max_time, pi_dict)
    eliminated_workers = [w for w in workers if w not in candidate_workers]

    # Print for each Recipient
    if eliminated_workers:
        logger.info(f"Recipient with entry {r_k} and req {s_k} {recipient_id:2d}: Candidate workers = {candidate_workers} (eliminated {eliminated_workers})")
    else:
        logger.info(f"Recipient with entry {r_k} and req {s_k} {recipient_id:2d}: Candidate workers = {candidate_workers} (no dominance)")
  
    time_until_end = max_time - r_k + 1

    # Pre-build constraint lookup structures for efficient access
    mp_cuts, left_patterns, right_patterns, forbidden_lookup = _parse_branching_constraints(branch_constraints, recipient_id, branching_variant)
    forbidden_schedules = mp_cuts  # Kept for backward compat
    use_branch_constraints = bool(forbidden_schedules)
    
    # Setup SP Branching (Patterns)
    use_sp_branching = bool(left_patterns) or bool(right_patterns)

    # DEBUG: Print SP Branching Constraints
    if use_sp_branching:
        logger.info(f"  [SP BRANCHING] Recipient {recipient_id}: Active Pattern Constraints")
        if left_patterns:
            logger.info(f"    Left Patterns (Limit Usage): {len(left_patterns)}")
        if right_patterns:
            logger.info(f"    Right Patterns (All-or-Nothing): {len(right_patterns)}")
            
    if use_branch_constraints:
        logger.print(f"\n  [MP BRANCHING] Recipient {recipient_id}: {len(forbidden_schedules)} no-good cut(s) active")
        # (Optional: Print details skipped for brevity in production, can restore if needed)
    else:
        if not use_sp_branching:
            logger.info(f"  [BRANCHING] No active constraints for recipient {recipient_id}")

    timers['init'] += time.time() - t_init_start

    # --- OPTIMIZED FAST PATH (Inner Function) ---
    def run_fast_path():
        """
        Optimized DP loop that skips all SP Branching overhead (rho, mu checks).
        Uses O(1) forbidden_lookup for MP branching instead of O(C) zeta tracking.
        """
        for j in candidate_workers:
            effective_min_duration = min(int(s_k), time_until_end)
            start_tau = r_k + effective_min_duration - 1

            for tau in range(start_tau, max_time + 1):
                is_timeout_scenario = (tau == max_time)
                
                # O(1) CHECK: Skip if initial assignment is forbidden
                if use_branch_constraints and (j, r_k) in forbidden_lookup:
                    continue
                
                start_cost = -pi_dict.get((j, r_k), 0)

                current_states = {}
                initial_history = (1,)
                # No zeta needed - using forbidden_lookup instead
                add_state_to_buckets(current_states, start_cost, 1.0, 0, initial_history, [1], 
                                     recipient_id, pruning_stats, dominance_mode, 
                                     None, None, None, epsilon)

                t_dp_start = time.time()
                for t in range(r_k + 1, tau):
                    next_states = {}
                    
                    # Simple bucket structure: (ai_count, hist) - no zeta!
                    for bucket_key, bucket_list in current_states.items():
                        ai_count, hist = bucket_key
                        
                        for item in bucket_list:
                            cost, prog, path = item

                            if use_bound_pruning:
                                lb = compute_lower_bound(cost, r_k, tau, gamma_k, obj_mode,
                                                         worker=j, pi_dict=pi_dict, max_time=max_time)
                                if lb >= 0:
                                    pruning_stats['lb'] += 1
                                    continue

                            remaining_steps = tau - t + 1
                            if not is_timeout_scenario:
                                if obj_mode > 0.5:
                                    if prog + remaining_steps * 1.0 < s_k - epsilon:
                                        continue

                            # A: Therapist
                            if check_strict_feasibility(hist, 1, ms, min_ms):
                                # O(1) CHECK: Skip if forbidden
                                if use_branch_constraints and (j, t) in forbidden_lookup:
                                    continue
                                
                                cost_ther = cost - pi_dict.get((j, t), 0)
                                prog_ther = prog + 1.0
                                new_hist_ther = (hist + (1,))
                                if len(new_hist_ther) > ms - 1: 
                                    new_hist_ther = new_hist_ther[-(ms - 1):]

                                add_state_to_buckets(next_states, cost_ther, prog_ther, ai_count, new_hist_ther, 
                                                   path + [1], recipient_id, pruning_stats, dominance_mode, 
                                                   None, None, None, epsilon)

                            # B: AI
                            if check_strict_feasibility(hist, 0, ms, min_ms):
                                # AI assignments are never in forbidden_lookup (only therapist assignments)
                                cost_ai = cost
                                efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                                prog_ai = prog + efficiency
                                ai_count_new = ai_count + 1
                                new_hist_ai = (hist + (0,))
                                if len(new_hist_ai) > ms - 1: 
                                    new_hist_ai = new_hist_ai[-(ms - 1):]

                                add_state_to_buckets(next_states, cost_ai, prog_ai, ai_count_new, new_hist_ai, 
                                                   path + [0], recipient_id, pruning_stats, dominance_mode, 
                                                   None, None, None, epsilon)

                            # C: Gap (Treat as non-human (0) for history, but keeps ai_count unchanged)
                            if allow_gaps and check_strict_feasibility(hist, 0, ms, min_ms):
                                # Gaps are never forbidden
                                cost_gap = cost
                                prog_gap = prog
                                new_hist_gap = (hist + (0,))
                                if len(new_hist_gap) > ms - 1: 
                                    new_hist_gap = new_hist_gap[-(ms - 1):]

                                add_state_to_buckets(next_states, cost_gap, prog_gap, ai_count, new_hist_gap, 
                                                   path + [2], recipient_id, pruning_stats, dominance_mode, 
                                                   None, None, None, epsilon)

                    current_states = next_states
                    if not current_states: 
                        break
                timers['state_expansion'] += time.time() - t_dp_start

                # Final Step
                t_final_start = time.time()
                for bucket_key, bucket_list in current_states.items():
                    ai_count, hist = bucket_key

                    for item in bucket_list:
                         cost, prog, path = item
                         
                         possible_moves = []
                         if check_strict_feasibility(hist, 1, ms, min_ms): 
                             possible_moves.append(1)
                         if is_timeout_scenario and check_strict_feasibility(hist, 0, ms, min_ms): 
                             possible_moves.append(0)
                         if allow_gaps and check_strict_feasibility(hist, 0, ms, min_ms): 
                             possible_moves.append(2)

                         for move in possible_moves:
                             # O(1) CHECK: Skip if final therapist assignment is forbidden
                             if move == 1 and use_branch_constraints and (j, tau) in forbidden_lookup:
                                 continue
                             
                             if move == 1:
                                 final_cost_accum = cost - pi_dict.get((j, tau), 0)
                                 final_prog = prog + 1.0
                             elif move == 0:
                                 final_cost_accum = cost
                                 efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                                 final_prog = prog + efficiency
                             else:  # move == 2 (gap)
                                 final_cost_accum = cost
                                 final_prog = prog

                             final_path = path + [move]
                             condition_met = (final_prog >= s_k - epsilon)

                             is_focus_patient = (obj_mode > 0.5)
                             if is_focus_patient: 
                                 is_valid_end = condition_met
                             else: 
                                 is_valid_end = condition_met or is_timeout_scenario

                             if is_valid_end:
                                 duration = tau - r_k + 1
                                 reduced_cost = (obj_mode * duration) + final_cost_accum - gamma_k
                                 
                                 if reduced_cost < -epsilon:
                                     col_candidate = {
                                         'k': recipient_id,
                                         'worker': j,
                                         'start': r_k,
                                         'end': tau,
                                         'duration': duration,
                                         'reduced_cost': reduced_cost,
                                         'final_progress': final_prog,
                                         'x_vector': generate_full_column_vector(j, final_path, r_k, tau, max_time, len(workers)),
                                         'path_pattern': final_path
                                     }
                                     best_columns.append(col_candidate)
                timers['final_step'] += time.time() - t_final_start

    # --- PURE DP FAST PATH (Option 4) ---
    def run_pure_dp():
        """
        Pure DP loop with ZERO constraint overhead.
        Used when use_sp_branching is False AND use_branch_constraints is False.
        
        This is the fastest path - no zeta, rho, or mu tracking.
        Maximum performance for root node and constraint-free profiles.
        """
        for j in candidate_workers:
            effective_min_duration = min(int(s_k), time_until_end)
            start_tau = r_k + effective_min_duration - 1

            for tau in range(start_tau, max_time + 1):
                is_timeout_scenario = (tau == max_time)
                start_cost = -pi_dict.get((j, r_k), 0)

                current_states = {}
                initial_history = (1,)
                # Pure DP: All constraint parameters are None
                add_state_to_buckets(current_states, start_cost, 1.0, 0, initial_history, [1], 
                                     recipient_id, pruning_stats, dominance_mode, 
                                     None, None, None, epsilon)

                t_dp_start = time.time()
                for t in range(r_k + 1, tau):
                    next_states = {}
                    
                    # Iterate over all buckets - Simple structure: (ai_count, hist)
                    for bucket_key, bucket_list in current_states.items():
                        ai_count, hist = bucket_key  # Always 2 elements
                        
                        for item in bucket_list:
                            cost, prog, path = item  # Always 3 elements

                            if use_bound_pruning:
                                lb = compute_lower_bound(cost, r_k, tau, gamma_k, obj_mode,
                                                         worker=j, pi_dict=pi_dict, max_time=max_time)
                                if lb >= 0:
                                    pruning_stats['lb'] += 1
                                    continue

                            remaining_steps = tau - t + 1
                            if not is_timeout_scenario:
                                if obj_mode > 0.5:
                                    if prog + remaining_steps * 1.0 < s_k - epsilon:
                                        continue

                            # A: Therapist
                            if check_strict_feasibility(hist, 1, ms, min_ms):
                                cost_ther = cost - pi_dict.get((j, t), 0)
                                prog_ther = prog + 1.0
                                new_hist_ther = (hist + (1,))
                                if len(new_hist_ther) > ms - 1: new_hist_ther = new_hist_ther[-(ms - 1):]

                                add_state_to_buckets(next_states, cost_ther, prog_ther, ai_count, new_hist_ther, 
                                                   path + [1], recipient_id, pruning_stats, dominance_mode, 
                                                   None, None, None, epsilon)

                            # B: AI
                            if check_strict_feasibility(hist, 0, ms, min_ms):
                                cost_ai = cost
                                efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                                prog_ai = prog + efficiency
                                ai_count_new = ai_count + 1
                                new_hist_ai = (hist + (0,))
                                if len(new_hist_ai) > ms - 1: new_hist_ai = new_hist_ai[-(ms - 1):]

                                add_state_to_buckets(next_states, cost_ai, prog_ai, ai_count_new, new_hist_ai, 
                                                   path + [0], recipient_id, pruning_stats, dominance_mode, 
                                                   None, None, None, epsilon)

                            # C: Gap (Treat as non-human (0) for history, but keeps ai_count unchanged)
                            if allow_gaps and check_strict_feasibility(hist, 0, ms, min_ms):
                                cost_gap = cost
                                prog_gap = prog
                                new_hist_gap = (hist + (0,))
                                if len(new_hist_gap) > ms - 1: new_hist_gap = new_hist_gap[-(ms - 1):]

                                add_state_to_buckets(next_states, cost_gap, prog_gap, ai_count, new_hist_gap, 
                                                   path + [2], recipient_id, pruning_stats, dominance_mode, 
                                                   None, None, None, epsilon)

                    current_states = next_states
                    if not current_states: break
                timers['state_expansion'] += time.time() - t_dp_start

                # Final Step
                t_final_start = time.time()
                for bucket_key, bucket_list in current_states.items():
                    ai_count, hist = bucket_key  # Simple unpacking

                    for item in bucket_list:
                         cost, prog, path = item
                         
                         possible_moves = []
                         if check_strict_feasibility(hist, 1, ms, min_ms): possible_moves.append(1)
                         if is_timeout_scenario and check_strict_feasibility(hist, 0, ms, min_ms): possible_moves.append(0)
                         # Add Gap (2) if allowed and feasible (treat as 0)
                         if allow_gaps and check_strict_feasibility(hist, 0, ms, min_ms): possible_moves.append(2)

                         for move in possible_moves:
                             if move == 1:
                                 final_cost_accum = cost - pi_dict.get((j, tau), 0)
                                 final_prog = prog + 1.0
                             elif move == 0:
                                 final_cost_accum = cost
                                 efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                                 final_prog = prog + efficiency
                             elif move == 2:
                                 final_cost_accum = cost
                                 final_prog = prog

                             final_path = path + [move]
                             condition_met = (final_prog >= s_k - epsilon)

                             is_focus_patient = (obj_mode > 0.5)
                             if is_focus_patient: is_valid_end = condition_met
                             else: is_valid_end = condition_met or is_timeout_scenario

                             if is_valid_end:
                                 duration = tau - r_k + 1
                                 reduced_cost = (obj_mode * duration) + final_cost_accum - gamma_k
                                 
                                 if reduced_cost < -epsilon:
                                     col_candidate = {
                                         'k': recipient_id,
                                         'worker': j,
                                         'start': r_k,
                                         'end': tau,
                                         'duration': duration,
                                         'reduced_cost': reduced_cost,
                                         'final_progress': final_prog,
                                         'x_vector': generate_full_column_vector(j, final_path, r_k, tau, max_time, len(workers)),
                                         'path_pattern': final_path
                                     }
                                     best_columns.append(col_candidate)
                timers['final_step'] += time.time() - t_final_start

    # --- RELAXED DP PATH (State-Space Relaxation) ---
    def run_relaxed_dp():
        """
        Relaxed DP WITHOUT history tracking - experimental optimization.
        Bucket key: (ai_count,) only - NO hist component!
        Validates columns afterward with validate_column_history().
        """
        for j in candidate_workers:
            effective_min_duration = min(int(s_k), time_until_end)
            start_tau = r_k + effective_min_duration - 1


            for tau in range(start_tau, max_time + 1):
                is_timeout_scenario = (tau == max_time)
                start_cost = -pi_dict.get((j, r_k), 0)

                current_states = {(0,): [(start_cost, 1.0, [1])]}  # Bucket: (ai_count,)

                t_dp_start = time.time()
                for t in range(r_k + 1, tau):
                    next_states = {}
                    
                    for bucket_key, bucket_list in current_states.items():
                        ai_count = bucket_key[0]
                        
                        for cost, prog, path in bucket_list:
                            if use_bound_pruning:
                                lb = compute_lower_bound(cost, r_k, tau, gamma_k, obj_mode,
                                                         worker=j, pi_dict=pi_dict, max_time=max_time)
                                if lb >= 0:
                                    pruning_stats['lb'] += 1
                                    continue

                            remaining_steps = tau - t + 1
                            if not is_timeout_scenario and obj_mode > 0.5:
                                if prog + remaining_steps * 1.0 < s_k - epsilon:
                                    continue

                            # Therapist (NO history check!)
                            cost_ther = cost - pi_dict.get((j, t), 0)
                            prog_ther = prog + 1.0
                            bucket_key_ther = (ai_count,)
                            if bucket_key_ther not in next_states:
                                next_states[bucket_key_ther] = []
                            next_states[bucket_key_ther].append((cost_ther, prog_ther, path + [1]))

                            # AI (NO history check!)
                            efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                            cost_ai = cost
                            prog_ai = prog + efficiency
                            bucket_key_ai = (ai_count + 1,)
                            if bucket_key_ai not in next_states:
                                next_states[bucket_key_ai] = []
                            next_states[bucket_key_ai].append((cost_ai, prog_ai, path + [0]))

                    current_states = next_states
                    if not current_states: break
                timers['state_expansion'] += time.time() - t_dp_start

                # Final Step with validation
                t_final_start = time.time()
                for bucket_key, bucket_list in current_states.items():
                    ai_count = bucket_key[0]

                    for cost, prog, path in bucket_list:
                        possible_moves = [1]
                        if is_timeout_scenario: possible_moves.append(0)

                        for move in possible_moves:
                            if move == 1:
                                final_cost = cost - pi_dict.get((j, tau), 0)
                                final_prog = prog + 1.0
                            else:
                                eff = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                                final_cost = cost
                                final_prog = prog + eff

                            final_path = path + [move]
                            condition_met = (final_prog >= s_k - epsilon)
                            is_valid_end = condition_met if obj_mode > 0.5 else (condition_met or is_timeout_scenario)

                            if is_valid_end:
                                duration = tau - r_k + 1
                                reduced_cost = (obj_mode * duration) + final_cost - gamma_k
                                
                                if reduced_cost < -epsilon:
                                    pruning_stats['relaxation_generated'] += 1
                                    
                                    # POST-VALIDATION
                                    if validate_column_history(final_path, ms, min_ms):
                                        col_candidate = {
                                            'k': recipient_id,
                                            'worker': j,
                                            'start': r_k,
                                            'end': tau,
                                            'duration': duration,
                                            'reduced_cost': reduced_cost,
                                            'final_progress': final_prog,
                                            'x_vector': generate_full_column_vector(j, final_path, r_k, tau, max_time, len(workers)),
                                            'path_pattern': final_path
                                        }
                                        best_columns.append(col_candidate)
                                    else:
                                        pruning_stats['relaxation_rejected'] += 1
                timers['final_step'] += time.time() - t_final_start

    # --- EXECUTION CONTROL: Choose the fastest applicable path ---
    if use_pure_dp_optimization and use_relaxed_history and not use_sp_branching and not use_branch_constraints:
        # Path 0: RELAXED DP (Fastest, but experimental - no history tracking!)
        run_relaxed_dp()
        candidate_workers = []  # Skip sequential loop
    elif use_pure_dp_optimization and not use_sp_branching and not use_branch_constraints:
        # Path 1: PURE DP (Fastest - No constraints at all)
        run_pure_dp()
        candidate_workers = []  # Skip sequential loop
    elif not use_sp_branching:
        # Path 2: FAST PATH (MP Constraints only)
        run_fast_path()
        candidate_workers = []  # Skip sequential loop
    # else: Path 3: FULL SP PATH (Fall through to sequential loop below)


    for j in candidate_workers:
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1

        for tau in range(start_tau, max_time + 1):
            is_timeout_scenario = (tau == max_time)

            start_cost = -pi_dict.get((j, r_k), 0)
            num_cuts = len(forbidden_schedules)

            initial_zeta = tuple([0] * num_cuts) if use_branch_constraints else None

            # SP Branching Initialization
            initial_rho = None
            initial_mu = None
            compatible_right_indices = set()

            if use_sp_branching:
                # Initialize vectors
                if left_patterns:
                    initial_rho = [0] * len(left_patterns)
                if right_patterns:
                    initial_mu = [0] * len(right_patterns) # 0: Exclude, 1: Cover
                    
                    # Pre-check compatibility: 
                    # A pattern is compatible with worker j only if ALL its elements belong to j.
                    # If incompatible, we can never Cover it, so Mode must be 0 (Exclude).
                    for idx, pat in enumerate(right_patterns):
                        is_compatible = True
                        for (wj, wt) in pat['elements']:
                            if wj != j:
                                is_compatible = False
                                break
                        if is_compatible:
                            compatible_right_indices.add(idx)

                # Process state for Start Time (r_k) - Action 1 (Therapist) is assumed at start
                # Left Patterns (Update Counts)
                if initial_rho:
                    for idx, pat in enumerate(left_patterns):
                        if (j, r_k) in pat['elements']:
                            initial_rho[idx] += 1
                            # Immediate Pruning Check (Unlikely at start but possible if limit=0)
                            if initial_rho[idx] > pat['limit']:
                                # Start state is invalid!
                                initial_rho = None # Signal to skip
                                break
                    if initial_rho:
                        initial_rho = tuple(initial_rho)

                # Right Patterns (Update Modes)
                # Only iterate compatible patterns (others stay 0)
                if initial_mu and (initial_rho is not None or not initial_rho):
                    for idx in compatible_right_indices:
                        pat = right_patterns[idx]
                        first_elt = pat['first_element']
                        if first_elt == (j, r_k):
                            initial_mu[idx] = 1 # Start Covering
                    initial_mu = tuple(initial_mu)

            # Show zeta initialization for recipients with active branching constraints
            if use_branch_constraints and tau == start_tau:
                first_candidate = candidate_workers[0] if candidate_workers else None
                if j == first_candidate:
                    logger.print(f"\n  [ZETA VECTOR] Recipient {recipient_id} (BRANCHING PROFILE): Initialized with {num_cuts} elements")
                    logger.print(f"    Initial ζ = {initial_zeta}")

            current_states = {}
            
            # Only add start state if valid (rho check passed)
            valid_start = True
            if use_sp_branching and left_patterns and initial_rho is None:
                valid_start = False
            
            if valid_start:
                # Initialize with start state
                initial_history = (1,)  # First action is always 1 (Therapist)
                add_state_to_buckets(current_states, start_cost, 1.0, 0, initial_history, [1], 
                                   recipient_id, pruning_stats, dominance_mode, 
                                   initial_zeta, initial_rho, initial_mu, epsilon)

            # Helper for SP State Update
            def get_next_sp_state(rho, mu, is_therapist, time_t):
                if not use_sp_branching:
                    return rho, mu, False

                next_rho = list(rho) if rho else None
                next_mu = list(mu) if mu else None
                
                # 1. Update Left Patterns (Rho)
                if next_rho:
                    for idx, pat in enumerate(left_patterns):
                        # Ensure limit check is robust
                        if is_therapist and (j, time_t) in pat['elements']:
                            next_rho[idx] += 1
                            print(f"      [SP DEBUG] Worker {j} t={time_t}: Left Pattern #{idx} Hit! Count {next_rho[idx]}/{pat['limit']}")
                            if next_rho[idx] > pat['limit']:
                                print(f"      [SP DEBUG] Worker {j} t={time_t}: PRUNED (Left Pattern #{idx} limit exceeded: Count {next_rho[idx]} > Limit {pat['limit']}, Pattern Elements: {sorted(list(pat['elements']))})")
                                return None, None, True # Pruned (Limit Reached)
                
                # 2. Update Right Patterns (Mu)
                if next_mu:
                    for idx, pat in enumerate(right_patterns):
                        # Skip if not compatible (always 0)
                        if idx not in compatible_right_indices:
                            continue
                            
                        current_mode = mu[idx]
                        t_start = pat['first_element'][1] if pat['first_element'] else 9999
                        
                        in_pattern = is_therapist and ((j, time_t) in pat['elements_set'])
                        
                        if time_t < t_start:
                            continue # Wait for start
                            
                        if time_t == t_start:
                            # Start of pattern window
                            if in_pattern:
                                next_mu[idx] = 1 # Enter Cover Mode
                                remaining_elements = sorted([e for e in pat['elements'] if e[1] > time_t], key=lambda x: x[1])
                                print(f"      [SP DEBUG] Worker {j} t={time_t}: Right Pattern #{idx} ENTER COVER MODE (Remaining: {len(remaining_elements)} elements: {remaining_elements})")
                            else:
                                next_mu[idx] = 0 # Enter Exclude Mode
                                print(f"      [SP DEBUG] Worker {j} t={time_t}: Right Pattern #{idx} ENTER EXCLUDE MODE (Reason: Pattern starts here, but current action is NOT in_pattern (is_therapist={is_therapist}, in_pattern={in_pattern}))")
                                
                        elif time_t > t_start:
                            if current_mode == 0: # Exclude Mode
                                if in_pattern:
                                    print(f"      [SP DEBUG] Worker {j} t={time_t}: PRUNED (Right Pattern #{idx} forbidden pick in Exclude Mode)")
                                    return None, None, True # Pruned (Forbidden pick)
                            elif current_mode == 1: # Cover Mode
                                # If pattern has requirement at this time, we MUST pick it
                                elt_at_t = (j, time_t) in pat['elements_set']
                                if elt_at_t and not in_pattern:
                                    print(f"      [SP DEBUG] Worker {j} t={time_t}: PRUNED (Right Pattern #{idx} broken chain in Cover Mode)")
                                    return None, None, True # Pruned (Broken chain)
                
                return (tuple(next_rho) if next_rho else None, 
                        tuple(next_mu) if next_mu else None, 
                        False)

            # DP Loop until just before Tau
            pruned_count_total = 0

            # DP Loop until just before Tau
            pruned_count_total = 0

            t_dp_start = time.time()
            for t in range(r_k + 1, tau):
                next_states = {}
                pruned_count_this_period = 0

                # Iterate over all buckets
                for bucket_key, bucket_list in current_states.items():
                    # Extract components from bucket key
                    # Structure depends on what constraints are active
                    # Key structure: (ai, hist, [zeta], [mu])
                    key_idx = 0
                    ai_count = bucket_key[key_idx]; key_idx += 1
                    hist = bucket_key[key_idx]; key_idx += 1
                    
                    zeta = None
                    if use_branch_constraints:
                        zeta = bucket_key[key_idx]; key_idx += 1
                        
                    mu = None
                    if use_sp_branching and right_patterns:
                        mu = bucket_key[key_idx]; key_idx += 1
                    
                    # Iterate over all states in the bucket
                    for item in bucket_list:
                        # Extract components from item
                        # Structure: (cost, prog, [rho], path)
                        if use_sp_branching and left_patterns:
                             cost, prog, rho, path = item
                        else:
                             cost, prog, path = item
                             rho = None

                        # BOUND PRUNING: Check if state is promising
                        if use_bound_pruning:
                            lb = compute_lower_bound(cost, r_k, tau, gamma_k, obj_mode,
                                                     worker=j, pi_dict=pi_dict, max_time=max_time)
                            if lb >= 0:
                                pruned_count_this_period += 1
                                pruned_count_total += 1
                                pruning_stats['lb'] += 1
                                continue  # State is pruned!

                        # Feasibility Check
                        remaining_steps = tau - t + 1
                        if not is_timeout_scenario:
                            if obj_mode > 0.5:
                                if prog + remaining_steps * 1.0 < s_k - epsilon:
                                    continue

                        # --- TRANSITIONS ---
                        

                        # Helper for SP State Update (Defined outside loop)
                        # ...



                        # A: Therapist
                        if check_strict_feasibility(hist, 1, ms, min_ms):
                            new_rho, new_mu, pruned = get_next_sp_state(rho, mu, True, t)
                            
                            if not pruned:
                                cost_ther = cost - pi_dict.get((j, t), 0)
                                prog_ther = prog + 1.0
                                new_hist_ther = (hist + (1,))
                                if len(new_hist_ther) > ms - 1: 
                                    new_hist_ther = new_hist_ther[-(ms - 1):]
                                
                                # Update deviation vector ζ_t if branch constraints are active
                                new_zeta_ther = zeta
                                if use_branch_constraints:
                                    new_zeta_list = list(zeta)
                                    for cut_idx, cut in enumerate(forbidden_schedules):
                                        if new_zeta_list[cut_idx] == 0:  # Not yet deviated
                                            forbidden_val = cut.get((j, t), 0)
                                            if forbidden_val != 1:
                                                new_zeta_list[cut_idx] = 1
                                    new_zeta_ther = tuple(new_zeta_list)

                                add_state_to_buckets(next_states, cost_ther, prog_ther, ai_count, new_hist_ther, 
                                                   path + [1], recipient_id, pruning_stats, dominance_mode, 
                                                   new_zeta_ther, new_rho, new_mu, epsilon)

                        # B: AI
                        if check_strict_feasibility(hist, 0, ms, min_ms):
                            new_rho, new_mu, pruned = get_next_sp_state(rho, mu, False, t)
                            
                            if not pruned:
                                cost_ai = cost
                                efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                                prog_ai = prog + efficiency
                                ai_count_new = ai_count + 1
                                new_hist_ai = (hist + (0,))
                                if len(new_hist_ai) > ms - 1: 
                                    new_hist_ai = new_hist_ai[-(ms - 1):]
                                
                                # Update deviation vector ζ_t if branch constraints are active
                                new_zeta_ai = zeta
                                if use_branch_constraints:
                                    new_zeta_list = list(zeta)
                                    for cut_idx, cut in enumerate(forbidden_schedules):
                                        if new_zeta_list[cut_idx] == 0:  # Not yet deviated
                                            forbidden_val = cut.get((j, t), None)
                                            if forbidden_val is not None and forbidden_val != 0:
                                                new_zeta_list[cut_idx] = 1
                                    new_zeta_ai = tuple(new_zeta_list)

                                add_state_to_buckets(next_states, cost_ai, prog_ai, ai_count_new, new_hist_ai, 
                                                   path + [0], recipient_id, pruning_stats, dominance_mode, 
                                                   new_zeta_ai, new_rho, new_mu, epsilon)

                current_states = next_states
                if not current_states: 
                    break
            timers['state_expansion'] += time.time() - t_dp_start

            # Final Step (Transition to Tau)
            # Final Step (Transition to Tau)
            t_final_start = time.time()
            for bucket_key, bucket_list in current_states.items():
                # Extract components from bucket key
                # Structure: (ai, hist, [zeta], [mu])
                key_idx = 0
                ai_count = bucket_key[key_idx]; key_idx += 1
                hist = bucket_key[key_idx]; key_idx += 1
                
                zeta = None
                if use_branch_constraints:
                    zeta = bucket_key[key_idx]; key_idx += 1
                    
                mu = None
                if use_sp_branching and right_patterns:
                    mu = bucket_key[key_idx]; key_idx += 1
                    
                for item in bucket_list:
                    if use_sp_branching and left_patterns:
                         cost, prog, rho, path = item
                    else:
                         cost, prog, path = item
                         rho = None
                    
                    # Collect possible end steps for this state
                    possible_moves = []

                    # Option 1: End with Therapist (1) - Standard
                    if check_strict_feasibility(hist, 1, ms, min_ms):
                        possible_moves.append(1)

                    # Option 2: End with App (0) - ONLY if Timeout
                    if is_timeout_scenario:
                        if check_strict_feasibility(hist, 0, ms, min_ms):
                            possible_moves.append(0)

                    for move in possible_moves:
                        # 1. Update SP Branching State (Check Pruning)
                        final_rho, final_mu, sp_pruned = get_next_sp_state(rho, mu, move == 1, tau)
                        if sp_pruned:
                            continue

                        # Calculate values based on Move type
                        if move == 1:
                            final_cost_accum = cost - pi_dict.get((j, tau), 0)
                            final_prog = prog + 1.0
                            final_ai_count = ai_count
                        else:  # move == 0
                            final_cost_accum = cost
                            efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                            final_prog = prog + efficiency
                            final_ai_count = ai_count + 1

                        final_path = path + [move]
                        condition_met = (final_prog >= s_k - epsilon)
                        
                        # 2. Update MP Branching State (Zeta)
                        final_zeta = zeta
                        if use_branch_constraints:
                            final_zeta_list = list(zeta)
                            for cut_idx, cut in enumerate(forbidden_schedules):
                                if final_zeta_list[cut_idx] == 0:  # Not yet deviated
                                    forbidden_val = cut.get((j, tau), None)
                                    if forbidden_val is not None and forbidden_val != move:
                                        final_zeta_list[cut_idx] = 1  # Deviated!
                            final_zeta = tuple(final_zeta_list)
                        
                            # TERMINAL FEASIBILITY CHECK: All deviation vector entries must equal 1
                            if not all(z == 1 for z in final_zeta):
                                continue

                        is_focus_patient = (obj_mode > 0.5)

                        if is_focus_patient:
                            # Focus-Patient (E_k=1):
                            is_valid_end = condition_met
                        else:
                            # Post-Patient (E_k=0):
                            is_valid_end = condition_met or is_timeout_scenario

                        if is_valid_end:
                            duration = tau - r_k + 1
                            reduced_cost = (obj_mode * duration) + final_cost_accum - gamma_k
                            
                            # 3. Apply Duals for SP Right Branching (Rewards for covering patterns)
                            if final_mu:
                                for idx, mode in enumerate(final_mu):
                                    if mode == 1: # Cover Mode
                                        pat = right_patterns[idx]
                                        
                                        # Check if pattern is FULLY covered by this schedule ending at tau
                                        # If pattern has any element with t > tau, it is NOT fully covered.
                                        remaining_elements = [e for e in pat['elements'] if e[1] > tau]
                                        
                                        if not remaining_elements:
                                            reduced_cost -= pat['dual']
                                            if pat['dual'] != 0:
                                                print(f"      [SP DEBUG] Worker {j} END: Applied Dual {pat['dual']} for Right Pattern #{idx} (Covered)")
                                        else:
                                            # Print unconditionally for debugging purposes
                                            print(f"      [SP DEBUG] Worker {j} END t={tau}: Right Pattern #{idx} Incomplete! (Remaining: {remaining_elements}) - Dual NOT applied (Dual val: {pat['dual']})")

                            col_candidate = {
                                'k': recipient_id,
                                'worker': j,
                                'start': r_k,
                                'end': tau,
                                'duration': duration,
                                'reduced_cost': reduced_cost,
                                'final_progress': final_prog,
                                'x_vector': generate_full_column_vector(j, final_path, r_k, tau, max_time, len(workers)),
                                'path_pattern': final_path
                            }

                            if reduced_cost < -epsilon:
                                best_columns.append(col_candidate)
            timers['final_step'] += time.time() - t_final_start
            
            # Debug Output: Bound Pruning Statistics
            if pruned_count_total > 0:
                logger.print(f"    Worker {j}, tau={tau}: Pruned {pruned_count_total} states by Lower Bound")

    timers['total'] = time.time() - t_start_total
    
    # Print timing summary
    if timers['total'] > 0.001:  # Only print if measurable time was spent
        print(f"  [Timing] Recipient {recipient_id}: Init: {timers['init']:.4f}s | Expansion: {timers['state_expansion']:.4f}s | Final: {timers['final_step']:.4f}s | Total: {timers['total']:.4f}s")

    # Sort columns by reduced cost (ascending, most negative first)
    best_columns.sort(key=lambda x: x['reduced_cost'])
    
    # Keep only unique columns (based on path pattern and worker)
    unique_columns = []
    seen_patterns = set()
    
    for col in best_columns:
        pattern_key = (col['worker'], tuple(col['path_pattern']))
        if pattern_key not in seen_patterns:
            unique_columns.append(col)
            seen_patterns.add(pattern_key)
            
        if len(unique_columns) >= max_columns:
            break
            
    best_columns = unique_columns

    # Print negative reduced costs if found
    #for col in best_columns:
        #print(f"  [Labeling] Recipient {recipient_id} with gamma {gamma_k}: Negative red. cost: {col['reduced_cost']:.2f} with {col['duration']} and {col['x_vector']}")

    # Debug output for forbidden schedules and generated columns
    if use_branch_constraints and forbidden_schedules:
        logger.print(f"\n{'='*100}")
        logger.print(f"  [FORBIDDEN vs GENERATED] Recipient {recipient_id}: {len(forbidden_schedules)} No-Good Cut(s) Active")
        logger.print(f"{'='*100}")
        
        # First, show all forbidden schedules in detail
        for cut_idx, cut in enumerate(forbidden_schedules):
            logger.print(f"\n  Forbidden Schedule #{cut_idx+1}:")
            # Group by worker
            workers_in_cut = {}
            for (worker, t_val), val in sorted(cut.items()):
                if worker not in workers_in_cut:
                    workers_in_cut[worker] = []
                workers_in_cut[worker].append((t_val, val))
            
            for worker in sorted(workers_in_cut.keys()):
                times_vals = workers_in_cut[worker]
                pattern_str = "".join([str(int(v)) for _, v in sorted(times_vals)])
                time_range = f"[{min(t for t, _ in times_vals)}→{max(t for t, _ in times_vals)}]"
                logger.print(f"    Worker {worker:2d} {time_range:8s}: {pattern_str}")
        
        # Now, show all generated columns
        if best_columns:
            logger.print(f"\n  Generated Columns ({len(best_columns)} found):")
            for col_idx, col in enumerate(best_columns):
                logger.print(f"\n    Column #{col_idx+1}:")
                logger.print(f"      Worker: {col['worker']:2d}, Period: [{col['start']}→{col['end']}], Reduced Cost: {col['reduced_cost']:.4f}")
                pattern_str = "".join([str(int(v)) for v in col['path_pattern']])
                logger.print(f"      Pattern: {pattern_str}")
                
                # Compare with each forbidden schedule
                for cut_idx, cut in enumerate(forbidden_schedules):
                    comparison = []
                    deviations = []
                    all_times = list(range(col['start'], col['end'] + 1))
                    
                    for t_step_idx, t in enumerate(all_times):
                        forbidden_val = cut.get((col['worker'], t), None)
                        generated_val = col['path_pattern'][t_step_idx]
                        
                        if forbidden_val is not None:
                            if forbidden_val == generated_val:
                                comparison.append(f"t{t}:✓")  # Match
                            else:
                                comparison.append(f"t{t}:✗({int(forbidden_val)}→{int(generated_val)})")  # Deviation
                                deviations.append(f"t{t}")
                    
                    if comparison:  # Only show if there's overlap
                        match_status = "IDENTICAL - REJECTED!" if not deviations else f"DEVIATES (at {', '.join(deviations)})"
                        logger.print(f"      vs Cut #{cut_idx+1}: {match_status}")
                        if len(comparison) <= 10:
                            logger.print(f"        {' '.join(comparison)}")
                        else:
                            logger.print(f"        {' '.join(comparison[:5])} ... {' '.join(comparison[-5:])}")
        else:
            logger.print(f"\n  ⚠️  No columns generated for this recipient!")
        
        logger.print(f"\n{'='*100}\n")


    
    return best_columns


def solve_pricing_multi_phase(
    recipient_id, r_k, s_k, gamma_k, obj_mode, pi_dict,
    workers, max_time, ms, min_ms, theta_lookup,
    use_bound_pruning=True, dominance_mode='bucket',
    branch_constraints=None, branching_variant='mp',
    max_columns=10, use_pure_dp_optimization=True,
    use_heuristic_pricing=True, heuristic_max_labels=20,
    use_relaxed_history=False, allow_gaps=False
):
    """
    Multi-phase pricing with heuristic + exact fallback.
    
    Phase 1: Try heuristic pricing (fast, aggressive pruning)
    Phase 2: Fall back to exact pricing if heuristic fails
    
    Args:
        Same as solve_pricing_for_recipient, plus:
        use_heuristic_pricing: Enable heuristic phase
        heuristic_max_labels: Max labels per bucket in heuristic mode
    
    Returns:
        List of best columns (guaranteed to be optimal via fallback)
    """
    
    if use_heuristic_pricing:
        # Phase 1: Heuristic pricing with aggressive pruning
        heuristic_cols = solve_pricing_for_recipient(
            recipient_id, r_k, s_k, gamma_k, obj_mode, pi_dict,
            workers, max_time, ms, min_ms, theta_lookup,
            use_bound_pruning=use_bound_pruning,
            dominance_mode=dominance_mode,
            branch_constraints=branch_constraints,
            branching_variant=branching_variant,
            max_columns=max_columns,
            use_pure_dp_optimization=use_pure_dp_optimization,
            # Heuristic settings:
            use_heuristic_pricing=True,
            max_labels_per_bucket=heuristic_max_labels,
            stop_at_first_negative=True,
            use_relaxed_history=use_relaxed_history,
            allow_gaps=allow_gaps
        )
        
        # Check if heuristic found columns with negative reduced cost
        if heuristic_cols and any(col['reduced_cost'] < -1e-6 for col in heuristic_cols):
            print(f"  ✓ [Heuristic] Found {len(heuristic_cols)} columns for profile {recipient_id}")
            return heuristic_cols
        
        # Heuristic failed, fall back to exact
        print(f"  → [Fallback] Running exact pricing for profile {recipient_id}")
    
    # Phase 2: Exact pricing (no limits, guaranteed optimal)
    return solve_pricing_for_recipient(
        recipient_id, r_k, s_k, gamma_k, obj_mode, pi_dict,
        workers, max_time, ms, min_ms, theta_lookup,
        use_bound_pruning=use_bound_pruning,
        dominance_mode=dominance_mode,
        branch_constraints=branch_constraints,
        branching_variant=branching_variant,
        max_columns=max_columns,
        use_pure_dp_optimization=use_pure_dp_optimization,
        # Exact settings (no heuristic):
        use_heuristic_pricing=False,
        max_labels_per_bucket=None,
        stop_at_first_negative=False,
        use_relaxed_history=use_relaxed_history,
        allow_gaps=allow_gaps
    )



# --- 3b. Wrapper for Branch-and-Price Integration ---

def solve_pricing_for_profile_bnp(
    profile,
    duals_pi,
    duals_gamma,
    r_k,
    s_k,
    obj_multiplier,
    workers,
    max_time,
    theta_lookup,
    MS,
    MIN_MS,
    col_id,
    branching_constraints=None,
    max_columns=10,
    use_pure_dp_optimization=True,
    use_heuristic_pricing=False,
    heuristic_max_labels=20,
    use_relaxed_history=False,
    use_numba_labeling=False,
    allow_gaps=False,
    stop_at_first_negative=False  # Early termination for child nodes
):
    """
    Wrapper function for Branch-and-Price integration.
    
    Solves the pricing problem for a single profile using the labeling algorithm
    and returns results in the format expected by branch_and_price.py.
    
    Note: Right-pattern branching duals are extracted directly from branching_constraints
    and applied per-pattern when covered (not via a single aggregated duals_delta).
    
    Args:
        profile: Profile index (k)
        duals_pi: Dict of (worker, time) -> dual value
        duals_gamma: Float, dual value for this profile's convexity constraint
        r_k: Release time for this profile
        s_k: Service requirement
        obj_multiplier: Objective mode (0 or 1)
        workers: List of available workers
        max_time: Time horizon
        theta_lookup: Lookup table for learning curve
        MS: Milestone window size
        MIN_MS: Minimum therapist sessions in window
        col_id: Next column ID to use
        branching_constraints: Optional list of branching constraints
        max_columns: Maximum number of columns to return
        use_pure_dp_optimization: Enable Pure DP fast path when no constraints (Option 4)
        use_heuristic_pricing: Enable heuristic pricing mode
        heuristic_max_labels: Max labels per bucket in heuristic mode
    
    Returns:
        list: List of best columns in subproblem format, or empty list if no improving column found
    """
    # Set global variables (temporary solution until we refactor to pass all parameters)
    global MAX_TIME, WORKERS, pi, gamma
    
    MAX_TIME = max_time
    WORKERS = workers
    
    # Convert duals_pi to global pi format expected by labeling algorithm
    pi = duals_pi
    # gamma for this profile
    gamma_k = duals_gamma
    


    # Gaps Fallback Logic for Numba
    if allow_gaps and max_time > 31 and use_numba_labeling:
        # Numba implementation uses 2-bit encoding which limits duration to 31 days (64 bits / 2)
        # Fallback to Python implementation for longer horizons
        print(f"[NUMBA WARNING] Profile {profile}: Gaps allowed with T={max_time} > 31. Falling back to Python implementation.")
        use_numba_labeling = False

    # Call multi-phase pricing (heuristic + exact fallback)
    best_columns = solve_pricing_multi_phase(
        recipient_id=profile,
        r_k=r_k,
        s_k=s_k,
        gamma_k=gamma_k,
        obj_mode=obj_multiplier,
        pi_dict=pi,
        workers=workers,
        max_time=max_time,
        ms=MS,
        min_ms=MIN_MS,
        theta_lookup=theta_lookup,
        use_bound_pruning=False,  # Disable for now
        dominance_mode='bucket',
        branch_constraints=branching_constraints,
        branching_variant='mp',
        max_columns=max_columns,
        use_pure_dp_optimization=use_pure_dp_optimization,
        use_heuristic_pricing=use_heuristic_pricing,
        heuristic_max_labels=heuristic_max_labels,
        use_relaxed_history=use_relaxed_history,
        allow_gaps=allow_gaps
    ) if not (use_numba_labeling and HAS_NUMBA) else []
    
    # === NUMBA OPTIMIZATION ===
    if use_numba_labeling and HAS_NUMBA:
        # Prepare data for Numba
        # Convert pi_dict to matrix
        max_worker_id = max(workers) if workers else 0
        pi_matrix = np.zeros((max_worker_id + 1, max_time + 1), dtype=np.float64)
        for (w, t), val in duals_pi.items():
            if w <= max_worker_id and t <= max_time:
                pi_matrix[w, t] = val

        # Worker Dominance Pre-Elimination (using Numba-optimized version)
        workers_array_full = np.array(workers, dtype=np.int64)
        workers_arr = label_numba.compute_candidate_workers_numba(
            workers_array_full, r_k, max_time, pi_matrix
        )  # Returns numpy array directly
        
        # Log eliminated workers
        eliminated = set(workers) - set(workers_arr)
        if eliminated:
            logger.info(f"[NUMBA] Profile {profile}: Workers {list(workers_arr)} (eliminated {list(eliminated)})")
        
        theta_arr = np.array(theta_lookup, dtype=np.float64)
        
        # === FLATTEN BRANCHING CONSTRAINTS FOR NUMBA ===
        has_sp_fixing = False
        has_nogood_cuts = False
        has_left_patterns = False
        
        # Default empty arrays (will be passed even if not used)
        forbidden_mask = np.zeros((max_worker_id + 1, max_time + 1), dtype=np.bool_)
        required_mask = np.zeros((max_worker_id + 1, max_time + 1), dtype=np.bool_)
        # O(1) MP BRANCHING: Use 2D boolean map instead of 3D nogood_patterns
        mp_forbidden_map = np.zeros((max_worker_id + 1, max_time + 1), dtype=np.bool_)
        has_mp_branching = False
        
        left_pattern_elements = np.full((1, 1), -1, dtype=np.int64)
        left_pattern_limits = np.zeros(1, dtype=np.int64)
        num_left_patterns = 0
        
        # Right Pattern Arrays (B.3.2)
        right_pattern_elements = np.full((1, 1), -1, dtype=np.int64)
        right_pattern_starts = np.zeros(1, dtype=np.int64)
        right_pattern_duals = np.zeros(1, dtype=np.float64)
        right_pattern_counts = np.zeros(1, dtype=np.int64)
        num_right_patterns = 0
        has_right_patterns = False
        
        if branching_constraints:
            from branching_constraints import MPVariableBranching, SPPatternBranching
            
            # B.2: MP No-Good Cuts → O(1) FORBIDDEN MAP
            mp_nogood_constraints = [c for c in branching_constraints 
                                     if isinstance(c, MPVariableBranching) and c.profile == profile 
                                     and c.direction == 'left' and c.original_schedule]
            if mp_nogood_constraints:
                has_mp_branching = True
                # Combine ALL no-good cuts into single 2D forbidden map
                for c in mp_nogood_constraints:
                    for key, val in c.original_schedule.items():
                        if len(key) >= 3 and val > 0.5:
                            j, t = key[1], key[2]
                            if j <= max_worker_id and t <= max_time:
                                mp_forbidden_map[j, t] = True  # O(1) mark as forbidden
            
            # B.3: SP Left Pattern Branching
            sp_left_patterns = [c for c in branching_constraints 
                                if isinstance(c, SPPatternBranching) and c.profile == profile 
                                and c.direction == 'left']
            if sp_left_patterns:
                # SIZE=1 OPTIMIZATION: Use forbidden_mask for single-element patterns
                for c in sp_left_patterns:
                    if len(c.pattern) == 1:
                        (j, t) = next(iter(c.pattern))
                        if j <= max_worker_id and t <= max_time:
                            forbidden_mask[j, t] = True
                            has_sp_fixing = True
                
                # Filter to only patterns with size > 1 for left_pattern_elements
                sp_left_patterns_multi = [c for c in sp_left_patterns if len(c.pattern) > 1]
                
                if sp_left_patterns_multi:
                    has_left_patterns = True
                    num_left_patterns = len(sp_left_patterns_multi)
                    max_pattern_size = max(len(c.pattern) for c in sp_left_patterns_multi)
                    left_pattern_elements = np.full((num_left_patterns, max_pattern_size), -1, dtype=np.int64)
                    left_pattern_limits = np.zeros(num_left_patterns, dtype=np.int64)
                    
                    for pat_idx, c in enumerate(sp_left_patterns_multi):
                        left_pattern_limits[pat_idx] = len(c.pattern) - 1  # limit = |P| - 1
                        for elem_idx, (j, t) in enumerate(sorted(c.pattern)):
                            # Encode (j, t) as j*1000000 + t
                            left_pattern_elements[pat_idx, elem_idx] = j * 1000000 + t

            # B.3.2: SP Right Pattern Branching
            sp_right_patterns = [c for c in branching_constraints 
                                 if isinstance(c, SPPatternBranching) and c.profile == profile 
                                 and c.direction == 'right']
            if sp_right_patterns:
                # SIZE=1 OPTIMIZATION: Use required_mask for single-element patterns
                for c in sp_right_patterns:
                    if len(c.pattern) == 1:
                        (j, t) = next(iter(c.pattern))
                        if j <= max_worker_id and t <= max_time:
                            required_mask[j, t] = True
                            has_sp_fixing = True
                
                # Filter to only patterns with size > 1 for right_pattern_elements
                sp_right_patterns_multi = [c for c in sp_right_patterns if len(c.pattern) > 1]
                
                if sp_right_patterns_multi:
                    has_right_patterns = True
                    num_right_patterns = len(sp_right_patterns_multi)
                    max_elems = max(len(c.pattern) for c in sp_right_patterns_multi)
                    right_pattern_elements = np.full((num_right_patterns, max_elems), -1, dtype=np.int64)
                    right_pattern_starts = np.zeros(num_right_patterns, dtype=np.int64)
                    right_pattern_duals = np.zeros(num_right_patterns, dtype=np.float64)
                    right_pattern_counts = np.zeros(num_right_patterns, dtype=np.int64)
                    
                    # Collect info for printing
                    pattern_info_list = []
                    
                    for pat_idx, c in enumerate(sp_right_patterns_multi):
                        # Sort pattern elements by time
                        sorted_pat = sorted(c.pattern, key=lambda x: x[1])
                        right_pattern_counts[pat_idx] = len(sorted_pat)
                        
                        # Extract dual from master_constraint if available
                        # Right branch (>=) should have dual >= 0
                        # If column covers pattern, RC is reduced by this dual
                        dual_val = 0.0
                        if hasattr(c, 'master_constraint') and c.master_constraint is not None:
                            try:
                                dual_val = c.master_constraint.Pi
                            except Exception as e:
                                logger.warning(f"  [Numba] Could not extract dual for Right Pattern {pat_idx}: {e}")
                                dual_val = 0.0
                        right_pattern_duals[pat_idx] = dual_val
                        
                        # Store pattern info for printing
                        pattern_str = "{" + ", ".join(f"({j},{t})" for j, t in sorted_pat) + "}"
                        pattern_info_list.append((pat_idx, pattern_str, dual_val))
                        
                        if sorted_pat:
                             _, first_t = sorted_pat[0]
                             right_pattern_starts[pat_idx] = first_t
                        
                        for elem_idx, (j, t) in enumerate(sorted_pat):
                             right_pattern_elements[pat_idx, elem_idx] = j * 1000000 + t
                    
                    # Print all Right Pattern Duals for this profile
                    print(f"\n  [NUMBA RIGHT PATTERNS] Profile {profile}: {num_right_patterns} Right Pattern(s)")
                    for pat_idx, pattern_str, dual_val in pattern_info_list:
                        dual_status = f"δ = {dual_val:+.6f}" if dual_val != 0.0 else "δ = 0.0 (inactive)"
                        print(f"    Pattern #{pat_idx}: {pattern_str} → {dual_status} and original dual {dual_val}")
        
        # Decide which Numba function to call
        use_branching_numba = has_sp_fixing or has_mp_branching or has_left_patterns or has_right_patterns
        
        if use_branching_numba:
            # Call extended Numba function with branching support
            raw_cols = label_numba.run_with_branching_constraints_numba(
                int(r_k), float(s_k), float(duals_gamma), float(obj_multiplier),
                pi_matrix, workers_arr, int(max_time),
                int(MS), int(MIN_MS), theta_arr, 1e-6,
                # SP Variable Fixing
                forbidden_mask, required_mask, has_sp_fixing,
                # MP No-Good Cuts → O(1) FORBIDDEN MAP
                mp_forbidden_map, has_mp_branching,
                # SP Left Patterns
                left_pattern_elements, left_pattern_limits, num_left_patterns, has_left_patterns,
                # SP Right Patterns
                right_pattern_elements, right_pattern_starts, right_pattern_duals, right_pattern_counts, num_right_patterns, has_right_patterns,
                stop_at_first_negative,
                allow_gaps
            )
        else:
            # Call original fast path (no constraints)
            raw_cols = label_numba.run_fast_path_numba(
                int(r_k), float(s_k), float(duals_gamma), float(obj_multiplier),
                pi_matrix, workers_arr, int(max_time), 
                int(MS), int(MIN_MS), theta_arr, 1e-6,
                stop_at_first_negative,
                allow_gaps
            )

        
        # Convert to best_columns format
        for col_tuple in raw_cols:
            w_id = int(col_tuple[0])
            rc = float(col_tuple[1])
            start_t = int(col_tuple[2])
            end_t = int(col_tuple[3])
            path_mask = int(col_tuple[4])
            final_prog = float(col_tuple[5])
            
            # Reconstruct path pattern
            # Duration is end_t - start_t
            # Wait, end_t in Numba was 'tau', and loop was range(r_k+1, tau).
            # Duration was calculated as tau - r_k.
            # So start=r_k, end=tau-1?
            # Original code: start=r_k.
            # If path has length D. t goes from r_k to r_k + D - 1.
            # My Numba code shifts: (1 << (t - r_k)).
            # If t = r_k, shift 0.
            # If t = tau - 1, shift tau - 1 - r_k.
            # So bitmask covers bits 0 to duration-1.
            
            duration = end_t - start_t + 1
            path_pattern = []
            for i in range(duration):
                # 2-bit decoding: 0=AI, 1=Therapist, 2=Gap
                val = (path_mask >> (2 * i)) & 3
                path_pattern.append(val)
            
            best_columns.append({
                'worker': w_id,
                'reduced_cost': rc,
                'start': start_t,
                'end': end_t, # Inclusive end
                'path_pattern': path_pattern,
                'final_progress': final_prog,
                'duration': duration,
                'x_vector': generate_full_column_vector(w_id, path_pattern, start_t, end_t, max_time, len(workers))
            })
        
        # Sort by RC, then Duration, then Start
        best_columns.sort(key=lambda x: (x['reduced_cost'], x['duration'], x['start'], x['worker']))
        if len(best_columns) > max_columns:
            best_columns = best_columns[:max_columns]
    
    if not best_columns:
        return []
    
    formatted_columns = []
    current_col_id = col_id
    
    for col in best_columns:
        # Convert to subproblem format expected by branch_and_price.py
        worker = col['worker']
        start = col['start']
        end = col['end']
        path_pattern = col['path_pattern']
    
        # Build schedules_x: {(profile, worker, time, col_id): value}
        # IMPORTANT: Initialize ALL possible (profile, worker, time, col_id) combinations with 0
        # to match the format of Gurobi-generated subproblems
        schedules_x = {}
        
        # Initialize all combinations with 0 for ALL workers and times
        for w in workers:
            for t in range(1, max_time + 1):
                schedules_x[(profile, w, t, current_col_id)] = 0.0
        
        # Now set the actual values from the path_pattern
        # path=1 → Human session (x=1), path=0 → AI session (x=0), path=2 → Gap (x=0)
        for t_idx, val in enumerate(path_pattern):
            current_time = start + t_idx
            # val==1 means Human session -> x=1, val==0 means AI -> x=0, val==2 means Gap -> x=0
            x_val = 1.0 if val == 1 else 0.0
            schedules_x[(profile, worker, current_time, current_col_id)] = x_val
        
        # Build schedules_y: {(profile, time, col_id): value}
        # path=0 → AI session (y=1), path=1 → Human session (y=0), path=2 → Gap (y=0)
        schedules_y = {}
        for t_idx, val in enumerate(path_pattern):
            current_time = start + t_idx
            # val==0 means AI session -> y=1, val==1 means Human -> y=0, val==2 means Gap -> y=0
            y_val = 1.0 if val == 0 else 0.0
            schedules_y[(profile, current_time, current_col_id)] = y_val
        
        # Build schedules_los: {(profile, col_id): los_value}
        duration = col['duration']
        schedules_los = {(profile, current_col_id): duration}
        
        # Build x_list (list of all x values for this column)
        x_list = list(schedules_x.values())
        
        # Build los_list (single element list with duration)
        los_list = [duration]
        
        formatted_columns.append({
            'reduced_cost': col['reduced_cost'],
            'schedules_x': schedules_x,
            'schedules_y': schedules_y,
            'schedules_los': schedules_los,
            'x_list': x_list,
            'los_list': los_list,
            'path_pattern': path_pattern,
            'worker': worker,
            'start': start,
            'end': end,
            'final_progress': col['final_progress'],
            'x_vector': col['x_vector']
        })
        
        # Increment col_id for next column? 
        # Note: The caller (branch_and_price) manages col_ids. 
        # Here we just use the passed col_id as a placeholder or base.
        # Ideally, branch_and_price should re-assign IDs when adding to master.
        # But for now, let's keep using the passed col_id to avoid breaking structure,
        # assuming branch_and_price handles the actual ID assignment or we just return data.
        # Actually, schedules_x keys use col_id. If we return multiple, they need unique IDs locally?
        # Let's increment it locally to be safe, though BnP might overwrite it.
        current_col_id += 1
        
    return formatted_columns


def run_labeling_algorithm(recipients_r, recipients_s, gamma_dict, obj_mode_dict, 
                           pi_dict, workers, max_time, ms, min_ms, theta_lookup,
                           print_worker_selection=True, use_bound_pruning=True, 
                           dominance_mode='bucket', branch_constraints=None, 
                           branching_variant='mp', n_workers=None, allow_gaps=False):
    """
    Global Labeling Algorithm Function.
    
    Labeling Algorithm for Column Generation (Pricing Problem Solver)
    
    Args:
        recipients_r: Release times {recipient_id: r_k}
        recipients_s: Service targets {recipient_id: s_k}
        gamma_dict: Dual values gamma {recipient_id: gamma_k}
        obj_mode_dict: Objective multipliers {recipient_id: multiplier}
        pi_dict: Dual values pi {(worker_id, time): pi_jt}
        workers: List of worker IDs
        max_time: Planning horizon
        ms: Rolling window size
        min_ms: Minimum human services in window
        theta_lookup: AI efficiency lookup table
        print_worker_selection: Print worker dominance info per recipient
        use_bound_pruning: Enable/Disable lower bound pruning
        dominance_mode: 'bucket' (default) or 'global' dominance strategy
        branch_constraints: Optional branch constraints dictionary
        branching_variant: Branching strategy ('mp' or 'sp')
        n_workers: Number of parallel workers (None = sequential)
        
    Returns:
        List of best columns (can be multiple per recipient if alternatives exist)
    """
    t0 = time.time()
    results = []
    
    # Pruning Statistics
    pruning_stats = {
        'lb': 0,
        'dominance': 0,
        'printed_dominance': {}
    }
    
    # === PARALLEL OR SEQUENTIAL PROCESSING ===
    
    if n_workers is not None and n_workers > 1:
        # --- PARALLEL PROCESSING ---
        from multiprocessing import Pool
        
        logger.print(f"\n[PARALLEL MODE] Using {n_workers} workers for {len(recipients_r)} recipients")
        
        # Prepare arguments for each recipient
        recipient_args = []
        for k in recipients_r:
            gamma_val = gamma_dict.get(k, 0.0)
            multiplier = obj_mode_dict.get(k, 1)
            recipient_args.append((
                k, recipients_r[k], recipients_s[k], 
                gamma_val, multiplier, pi_dict, workers, 
                max_time, ms, min_ms, theta_lookup,
                use_bound_pruning, dominance_mode, 
                use_bound_pruning, dominance_mode, 
                branch_constraints, branching_variant,
                10, True, False, None, False, False, # Defaults for intermediate args: max_cols, pure_dp, heuristic, max_labels, stop_first, relaxed
                allow_gaps
            ))
        
        # Execute in parallel
        with Pool(processes=n_workers) as pool:
            all_cols = pool.starmap(solve_pricing_for_recipient, recipient_args)
        
        # Merge results
        recipient_keys = list(recipients_r.keys())
        for k, cols in zip(recipient_keys, all_cols):
            if cols:
                results.extend(cols)
    
    else:
        # --- SEQUENTIAL PROCESSING ---
        for k in recipients_r:
            gamma_val = gamma_dict.get(k, 0.0)
            multiplier = obj_mode_dict.get(k, 1)
            
            cols = solve_pricing_for_recipient(k, recipients_r[k], recipients_s[k], 
                                              gamma_val, multiplier, pi_dict, workers, 
                                              max_time, ms, min_ms, theta_lookup,
                                              use_bound_pruning=use_bound_pruning, 
                                              dominance_mode=dominance_mode, 
                                              branch_constraints=branch_constraints, 
                                              branching_variant=branching_variant,
                                              allow_gaps=allow_gaps)
            
            if cols:
                results.extend(cols)
    
    runtime = time.time() - t0
    
    logger.print(f"\nRuntime: {runtime:.4f}s")
    logger.print(f"Pruning Stats: Lower Bound = {pruning_stats['lb']}, State Dominance = {pruning_stats['dominance']}")
    logger.print(f"\n--- Final Results ({len(results)} optimal schedules) ---")
    
    return results
