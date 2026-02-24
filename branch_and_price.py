import sys
import math
import time
from bnp_node import BnPNode
import gurobipy as gu
import copy
import logging
import os
import threading


# ============================================================================
# GLOBAL WORKER FUNCTION FOR PARALLEL PRICING
# ============================================================================
def _parallel_pricing_worker(profile, node_data, duals_pi, duals_gamma, branching_duals, 
                             threshold, cg_solver_data, next_col_id):
    """
    Global worker function for parallel pricing (can be pickled by multiprocessing).
    
    This function is called by multiprocessing.Pool to solve pricing problems in parallel.
    It must be a top-level function (not a class method) to be picklable.
    
    Args:
        profile: Profile index
        node_data: Dictionary with node information
        duals_pi: Dual variables for capacity constraints
        duals_gamma: Dual variable for this profile
        branching_duals: Dict of branching constraint duals
        threshold: Reduced cost threshold
        cg_solver_data: Dict with CG solver data needed for pricing
        next_col_id: Next column ID to use for this profile
    
    Returns:
        Tuple of (profile, col_data_list) where col_data_list is list of columns or None
    """
    from label import solve_pricing_for_profile_bnp
    from bnp_node import BnPNode
    
    # Reconstruct minimal node object for pricing
    logger = logging.getLogger(__name__)
    node = BnPNode(
        node_id=node_data['node_id'],
        parent_id=node_data['parent_id'],
        depth=node_data['depth'],
        path=node_data['path']
    )
    node.branching_constraints = node_data['branching_constraints']
    node.column_pool = {}
    
    # Extract profile-specific data
    r_k = cg_solver_data['Entry_agg'][profile]
    s_k = cg_solver_data['Req_agg'][profile]
    obj_multiplier = cg_solver_data['E_dict'].get(profile, 1)
    
    # Extract pattern duals for this profile (Right-Pattern duals are >= 0, reduce RC when covered)
    pattern_dual_sum = 0.0
    if branching_duals:
        # 1. Apply duals to constraint objects (CRITICAL FIX)
        if node.branching_constraints:
            for constraint in node.branching_constraints:
                # Check if it's an SP pattern constraint
                if hasattr(constraint, 'pattern') and hasattr(constraint, 'level'):
                     # Reconstruct stable key
                    pattern_str = str(sorted(list(constraint.pattern)))
                    key = (profile, 'pattern', pattern_str, constraint.level)
                    
                    if key in branching_duals:
                        constraint.dual_var = branching_duals[key]
                        # logger.debug(f"    [Worker] Applied dual {constraint.dual_var:.6f} to pattern {pattern_str}")

        # 2. Calculate sum for A Priori Bound (only for THIS profile)
        pattern_duals = {k: v for k, v in branching_duals.items() 
                        if k[0] == profile and len(k) == 4 and k[1] == 'pattern'}
        if pattern_duals:
            # Right-branch pattern duals are >= 0 and reduce RC when pattern is covered
            # For conservative bound, assume all patterns COULD be covered
            pattern_dual_sum = sum(pattern_duals.values())
    
    # --- EARLY PRUNING CHECK (A Priori Bound) ---
    if cg_solver_data.get('use_apriori_pruning', True):
        # Logic: Reduced Cost = (Cost - Sum(Pi*Use)) - Gamma - Pattern_Duals
        # Since Pi <= 0 and Use >= 0, the term -Sum(Pi*Use) is always >= 0.
        # Therefore, Min Possible RC >= Min Possible Cost - Gamma - Pattern_Duals.
        # For a valid column, Cost = Duration * obj_multiplier.
        # Min Duration is approx s_k (if efficiency=1.0).
        # Pattern duals (right-branch, >= 0) reduce RC when pattern is covered.
        # For conservative bound, assume patterns could be covered.
        
        lb_pruning = (s_k * obj_multiplier) - duals_gamma - pattern_dual_sum
        # Safety margin
        if lb_pruning > -1e-9:
            # Prune this profile! No negative reduced cost possible.
            logger.debug(f"    [Pruning] Profile {profile} skipped (A Priori Bound: {lb_pruning:.4f} > 0)")
            return (profile, [])

    # Call labeling algorithm
    col_data_list = solve_pricing_for_profile_bnp(
        profile=profile,
        duals_pi=duals_pi,
        duals_gamma=duals_gamma,
        r_k=r_k,
        s_k=s_k,
        obj_multiplier=obj_multiplier,
        workers=cg_solver_data['workers'],
        max_time=cg_solver_data['max_time'],
        theta_lookup=cg_solver_data['theta_lookup'],
        MS=cg_solver_data['MS'],
        MIN_MS=cg_solver_data['MIN_MS'],
        col_id=next_col_id,
        branching_constraints=node.branching_constraints,
        max_columns=cg_solver_data['max_columns_per_iter'],
        use_pure_dp_optimization=cg_solver_data.get('use_pure_dp_optimization', True),
        use_heuristic_pricing=cg_solver_data.get('use_heuristic_pricing', False),
        heuristic_max_labels=cg_solver_data.get('heuristic_max_labels', 20),
        use_relaxed_history=cg_solver_data.get('use_relaxed_history', False),
        use_numba_labeling=cg_solver_data.get('use_numba_labeling', False),
        allow_gaps=cg_solver_data.get('allow_gaps', False),
        stop_at_first_negative=(node.depth > 0)  # Early termination in child nodes only
    )
    
    return (profile, col_data_list)


class BranchAndPrice:
    """
    Branch-and-Price Algorithm

    Attributes:
        nodes: Dictionary of all nodes {node_id -> BnPNode}
        node_counter: Counter for unique node IDs
        open_nodes: List of open nodes (later: queue for DFS)
        incumbent: Best found IP solution (upper bound)
        best_lp_bound: Best LP bound of all nodes (lower bound)
        gap: Optimality gap
        cg_solver: Reference to Column Generation solver
    """

    def __init__(self, cg_solver, branching_strategy='mp', search_strategy='dfs', verbose=True,
                 ip_heuristic_frequency=10, early_incumbent_iteration=0, save_lps=True, label_dict=None):
        """
        Initialize Branch-and-Price with existing CG solver.

        Args:
            cg_solver: ColumnGeneration object (already initialized with setup())
            branching_strategy: 'mp' for MP variable branching, 'sp' for SP variable branching
            search_strategy: 'dfs' for Depth-First-Search or 'bfs' for Best-Fit-Search.
            verbose: If True, print detailed progress
            ip_heuristic_frequency: Solve RMP as IP every N nodes (0 to disable)
            early_incumbent_iteration: CG iteration to compute initial incumbent
                                      - If 0 or None: solve final RMP as IP (after CG converges)
                                      - If > 0: solve RMP as IP after this iteration,
                                               then continue CG without further IP solves
            save_lps: If True, save LP and SOL files during solving
            label_dict: Dictionary of labels configurations
        """
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Create sols/Integral directory
        os.makedirs('sols/Integral', exist_ok=True)

        # Node management
        self.nodes = {}  # {node_id -> BnPNode}
        self.node_counter = 0
        self.open_nodes = []  # For DFS: list of IDs. For BFS: list of (bound, ID)

        # Output control
        self.verbose = verbose
        self.save_lps = save_lps
        self.debug_mode = label_dict['debug_mode']  if label_dict is not None else True # DEBUG MODE: If True, re-raise exceptions

        # Global bounds
        self.incumbent = float('inf')  # Best IP solution (upper bound)
        self.incumbent_solution = None
        self.incumbent_lambdas = None
        self.best_lp_bound = float('inf')  # Best LP bound (lower bound)
        self.gap = float('inf')

        # Reference to CG solver
        self.cg_solver = cg_solver

        # Search and Branching Configuration
        self.branching_strategy = branching_strategy
        self.search_strategy = search_strategy
        self.use_labeling = label_dict['use_labeling'] if label_dict is not None else False # Use labeling algorithm for pricing
        self.max_columns_per_iter = label_dict['max_columns_per_iter'] if label_dict is not None else 5
        
        # Parallelization Configuration
        self.use_parallel_pricing = label_dict['use_parallel_pricing'] and label_dict['use_labeling'] if label_dict is not None else False # Only works with labeling
        self.n_pricing_workers = label_dict['n_pricing_workers']if label_dict is not None else 1
        self.use_apriori_pruning = label_dict['use_apriori_pruning'] if label_dict is not None else True # Default: True
        self.use_pure_dp_optimization = label_dict['use_pure_dp_optimization'] if label_dict is not None else True # Option 4: Pure DP fast path
        self.use_persistent_pool = label_dict['use_persistent_pool'] if label_dict is not None else True
        self.use_heuristic_pricing = label_dict['use_heuristic_pricing'] if label_dict is not None else False
        self.heuristic_max_labels = label_dict['heuristic_max_labels'] if label_dict is not None else 20
        self.use_relaxed_history = label_dict['use_relaxed_history'] if label_dict is not None else True
        self.use_numba_labeling = label_dict['use_numba_labeling'] if label_dict is not None else False
        self.allow_gaps = label_dict.get('allow_gaps', False) if label_dict is not None else False
        self.use_label_recycling = label_dict.get('use_label_recycling', False) if label_dict is not None else False
        
        # Parallel Tree Exploration Configuration
        self.use_parallel_tree = label_dict.get('use_parallel_tree', False) if label_dict is not None else False
        self.n_tree_workers = label_dict.get('n_tree_workers', 1) if label_dict is not None else 1
        
        if self.use_parallel_tree and self.n_tree_workers > 1:
            self.logger.info(f"✓ Parallel tree exploration enabled with {self.n_tree_workers} worker threads")
        
        # Create persistent multiprocessing pool
        self.pricing_pool = None
        if self.use_parallel_pricing and self.use_persistent_pool:
            from multiprocessing import Pool
            self.pricing_pool = Pool(processes=self.n_pricing_workers)
            self.logger.info(f"✓ Created persistent pricing pool with {self.n_pricing_workers} workers")
        
        if self.use_parallel_pricing and not self.use_labeling:
            self.logger.warning("⚠️  Parallel pricing requires use_labeling=True. Disabling parallelization.")
            self.use_parallel_pricing = False

        # IP Heuristic
        self.ip_heuristic_frequency = ip_heuristic_frequency

        # Early incumbent computation
        self.early_incumbent_iteration = early_incumbent_iteration if early_incumbent_iteration else 0
        self.incumbent_computed_early = False

        # Start solutions
        self.start_x = self.cg_solver.start_x
        self.start_los = self.cg_solver.start_los

        # Statistics
        self.stats = {
            'nodes_explored': 0,
            'nodes_fathomed': 0,
            'nodes_branched': 0,
            'total_cg_iterations': 0,
            'total_time': 0,
            'incumbent_updates': 0,
            'ip_solves': 0,
            'node_processing_order': [],
            'bfs_decision_log': [],
            'tree_complete': False,  # True if all nodes were processed (no open nodes remain)
            # Timing statistics
            'time_in_mp': 0.0,
            'time_in_sp': 0.0,
            'time_in_ip_heuristic': 0.0,
            'time_in_root': 0.0,
            'time_in_branching': 0.0,
            'time_to_first_incumbent': None,
            # Pattern size counts (SP branching only)
            'pattern_size_counts': {},  # {size: count} e.g. {1: 5, 2: 3} means 5 patterns of size 1, 3 of size 2
            # Additional metrics
            'max_tree_depth': 0,  # Maximum depth reached in the search tree
            'nodes_pruned': 0,  # Number of nodes pruned by bound or infeasibility
            'integer_solutions_found': 0  # Number of times an integer solution was found
        }

        # Timing
        self.start_time = None

        self.logger.info("\n" + "=" * 100)
        self.logger.info(" BRANCH-AND-PRICE INITIALIZED ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"CG Solver ready with {len(self.cg_solver.P_Join)} patients")
        self.logger.info(f"Branching strategy: {self.branching_strategy.upper()}")
        self.logger.info(f"Search strategy: {'Depth-First (DFS)' if self.search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
        if self.early_incumbent_iteration > 0:
            self.logger.info(f"Incumbent strategy: Compute after CG iteration {self.early_incumbent_iteration}")
        else:
            self.logger.info(f"Incumbent strategy: Compute after CG convergence (final RMP as IP)")

        self.logger.info("=" * 100 + "\n")

        # Initialize LP-folder
        os.makedirs("results", exist_ok=True)
        os.makedirs("LPs/MP/LPs", exist_ok=True)
        os.makedirs("LPs/MP/Root", exist_ok=True)
        os.makedirs("LPs/MP/SOLs", exist_ok=True)
        os.makedirs("LPs/SPs/pricing", exist_ok=True)

        # Logger init
        self.logger.info("=" * 100)
        self.logger.info(" BRANCH-AND-PRICE INITIALIZED ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"CG Solver ready with {len(self.cg_solver.P_Join)} patients")
        self.logger.info(f"Branching strategy: {self.branching_strategy.upper()}")

    def _print(self, *args, **kwargs):
        """Print only if verbose mode is enabled."""
        if self.verbose:
            print(*args, **kwargs)

    def _print_always(self, *args, **kwargs):
        """Always print (for critical messages and final results)."""
        print(*args, **kwargs)

    def _early_incumbent_callback(self, iteration, cg_solver):
        """
        Callback executed after each CG iteration at root node.

        If early_incumbent_iteration is set and we reach that iteration,
        solve the RMP as IP to get the initial incumbent.

        Args:
            iteration: Current CG iteration number
            cg_solver: Reference to ColumnGeneration instance
        """
        # Check if we should compute early incumbent
        if self.early_incumbent_iteration == 0:
            return  # No early incumbent requested

        if iteration != self.early_incumbent_iteration:
            return  # Not the right iteration yet

        if self.incumbent_computed_early:
            return  # Already computed

        # Compute incumbent at this iteration
        self.logger.info(f"\n{'─' * 100}")
        self.logger.info(f" COMPUTING EARLY INCUMBENT (after CG iteration {iteration}) ".center(100, "─"))
        self.logger.info(f"{'─' * 100}")
        self.logger.info(f"Solving RMP as IP with columns generated so far...")
        self.logger.info(f"CG will continue afterwards until convergence.\n")

        success = self._solve_rmp_as_ip(cg_solver.master, context="Early Incumbent", node_id=0)

        if success:
            self.incumbent_computed_early = True
            self.logger.info(f"\n[OK] Early incumbent computed successfully!")
            self.logger.info(f"   Incumbent: {self.incumbent:.6f}")
            self.logger.info(f"   CG will continue to convergence...\n")
        else:
            self.logger.warning(f"\n⚠️  Early incumbent computation unsuccessful")
            self.logger.warning(f"   CG will continue and we'll try again after convergence.\n")

        self.logger.info(f"{'─' * 100}\n")

    def create_root_node(self):
        """
        Create root node with initial columns from CG heuristic.

        Returns:
            BnPNode: The root node (ID=0, depth=0)
        """
        node = BnPNode(node_id=0, depth=0)

        # Transfer initial columns from CG solver
        for (p, old_col_id) in self.cg_solver.global_solutions.get('x', {}).keys():
            col_id = old_col_id

            # Extract schedules_x from global_solutions
            x_solution = self.cg_solver.global_solutions['x'][(p, old_col_id)]

            # Remap keys: (p, agent, period, old_iteration) -> (p, agent, period, col_id)
            schedules_x = {}
            for (p_key, agent, period, old_iter), value in x_solution.items():
                # Use col_id instead of old_iter
                schedules_x[(p_key, agent, period, col_id)] = value

            # Extract schedules_los
            schedules_los = {}
            if (p, old_col_id) in self.cg_solver.global_solutions.get('LOS', {}):
                los_solution = self.cg_solver.global_solutions['LOS'][(p, old_col_id)]
                # Remap keys: (p, old_iteration) -> (p, col_id)
                for (p_key, old_iter), value in los_solution.items():
                    schedules_los[(p_key, col_id)] = value

            # Create column data with CORRECT field names
            col_data = {
                'index': p,
                'column_id': col_id,
                'schedules_x': schedules_x,
                'schedules_los': schedules_los,
                'x_list': list(schedules_x.values()),
                'los_list': list(schedules_los.values()),
            }

            # Add other solution data
            for var_name in ['y', 'z', 'S', 'l']:
                if (p, old_col_id) in self.cg_solver.global_solutions.get(var_name, {}):
                    col_data[f'{var_name}_data'] = self.cg_solver.global_solutions[var_name][(p, old_col_id)]

            if 'App' in self.cg_solver.global_solutions:
                if (p, old_col_id) in self.cg_solver.global_solutions['App']:
                    col_data['App_data'] = self.cg_solver.global_solutions['App'][(p, old_col_id)]

            # Store in node pool with the correct column_id
            node.column_pool[(p, col_id)] = col_data

            # Validate key format consistency
            for key in schedules_x.keys():
                if len(key) != 4:
                    self.logger.warning(f"⚠️  Invalid schedules_x key format in root node: {key} (expected 4 components)")
                elif key[3] != col_id:
                    self.logger.warning(f"⚠️  Inconsistent col_id in root schedules_x key: {key} (expected col_id={col_id})")

        # Store node
        self.nodes[0] = node
        if self.search_strategy == 'dfs':
            self.open_nodes.append(0)
        else:  # bfs
            # Root node has no initial bound yet, will be computed in solve_root_node
            self.open_nodes.append((float('inf'), 0))

        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" ROOT NODE CREATED ".center(100, "="))
        self.logger.info(f"{'=' * 100}")
        self.logger.info(f"Root node initialized with {len(node.column_pool)} initial columns")
        self.logger.info(f"Columns distribution:")

        # Show column distribution by profiles
        col_per_profile = {}
        for (p, _) in node.column_pool.keys():
            col_per_profile[p] = col_per_profile.get(p, 0) + 1

        for p in sorted(col_per_profile.keys())[:5]:
            self.logger.info(f"  Profile {p}: {col_per_profile[p]} columns")
        if len(col_per_profile) > 5:
            self.logger.info(f"  ... and {len(col_per_profile) - 5} more profiles")

        # Debug: Show sample column structure
        if node.column_pool:
            sample_key = list(node.column_pool.keys())[0]
            sample_col = node.column_pool[sample_key]
            self.logger.info(f"\n  Sample column {sample_key}:")
            self.logger.info(f"    schedules_x: {len(sample_col.get('schedules_x', {}))} entries")
            self.logger.info(f"    schedules_los: {len(sample_col.get('schedules_los', {}))} entries")
            self.logger.info(f"    x_list: {len(sample_col.get('x_list', []))} values")
            self.logger.info(f"    los_list: {len(sample_col.get('los_list', []))} values")

            # Show first key format
            if sample_col.get('schedules_x'):
                first_key = list(sample_col['schedules_x'].keys())[0]
                self.logger.info(f"    First schedules_x key: {first_key}")

        self.logger.info(f"{'=' * 100}\n")

        return node

    def solve_root_node(self):
        """
        Solve root node via Column Generation.

        Depending on early_incumbent_iteration:
        - If 0: Solve RMP as IP after CG converges
        - If > 0: RMP is solved as IP during CG (via callback),
                  then solve final LP after convergence

        Returns:
            tuple: (lp_bound, is_integral, most_frac_info)
        """
        self.logger.info("\n" + "=" * 100)
        self.logger.info(" SOLVING ROOT NODE ".center(100, "="))
        self.logger.info("=" * 100 + "\n")

        # Log processing of root node
        self.stats['node_processing_order'].append(0)
        
        root_node = self.nodes[0]

        # Choose solving method based on use_labeling flag
        if self.use_labeling:
            # ===== USE BRANCH-AND-PRICE CG (with labeling) =====
            self.logger.info("[Root] Using Branch-and-Price CG with LABELING ALGORITHM\n")
            
            # Start timing root node
            root_start_time = time.time()
            
            # Solve root node using the same method as child nodes
            # This ensures labeling algorithm is used consistently
            lp_bound, is_integral, most_frac_info, lambda_list_root = self.solve_node_with_cg(
                root_node, 
                max_cg_iterations=100
            )
            
            # Record root node time
            self.stats['time_in_root'] = time.time() - root_start_time
            
            # Compute incumbent after CG converges (if not using early incumbent)
            if self.early_incumbent_iteration == 0:
                self.logger.info("\n" + "=" * 100)
                self.logger.info(" COMPUTING FINAL INCUMBENT ".center(100, "="))
                self.logger.info("=" * 100)
                self.logger.info("Column Generation converged. All columns generated.")
                self.logger.info("Solving final Root Master Problem as IP to get initial upper bound...\n")
                self._compute_final_incumbent()
        
        else:
            # ===== USE ORIGINAL CG SOLVER (with Gurobi) =====
            self.logger.info("[Root] Using original CG solver with GUROBI\n")
            
            # Setup callback if early incumbent is requested
            if self.early_incumbent_iteration > 0:
                self.cg_solver.callback_after_iteration = self._early_incumbent_callback
                self.logger.info(
                    f"[Root] Early incumbent will be computed after CG iteration {self.early_incumbent_iteration}\n")
            else:
                self.cg_solver.callback_after_iteration = None
                self.logger.info(f"[Root] Incumbent will be computed after CG convergence\n")

            # Solve with Column Generation
            self.cg_solver.solve_cg()

            # After CG converges: Check if we need to compute incumbent
            if not self.incumbent_computed_early:
                self.logger.info("\n" + "=" * 100)
                self.logger.info(" COMPUTING FINAL INCUMBENT ".center(100, "="))
                self.logger.info("=" * 100)
                self.logger.info("Column Generation converged. All columns generated.")
                self.logger.info("Solving final Root Master Problem as IP to get initial upper bound...\n")

                self._compute_final_incumbent()
            else:
                self.logger.info("\n" + "=" * 100)
                self.logger.info(" USING EARLY INCUMBENT ".center(100, "="))
                self.logger.info("=" * 100)
                self.logger.info(f"Incumbent was already computed at iteration {self.early_incumbent_iteration}")
                self.logger.info(f"Current incumbent: {self.incumbent:.6f}")
                self.logger.info("=" * 100 + "\n")

            # Final LP relaxation check
            self.logger.debug("\n[Root] Final LP relaxation check...")
            self.cg_solver.master.solRelModel()

            lambda_list_root = {key: {'value': var.X, 'obj': var.Obj} for key, var in self.cg_solver.master.lmbda.items() if var.X > 1e-6}

            is_integral, lp_bound, most_frac_info = self.cg_solver.master.check_fractionality()

            # Update root node
            root_node.lp_bound = lp_bound
            root_node.is_integral = is_integral
            root_node.most_fractional_var = most_frac_info

            # Update root node's column pool
            self._update_node_column_pool(root_node)

        # Update node status (common for both paths)
        if is_integral:
            root_node.status = 'integral'
            root_node.fathom_reason = 'integral'
            self.stats['integer_solutions_found'] += 1
            self.logger.info(f"\n[OK] ROOT NODE IS INTEGRAL (LP)!")
        else:
            root_node.status = 'solved'
            self.logger.warning(f"\n⚠️  ROOT NODE IS FRACTIONAL (LP) - Status: SOLVED")

        self.logger.info(f"   LP Bound: {lp_bound:.6f}")
        self.logger.info(f"   Incumbent: {self.incumbent:.6f}" if self.incumbent < float('inf') else "   Incumbent: None")

        # Update global bounds
        self.best_lp_bound = lp_bound
        self.update_gap()

        # Save initial Root-LP/SOL
        if self.save_lps and not self.use_labeling:
            # Only save if using CG solver (has master attribute)
            self.cg_solver.master.Model.write('LPs/MP/LPs/master_node_root.lp')
            self.cg_solver.master.Model.write('LPs/MP/SOLs/master_node_root.sol')

        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" ROOT NODE SOLVED ".center(100, "="))
        self.logger.info(f"{'=' * 100}\n")

        # Exit after root node for debugging
        # self.logger.info("🛑 STOPPING AFTER ROOT NODE (sys.exit() for debugging)")
        # sys.exit()

        return lp_bound, is_integral, most_frac_info, lambda_list_root


    def _compute_final_incumbent(self):
        """
        Compute incumbent from final RMP after CG convergence.

        This is the default behavior (when early_incumbent_iteration = 0).
        """
        master = self.cg_solver.master
        success = self._solve_rmp_as_ip(master, context="Final Incumbent", node_id=0)

        if success:
            self.logger.info(f"\n{'=' * 100}")
            self.logger.info("[OK] FINAL INCUMBENT FOUND ".center(100, "="))
            self.logger.info(f"{'=' * 100}")
            self.logger.info(f"IP Objective:     {self.incumbent:.6f}")
            self.logger.info(f"LP Bound (root):  {master.Model.objBound:.6f}" if hasattr(master.Model, 'objBound') else "")
            self.logger.info(f"Gap:              {self.gap:.4%}")
            self.logger.info(f"{'=' * 100}\n")
        else:
            self.logger.warning(f"\n⚠️  Could not compute final incumbent")

    def _solve_rmp_as_ip(self, master, context="IP Solve", node_id=None):
        """
        Solve the Restricted Master Problem as Integer Program.

        This is a helper method used by both early and final incumbent computation.

        Args:
            master: MasterProblem_d instance
            context: String describing the context (for logging)

        Returns:
            bool: True if successful and incumbent was updated
        """
        self.logger.info("=" * 100)
        self.logger.info(f"{context}: Solving RMP as Integer Program...".center(100))
        self.logger.info("=" * 100 + "\n")
        self.stats['ip_solves'] += 1

        try:
            # Save current variable types
            original_vtypes = {}
            for var in master.lmbda.values():
                original_vtypes[var.VarName] = var.VType
                var.VType = gu.GRB.INTEGER

            # Solve as IP
            master.Model.Params.OutputFlag = 0
            master.Model.Params.TimeLimit = 300
            master.Model.update()

            self.logger.info(f"[{context}] Starting optimization...")
            master.Model.optimize()

            success = False
            result_obj = float('inf')

            # Check solution status
            if master.Model.status == gu.GRB.OPTIMAL:
                ip_obj = master.Model.objVal

                # Update incumbent if better
                if ip_obj < self.incumbent:
                    self.incumbent = ip_obj
                    self.incumbent_solution = master.finalDicts(
                        self.cg_solver.global_solutions,
                        self.cg_solver.app_data, None
                    )
                    lambda_assignments = {}
                    for (p, a), var in master.lmbda.items():
                        if var.X > 1e-6:
                            lambda_assignments[(p, a)] = {'value': var.X, 'obj': var.Obj}

                    self.incumbent_lambdas = lambda_assignments

                    self.stats['incumbent_updates'] += 1
                    self.update_gap()

                    self.logger.info(f"\n[OK] New incumbent found: {self.incumbent:.6f}")
                    self.logger.info(f"   Gap: {self.gap:.4%}\n")
                    
                    # Save solution file
                    if master.Model.SolCount > 0:
                        sol_name = f"Node_{node_id if node_id is not None else 'root'}.sol"
                        sol_path = os.path.join('sols/Integral', sol_name)
                        master.Model.write(sol_path)
                        self.logger.info(f"   [IP Save] Solution saved to {sol_path}")

                    success = True
                    result_obj = ip_obj
                else:
                    self.logger.warning(f"\n⚠️  IP solution not better than current incumbent")
                    self.logger.info(f"   IP Objective:      {ip_obj:.6f}")
                    self.logger.info(f"   Current Incumbent: {self.incumbent:.6f}\n")
                    success = False
                    result_obj = ip_obj

            elif master.Model.status == gu.GRB.TIME_LIMIT:
                self.logger.warning(f"\n⚠️  IP solve hit time limit")
                if master.Model.SolCount > 0:
                    ip_obj = master.Model.objVal
                    self.logger.info(f"   Best found solution: {ip_obj:.6f}")
                    if ip_obj < self.incumbent:
                        self.incumbent = ip_obj
                        self.incumbent_solution = master.finalDicts(
                            self.cg_solver.global_solutions,
                            self.cg_solver.app_data, None
                        )
                        lambda_assignments = {}
                        for (p, a), var in master.lmbda.items():
                            if var.X > 1e-6:
                                lambda_assignments[(p, a)] = {'value': float(round(var.X)), 'obj': var.Obj}
                        self.incumbent_lambdas = lambda_assignments

                        self.stats['incumbent_updates'] += 1
                        self.update_gap()
                        self.logger.info(f"   Updated incumbent: {self.incumbent:.6f}\n")
                        
                        # Save solution file
                        if master.Model.SolCount > 0:
                            sol_name = f"Node_{node_id if node_id is not None else 'root'}.sol"
                            sol_path = os.path.join('sols/Integral', sol_name)
                            master.Model.write(sol_path)
                            self.logger.info(f"   [IP Save] Solution (Time Limit) saved to {sol_path}")
                        success = True
                        result_obj = ip_obj
                    else:
                        success = False
                        result_obj = ip_obj
                else:
                    self.logger.info(f"   No feasible solution found within time limit\n")
                    success = False
                    result_obj = float('inf')

            else:
                self.logger.error(f"❌ IP solve unsuccessful (status={master.Model.status})")
                success = False
                result_obj = float('inf')

            # Restore continuous relaxation
            for var in master.lmbda.values():
                var.VType = original_vtypes[var.VarName]

            master.Model.Params.OutputFlag = 0
            master.Model.Params.TimeLimit = float('inf')
            master.Model.update()

            return success

        except Exception as e:
            self.logger.error(f"❌ Error during {context}: {e}\n")

            # Restore original variable types
            for var in master.lmbda.values():
                if var.VarName in original_vtypes:
                    var.VType = original_vtypes[var.VarName]

            master.Model.Params.OutputFlag = 0
            master.Model.update()
            
            # DEBUG MODE: Re-raise exception instead of returning False
            if self.debug_mode:
                self.logger.error("🐛 DEBUG_MODE enabled - re-raising exception!")
                raise

            return False

    def should_fathom(self, node, lambdas_dict):
        """
        Determine if a node should be fathomed.

        A node is fathomed if:
        1. Its LP solution is integral (we found an IP solution)
        2. Its LP is infeasible
        3. Its LP bound >= incumbent (no better solution possible)

        Args:
            node: BnPNode to check

        Returns:
            bool: True if node should be fathomed
        """
        # Check 1: Integral solution
        if node.is_integral:
            node.status = 'fathomed'
            node.fathom_reason = 'integral'

            # Update incumbent if this is better
            if node.lp_bound < self.incumbent:
                self.logger.info(f"\n[OK] Node {node.node_id} found improving integral solution!")
                self.logger.info(f"   Previous incumbent: {self.incumbent:.6f}")
                self.logger.info(f"   New incumbent:      {node.lp_bound:.6f}")

                self.incumbent = node.lp_bound
                self.incumbent_solution = self.cg_solver.master.finalDicts(
                    self.cg_solver.global_solutions,
                    self.cg_solver.app_data, lambdas_dict
                )

                self.incumbent_lambdas = {k: v for k, v in lambdas_dict.items() if v['value'] > 1e-6}
                self.stats['incumbent_updates'] += 1
                
                # Update time to first incumbent if not yet set
                if self.stats.get('time_to_first_incumbent') is None:
                    self.stats['time_to_first_incumbent'] = time.time() - self.start_time
                    
                self.update_gap()

                self.logger.info(f"   New gap: {self.gap:.4%}\n")

            return True

        # Check 2: Infeasible
        if node.lp_bound == float('inf'):
            node.status = 'fathomed'
            node.fathom_reason = 'infeasible'
            self.logger.info(f"   Node {node.node_id} fathomed: LP infeasible")
            return True

        # Check 3: Bound worse than incumbent
        # Integer Pruning (Objective is always integer)
        # If ceil(LB) >= UB, we can prune because no integer solution can be strictly better than UB.
        if math.ceil(node.lp_bound - 1e-5) >= self.incumbent:
            node.status = 'fathomed'
            node.fathom_reason = 'bound'
            self.logger.info(f"   Node {node.node_id} fathomed by bound (Integer): "
                        f"ceil({node.lp_bound:.6f}) >= UB={self.incumbent:.6f}")
            return True

        # Check 4:
        if node.status == 'fathomed':
            self._check_and_fathom_parents(node.node_id)
            return True

        # Node cannot be fathomed
        return False

    def update_gap(self):
        """
        Calculate optimality gap: (UB - LB) / UB.
        """
        if self.incumbent < float('inf') and self.best_lp_bound < float('inf'):
            if abs(self.incumbent) > 1e-10:
                self.gap = (self.incumbent - self.best_lp_bound) / abs(self.incumbent)
            else:
                self.gap = abs(self.incumbent - self.best_lp_bound)
        else:
            self.gap = float('inf')

    def solve(self, time_limit=3600, max_nodes=10000, cutoff=None):
        """
        Main solve method for Branch-and-Price with full tree exploration.
        """
        self.start_time = time.time()

        self.logger.info("\n" + "=" * 100)
        self.logger.info(" BRANCH-AND-PRICE SOLVE ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"Time limit: {time_limit}s")
        self.logger.info(f"Max nodes: {max_nodes}")
        if cutoff is not None:
            self.logger.info(f"Cutoff: {cutoff}")
        self.logger.info(f"Branching strategy: {self.branching_strategy.upper()}")
        self.logger.info("=" * 100 + "\n")

        if self.ip_heuristic_frequency > 0:
            self.logger.info(f"IP heuristic: Every {self.ip_heuristic_frequency} nodes")
        else:
            self.logger.info(f"IP heuristic: Disabled")

        # ========================================
        # PHASE 1: CREATE AND SOLVE ROOT NODE
        # ========================================

        root_node = self.create_root_node()
        # For 'bfs', open_nodes contains a tuple, so we extract the ID
        if self.search_strategy == 'bfs':
            self.open_nodes.pop()  # Remove placeholder

        lp_bound, is_integral, frac_info, root_lambdas = self.solve_root_node()
        self.stats['nodes_explored'] = 1

        if cutoff is not None and math.ceil(lp_bound - 1e-5) > cutoff:
            self.logger.info(f"\n[OK] Root node LP bound {lp_bound:.6f} exceeds cutoff {cutoff}. Early termination.")
            self.stats['nodes_fathomed'] = 1
            self.stats['tree_complete'] = True
            
            # Record that we were cut off
            res = self._get_results_dict()
            res['cutoff_exceeded'] = True
            
            self._finalize_and_print_results()
            self._cleanup_pricing_pool()
            return res

        # Add root node to open list with its solved bound for 'bfs'
        if self.search_strategy == 'bfs' and not is_integral:
            self.open_nodes.append((lp_bound, 0))

        # Check if root can be fathomed
        if self.should_fathom(root_node, root_lambdas):
            self.logger.info(f"[OK] Root node fathomed: {root_node.fathom_reason}")
            self.logger.info(f"   Solution is optimal!\n")
            self.stats['nodes_fathomed'] = 1
            if 'by bound' in root_node.fathom_reason.lower() or 'infeasible' in root_node.fathom_reason.lower():
                self.stats['nodes_pruned'] += 1  # Track pruned nodes
            self.stats['tree_complete'] = True  # Root was fathomed, so tree is complete
            self.stats['max_tree_depth'] = max(self.stats['max_tree_depth'], 0)  # Root is depth 0
            # When tree is complete (only root node), lower bound equals incumbent
            if self.incumbent < float('inf'):
                self.best_lp_bound = self.incumbent
                self.update_gap()  # Gap should be 0 now
            if self.open_nodes:
                self.open_nodes.pop()
            self._finalize_and_print_results()
            self._cleanup_pricing_pool()
            return self._get_results_dict()

        # Root needs branching
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" ROOT NODE REQUIRES BRANCHING ".center(100, "="))
        self.logger.info(f"{'=' * 100}\n")

        # Remove root from open nodes before branching
        if self.open_nodes:
            self.open_nodes.pop()

        # Branch on root
        self.logger.info(f"DEBUG: Root node most_fractional_var: {root_node.most_fractional_var}")
        branching_type, branching_info = self.select_branching_candidate(root_node, root_lambdas)
        #print(branching_info, branching_type)

        if not branching_type:
            self.logger.warning(f"⚠️  Could not find branching candidate despite fractional solution!")
            self._finalize_and_print_results()
            self._cleanup_pricing_pool()
            return self._get_results_dict()

        # Create child nodes
        if branching_type == 'mp':
            left_child, right_child = self.branch_on_mp_variable(root_node, branching_info)
        else:  # 'sp'
            left_child, right_child = self.branch_on_sp_pattern(root_node, branching_info)
            # Track pattern size for SP branching
            pattern_size = branching_info.get('pattern_size', 1)
            if pattern_size not in self.stats['pattern_size_counts']:
                self.stats['pattern_size_counts'][pattern_size] = 0
            self.stats['pattern_size_counts'][pattern_size] += 1
        #print(left_child, right_child)
        # Mark root as branched
        root_node.status = 'branched'
        self.stats['nodes_branched'] += 1

        # ========================================
        # PHASE 2: DISPATCH TO PARALLEL OR SEQUENTIAL TREE EXPLORATION
        # ========================================
        if self.use_parallel_tree and self.n_tree_workers > 1:
            self.logger.info(f"\n{'=' * 100}")
            self.logger.info(" PARALLEL TREE EXPLORATION ".center(100, "="))
            self.logger.info(f"{'=' * 100}")
            self.logger.info(f"Using {self.n_tree_workers} worker threads\n")
            return self._solve_parallel(time_limit, max_nodes)
        else:
            # Use sequential implementation (current code)
            return self._solve_sequential(time_limit, max_nodes)

    def _solve_sequential(self, time_limit, max_nodes):
        """
        Sequential tree exploration (original implementation).
        
        This is the current while-loop based exploration, refactored into its own method.
        """
        # ========================================
        # PHASE 2: MAIN BRANCH-AND-PRICE LOOP
        # ========================================
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" MAIN BRANCH-AND-PRICE LOOP ".center(100, "="))
        self.logger.info(f"{'=' * 100}\n")

        iteration = 0

        while self.open_nodes and iteration < max_nodes:
            iteration += 1

            # Check time limit
            elapsed = time.time() - self.start_time
            if elapsed > time_limit:
                self.logger.info(f"\n⏱️  Time limit reached: {elapsed:.2f}s > {time_limit}s")
                break

            # ========================================
            # PERIODIC IP HEURISTIC
            # ========================================
            # Run BEFORE processing the next node
            if self.ip_heuristic_frequency > 0 and iteration > 1:
                improved = self._run_ip_heuristic(iteration)

                # If incumbent improved significantly and no more open nodes, we're done
                if improved and not self.open_nodes:
                    self.logger.info("\n[OK] All nodes fathomed after IP heuristic improvement!")
                    break

            # If all nodes fathomed, terminate
            if not self.open_nodes:
                break

            # ========================================
            # SELECT NEXT NODE (BEST-FIRST SEARCH)
            # ========================================
            self.logger.info(f"\n{'─' * 100}")
            self.logger.info(f" NODE SELECTION (Iteration {iteration}) ".center(100, "─"))
            self.logger.info(f"{'─' * 100}")
            self.logger.info(f"Strategy: {self.search_strategy.upper()}")
            self.logger.info(f"Open nodes before selection: {len(self.open_nodes)}")

            if self.search_strategy == 'bfs':
                # Best-first: sort by bound (ascending) and pop the best (lowest bound)
                # Tie-breaking: if bounds are equal, select node with lower ID (was created first)
                # We sort descending and pop from the end for efficiency (O(1))
                sorted_open_nodes = sorted(self.open_nodes, key=lambda x: (round(x[0], 4), x[1]), reverse=True)

                # Log all open nodes with their bounds
                self.logger.info(f"\nAll open nodes (sorted by bound, ascending):")
                for idx, (bound, node_id) in enumerate(reversed(sorted_open_nodes[-5:])):  # Show top 5
                    node = self.nodes[node_id]
                    self.logger.info(f"  {idx+1}. Node {node_id}: bound={bound:.6f}, depth={node.depth}, path='{node.path}'")
                if len(sorted_open_nodes) > 5:
                    self.logger.info(f"  ... and {len(sorted_open_nodes)-5} more nodes")

                # Select best node (lowest bound, lowest ID for tie-breaking)
                bound, current_node_id = sorted_open_nodes.pop()
                self.open_nodes = sorted_open_nodes  # update the list

                self.logger.info(f"\n✓ SELECTED: Node {current_node_id} with LP bound {bound:.6f} (BEST)")
                self.logger.info(f"  Incumbent: {self.incumbent:.6f}")
                self.logger.info(f"  Gap to incumbent: {bound - self.incumbent:.6f}")

                # Detailed selection decision output
                print("\n" + "╔" + "═" * 98 + "╗")
                print("║" + " NODE SELECTION DECISION ".center(98) + "║")
                print("╠" + "═" * 98 + "╣")
                print(f"║ 🎯 SELECTED NODE: {current_node_id}".ljust(99) + "║")
                print(f"║    LP Bound:      {bound:.6f}".ljust(99) + "║")
                print(f"║    Depth:         {self.nodes[current_node_id].depth}".ljust(99) + "║")
                print(f"║    Path:          '{self.nodes[current_node_id].path}'".ljust(99) + "║")
                print(f"║    Status:        {self.nodes[current_node_id].status}".ljust(99) + "║")
                print("╠" + "═" * 98 + "╣")
                print(f"║ 📊 GLOBAL STATE:".ljust(99) + "║")
                print(f"║    Incumbent (UB): {self.incumbent:.6f}".ljust(99) + "║")
                print(f"║    Best LB:        {self.best_lp_bound:.6f}".ljust(99) + "║")
                print(f"║    Gap:            {self.gap:.6%}".ljust(99) + "║")
                print(f"║    Remaining open: {len(self.open_nodes)}".ljust(99) + "║")
                print("╠" + "═" * 98 + "╣")
                print(f"║ 🔍 WHY THIS NODE?".ljust(99) + "║")
                print(f"║    ➤ Lowest LP bound among all open nodes (Best-First Search)".ljust(99) + "║")
                print(f"║    ➤ Most promising for finding better solutions".ljust(99) + "║")
                # Check if there are ties
                tied_nodes = [nid for b, nid in self.open_nodes if abs(b - bound) < 1e-10]
                if tied_nodes:
                    print(f"║    ➤ Tie-breaking: Selected lowest node ID ({current_node_id}) among tied nodes".ljust(99) + "║")
                if len(sorted_open_nodes) > 0:
                    next_best_bound = sorted_open_nodes[-1][0]
                    diff = next_best_bound - bound
                    if diff > 1e-10:
                        print(f"║    ➤ Better than next best by {diff:.6f}".ljust(99) + "║")
                print("╚" + "═" * 98 + "╝\n")

                # Log the decision process
                decision_log_entry = {
                    'iteration': iteration,
                    'open_nodes_state': copy.deepcopy(sorted_open_nodes),
                    'chosen_node_id': current_node_id,
                    'chosen_node_bound': bound
                }
                self.stats['bfs_decision_log'].append(decision_log_entry)

            else:  # DFS (default)
                current_node_id = self.open_nodes.pop()
                node = self.nodes[current_node_id]
                self.logger.info(f"\n✓ SELECTED: Node {current_node_id} (DFS - most recent)")
                self.logger.info(f"  LP Bound: {node.lp_bound:.6f}")
                self.logger.info(f"  Depth: {node.depth}, Path: '{node.path}'")

            # Log the processing order for all strategies
            self.stats['node_processing_order'].append(current_node_id)

            self.logger.info(f"{'─' * 100}\n")
            current_node = self.nodes[current_node_id]

            print(f"\n{'╔' + '═' * 98 + '╗'}")
            print(f"║{f' PROCESSING NODE {current_node_id} (Iteration {iteration}) ':^98}║")
            print(f"║{f' Path: {current_node.path}, Depth: {current_node.depth} ':^98}║")
            print(f" Open Nodes: {len(self.open_nodes)}, Explored: {self.stats['nodes_explored']} ")
            print(f"║{f' Incumbent: {self.incumbent:.4f}, Best LB: {self.best_lp_bound:.4f} ':^98}║")
            print(f"{'╚' + '═' * 98 + '╝'}\n")

            # ========================================
            # NODE IS ALREADY SOLVED - RETRIEVE SOLUTION
            # ========================================
            # With the new correct implementation, ALL nodes in open_nodes have been
            # fully solved with Column Generation when they were created.
            # We just need to retrieve their stored values.

            if current_node.status != 'solved':
                self.logger.error(f"❌ ERROR: Node {current_node_id} in open_nodes but status is '{current_node.status}', not 'solved'!")
                self.logger.error(f"   This should not happen with the corrected implementation!")
                self.logger.error(f"   Attempting to solve node now as fallback...")

                try:
                    lp_bound, is_integral, most_frac_info, node_lambdas = self.solve_node_with_cg(
                        current_node, max_cg_iterations=50
                    )
                    self.stats['nodes_explored'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Error solving node {current_node_id}: {e}")
                    
                    # DEBUG MODE: Re-raise exception instead of continuing
                    if self.debug_mode:
                        self.logger.error("🐛 DEBUG_MODE enabled - re-raising exception!")
                        raise
                    
                    current_node.status = 'fathomed'
                    current_node.fathom_reason = 'error'
                    self.stats['nodes_fathomed'] += 1
                    self.stats['nodes_pruned'] += 1  # Error counts as pruning
                    continue
            else:
                # Node was already solved - retrieve stored values
                print(f'Current Node: {current_node_id} with {current_node.lp_bound} is already solved.')
                lp_bound = current_node.lp_bound
                is_integral = current_node.is_integral
                most_frac_info = current_node.most_fractional_var

                print(f"  ✓ Using pre-computed solution from node creation:")
                print(f"    LP Bound: {lp_bound:.6f}")
                print(f"    Integral: {is_integral}")

                # Retrieve lambda values from master
                # We need to rebuild master to get lambda values for branching
                print(f"\n  🔧 Rebuilding master for Node {current_node_id, current_node.node_id} to extract lambda values...")
                master = self._build_master_for_node(current_node)
                master.solRelModel()

                # Track max tree depth
                self.stats['max_tree_depth'] = max(self.stats['max_tree_depth'], current_node.depth)

                if self.save_lps:
                    master.Model.write(f"LPs/MP/SOLs/new_node_{current_node.node_id}.sol")
                    master.Model.write(f"LPs/MP/LPs/new_node_{current_node.node_id}.lp")


                if master.Model.status != 2:  # Not optimal
                    print(f"  ⚠️  WARNING: Master rebuild status = {master.Model.status} (not optimal)")
                    self.logger.warning(f"  Master rebuild for Node {current_node_id} not optimal: status={master.Model.status}")

                node_lambdas = {key: var.X for key, var in master.lmbda.items()}
                rebuilt_obj = master.Model.objVal if master.Model.status == 2 else float('inf')

                print(f"  ✓ Master rebuilt for {current_node.node_id, current_node_id}: obj={rebuilt_obj:.6f}, lambda count={len(node_lambdas)}")

                # Print active branching constraints with duals (since master is fresh)
                if current_node.branching_constraints:
                    print(f"\n   Active Branching Constraints ({len(current_node.branching_constraints)}):")
                    for idx, c in enumerate(current_node.branching_constraints, 1):

                        # Manual str representation for clear output
                        if "SPPatternBranching" in str(type(c)):
                            c_type = "SP Pattern"
                            pat_str = str(sorted(list(c.pattern)))
                            details = f"Profile {c.profile}, Pattern {pat_str}, {c.direction.upper()} (Level {c.level})"
                        elif "MPVariableBranching" in str(type(c)):
                            c_type = "MP Var"
                            details = f"Profile {c.profile}, Column {c.column}, {c.direction.upper()} (Bound {c.bound})"
                        else:
                            c_type = "Unknown"
                            details = str(c)
                        
                        print(f"   {idx}. [{c_type}] {details}")
                else:
                    print("\n   Active Branching Constraints: None (Root Node)")
                print("")

                # Check if objective differs
                if abs(rebuilt_obj - lp_bound) > 1e-4:
                    print(f"  ⚠️  WARNING: Rebuilt objective {rebuilt_obj:.6f} differs from stored bound {lp_bound:.6f}")
                    self.logger.warning(f"  Objective mismatch for Node {current_node_id}: rebuilt={rebuilt_obj:.6f}, stored={lp_bound:.6f}")

                # Check integrality again on rebuilt solution
                is_integral_check, rebuilt_bound, most_frac_info_check = master.check_fractionality()
                if is_integral_check != is_integral:
                    print(f"  ⚠️  WARNING: Integrality changed! Stored={is_integral}, Rebuilt={is_integral_check}")
                    self.logger.warning(f"  Integrality mismatch for Node {current_node_id}: stored={is_integral}, rebuilt={is_integral_check}")
                    # Use the rebuilt value
                    is_integral = is_integral_check
                    most_frac_info = most_frac_info_check
                    print(f"  → Using rebuilt integrality value: {is_integral}")
            
            # DEBUG: Log fractionality status
            self.logger.info(f"    [Debug] Node {current_node_id} Fractionality Check:")
            self.logger.info(f"      is_integral: {is_integral}")
            if most_frac_info:
                self.logger.info(f"      most_frac_info: {most_frac_info}")
            else:
                self.logger.info(f"      most_frac_info: None")

            # Update best LP bound if needed (though should have been done at creation)
            if lp_bound < self.best_lp_bound:
                self.best_lp_bound = lp_bound
                self.update_gap()

            # ========================================
            # CHECK FATHOMING
            # ========================================
            print("\n" + "╔" + "═" * 98 + "╗")
            print("║" + " FATHOMING CHECK ".center(98) + "║")
            print("╠" + "═" * 98 + "╣")
            print(f"║ Node {current_node_id}:".ljust(99) + "║")
            print(f"║   LP Bound:       {lp_bound:.6f}".ljust(99) + "║")
            print(f"║   Is Integral:    {is_integral}".ljust(99) + "║")
            print(f"║   Incumbent:      {self.incumbent:.6f}".ljust(99) + "║")
            print(f"║   Status:         {current_node.status}".ljust(99) + "║")
            print("╠" + "═" * 98 + "╣")

            should_fathom_result = self.should_fathom(current_node, node_lambdas)
            self.logger.info(f"    [Debug] should_fathom result: {should_fathom_result}")
            if should_fathom_result:
                print(f"║ ❌ DECISION: FATHOM".ljust(99) + "║")
                print(f"║    Reason: {current_node.fathom_reason}".ljust(99) + "║")
                print("╚" + "═" * 98 + "╝\n")

                self.logger.info(f"[OK] Node {current_node_id} fathomed: {current_node.fathom_reason}")
                self.stats['nodes_fathomed'] += 1
                # Track pruned nodes (by bound or infeasibility)
                if 'by bound' in current_node.fathom_reason.lower() or 'infeasible' in current_node.fathom_reason.lower():
                    self.stats['nodes_pruned'] += 1

                # Print current status
                self.logger.info(f"\n   Status after fathoming:")
                self.logger.info(f"   ├─ Best LB: {self.best_lp_bound:.6f}")
                self.logger.info(f"   ├─ Incumbent: {self.incumbent:.6f}" if self.incumbent < float(
                    'inf') else "   ├─ Incumbent: None")
                self.logger.info(f"   ├─ Gap: {self.gap:.4%}" if self.gap < float('inf') else "   ├─ Gap: ∞")
                self.logger.info(f"   └─ Open nodes: {len(self.open_nodes)}\n")

                continue

            print(f"║ [OK] DECISION: BRANCH".ljust(99) + "║")
            print(f"║    Node is fractional and not fathomable".ljust(99) + "║")
            print("╚" + "═" * 98 + "╝\n")

            # ========================================
            # NODE NOT FATHOMED → BRANCH
            # ========================================
            self.logger.warning(f"\n⚠️  Node {current_node_id} requires branching (LP is fractional)")

            # Select branching candidate
            branching_type, branching_info = self.select_branching_candidate(current_node, node_lambdas)

            if not branching_type:
                self.logger.error(f"❌ Could not find branching candidate at node {current_node_id}")
                self.logger.error(f"   Marking as fathomed (should not happen!)")
                current_node.status = 'fathomed'
                current_node.fathom_reason = 'no_branching_candidate'
                self.stats['nodes_fathomed'] += 1
                self.stats['nodes_pruned'] += 1  # No branching candidate -> prune
                continue

            # ========================================
            # BRANCHING DETAILS BOX
            # ========================================
            print("\n" + "╔" + "═" * 98 + "╗")
            print("║" + " BRANCHING DETAILS ".center(98) + "║")
            print("╠" + "═" * 98 + "╣")

            if branching_type == 'mp':
                # MP Variable Branching
                n = branching_info['profile']
                a = branching_info['column']
                lambda_value = branching_info['value']
                floor_val = branching_info['floor']
                ceil_val = branching_info['ceil']

                print(f"║ 🔀 Branching Strategy: MP VARIABLE BRANCHING".ljust(99) + "║")
                print("╠" + "═" * 98 + "╣")
                print(f"║ Branching Variable: Lambda[{n},{a}] = {lambda_value:.6f}".ljust(99) + "║")
                print(f"║   Profile (n):  {n}".ljust(99) + "║")
                print(f"║   Column (a):   {a}".ljust(99) + "║")
                print(f"║   Value:        {lambda_value:.6f} (fractional)".ljust(99) + "║")
                print("╠" + "═" * 98 + "╣")
                print(f"║ 📋 CONSTRAINTS TO BE ADDED:".ljust(99) + "║")
                print("╠" + "═" * 98 + "╣")
                print(f"║ LEFT CHILD (Node {self.node_counter + 1}):".ljust(99) + "║")
                print(f"║   Master Problem (MP):  Lambda[{n},{a}] ≤ {floor_val}".ljust(99) + "║")
                print(f"║   Subproblem (SP):      No-good cut (forbids regenerating this column)".ljust(99) + "║")
                print(f"║                         Prevents column with same schedule as column {a}".ljust(99) + "║")
                print("╠" + "═" * 98 + "╣")
                print(f"║ RIGHT CHILD (Node {self.node_counter + 2}):".ljust(99) + "║")
                print(f"║   Master Problem (MP):  Lambda[{n},{a}] ≥ {ceil_val}".ljust(99) + "║")
                print(f"║   Subproblem (SP):      No constraint (column required by MP bound)".ljust(99) + "║")

            else:  # SP Branching (Variable or Pattern)
                n = branching_info['profile']
                beta_val = branching_info['beta_value']
                floor_val = branching_info['floor']
                ceil_val = branching_info['ceil']

                if 'pattern' in branching_info:
                    # SP Pattern Branching
                    pattern = branching_info['pattern']
                    pattern_size = branching_info.get('pattern_size', len(pattern))
                    pattern_str = "{" + ", ".join(f"({j},{t})" for j, t in sorted(pattern)) + "}"
                    
                    print(f"║ 🔀 Branching Strategy: SP PATTERN BRANCHING".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ Branching Pattern: P({n}) = {pattern_str}".ljust(99) + "║")
                    print(f"║   Profile (n):  {n}".ljust(99) + "║")
                    print(f"║   Size:         {pattern_size}".ljust(99) + "║")
                    print(f"║   beta_P(k):    {beta_val:.6f} (fractional)".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ 📋 CONSTRAINTS TO BE ADDED:".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ LEFT CHILD (Node {self.node_counter + 1}):".ljust(99) + "║")
                    print(f"║   SP Constraint: sum x_kjt <= {pattern_size - 1} (No full coverage)".ljust(99) + "║")
                    print(f"║   MP Constraint: sum Lambda <= {floor_val}".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ RIGHT CHILD (Node {self.node_counter + 2}):".ljust(99) + "║")
                    print(f"║   SP Constraint: sum x_kjt = {pattern_size} * w^Full (All-or-nothing)".ljust(99) + "║")
                    print(f"║   MP Constraint: sum Lambda >= {ceil_val}".ljust(99) + "║")
                    
                else:
                    # SP Variable Branching (Legacy/Singleton)
                    j = branching_info['agent']
                    t = branching_info['period']
    
                    print(f"║ 🔀 Branching Strategy: SP VARIABLE BRANCHING".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ Branching Variable: x[{n},{j},{t}] (aggregated from columns)".ljust(99) + "║")
                    print(f"║   Profile (n):  {n}".ljust(99) + "║")
                    print(f"║   Agent (j):    {j}".ljust(99) + "║")
                    print(f"║   Period (t):   {t}".ljust(99) + "║")
                    print(f"║   Value:        {beta_val:.6f} (fractional)".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ 📋 CONSTRAINTS TO BE ADDED:".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ LEFT CHILD (Node {self.node_counter + 1}):".ljust(99) + "║")
                    print(f"║   Subproblem (SP):      x[{n},{j},{t}] = 0 (Fixed to 0)".ljust(99) + "║")
                    print(f"║   Master Problem (MP):  Σ Lambda[{n},a] * chi[{n},{j},{t},a] ≤ {floor_val}".ljust(99) + "║")
                    print(f"║                         (sum over columns where patient {n} sees agent {j} at time {t})".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ RIGHT CHILD (Node {self.node_counter + 2}):".ljust(99) + "║")
                    print(f"║   Subproblem (SP):      x[{n},{j},{t}] = 1 (Fixed to 1)".ljust(99) + "║")
                    print(f"║   Master Problem (MP):  Σ Lambda[{n},a] * chi[{n},{j},{t},a] ≥ {ceil_val}".ljust(99) + "║")
                    print(f"║                         (sum over columns where patient {n} sees agent {j} at time {t})".ljust(99) + "║")

            print("╠" + "═" * 98 + "╣")
            print(f"║ 📊 BOTH CHILDREN WILL BE SOLVED IMMEDIATELY WITH FULL CG".ljust(99) + "║")
            print("╚" + "═" * 98 + "╝\n")

            # Create child nodes (with timing)
            branching_start = time.time()
            if branching_type == 'mp':
                left_child, right_child = self.branch_on_mp_variable(current_node, branching_info)
            else:  # 'sp'
                left_child, right_child = self.branch_on_sp_pattern(current_node, branching_info)
                # Track pattern size for SP branching
                pattern_size = branching_info.get('pattern_size', 1)
                if pattern_size not in self.stats['pattern_size_counts']:
                    self.stats['pattern_size_counts'][pattern_size] = 0
                self.stats['pattern_size_counts'][pattern_size] += 1
            
            # Record branching time
            self.stats['time_in_branching'] += time.time() - branching_start

            # Mark current node as branched
            current_node.status = 'branched'
            self.stats['nodes_branched'] += 1

            self.logger.info(f"\n[OK] Created child nodes:")
            self.logger.info(f"   ├─ Left:  Node {left_child.node_id} (path: '{left_child.path}')")
            self.logger.info(f"   └─ Right: Node {right_child.node_id} (path: '{right_child.path}')")
            self.logger.info(f"\n   Open nodes queue: {self.open_nodes}")

        # ========================================
        # FINALIZATION
        # ========================================
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" BRANCH-AND-PRICE TERMINATED ".center(100, "="))
        self.logger.info(f"{'=' * 100}")

        # Determine termination reason
        if not self.open_nodes:
            self.logger.info(f"[OK] All nodes explored - Tree complete!")
            self.stats['tree_complete'] = True
            # When tree is complete, lower bound equals incumbent (all nodes fathomed)
            if self.incumbent < float('inf'):
                self.best_lp_bound = self.incumbent
                self.update_gap()  # Gap should be 0 now
        elif iteration >= max_nodes:
            self.logger.warning(f"⚠️  Node limit reached: {iteration} >= {max_nodes}")
            self.logger.warning(f"   {len(self.open_nodes)} nodes remain open")
        else:
            self.logger.info(f"⏱️  Time limit reached")
            self.logger.info(f"   {len(self.open_nodes)} nodes remain open")

        self._finalize_and_print_results()
        self._cleanup_pricing_pool()
        return self._get_results_dict()

    def _solve_parallel(self, time_limit, max_nodes):
        """
        Parallel tree exploration using multiple worker threads.
        
        This method creates a thread-safe shared state manager and spawns
        worker threads to process nodes concurrently from the open queue.
        
        Args:
            time_limit: Maximum time in seconds
            max_nodes: Maximum number of nodes to explore
            
        Returns:
            dict: Results dictionary
        """
        from thread_safe_state import ThreadSafeSharedState
        import time as time_module
        
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" INITIALIZING PARALLEL TREE EXPLORATION ".center(100, "="))
        self.logger.info(f"{'=' * 100}\n")
        
        # Create shared state manager
        shared_state = ThreadSafeSharedState(
            search_strategy=self.search_strategy,
            initial_stats=self.stats,
            initial_nodes=self.nodes
        )
        
        # Copy current state to shared
        shared_state.incumbent = self.incumbent
        shared_state.incumbent_solution = self.incumbent_solution
        shared_state.incumbent_lambdas = self.incumbent_lambdas
        shared_state.best_lp_bound = self.best_lp_bound
        shared_state.gap = self.gap
        shared_state.open_nodes = self.open_nodes.copy()
        
        self.logger.info(f"Initial state:")
        self.logger.info(f"  Incumbent: {shared_state.incumbent:.6f}")
        self.logger.info(f"  Best LB:   {shared_state.best_lp_bound:.6f}")
        self.logger.info(f"  Open nodes: {len(shared_state.open_nodes)}")
        self.logger.info(f"  Workers:   {self.n_tree_workers}\n")
        
        # Create worker threads
        workers = []
        threads = []
        
        for i in range(self.n_tree_workers):
            worker = _NodeWorker(i, self, shared_state)
            thread = threading.Thread(target=worker.run, name=f"Worker-{i}")
            workers.append(worker)
            threads.append(thread)
        
        # Start all workers
        start_time = time_module.time()
        self.logger.info(f"{'=' * 100}")
        self.logger.info(" STARTING WORKER THREADS ".center(100, "="))
        self.logger.info(f"{'=' * 100}\n")
        
        for thread in threads:
            thread.start()
            self.logger.info(f"Started {thread.name}")
        
        # Monitor progress
        last_report_time = start_time
        report_interval = 10.0  # Report every 10 seconds
        
        while any(t.is_alive() for t in threads):
            time_module.sleep(1.0)
            
            # Check termination conditions
            elapsed = time_module.time() - start_time
            
            # Periodic progress report
            if time_module.time() - last_report_time >= report_interval:
                snapshot = shared_state.get_state_snapshot()
                self.logger.info(f"\n{'─' * 100}")
                self.logger.info(f" PROGRESS REPORT (T+{elapsed:.1f}s) ".center(100, "─"))
                self.logger.info(f"{'─' * 100}")
                self.logger.info(f"  Incumbent:      {snapshot['incumbent']:.6f}")
                self.logger.info(f"  Best LB:        {snapshot['best_lp_bound']:.6f}")
                self.logger.info(f"  Gap:            {snapshot['gap']:.4%}")
                self.logger.info(f"  Nodes explored: {snapshot['nodes_explored']}")
                self.logger.info(f"  Nodes fathomed: {snapshot['nodes_fathomed']}")
                self.logger.info(f"  Nodes branched: {snapshot['nodes_branched']}")
                self.logger.info(f"  Queue size:     {snapshot['queue_size']}")
                active_threads = sum(1 for t in threads if t.is_alive())
                self.logger.info(f"  Active workers: {active_threads}/{self.n_tree_workers}")
                self.logger.info(f"{'─' * 100}\n")
                last_report_time = time_module.time()
            
            if elapsed > time_limit:
                self.logger.info(f"\n⏱️  Time limit reached: {elapsed:.2f}s > {time_limit}s")
                self.logger.info("Requesting graceful shutdown...\n")
                shared_state.request_shutdown()
                break
            
            if shared_state.stats['nodes_explored'] >= max_nodes:
                self.logger.info(f"\n📊 Node limit reached: {shared_state.stats['nodes_explored']} >= {max_nodes}")
                self.logger.info("Requesting graceful shutdown...\n")
                shared_state.request_shutdown()
                break
        
        # Wait for all workers to finish
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(" WAITING FOR WORKERS TO COMPLETE ".center(100, "="))
        self.logger.info(f"{'=' * 100}\n")
        
        for i, thread in enumerate(threads):
            thread.join(timeout=10)
            if thread.is_alive():
                self.logger.warning(f"Worker-{i} did not terminate gracefully")
            else:
                self.logger.info(f"Worker-{i} completed ({workers[i].nodes_processed} nodes)")
        
        # Copy shared state back to instance
        self.incumbent = shared_state.incumbent
        self.incumbent_solution = shared_state.incumbent_solution
        self.incumbent_lambdas = shared_state.incumbent_lambdas
        self.best_lp_bound = shared_state.best_lp_bound
        self.gap = shared_state.gap
        self.open_nodes = shared_state.open_nodes
        self.stats = shared_state.stats
        
        # Check if tree is complete
        if not self.open_nodes:
            self.stats['tree_complete'] = True
            if self.incumbent < float('inf'):
                self.best_lp_bound = self.incumbent
                self.gap = 0.0
        
        # Finalize
        self._finalize_and_print_results()
        self._cleanup_pricing_pool()
        return self._get_results_dict()

    def _cleanup_pricing_pool(self):
        """Cleanup persistent multiprocessing pool if it exists."""
        if self.pricing_pool is not None:
            self.pricing_pool.close()
            self.pricing_pool.join()
            self.logger.info("✓ Closed persistent pricing pool")
            self.pricing_pool = None

    def _print_final_results(self):
        """Print final results."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info(" BRANCH-AND-PRICE RESULTS ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"Status: Phase 1 Complete (Root Node Only)")
        self.logger.info(f"")
        self.logger.info(f"Bounds:")
        self.logger.info(f"  LP Bound (LB):  {self.best_lp_bound:.6f}")
        self.logger.info(
            f"  Incumbent (UB): {self.incumbent:.6f}" if self.incumbent < float('inf') else "  Incumbent (UB): None")
        self.logger.info(f"  Gap:            {self.gap:.4%}" if self.gap < float('inf') else "  Gap:            ∞")
        self.logger.info(f"")
        self.logger.info(f"Statistics:")
        self.logger.info(f"  Nodes Explored:   {self.stats['nodes_explored']}")
        self.logger.info(f"  Nodes Fathomed:   {self.stats['nodes_fathomed']}")
        self.logger.info(f"  Nodes Pruned:     {self.stats['nodes_pruned']}")
        self.logger.info(f"  Max Tree Depth:   {self.stats['max_tree_depth']}")
        self.logger.info(f"  Integer Solutions: {self.stats['integer_solutions_found']}")
        self.logger.info(f"  CG Iterations:    {self.stats['total_cg_iterations']}")
        self.logger.info(f"  IP Solves:        {self.stats['ip_solves']}")
        self.logger.info(f"  Incumbent Updates: {self.stats['incumbent_updates']}")
        self.logger.info(f"  Total Time:       {self.stats['total_time']:.2f}s")
        self.logger.info(f"")
        self.logger.info(f"Root Node Info:")
        root = self.nodes[0]
        self.logger.info(f"  Status:         {root.status}")
        self.logger.info(f"  Is Integral:    {root.is_integral}")
        self.logger.info(f"  LP Bound:       {root.lp_bound:.6f}")
        if root.most_fractional_var:
            frac = root.most_fractional_var
            self.logger.info(
                f"  Most Frac Var:  {frac['var_name']} = {frac['value']:.6f} (dist={frac['fractionality']:.6f})")
        if root.fathom_reason:
            self.logger.info(f"  Fathom Reason:  {root.fathom_reason}")
        self.logger.info("=" * 100 + "\n")

    def _get_results_dict(self):
        """Create results dictionary."""
        # Build list of CG iterations per node (sorted by node_id)
        iterations_per_node = [
            self.nodes[node_id].cg_iterations 
            for node_id in sorted(self.nodes.keys())
            if hasattr(self.nodes[node_id], 'cg_iterations')
        ]
        
        # Find incumbent node
        incumbent_node_id = self._find_incumbent_node() if hasattr(self, '_find_incumbent_node') else None
        
        # Determine if solved optimally (tree complete and incumbent found)
        is_optimal = self.stats['tree_complete'] and self.incumbent < float('inf')
        
        # Calculate overhead time
        time_accounted = (self.stats['time_in_mp'] + self.stats['time_in_sp'] + 
                          self.stats['time_in_ip_heuristic'] + self.stats['time_in_branching'])
        time_overhead = self.stats['total_time'] - time_accounted
        
        return {
            'lp_bound': self.best_lp_bound,
            'incumbent': self.incumbent if self.incumbent < float('inf') else None,
            'gap': self.gap if self.gap < float('inf') else None,
            'is_integral': self.nodes[0].is_integral,
            'nodes_explored': self.stats['nodes_explored'],
            'nodes_fathomed': self.stats['nodes_fathomed'],
            'cg_iterations': self.stats['total_cg_iterations'],
            'ip_solves': self.stats['ip_solves'],
            'incumbent_updates': self.stats['incumbent_updates'],
            'total_time': self.stats['total_time'],
            'root_node': self.nodes[0],
            'tree_complete': self.stats['tree_complete'],
            'incumbent_lambdas': self.incumbent_lambdas,
            'total_nodes': len(self.nodes),
            'iterations_per_node': iterations_per_node,
            'root_integral': self.nodes[0].is_integral,
            'is_optimal': is_optimal,
            'incumbent_node_id': incumbent_node_id,
            # Timing statistics
            'time_in_mp': self.stats['time_in_mp'],
            'time_in_sp': self.stats['time_in_sp'],
            'time_in_ip_heuristic': self.stats['time_in_ip_heuristic'],
            'time_in_root': self.stats['time_in_root'],
            'time_in_branching': self.stats['time_in_branching'],
            'time_to_first_incumbent': self.stats['time_to_first_incumbent'],
            'time_overhead': time_overhead,
            # Pattern size counts (SP only, None for MP)
            'pattern_size_counts': self.stats['pattern_size_counts'] if self.branching_strategy == 'sp' else None,
            # Additional BnP metrics
            'max_tree_depth': self.stats['max_tree_depth'],
            'nodes_pruned': self.stats['nodes_pruned'],
            'integer_solutions_found': self.stats['integer_solutions_found'],
            # Final solution
            'incumbent_solution': self.incumbent_solution
        }

    # ============================================================================
    # BRANCHING LOGIC
    # ============================================================================

    def select_branching_candidate(self, node, node_lambda):
        """
        Select the most fractional variable for branching.

        Strategy depends on self.branching_strategy:
        - 'mp': Branch on Lambda_{na} (master variable)
        - 'sp': Branch on Pattern P(k) (hierarchical pattern-based branching)

        Tie-breaking: smallest n, then smallest a/j/t

        Args:
            node: BnPNode to select branching candidate from

        Returns:
            tuple: (branching_type, branching_info) or (None, None) if no fractional var
        """
        if self.branching_strategy == 'mp':
            self.logger.info("DEBUG: Calling _select_mp_branching_candidate")
            result = self._select_mp_branching_candidate(node)
            self.logger.info(f"DEBUG: _select_mp_branching_candidate returned: {result}")
            return result
        elif self.branching_strategy == 'sp':
            # Pattern-based branching (hierarchical)
            self.logger.info(f"DEBUG: Calling _select_sp_branching_candidate with {len(node_lambda)} lambdas")
            result = self._select_sp_branching_candidate(node, node_lambda)
            self.logger.info(f"DEBUG: _select_sp_branching_candidate returned: {result}")
            return result
        else:
            raise ValueError(f"Unknown branching strategy: {self.branching_strategy}")

    def _select_mp_branching_candidate(self, node):
        """
        Select most fractional Lambda_{na} for MP branching.

        Returns:
            tuple: ('mp', branching_info) or (None, None)
        """
        if node.most_fractional_var is None:
            return None, None

        frac_info = node.most_fractional_var
        var_name = frac_info['var_name']

        if 'lmbda' not in var_name:
            self.logger.warning(f"⚠️  Unknown variable type: {var_name}")
            return None, None

        # Parse Lambda[n,a]
        parts = var_name.split('[')[1].split(']')[0].split(',')
        n = int(parts[0])
        a = int(parts[1])

        branching_info = {
            'profile': n,
            'column': a,
            'value': frac_info['value'],
            'floor': frac_info['floor'],
            'ceil': frac_info['ceil'],
            'fractionality': frac_info['fractionality']
        }

        return 'mp', branching_info

    def _find_fractional_pattern(self, node, lambdas, max_pattern_size=5):
        """
        Hierarchical pattern search as described in paper.

        Search for fractional beta_P(k) starting with singletons (|P(k)|=1),
        incrementally increasing pattern size until a fractional value is found.

        Args:
            node: Current BnPNode
            lambdas: Current lambda values {(k,a): value}
            max_pattern_size: Maximum pattern size to consider

        Returns:
            tuple: (pattern_set, beta_value, floor, ceil, profile)
            or (None, None, None, None, None)
        """
        self.logger.info(f"\n[Pattern Search] Starting hierarchical pattern search...")
        self.logger.info(f"  Max pattern size: {max_pattern_size}")

        # Loop over increasing pattern sizes
        for pattern_size in range(1, max_pattern_size + 1):
            self.logger.info(f"\n  Searching patterns of size {pattern_size}...")

            pattern, beta_val, floor_val, ceil_val, profile = self._search_patterns_of_size(
                node, lambdas, pattern_size
            )

            if pattern is not None:
                return pattern, beta_val, floor_val, ceil_val, profile

            self.logger.info(f"  → No fractional pattern of size {pattern_size}, trying larger...")

        self.logger.warning(f"  ❌ No fractional pattern found up to size {max_pattern_size}")
        return None, None, None, None, None
    
    def _search_patterns_of_size(self, node, lambdas, pattern_size):
        """
        Search for fractional patterns of a specific size.
        
        For each profile k, generate combinations of (j,t) pairs of given size,
        compute beta_P(k) for each pattern, and return the most fractional one.
        
        Args:
            node: Current BnPNode
            lambdas: Current lambda values {(k,a): value}
            pattern_size: Size of patterns to search for
            
        Returns:
            tuple: (pattern, beta_value, floor, ceil, profile) or (None, None, None, None, None)
        """
        from itertools import combinations
        
        # First, collect all (j,t) assignments per profile
        profile_assignments = {}  # profile -> set of (j,t) tuples
        
        for (k, a), lambda_val in lambdas.items():
            # Handle both dict format {'value': ..., 'obj': ...} and direct float
            if isinstance(lambda_val, dict):
                val = lambda_val.get('value', 0.0)
            else:
                val = lambda_val
            if val < 1e-6:
                continue
            
            if (k, a) not in node.column_pool:
                continue
            
            col_data = node.column_pool[(k, a)]
            schedules_x = col_data.get('schedules_x', {})
            
            if k not in profile_assignments:
                profile_assignments[k] = set()
            
            # Extract all (j,t) assignments from this column
            for (p, j, t, _), chi_val in schedules_x.items():
                if p == k and chi_val > 0.5:
                    profile_assignments[k].add((j, t))
        
        # Now search for fractional patterns
        best_candidate = None
        max_fractionality = 0.0
        
        for profile_k, assignments in profile_assignments.items():
            if len(assignments) < pattern_size:
                continue  # Not enough assignments for this pattern size
            
            # Generate all combinations of pattern_size
            for pattern_tuple in combinations(sorted(assignments), pattern_size):
                pattern = frozenset(pattern_tuple)
                
        # Compute beta_P(k) for this pattern
                # beta_P(k) = sum_{a in A(k,P(k))} Lambda_{ka}
                # where A(k,P(k)) = {a : chi^a_{kjt}=1 for all (j,t) in P(k)}
                
                beta_val = 0.0
                contrib_count = 0
                contributing_lambdas = []
                
                for (k2, a), lambda_val in lambdas.items():
                    # Handle both dict format and direct float
                    lval = lambda_val.get('value', 0.0) if isinstance(lambda_val, dict) else lambda_val
                    if k2 != profile_k or lval < 1e-6:
                        continue
                    
                    if (k2, a) not in node.column_pool:
                        continue
                    
                    col_data = node.column_pool[(k2, a)]
                    schedules_x = col_data.get('schedules_x', {})
                    
                    # Check if this column covers ALL elements of the pattern
                    covers_all = True
                    for (j_pat, t_pat) in pattern:
                        found = False
                        for (p, j, t, _), chi_val in schedules_x.items():
                            if p == k2 and j == j_pat and t == t_pat and chi_val > 0.5:
                                found = True
                                break
                        if not found:
                            covers_all = False
                            break
                    
                    if covers_all:
                        beta_val += lval
                        contrib_count += 1
                        # Get original ID if possible, otherwise use 'a'
                        col_id = col_data.get('original_id', f"a={a}")
                        contributing_lambdas.append(f"Lambda[{k2},{col_id}]={lval:.4f}")
                
                # Check fractionality
                floor_val = int(beta_val)
                ceil_val = floor_val + 1
                
                dist_to_floor = beta_val - floor_val
                dist_to_ceil = ceil_val - beta_val
                fractionality = min(dist_to_floor, dist_to_ceil)
                
                if fractionality > 1e-5:  # Fractional
                    # Log candidate
                   # print(f"      Candidate: P={profile_k}, Pattern={pattern}, Beta={beta_val:.4f}, Count={contrib_count}, Frac={fractionality:.4f}")

                    is_better = False
                    reason = ""
                    
                    if fractionality > max_fractionality + 1e-10:
                        is_better = True
                        reason = f"Better fractionality ({fractionality:.6f} > {max_fractionality:.6f}) - Lambdas: {contributing_lambdas}"
                    elif abs(fractionality - max_fractionality) < 1e-10:
                        # Tie: prefer smaller profile, then lexicographically smaller pattern
                        if best_candidate is None:
                            is_better = True
                            reason = "First valid candidate"
                        else:
                            current_key = (profile_k, sorted(pattern))
                            best_key = (best_candidate['profile'], sorted(best_candidate['pattern']))
                            if current_key < best_key:
                                is_better = True
                                reason = f"Tie-break: {current_key} < {best_key}"
                            else:
                                pass
                                # reason = f"Tie-break lost: {current_key} >= {best_key}"
                    
                    if is_better:
                        print(f"        -> NEW BEST: {reason}")
                        max_fractionality = fractionality
                        best_candidate = {
                            'profile': profile_k,
                            'pattern': pattern,
                            'beta_value': beta_val,
                            'floor': floor_val,
                            'ceil': ceil_val,
                            'fractionality': fractionality
                        }
        
        if best_candidate is None:
            return None, None, None, None, None
        
        pattern_str = "{" + ", ".join(f"({j},{t})" for j, t in sorted(best_candidate['pattern'])) + "}"
        self.logger.info(f"    Found fractional pattern for profile {best_candidate['profile']}: {pattern_str}")
        self.logger.info(f"    beta_P(k) = {best_candidate['beta_value']:.6f}, fractionality = {best_candidate['fractionality']:.6f}")
        
        return (best_candidate['pattern'], 
                best_candidate['beta_value'], 
                best_candidate['floor'], 
                best_candidate['ceil'],
                best_candidate['profile'])

    def _select_sp_branching_candidate(self, node, lambdas):
        """
        Select most fractional pattern P(k) for SP Pattern Branching.
        
        Uses hierarchical pattern search: starts with singletons |P(k)|=1,
        incrementally increases pattern size until fractional beta_P(k) is found.

        Returns:
            tuple: ('sp', branching_info) or (None, None)
        """
        self.logger.info(f"\n[SP Branching] Starting pattern-based candidate selection...")
        self.logger.info(f"  Column pool size: {len(node.column_pool)}")
        self.logger.info(f"  Lambda values size: {len(lambdas)}")
        
        # Use hierarchical pattern search
        pattern, beta_val, floor_val, ceil_val, profile = self._find_fractional_pattern(
            node, lambdas, max_pattern_size=10
        )
        
        if pattern is None:
            self.logger.error(f"  ❌ No fractional pattern found!")
            return None, None
        
        # Build branching info
        branching_info = {
            'profile': profile,
            'pattern': pattern,
            'beta_value': beta_val,
            'floor': floor_val,
            'ceil': ceil_val,
            'fractionality': min(beta_val - floor_val, ceil_val - beta_val),
            'pattern_size': len(pattern)
        }
        
        pattern_str = "{" + ", ".join(f"({j},{t})" for j, t in sorted(pattern)) + "}"
        self.logger.info(f"\n  [OK] Selected pattern for branching:")
        self.logger.info(f"     Profile: {profile}")
        self.logger.info(f"     Pattern: {pattern_str}")
        self.logger.info(f"     beta_P(k) = {beta_val:.6f}")
        self.logger.info(f"     Fractionality: {branching_info['fractionality']:.6f}")
        self.logger.info(f"     Floor: {floor_val}, Ceil: {ceil_val}")
        
        return 'sp', branching_info


    def branch_on_mp_variable(self, parent_node, branching_info):
        """
        Branch on Master Problem Variable Lambda_{na}.

        Creates two child nodes:
        - Left:  Lambda_{na} <= floor(Lambda_hat)
        - Right: Lambda_{na} >= ceil(Lambda_hat)

        Paper Section 3.2.4, Equation (branch_mp1)

        Args:
            parent_node: BnPNode to branch from
            branching_info: Dict with 'profile', 'column', 'value', 'floor', 'ceil'

        Returns:
            tuple: (left_child, right_child) - two new BnPNode objects
        """
        n = branching_info['profile']
        a = branching_info['column']
        lambda_value = branching_info['value']
        floor_val = branching_info['floor']
        ceil_val = branching_info['ceil']

        self.logger.info("DEBUG: Entering branch_on_mp_variable")

        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(f" BRANCHING ON MP VARIABLE ".center(100, "="))
        self.logger.info(f"{'=' * 100}")
        self.logger.info(f"Branching on Lambda[{n},{a}] = {lambda_value:.6f}")
        self.logger.info(f"  Left:  Lambda[{n},{a}] <= {floor_val}")
        self.logger.info(f"  Right: Lambda[{n},{a}] >= {ceil_val}")

        # Get original schedule for no-good cut
        original_schedule = None
        if (n, a) in parent_node.column_pool:
            original_schedule = parent_node.column_pool[(n, a)].get('schedules_x', {})
            self.logger.info(f"\n  [OK] Found column ({n},{a}) in parent's column pool")
            self.logger.info(f"     Schedule has {len(original_schedule)} assignments")

            # Show first few assignments
            if original_schedule:
                sample_assignments = list(original_schedule.items())[:3]
                for key, val in sample_assignments:
                    self.logger.info(f"       {key}: {val}")
        else:
            self.logger.error(f"\n  ❌ ERROR: Column ({n},{a}) NOT found in parent's column pool!")
            self.logger.error(
                f"     Available columns for profile {n}: {[col_id for (p, col_id) in parent_node.column_pool.keys() if p == n]}")
            self.logger.error(f"     No-good cut cannot be added!")

        # -------------------------
        # LEFT CHILD
        # -------------------------
        self.node_counter += 1
        left_child = BnPNode(
            node_id=self.node_counter,
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            path=parent_node.path + 'l'
        )

        # Create left branching constraint
        from branching_constraints import MPVariableBranching

        left_constraint = MPVariableBranching(
            profile_n=n,
            column_a=a,
            bound=floor_val,
            direction='left',
            original_schedule=original_schedule
        )

        # Inherit branching constraints from parent + add new one
        left_child.branching_constraints = parent_node.branching_constraints.copy()
        left_child.branching_constraints.append(left_constraint)

        # Inherit compatible columns from parent
        self._inherit_columns_from_parent(left_child, parent_node)

        # -------------------------
        # RIGHT CHILD
        # -------------------------
        self.node_counter += 1
        right_child = BnPNode(
            node_id=self.node_counter,
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            path=parent_node.path + 'r'
        )

        # Create right branching constraint
        right_constraint = MPVariableBranching(
            profile_n=n,
            column_a=a,
            bound=ceil_val,
            direction='right'
        )

        # Inherit branching constraints from parent + add new one
        right_child.branching_constraints = parent_node.branching_constraints.copy()
        right_child.branching_constraints.append(right_constraint)

        # Inherit all columns (no restriction on right branch)
        self._inherit_columns_from_parent(right_child, parent_node)

        # Store nodes in the main dictionary first
        self.nodes[left_child.node_id] = left_child
        self.nodes[right_child.node_id] = right_child

        # -------------------------
        # IMMEDIATELY SOLVE BOTH CHILDREN WITH FULL COLUMN GENERATION
        # This is the CORRECT Branch-and-Price approach!
        # -------------------------
        self.logger.info("\n" + "=" * 100)
        print(" SOLVING BOTH CHILDREN IMMEDIATELY WITH FULL CG ".center(100, "="))
        self.logger.info("=" * 100)

        # Solve left child
        left_is_active, left_bound, left_lambdas = self._solve_and_evaluate_child_node(left_child)

        # Solve right child
        right_is_active, right_bound, right_lambdas = self._solve_and_evaluate_child_node(right_child)

        self.logger.info(f"\n  Child Node Evaluation Summary:")
        self.logger.info(f"    Left  (Node {left_child.node_id}): Bound={left_bound:.6f}, Active={left_is_active}, Status={left_child.status}")
        self.logger.info(f"    Right (Node {right_child.node_id}): Bound={right_bound:.6f}, Active={right_is_active}, Status={right_child.status}")

        # Add to open nodes ONLY if they are active (not fathomed)
        # For BFS: Store (bound, node_id) tuple
        # For DFS: Store node_id only
        if self.search_strategy == 'bfs':
            if left_is_active:
                self.open_nodes.append((left_bound, left_child.node_id))
                self.logger.info(f"    ✓ Left child added to open_nodes with bound {left_bound:.6f}")
            if right_is_active:
                self.open_nodes.append((right_bound, right_child.node_id))
                self.logger.info(f"    ✓ Right child added to open_nodes with bound {right_bound:.6f}")
        else:  # DFS
            # Add right first, then left (so left is processed first in LIFO)
            if right_is_active:
                self.open_nodes.append(right_child.node_id)
                self.logger.info(f"    ✓ Right child added to open_nodes")
            if left_is_active:
                self.open_nodes.append(left_child.node_id)
                self.logger.info(f"    ✓ Left child added to open_nodes")

        # Update parent status
        parent_node.status = 'branched'

        self.logger.info(f"  Created left child:  Node {left_child.node_id} (depth {left_child.depth})")
        self.logger.info(f"  Created right child: Node {right_child.node_id} (depth {right_child.depth})")
        self.logger.info(f"{'=' * 100}\n")

        # -------------------------
        # DETAILED OUTPUT OF OPEN_NODES STRUCTURE
        # -------------------------
        print("\n" + "╔" + "═" * 98 + "╗")
        print("║" + " OPEN_NODES STRUCTURE AFTER BRANCHING ".center(98) + "║")
        print("╠" + "═" * 98 + "╣")
        incumbent_str = f"{self.incumbent:.6f}" if self.incumbent < float('inf') else "None"
        print(f"║ Current Incumbent (UB): {incumbent_str}".ljust(99) + "║")
        print("╠" + "═" * 98 + "╣")

        if self.search_strategy == 'bfs':
            print("║ Strategy: BEST-FIRST SEARCH (BFS)".ljust(99) + "║")
            print("║ Format: List of tuples [(bound, node_id), ...]".ljust(99) + "║")
            print("╠" + "═" * 98 + "╣")
            print(f"║ Total nodes in open_nodes: {len(self.open_nodes)}".ljust(99) + "║")
            print("╠" + "═" * 98 + "╣")

            # Sort for display (ascending bound = best first)
            sorted_nodes = sorted(self.open_nodes, key=lambda x: x[0])

            for idx, (bound, node_id) in enumerate(sorted_nodes, 1):
                node = self.nodes[node_id]
                marker = "🏆 BEST" if idx == 1 else ""
                # Show if node can improve incumbent
                if self.incumbent < float('inf'):
                    can_improve = " ✓" if bound < self.incumbent - 1e-5 else " ❌"
                    gap_to_inc = abs(bound - self.incumbent)
                    print(f"║ {idx}. Node {node_id:3d}: bound={bound:10.6f}, depth={node.depth}, path='{node.path:10s}' {marker} {can_improve} (Δ={gap_to_inc:.4f})".ljust(99)[:99] + "║")
                else:
                    print(f"║ {idx}. Node {node_id:3d}: bound={bound:10.6f}, depth={node.depth}, path='{node.path:10s}' {marker}".ljust(99) + "║")

            print("╠" + "═" * 98 + "╣")
            print(f"║ Raw open_nodes list: {self.open_nodes}".ljust(99)[:99] + "║")

        else:  # DFS
            print("║ Strategy: DEPTH-FIRST SEARCH (DFS)".ljust(99) + "║")
            print("║ Format: List of node IDs [node_id, ...]".ljust(99) + "║")
            print("╠" + "═" * 98 + "╣")
            print(f"║ Total nodes in open_nodes: {len(self.open_nodes)}".ljust(99) + "║")
            print("╠" + "═" * 98 + "╣")

            for idx, node_id in enumerate(reversed(self.open_nodes), 1):
                node = self.nodes[node_id]
                marker = "👉 NEXT" if idx == 1 else ""
                # Show if node can improve incumbent
                if self.incumbent < float('inf'):
                    can_improve = " ✓" if node.lp_bound < self.incumbent - 1e-5 else " ❌"
                    gap_to_inc = abs(node.lp_bound - self.incumbent)
                    print(f"║ {idx}. Node {node_id:3d}: bound={node.lp_bound:10.6f}, depth={node.depth}, path='{node.path:10s}' {marker} {can_improve} (Δ={gap_to_inc:.4f})".ljust(99)[:99] + "║")
                else:
                    print(f"║ {idx}. Node {node_id:3d}: bound={node.lp_bound:10.6f}, depth={node.depth}, path='{node.path:10s}' {marker}".ljust(99) + "║")

            print("╠" + "═" * 98 + "╣")
            print(f"║ Raw open_nodes list: {self.open_nodes}".ljust(99)[:99] + "║")

        print("╚" + "═" * 98 + "╝\n")

        self.stats['nodes_branched'] += 1

        return left_child, right_child

    def _solve_and_evaluate_child_node(self, child_node, max_cg_iterations=50):
        """
        Solve a newly created child node with FULL Column Generation immediately.
        This is the CORRECT way to evaluate nodes in Branch-and-Price.

        After solving:
        - Check if node should be fathomed
        - Update node status appropriately
        - Return whether node is still active (not fathomed)

        Args:
            child_node: BnPNode to solve
            max_cg_iterations: Max CG iterations for this node

        Returns:
            tuple: (is_active, lp_bound, node_lambdas)
                - is_active: True if node should be added to open_nodes (not fathomed)
                - lp_bound: LP bound after full CG convergence
                - node_lambdas: Lambda values from solution
        """
        self.logger.info(f"DEBUG: Entering _solve_and_evaluate_child_node for Node {child_node.node_id}")
        self.logger.info(f"\n  [Immediate Child Solving] Solving Node {child_node.node_id} with full CG...")

        # Add to processing order (since we're solving it immediately, not from open_nodes)
        self.stats['node_processing_order'].append(child_node.node_id)

        try:
            # Solve node with FULL Column Generation until convergence
            lp_bound, is_integral, most_frac_info, node_lambdas = self.solve_node_with_cg(
                child_node, max_cg_iterations=max_cg_iterations
            )

            # Update statistics
            self.stats['nodes_explored'] += 1

            # Update global best LP bound if this is better
            if lp_bound < self.best_lp_bound:
                self.best_lp_bound = lp_bound
                self.update_gap()

            # Check fathoming
            if self.should_fathom(child_node, node_lambdas):
                self.logger.info(f"  ✓ Node {child_node.node_id} FATHOMED immediately: {child_node.fathom_reason}")
                self.logger.info(f"    LP Bound: {lp_bound:.6f}, Incumbent: {self.incumbent:.6f}")
                self.stats['nodes_fathomed'] += 1
                return False, lp_bound, node_lambdas  # Node is fathomed, not active

            # Node is not fathomed - mark as solved and ready for branching
            child_node.status = 'solved'
            self.logger.info(f"  ✓ Node {child_node.node_id} SOLVED and ACTIVE")
            self.logger.info(f"    LP Bound: {lp_bound:.6f}, Integral: {is_integral}")

            return True, lp_bound, node_lambdas  # Node is active

        except Exception as e:
            self.logger.error(f"  ✗ Error solving node {child_node.node_id}: {e}")
            
            # DEBUG MODE: Re-raise exception instead of fathoming
            if self.debug_mode:
                self.logger.error("🐛 DEBUG_MODE enabled - re-raising exception!")
                raise
            
            child_node.status = 'fathomed'
            child_node.fathom_reason = 'error'
            self.stats['nodes_fathomed'] += 1
            return False, float('inf'), {}  # Node fathomed due to error

    def _inherit_columns_from_parent(self, child_node, parent_node):
        """
        Inherit columns from parent to child node.

        Filters out columns that are incompatible with the child's
        branching constraints.

        Args:
            child_node: BnPNode receiving columns
            parent_node: BnPNode providing columns
        """
        child_node.column_pool = {}
        inherited_count = 0
        filtered_count = 0

        for (p, col_id), col_data in parent_node.column_pool.items():
            # Check compatibility with all branching constraints
            is_compatible = True

            for constraint in child_node.branching_constraints:
                if not constraint.is_column_compatible(col_data):
                    is_compatible = False
                    break

            if is_compatible:
                child_node.column_pool[(p, col_id)] = col_data.copy()
                inherited_count += 1
            else:
                filtered_count += 1

        self.logger.info(f"    Column inheritance: {inherited_count} inherited, {filtered_count} filtered")

    def solve_node_with_cg(self, node, max_cg_iterations=100):
        """
        Solve a node using Column Generation with branching constraints.

        This performs full CG at a node:
        1. Build master with inherited columns + branching constraints
        2. CG loop: solve LP, get duals, price subproblems, add columns
        3. Check integrality
        4. Return LP bound

        Args:
            node: BnPNode to solve
            max_cg_iterations: Maximum CG iterations at this node

        Returns:
            tuple: (lp_bound, is_integral, most_frac_info)
        """
        self.logger.info(f"\n{'─' * 100}")
        self.logger.info(f" SOLVING NODE {node.node_id} (path: '{node.path}', depth {node.depth}) ".center(100, "─"))
        self.logger.info(f"{'─' * 100}")
        self.logger.info(f"Branching constraints: {len(node.branching_constraints)}")
        self.logger.info(f"Column pool size: {len(node.column_pool)}")

        # Show column distribution
        cols_per_profile = {}
        for (p, _) in node.column_pool.keys():
            cols_per_profile[p] = cols_per_profile.get(p, 0) + 1
        self.logger.info(f"Columns per profile (sample): {dict(list(cols_per_profile.items())[:3])}")
        self.logger.info(f"{'─' * 100}\n")

        # 1. Build master problem and save LP for this node
        master = self._build_master_for_node(node)
        master.Model.update()


        # Determine branching profile (from constraints)
        branching_profile = self._get_branching_profile(node)
        if branching_profile:
            self.logger.info(f"    [SP Saving] Branching profile: {branching_profile}")

        # 2. Column Generation loop
        threshold = self.cg_solver.threshold  # Use same threshold as CG
        cg_iteration = 0
        pricing_filter_history = {}
        skipped_sp = 0
        last_lp_obj = float('inf')

        # Node time limit
        node_start_time = time.time()
        NODE_TIME_LIMIT = 300

        self.logger.debug(f"\n    [Debug] Constraints BEFORE CG loop:")
        for c in master.Model.getConstrs():
            if 'sp_branch' in c.ConstrName:
                self.logger.debug(f"      {c.ConstrName}: {c.Sense} {c.RHS}")

        while cg_iteration < max_cg_iterations:
            if time.time() - node_start_time > NODE_TIME_LIMIT:
                self.logger.debug(f"⏱️  Node {node.node_id} time limit reached")
                break

            cg_iteration += 1

            print(f"\n{'*' * 102}\n*{f'Begin Column Generation Iteration {cg_iteration} (Node {node.node_id})':^100}*\n{'*' * 102}")
            
            # Print parallelization info
            if self.use_labeling:
                if self.use_parallel_pricing:
                    print(f"* Parallel Labeling: ENABLED ({self.n_pricing_workers} workers){' ' * (100 - len(f'Parallel Labeling: ENABLED ({self.n_pricing_workers} workers)'))}*")
                else:
                    print(f"* Parallel Labeling: DISABLED (sequential){' ' * (100 - len('Parallel Labeling: DISABLED (sequential)'))}*")
            else:
                print(f"* Using Gurobi Subproblems (no labeling){' ' * (100 - len('Using Gurobi Subproblems (no labeling)'))}*")
            print(f"{'*' * 102}")

            self.logger.info(f"    [CG Iter {cg_iteration}] Solving master LP...")

            # Solve master as LP with timing
            mp_start_time = time.time()
            master.solRelModel()
            self.stats['time_in_mp'] += time.time() - mp_start_time
            
            if master.Model.status != 2:  # GRB.OPTIMAL
                self.logger.warning(f"    ⚠️  Master in CG-iterations infeasible or unbounded at node {node.node_id}")
                return float('inf'), False, None, {}

            current_lp_obj = master.Model.objVal
            
            # Calculate improvement
            if last_lp_obj != float('inf'):
                improvement = last_lp_obj - current_lp_obj
                relative_improvement = improvement / abs(last_lp_obj) if last_lp_obj != 0 else 0.0
                print(f"    [Dual] LP objective improved: {current_lp_obj:.6f} (Δ={improvement:.6f}, rel={relative_improvement:.4%})")
            
            last_lp_obj = current_lp_obj
            
            self.logger.info(f"    [CG Iter {cg_iteration}] LP objective: {current_lp_obj:.6f}")

            # Get duals from master
            duals_pi, duals_gamma = master.getDuals()
            self.logger.info(self.branching_strategy)

            # Get branching constraint duals if SP-branching is used
            branching_duals = {}
            if self.branching_strategy == 'sp':
                branching_duals = self._get_branching_constraint_duals(master, node)

                # Print active branching constraints with duals (inside CG iteration)
                if cg_iteration == 1 and node.branching_constraints:
                    print(f"\n   Active Branching Constraints ({len(node.branching_constraints)}):")
                    for idx, c in enumerate(node.branching_constraints, 1):

                        # Manual str representation for clear output
                        if "SPPatternBranching" in str(type(c)):
                            c_type = "SP Pattern"
                            pat_str = str(sorted(list(c.pattern)))
                            details = f"Profile {c.profile}, Pattern {pat_str}, {c.direction.upper()} (Level {c.level})"
                        elif "MPVariableBranching" in str(type(c)):
                            c_type = "MP Var"
                            details = f"Worker {c.agent}, Time {c.period}, {c.direction.upper()}"
                        else:
                            c_type = "Unknown"
                            details = str(c)
                        
                        print(f"   {idx}. [{c_type}] {details}")
                print("")

            # 3. Solve subproblems
            # Pricing Filter Logic
            patients_to_solve = []
            if self.cg_solver.pricing_filtering and cg_iteration > 1:
                for index in self.cg_solver.P_Join:
                    skip_subproblem = False
                    previous_iterations = [ell for (n, ell) in pricing_filter_history.keys()
                                           if n == index and ell < cg_iteration]

                    if previous_iterations:
                        ell = max(previous_iterations)
                        hist = pricing_filter_history[(index, ell)]
                        lb = hist['bar_c'] + hist['gamma_n'] - duals_gamma.get(index, 0)
                        sum_term = sum(min(0, hist['pi_td'].get(key, 0) - duals_pi.get(key, 0))
                                       for key in hist['pi_td'])
                        lb += sum_term

                        if lb >= -threshold:
                            skip_subproblem = True
                            skipped_sp += 1

                    if not skip_subproblem:
                        patients_to_solve.append(index)
                
                if not patients_to_solve:
                    self.logger.info(f"    [Pricing Filter] All subproblems skipped. Terminating CG.")
                    break
                    
                skipped_profiles = sorted(list(set(self.cg_solver.P_Join) - set(patients_to_solve)))
                self.logger.info(f"    [Pricing Filter] Solving {len(patients_to_solve)} of {len(self.cg_solver.P_Join)} profiles (skipped {len(skipped_profiles)}: {skipped_profiles})")
            else:
                patients_to_solve = self.cg_solver.P_Join

            new_columns_found = False
            columns_added_this_iter = 0

            # Start timing SPs
            sp_start_time = time.time()

            # ========================================================================
            # PARALLEL PRICING (only with labeling algorithm)
            # ========================================================================
            if self.use_parallel_pricing and self.use_labeling:
                from multiprocessing import Pool
                
                self.logger.info(f"    [Parallel Pricing] Solving {len(patients_to_solve)} profiles with {self.n_pricing_workers} workers")
                
                # Prepare node data (without Gurobi models which can't be pickled)
                # CRITICAL: Deep copy branching constraints and remove 'master_constraint' attribute
                sanitized_constraints = []
                for c in node.branching_constraints:
                    # Create a copy (shallow copy of object structure is enough if we unset the attribute)
                    c_copy = copy.copy(c)
                    if hasattr(c_copy, 'master_constraint'):
                        c_copy.master_constraint = None
                    sanitized_constraints.append(c_copy)

                node_data = {
                    'node_id': node.node_id,
                    'parent_id': node.parent_id,
                    'depth': node.depth,
                    'path': node.path,
                    'branching_constraints': sanitized_constraints
                }
                
                # Prepare CG solver data (picklable data only)
                # Build theta_lookup
                theta_base = self.cg_solver.app_data['theta_base'][0]
                learn_type = self.cg_solver.app_data['learn_type'][0]
                
                theta_lookup = []
                if learn_type == 'lin':
                    lin_increase = self.cg_solver.app_data['lin_increase'][0]
                    theta_lookup = [theta_base + lin_increase * k for k in range(50)]
                elif learn_type == 'exp':
                    k_learn = self.cg_solver.app_data['k_learn'][0]
                    theta_lookup = [theta_base + (1 - theta_base) * (1 - math.exp(-k_learn * k)) for k in range(50)]
                elif learn_type == 'sigmoid':
                    k_learn = self.cg_solver.app_data['k_learn'][0]
                    infl_point = self.cg_solver.app_data['infl_point'][0]
                    theta_lookup = [theta_base + (1 - theta_base) / (1 + math.exp(-k_learn * (k - infl_point))) for k in range(50)]
                else:
                    try:
                        const_val = float(learn_type)
                        theta_lookup = [const_val] * 50
                    except ValueError:
                        # DEBUG MODE: Re-raise exception for debugging
                        if self.debug_mode:
                            self.logger.error("🐛 DEBUG_MODE enabled - re-raising ValueError!")
                            raise
                        theta_lookup = [theta_base] * 50
                theta_lookup = [min(x, 1.0) for x in theta_lookup]
                
                cg_solver_data = {
                    'Entry_agg': self.cg_solver.Entry_agg,
                    'Req_agg': self.cg_solver.Req_agg,
                    'E_dict': self.cg_solver.E_dict,
                    'workers': list(range(1, len(self.cg_solver.T) + 1)),
                    'max_time': max(self.cg_solver.D_Ext),
                    'theta_lookup': theta_lookup,
                    'MS': self.cg_solver.app_data['MS'][0],
                    'MIN_MS': self.cg_solver.app_data['MS_min'][0],
                    'max_columns_per_iter': self.max_columns_per_iter,
                    'use_apriori_pruning': self.use_apriori_pruning,
                    'use_pure_dp_optimization': self.use_pure_dp_optimization,
                    'use_heuristic_pricing': self.use_heuristic_pricing,
                    'heuristic_max_labels': self.heuristic_max_labels,
                    'use_relaxed_history': self.use_relaxed_history,
                    'use_numba_labeling': self.use_numba_labeling,
                    'allow_gaps': self.allow_gaps
                }
                
                # Prepare arguments for each profile
                # CRITICAL: Compute correct next_col_id for each profile FIRST
                profile_args = []
                for profile in patients_to_solve:
                    profile_columns = [col_id for (p, col_id) in node.column_pool.keys() if p == profile]
                    next_col_id = max(profile_columns) + 1 if profile_columns else 1
                    
                    profile_args.append((
                        profile, node_data, duals_pi, duals_gamma.get(profile, 0.0), 
                        branching_duals, threshold, cg_solver_data, next_col_id
                    ))
                
                
                # Solve all pricing problems in parallel using the GLOBAL function
                if self.use_persistent_pool and self.pricing_pool:
                    # Use persistent pool
                    pricing_results = self.pricing_pool.starmap(_parallel_pricing_worker, profile_args)
                else:
                    # Fallback: create temporary pool (backward compatibility)
                    from multiprocessing import Pool
                    with Pool(processes=self.n_pricing_workers) as pool:
                        pricing_results = pool.starmap(_parallel_pricing_worker, profile_args)
                
                # Now add all columns sequentially (to avoid race conditions)
                for profile, col_data_list in pricing_results:
                    # DEBUGGING: Explicit None check
                    if col_data_list is None:
                        print(f"\n{'='*80}")
                        print(f"ERROR DETECTED: col_data_list is None for profile {profile}")
                        print(f"Node ID: {node.node_id}")
                        print(f"Pricing results: {pricing_results}")
                        print(f"{'='*80}\n")
                        sys.exit(1)
                    
                    if not col_data_list:
                        self.logger.debug(f"    [Parallel] Profile {profile}: no column found")
                        continue
                    
                    # Ensure it's a list
                    if isinstance(col_data_list, dict):
                        col_data_list = [col_data_list]
                    
                    # CRITICAL: Get current max col_id for this profile to ensure unique IDs
                    profile_columns = [col_id for (p, col_id) in node.column_pool.keys() if p == profile]
                    current_max_col_id = max(profile_columns) if profile_columns else 0
                    
                    columns_added_for_profile = 0
                    negative_printed = False
                    for col_idx, col_data in enumerate(col_data_list):
                        # Print reduced cost if negative
                        if col_data['reduced_cost'] < 0 and not negative_printed:
                            #print(f'    [Parallel] ⭐ NEGATIVE Red. cost for profile {profile}: {col_data["reduced_cost"]:.6f}')
                            negative_printed = True
                        
                        # Check if column has negative reduced cost below threshold
                        if col_data['reduced_cost'] < -threshold:
                            self.logger.info(f'    [Parallel] [OK] Adding column (below threshold -{threshold})')
                            
                            # CRITICAL: Reassign col_id to ensure uniqueness
                            # The labeling algorithm returns columns with sequential IDs starting from next_col_id
                            # But we need to track the ACTUAL col_id based on what's already added
                            correct_col_id = current_max_col_id + col_idx + 1
                            
                            # Update schedules_x keys with correct col_id
                            original_schedules_x = col_data['schedules_x']
                            corrected_schedules_x = {}
                            for key, val in original_schedules_x.items():
                                # key format: (profile, worker, time, old_col_id)
                                corrected_schedules_x[(key[0], key[1], key[2], correct_col_id)] = val
                            col_data['schedules_x'] = corrected_schedules_x
                            
                            # Update schedules_los keys with correct col_id
                            original_schedules_los = col_data['schedules_los']
                            corrected_schedules_los = {}
                            for key, val in original_schedules_los.items():
                                # key format: (profile, old_col_id)
                                corrected_schedules_los[(key[0], correct_col_id)] = val
                            col_data['schedules_los'] = corrected_schedules_los
                            
                            # Add column to node and master (sequentially!)
                            self._add_column_from_labeling(col_data, profile, node, master)
                            new_columns_found = True
                            columns_added_this_iter += 1
                            columns_added_for_profile += 1
                        else:
                            self.logger.debug(f"    [Parallel] Profile {profile}: red. cost {col_data['reduced_cost']:.6f} >= threshold")
                    
                    if columns_added_for_profile > 0:
                        master.Model.update()
                    
                    # Update pricing filter history
                    if col_data_list:
                        best_col = min(col_data_list, key=lambda c: c['reduced_cost'])
                        pricing_filter_history[(profile, cg_iteration)] = {
                            'bar_c': best_col['reduced_cost'],
                            'gamma_n': duals_gamma.get(profile, 0.0),
                            'pi_td': duals_pi.copy()
                        }
            
            # ========================================================================
            # SEQUENTIAL PRICING (fallback or when parallelization is disabled)
            # ========================================================================
            else:
                for profile in patients_to_solve:
                    # Choose between labeling algorithm and Gurobi
                    if self.use_labeling:
                        # ===== LABELING ALGORITHM =====
                        self.logger.info(f"    [Labeling] Solving pricing for profile {profile}...")
                        
                        col_data_list = self._solve_with_labeling(
                            profile, node, duals_pi, duals_gamma, branching_duals
                        )
                        
                        # DEBUGGING: Explicit None check
                        if col_data_list is None:
                            print(f"\n{'='*80}")
                            print(f"ERROR DETECTED: col_data_list is None for profile {profile}")
                            print(f"Node ID: {node.node_id}")
                            print(f"CG Iteration: {cg_iteration}")
                            print(f"{'='*80}\n")
                            sys.exit(1)
                        
                        # Handle list of columns (or single None if no columns found)
                        if not col_data_list:
                             self.logger.debug(f"    [Labeling] Profile {profile}: no column found")
                        else:
                            # Ensure it's a list (backward compatibility if _solve_with_labeling returns dict)
                            if isinstance(col_data_list, dict):
                                col_data_list = [col_data_list]
                                
                            # CRITICAL: Get current max col_id for this profile to ensure unique IDs
                            profile_columns = [col_id for (p, col_id) in node.column_pool.keys() if p == profile]
                            current_max_col_id = max(profile_columns) if profile_columns else 0
                                
                            columns_added_for_profile = 0
                            negative_printed = False
                            for col_idx, col_data in enumerate(col_data_list):
                                # Print reduced cost if negative
                                if col_data['reduced_cost'] < 0 and not negative_printed:
                                    print(f'    [Labeling] ⭐ NEGATIVE Red. cost for profile {profile}: {col_data["reduced_cost"]:.6f}')
                                    negative_printed = True
                                
                                # Check if column has negative reduced cost below threshold
                                if col_data['reduced_cost'] < -threshold:
                                    self.logger.info(f'    [Labeling] [OK] Adding column (below threshold -{threshold})')
                                    
                                    # CRITICAL: Reassign col_id to ensure uniqueness
                                    correct_col_id = current_max_col_id + col_idx + 1
                                    
                                    # Update schedules_x keys with correct col_id
                                    original_schedules_x = col_data['schedules_x']
                                    corrected_schedules_x = {}
                                    for key, val in original_schedules_x.items():
                                        # key format: (profile, worker, time, old_col_id)
                                        corrected_schedules_x[(key[0], key[1], key[2], correct_col_id)] = val
                                    col_data['schedules_x'] = corrected_schedules_x
                                    
                                    # Update schedules_los keys with correct col_id
                                    original_schedules_los = col_data['schedules_los']
                                    corrected_schedules_los = {}
                                    for key, val in original_schedules_los.items():
                                        # key format: (profile, old_col_id)
                                        corrected_schedules_los[(key[0], correct_col_id)] = val
                                    col_data['schedules_los'] = corrected_schedules_los
                                    
                                    # Add column to node and master
                                    self._add_column_from_labeling(col_data, profile, node, master)
                                    new_columns_found = True
                                    columns_added_this_iter += 1
                                    columns_added_for_profile += 1
                                else:
                                    self.logger.debug(f"    [Labeling] Profile {profile}: red. cost {col_data['reduced_cost']:.6f} >= threshold")
                            
                            if columns_added_for_profile > 0:
                                 master.Model.update()
                            
                            # Update pricing filter history
                            if col_data_list:
                                best_col = min(col_data_list, key=lambda c: c['reduced_cost'])
                                pricing_filter_history[(profile, cg_iteration)] = {
                                    'bar_c': best_col['reduced_cost'],
                                    'gamma_n': duals_gamma.get(profile, 0.0),
                                    'pi_td': duals_pi.copy()
                                }
                    
                    else:
                        # ===== GUROBI SUBPROBLEM =====
                        # Build and solve subproblem with branching constraints
                        sp = self._build_subproblem_for_node(
                            profile, node, duals_pi, duals_gamma, branching_duals
                        )
                        # SAVE FIRST SP FOR BRANCHING PROFILE
                        if profile == branching_profile and self.save_lps:
                            sp_filename = f"LPs/SPs/pricing/sp_node_{node.node_id}_profile_{profile}_iter{cg_iteration}.lp"
                            sp.Model.write(sp_filename)
                            self.logger.info(f"    [OK] [SP Saved] First pricing SP for branching profile {profile}: {sp_filename}")
                        sp.solModel()

                        # Check reduced cost
                        if sp.Model.status == 2 and sp.Model.objVal < -threshold:
                            self.logger.info(f'    Red. cost for profile {profile} : {sp.Model.objVal}')

                            # Add column to node and master
                            self._add_column_from_subproblem(sp, profile, node, master)
                            new_columns_found = True
                            columns_added_this_iter += 1
                            master.Model.update()
                        
                        # Update pricing filter history (Gurobi)
                        if sp.Model.status == 2:
                             pricing_filter_history[(profile, cg_iteration)] = {
                                'bar_c': sp.Model.objVal,
                                'gamma_n': duals_gamma.get(profile, 0.0),
                                'pi_td': duals_pi.copy()
                            }



            # End timing SPs
            sp_duration = time.time() - sp_start_time
            self.stats['time_in_sp'] += sp_duration
            print(f"    [Performance] All Subproblems solved in {sp_duration:.4f}s")
            
            self.logger.info(f"    [CG Iter {cg_iteration}] Added {columns_added_this_iter} new columns")

            # Check convergence
            if not new_columns_found:
                self.logger.info(f"    [CG] Converged after {cg_iteration} iterations - no improving columns found")
                break
            master.Model.update()
            
            # Save LP per iteration for Root Node
            if self.save_lps and node.node_id == 0:
                lp_filename = f"LPs/MP/Root/Root_Iter_{cg_iteration}.lp"
                master.Model.write(lp_filename)
                self.logger.info(f"    [LP Saved] Saved root node LP to {lp_filename}")


        # 4. Final LP solve and integrality check
        self.logger.info(f"\n    [Node {node.node_id}] Final LP solve...")

        if self.save_lps:
            master.Model.write(f"LPs/MP/LPs/mp_final_{node.node_id}.lp")
        master.solRelModel()
        if master.Model.status != 2:  # GRB.OPTIMAL
            self.logger.warning(f"    ⚠️  Final Master infeasible or unbounded at node {node.node_id}")
            return float('inf'), False, None, {}

        if master.Model.status == 2:
            lambda_list_cg = {
                key: {'value': var.X, 'obj': var.Obj} for key, var in master.lmbda.items() if var.X > 1e-6
            }
        else:
            lambda_list_cg = {}

        is_integral, lp_obj, most_frac_info = master.check_fractionality()

        if is_integral:
            self.logger.info(f"\n[OK] INTEGRAL SOLUTION FOUND AT NODE {node.node_id}!")
            self.logger.info(f"   LP Bound: {lp_obj:.6f}")
            
            # Save solution file
            if master.Model.SolCount > 0:
                sol_name = f"Node_{node.node_id}.sol"
                sol_path = os.path.join('sols/Integral', sol_name)
                master.Model.write(sol_path)
                self.logger.info(f"   [IP Save] Integral node solution saved to {sol_path}")

        # Store results in node
        node.lp_bound = lp_obj
        node.is_integral = is_integral
        node.most_fractional_var = most_frac_info

        self.logger.info(f"\n    [Node {node.node_id}] Results:")
        self.logger.info(f"      LP Bound: {lp_obj:.6f}")
        self.logger.info(f"      Is Integral: {is_integral}")
        self.logger.info(f"      CG Iterations: {cg_iteration}")
        self.logger.info(f"      Final column pool: {len(node.column_pool)} columns")

        if most_frac_info:
            self.logger.info(f"      Most fractional: {most_frac_info['var_name']} = {most_frac_info['value']:.6f}")

        self.logger.info(f"{'─' * 100}\n")

        # Store CG iterations on the node for later retrieval
        node.cg_iterations = cg_iteration
        self.stats['total_cg_iterations'] += cg_iteration

        return lp_obj, is_integral, most_frac_info, lambda_list_cg

    def _build_master_for_node(self, node):
        """
        Build master problem for a node with inherited columns and branching constraints.
        """
        from masterproblem import MasterProblem_d

        self.logger.info(f"    [Master] Building master problem for node {node.node_id}...")

        # Create master
        master = MasterProblem_d(
            self.cg_solver.data,
            self.cg_solver.Max_t_cg,
            self.cg_solver.Nr_agg,
            self.cg_solver.Req_agg,
            self.cg_solver.pre_x,
            self.cg_solver.E_dict,
            use_warmstart=self.cg_solver.use_warmstart
        )

        # Build model with start sol (creates basic constraints)
        master.buildModel()
        master.startSol(self.start_x, self.start_los)
        master.Model.update()

        self.logger.info(f"    [Master] Basic model built with {len(master.Model.getConstrs())} constraints")

        # [OK] CRITICAL FIX: Add initial columns (col_id=1) to all_schedules for SP branching
        self.logger.info(f"    [Master] Adding initial columns (col_id=1) to all_schedules...")
        initial_cols_added = 0
        for (profile, col_id), col_data in node.column_pool.items():
            if col_id == 1:
                schedules_x = col_data.get('schedules_x', {})
                if schedules_x:
                    master.addSchedule(schedules_x)
                    initial_cols_added += 1

                    # Debug: Show first assignment
                    if initial_cols_added == 1:
                        sample_key = list(schedules_x.keys())[0]
                        self.logger.debug(f"      Sample initial column: {sample_key} = {schedules_x[sample_key]}")

        self.logger.info(f"    [Master] Added {initial_cols_added} initial columns to all_schedules")

        sp_branching_active = False

        # Load columns
        self.logger.info(f"    [Master] Loading {len(node.column_pool)} columns from pool...")

        for (profile, col_id), col_data in node.column_pool.items():

            if col_id >= 2:

                # Add schedules to master
                schedules_x = col_data.get('schedules_x', {})
                schedules_los = col_data.get('schedules_los', {})


                if not schedules_x:
                    self.logger.warning(f"      ⚠️ WARNING: Column ({profile},{col_id}) has empty schedules_x!")
                    continue

                master.addSchedule(schedules_x)
                master.addLOS(schedules_los)

                # Get pre-computed lists or create them
                x_list = col_data.get('x_list', list(schedules_x.values()))
                if profile in master.P_Post:
                    los_list = [0]
                elif profile in master.P_Focus and col_id >= 2:
                    los_list = col_data.get('los_list', list(schedules_los.values()))

                lambda_list = self._create_lambda_list(profile)

                # Build coefficient vector: [lambda_coefs, x_coefs]
                col_coefs = lambda_list + x_list

                # In MP branching, only variable bounds are set, no new constraints.
                if sp_branching_active:
                    branching_coefs = self._compute_branching_coefficients_for_column(
                        col_data, profile, col_id, node.branching_constraints
                    )
                    if all(x == 0 for x in branching_coefs):
                        self.logger.info(
                            f"      [Column with postive Chi {profile},{col_id}] Added {len(branching_coefs)} branching coefficients")
                        # sys.exit() # Removed debug exit
                    col_coefs = col_coefs + branching_coefs
                    self.logger.info(
                        f"      [Column {profile},{col_id}] Added {len(branching_coefs)} branching coefficients")

                master.addLambdaVar(profile, col_id, col_coefs, los_list, 
                                    pattern={'path': col_data.get('path_pattern'), 'start': col_data.get('start')})
        master.Model.update()



        # Update Obj coefficients for initial column
        for (profile, col_id), col_data in node.column_pool.items():
            schedules_los = col_data.get('schedules_los', {})
            if profile in master.P_Join and col_id == 1:
                master.lmbda[profile, 1].Obj = col_data.get('los_list', schedules_los.values())[0]
        master.Model.update()

        # SP-Branching: adds new constraints → need coefficients
        # MP-Branching: only sets variable bounds → NO new coefficients needed
        if node.branching_constraints:
            self.logger.info(f"    [Master] Applying {len(node.branching_constraints)} branching constraints...")

            for constraint in node.branching_constraints:
                self.logger.info(f'Cons {constraint}')

                constraint.apply_to_master(master, node)  # Pass node to constraint

                # Check if this is SP branching (adds constraints)
                if hasattr(constraint, 'master_constraint') and constraint.master_constraint is not None:
                    sp_branching_active = True

            master.Model.update()
            self.logger.info(f"    [Master] Now have {len(master.Model.getConstrs())} constraints")
            self.logger.info(f"    [Master] SP-Branching constraints added: {sp_branching_active}")

            # DEBUG EXIT
            if sp_branching_active and node.node_id > 0:
                #master.Model.write(f"LPs/MP/LPs/master_branch_node_{node.node_id}.lp")

                # Show constraint details
                for c in master.Model.getConstrs():
                    if 'sp_branch' in c.ConstrName:
                        self.logger.info(f"  Constraint: {c.ConstrName}")


        self.logger.info(f"    [Master] Master problem ready:")
        self.logger.info(f"             - {len(master.lmbda)} lambda variables")
        self.logger.info(f"             - {len(master.Model.getConstrs())} constraints")

        return master

    def _compute_branching_coefficients_for_column(self, col_data, profile, col_id, branching_constraints):
        """
        Compute coefficients for branching constraints for an existing column.

        CRITICAL: A branching constraint on profile n only affects columns for profile n!
        """
        from branching_constraints import SPPatternBranching, MPVariableBranching

        coefs = []
        schedules_x = col_data.get('schedules_x', {})

        for constraint in branching_constraints:
            if isinstance(constraint, SPPatternBranching):
                k = constraint.profile
                pattern = constraint.pattern
                
                # If this column is not for the branched profile, coef = 0
                if profile != k:
                    coefs.append(0)
                    continue
                
                # Check if this column covers the ENTIRE pattern
                # coef = 1 if ALL pattern elements (j,t) are present in this column
                pattern_fully_covered = True
                for (j_pat, t_pat) in pattern:
                    found = False
                    for (p, j_sched, t_sched, a), val in schedules_x.items():
                        if p == profile and j_sched == j_pat and t_sched == t_pat and a == col_id and val > 0.5:
                            found = True
                            break
                    if not found:
                        pattern_fully_covered = False
                        break
                
                coef = 1 if pattern_fully_covered else 0
                coefs.append(coef)

            elif isinstance(constraint, MPVariableBranching):
                # MP branching uses variable bounds, not linear constraints
                # So we do NOT add any coefficient to the column vector
                pass

        return coefs

    def _create_lambda_list(self, profile):
        """
        Create lambda list for a profile (indicator vector).

        Args:
            profile: Profile index

        Returns:
            list: Lambda list with 1 at profile position, 0 elsewhere
        """
        if profile in self.cg_solver.P_Join:
            ind = self.cg_solver.P_Join.index(profile)
            lst = [0] * len(self.cg_solver.P_Join)
            lst[ind] = 1
            return lst
        return []

    def _get_branching_constraint_duals(self, master, node):
        """
        Extract dual variables from SP branching constraints (SPPatternBranching).

        According to Paper Eq. (branch:sub4):
        - Left branch (≤): δ^L ≤ 0
        - Right branch (≥): δ^R ≥ 0
        - Both are ADDED in pricing: - sum(δ^L + δ^R)
        """
        from branching_constraints import SPPatternBranching
        
        branching_duals = {}

        self.logger.info(f"\n      [Extracting Branching Duals] Node {node.node_id}, Path: '{node.path}'")
        self.logger.info(f"      Total branching constraints: {len(node.branching_constraints)}")

        sp_constraints_found = 0

        for constraint in node.branching_constraints:
            # Only SP Pattern branching constraints have master constraints
            if not isinstance(constraint, SPPatternBranching):
                continue
            if not hasattr(constraint, 'master_constraint') or constraint.master_constraint is None:
                continue

            try:
                dual_val = constraint.master_constraint.Pi
                sp_constraints_found += 1
                
                direction = constraint.direction

                # Validate dual sign (according to constraint direction)
                if direction == 'left' and dual_val > 1e-6:
                    self.logger.warning(f"      ⚠️  WARNING: Left branch (≤) has positive dual: {dual_val:.6f}")
                if direction == 'right' and dual_val < -1e-6:
                    self.logger.warning(f"      ⚠️  WARNING: Right branch (≥) has negative dual: {dual_val:.6f}")

                # Store dual with pattern identifier
                # NEW: Key using stable string representation (sorted tuples)
                # Key: (profile, 'pattern', pattern_str, level)
                pattern_str = str(sorted(list(constraint.pattern)))
                key = (constraint.profile, 'pattern', pattern_str, constraint.level)
                branching_duals[key] = dual_val
                
                pattern_str = "{" + ", ".join(f"({j},{t})" for j, t in sorted(constraint.pattern)) + "}"
                self.logger.info(f"      [Dual] Level {constraint.level:2d} ({direction:5s}): "
                            f"Pattern[{constraint.profile}] {pattern_str} "
                            f"→ π={dual_val:+.6f}")

            except Exception as e:
                self.logger.warning(f"      ⚠️  Could not extract dual from constraint: {e}")

        self.logger.info(f"      Found {sp_constraints_found} SP branching duals\n")
        print('Branching Duals', {k: v for k, v in branching_duals.items() if v != 0})
        return branching_duals

    def _build_subproblem_for_node(self, profile, node, duals_pi, duals_gamma, branching_duals=None):
        """
        Build subproblem for a profile at a node with branching constraints.

        Uses node-local column IDs and REAL duals from master LP.

        Args:
            profile: Profile index
            node: BnPNode
            duals_pi: Dual variables for capacity constraints
            duals_gamma: Dual variables for profile constraints
            branching_duals: Dict of branching constraint duals (for SP-branching)

        Returns:
            Subproblem: Subproblem with constraints
        """
        from subproblem import Subproblem

        if branching_duals is None:
            branching_duals = {}

        # Separate variable and pattern branching duals
        variable_duals = {}
        pattern_duals = {}
        
        for key, value in branching_duals.items():
            if key[0] == profile:
                # Check if this is a pattern dual (has 'pattern' as second element)
                if len(key) == 4 and key[1] == 'pattern':
                    pattern_duals[key] = value
                elif len(key) == 4:
                    # Variable branching: (profile, agent, period, level)
                    variable_duals[key] = value
        
        # Calculate duals_delta from variable branching constraints only
        if variable_duals:
            duals_delta = sum(variable_duals.values())
            print(f"\n      [SP Variable Duals] Profile {profile} has {len(variable_duals)} variable branching duals: {variable_duals}")
            for (p, j, t, level), dual_val in variable_duals.items():
                print(f"         Level {level}: x[{p},{j},{t}] → dual={dual_val:.6f}")
            print(f"      [SP Variable Duals] Total duals_delta = {duals_delta:.6f}\n")
        else:
            duals_delta = 0.0
            #print(f"      [SP Variable Duals] Profile {profile} has NO variable branching constraints\n")
        
        # Log pattern duals (these will be handled separately)
        if pattern_duals:
            print(f"\n      [SP Pattern Duals] Profile {profile} has {len(pattern_duals)} pattern branching duals: {pattern_duals}")
            for (p, pattern_type, pattern_id, level), dual_val in pattern_duals.items():
                self.logger.info(f"         Level {level}: Pattern {pattern_id} → dual={dual_val:.6f}")
            print(f"      [SP Pattern Duals] These will be integrated via w^Full variables\n")

        # Determine next col_id based on column_pool of this node
        profile_columns = [col_id for (p, col_id) in node.column_pool.keys() if p == profile]

        if profile_columns:
            next_col_id = max(profile_columns) + 1
        else:
            next_col_id = 1

        # Create subproblem with real duals
        sp = Subproblem(
            self.cg_solver.data,
            duals_gamma,
            duals_pi,
            duals_delta,
            profile,
            next_col_id,
            self.cg_solver.Req_agg,
            self.cg_solver.Entry_agg,
            self.cg_solver.app_data,
            self.cg_solver.W_coeff,
            self.cg_solver.E_dict,
            self.cg_solver.S_Bound,
            learn_method=self.cg_solver.learn_method,
            reduction=True,
            num_tangents=10,
            node_path=node.path,
            verbose=self.verbose,
            deterministic=self.cg_solver.deterministic
        )

        sp.buildModel()

        # Apply all branching constraints
        for constraint in node.branching_constraints:
            constraint.apply_to_subproblem(sp)

        sp.Model.update()
        
        # ========================================================================
        # INTEGRATE PATTERN BRANCHING DUALS WITH w^Full VARIABLES
        # ========================================================================
        # For pattern branching right branches, add dual coefficient to w^Full
        if pattern_duals and hasattr(sp, 'pattern_w_full_vars'):
            self.logger.info(f"\n      [Pattern Dual Integration] Adding dual coefficients to w^Full variables...")
            
            for (p, pattern_type, pattern_id, level), dual_val in pattern_duals.items():
                if level in sp.pattern_w_full_vars:
                    w_var = sp.pattern_w_full_vars[level]
                    # Add dual to w variable objective coefficient
                    # The dual enters the pricing problem as: obj_coeff = -dual
                    # (negative because we minimize reduced cost)
                    w_var.Obj = -dual_val
                    self.logger.info(f"         Level {level}: Set w^Full obj coefficient = {-dual_val:.6f}")
                else:
                    self.logger.warning(f"         ⚠️  Level {level} pattern dual found but no w^Full variable exists!")
            
            sp.Model.update()
            self.logger.info(f"      [Pattern Dual Integration] Complete\n")
        return sp

    def _solve_with_labeling(self, profile, node, duals_pi, duals_gamma, branching_duals=None):
        """
        Solve pricing problem using labeling algorithm instead of Gurobi.
        
        Args:
            profile: Profile index
            node: BnPNode
            duals_pi: Dual variables for capacity constraints
            duals_gamma: Dual variables for profile constraints
            branching_duals: Dict of branching constraint duals (not yet used)
        
        Returns:
            dict or None: Column data or None if no improving column found
        """
        from label import solve_pricing_for_profile_bnp
        
        # Extract profile-specific data from cg_solver
        r_k = self.cg_solver.Entry_agg[profile]  # Release time

        # Get service requirement from Req dictionary
        s_k = self.cg_solver.Req_agg[profile]
        
        # Get objective mode from E_dict - determines if duration is minimized (1) or not (0)
        # E_dict[profile] = 1 if profile in Focus Horizon (minimize duration)
        # E_dict[profile] = 0 if profile in Post Horizon (only minimize cost, not duration)
        obj_multiplier = self.cg_solver.E_dict.get(profile, 1)
        
        # Get next column ID
        profile_columns = [col_id for (p, col_id) in node.column_pool.keys() if p == profile]
        next_col_id = max(profile_columns) + 1 if profile_columns else 1
        
        # Configuration parameters from app_data
        workers = list(range(1, len(self.cg_solver.T) + 1))  # [1, 2, 3, ...] based on number of therapists
        max_time = max(self.cg_solver.D_Ext)  # Maximum time from extended day set (e.g., 42)
        
        # Learning curve parameters
        # Learning curve parameters
        theta_base = self.cg_solver.app_data['theta_base'][0]
        learn_type = self.cg_solver.app_data['learn_type'][0]
        
        theta_lookup = []
        
        if learn_type == 'lin':
            lin_increase = self.cg_solver.app_data['lin_increase'][0]
            theta_lookup = [theta_base + lin_increase * k for k in range(50)]
            
        elif learn_type == 'exp':
            k_learn = self.cg_solver.app_data['k_learn'][0]
            theta_lookup = [theta_base + (1 - theta_base) * (1 - math.exp(-k_learn * k)) for k in range(50)]
            
        elif learn_type == 'sigmoid':
            k_learn = self.cg_solver.app_data['k_learn'][0]
            infl_point = self.cg_solver.app_data['infl_point'][0]
            theta_lookup = [theta_base + (1 - theta_base) / (1 + math.exp(-k_learn * (k - infl_point))) for k in range(50)]
            
        else:
            # Constant learning factor (numeric value or string representation of float)
            try:
                const_val = float(learn_type)
                theta_lookup = [const_val] * 50
            except ValueError:
                # DEBUG MODE: Re-raise exception for debugging
                if self.debug_mode:
                    self.logger.error("🐛 DEBUG_MODE enabled - re-raising ValueError!")
                    raise
                # Fallback to theta_base if unknown string
                self.logger.warning(f"Unknown learn_type '{learn_type}', using theta_base constant.")
                theta_lookup = [theta_base] * 50

        theta_lookup = [min(x, 1.0) for x in theta_lookup]  # Cap at 1.0
        
        # Milestone parameters
        MS = self.cg_solver.app_data['MS'][0]
        MIN_MS = self.cg_solver.app_data['MS_min'][0]
        
        # Call labeling algorithm

        col_data_list = solve_pricing_for_profile_bnp(
            profile=profile,
            duals_pi=duals_pi,
            duals_gamma=duals_gamma.get(profile, 0.0),
            r_k=r_k,
            s_k=s_k,
            obj_multiplier=obj_multiplier,
            workers=workers,
            max_time=max_time,
            theta_lookup=theta_lookup,
            MS=MS,
            MIN_MS=MIN_MS,
            col_id=next_col_id,
            branching_constraints=node.branching_constraints,
            max_columns=self.max_columns_per_iter,  # Return up to N columns
            use_pure_dp_optimization=self.use_pure_dp_optimization,
            use_numba_labeling=self.use_numba_labeling,
            allow_gaps=self.allow_gaps,
            stop_at_first_negative=(node.depth > 0)  # Early termination in child nodes only
        )
        
        return col_data_list
    
    def _add_column_from_labeling(self, col_data, profile, node, master):
        """
        Add column generated by labeling algorithm to node and master.
        
        Args:
            col_data: Column data from labeling algorithm
            profile: Profile index
            node: BnPNode
            master: MasterProblem instance
        """
        # Extract column ID from schedules_x keys
        sample_key = next(iter(col_data['schedules_x'].keys()))
        col_id = sample_key[3]  # (profile, worker, time, col_id)
        
        # Add to node pool
        node.column_pool[(profile, col_id)] = {
            'schedules_x': col_data['schedules_x'],
            'schedules_y': col_data.get('schedules_y', {}),
            'schedules_los': col_data['schedules_los'],
            'x_list': col_data['x_list'],
            'los_list': col_data['los_list'],
        }
        
        # Update cg_solver.global_solutions
        solution_key = (profile, col_id)
        if 'solution_vars' in col_data:
            for var_name, var_value in col_data['solution_vars'].items():
                if solution_key not in self.cg_solver.global_solutions[var_name]:
                    self.cg_solver.global_solutions[var_name][solution_key] = var_value
        else:
            # Fallback for labeling which doesn't provide solution_vars
            self.cg_solver.global_solutions['x'][solution_key] = col_data['schedules_x']
            self.cg_solver.global_solutions['LOS'][solution_key] = col_data['schedules_los']
            # Store y values from labeling
            if 'schedules_y' in col_data and col_data['schedules_y']:
                self.cg_solver.global_solutions['y'][solution_key] = col_data['schedules_y']
        
        # Add to master
        master.addSchedule(col_data['schedules_x'])
        master.addLOS(col_data['schedules_los'])
        
        # Build coefficient vector for master
        lambda_list = self._create_lambda_list(profile)
        
        # Use x_vector directly from labeling algorithm
        # This vector is already full length (workers * max_time)
        x_list_full = col_data['x_vector']
        
        col_coefs = lambda_list + x_list_full
        
        # ========================================================================
        # ADD SP-BRANCHING COEFFICIENTS IF NEEDED
        # ========================================================================
        sp_branching_constraints = [c for c in node.branching_constraints
                                    if hasattr(c, 'master_constraint')
                                    and c.master_constraint is not None]

        if sp_branching_constraints:
            branching_coefs = self._compute_branching_coefficients_for_column(
                col_data, profile, col_id, node.branching_constraints
            )
            col_coefs = col_coefs + branching_coefs

            self.logger.info(f"        [Labeling] Added {len(branching_coefs)} branching coefficients "
                        f"for new column ({profile}, {col_id})")

        # DEBUG: Log array lengths before adding to master
        num_constraints = len(master.Model.getConstrs())
        self.logger.info(f"      [DEBUG Labeling Column {profile},{col_id}]:")
        self.logger.info(f"        lambda_list length: {len(lambda_list)}")
        self.logger.info(f"        x_list_full length: {len(x_list_full)}")
        self.logger.info(f"        col_coefs total length: {len(col_coefs)}")
        self.logger.info(f"        Master constraints count: {num_constraints}")
        self.logger.info(f"        Match: {len(col_coefs) == num_constraints}")

        # Add lambda variable to master
        if profile in self.cg_solver.P_F:
            master.addLambdaVar(profile, col_id, col_coefs, col_data['los_list'], 
                                pattern={'path': col_data.get('path_pattern'), 'start': col_data.get('start')})
        else:
            master.addLambdaVar(profile, col_id, col_coefs, [0], 
                                pattern={'path': col_data.get('path_pattern'), 'start': col_data.get('start')})
        
        self.logger.info(f"      [Labeling] Added column {col_id} for profile {profile}, "
                        f"reduced cost: {col_data['reduced_cost']:.6f}")


    def _add_column_from_subproblem(self, subproblem, profile, node, master):
        """
        Add a column generated from a subproblem to node and master.

        CRITICAL: Must compute branching coefficients for SP-branching constraints!
        """
        col_id = subproblem.col_id

        # Extract solution from subproblem
        schedules_x, x_list, _ = subproblem.getOptVals('x')
        schedules_los, los_list, _ = subproblem.getOptVals('LOS')

        # Create column data
        col_data = {
            'index': profile,
            'column_id': col_id,
            'schedules_x': schedules_x,
            'schedules_los': schedules_los,
            'x_list': x_list,
            'los_list': los_list,
            'reduced_cost': subproblem.Model.objVal
        }

        # Add to node's column pool
        node.column_pool[(profile, col_id)] = col_data

        # Validate key format consistency
        for key in schedules_x.keys():
            if len(key) != 4:
                self.logger.warning(f"⚠️  Invalid schedules_x key format: {key} (expected 4 components)")
            elif key[3] != col_id:
                self.logger.warning(f"⚠️  Inconsistent col_id in schedules_x key: {key} (expected col_id={col_id})")

        # Add to master
        master.addSchedule(schedules_x)
        master.addLOS(schedules_los)

        # Update cg_solver.global_solutions
        solution_vars = {
            var: subproblem.getVarSol(var, col_id)
            for var in ['x', 'LOS', 'y', 'z', 'S', 'l']
        }
        if self.cg_solver.app_data["learn_type"][0] in ['exp', 'sigmoid', 'lin']:
            solution_vars['App'] = subproblem.getVarSol('App', col_id)
            
        solution_key = (profile, col_id)
        for var_name, var_value in solution_vars.items():
            if solution_key not in self.cg_solver.global_solutions[var_name]:
                self.cg_solver.global_solutions[var_name][solution_key] = var_value

        # Create coefficient lists
        lambda_list = self._create_lambda_list(profile)

        # Basic coefficients
        col_coefs = lambda_list + x_list

        # ========================================================================
        # ADD SP-BRANCHING COEFFICIENTS IF NEEDED
        # ========================================================================
        sp_branching_constraints = [c for c in node.branching_constraints
                                    if hasattr(c, 'master_constraint')
                                    and c.master_constraint is not None]

        #print(col_data, profile, col_id, node.branching_constraints, sep="\n")

        if sp_branching_constraints:
            branching_coefs = self._compute_branching_coefficients_for_column(
                col_data, profile, col_id, node.branching_constraints
            )
            col_coefs = col_coefs + branching_coefs

            #print(f'New_Coeffs for profile {profile} are: {branching_coefs}')

            self.logger.info(f"        [Column] Added {len(branching_coefs)} branching coefficients "
                        f"for new column ({profile}, {col_id})")

        # Verify length
        expected_length = len(master.Model.getConstrs())
        actual_length = len(col_coefs)
        #print(expected_length, actual_length, sep="\n")


        if actual_length != expected_length:
            self.logger.error(f"        ❌ ERROR: Coefficient mismatch when adding new column!")
            self.logger.error(f"           Expected: {expected_length}, Got: {actual_length}")
            raise ValueError("Coefficient vector length mismatch!")

        # Add variable to master
        master.addLambdaVar(
            profile, col_id,
            col_coefs,
            los_list
        )

        self.logger.info(f"        [Column] Added column ({profile}, {col_id}) "
                    f"with reduced cost {subproblem.Model.objVal:.6f}")



    def branch_on_sp_pattern(self, parent_node, branching_info):
        """
        Branch on Pattern P(k) ⊆ J × T_k.
        
        Creates two child nodes:
        - Left:  sum_{(j,t) in P(k)} x_{kjt} <= |P(k)| - 1  (no full pattern coverage)
        - Right: sum_{(j,t) in P(k)} x_{kjt} = |P(k)| * w^Full  (all-or-nothing)
        
        Paper Section 3.2.4, Equations (branch:sp_mp) and (branch:sp_rest)
        
        Args:
            parent_node: BnPNode to branch from
            branching_info: Dict with 'profile', 'pattern', 'beta_value', 'floor', 'ceil'
            
        Returns:
            tuple: (left_child, right_child)
        """
        k = branching_info['profile']
        pattern = branching_info['pattern']
        beta_val = branching_info['beta_value']
        floor_val = branching_info['floor']
        ceil_val = branching_info['ceil']
        pattern_size = branching_info['pattern_size']
        
        pattern_str = "{" + ", ".join(f"({j},{t})" for j, t in sorted(pattern)) + "}"
        
        self.logger.info(f"\n{'=' * 100}")
        self.logger.info(f" BRANCHING ON SP PATTERN ".center(100, "="))
        self.logger.info(f"{'=' * 100}")
        self.logger.info(f"Branching on pattern P({k}) = {pattern_str}")
        self.logger.info(f"  Pattern size: {pattern_size}")
        self.logger.info(f"  beta_P(k) = {beta_val:.6f}")
        self.logger.info(f"  Left:  sum x_kjt <= {pattern_size - 1} (no full coverage)")
        self.logger.info(f"  Right: sum x_kjt = {pattern_size} * w^Full (all-or-nothing)")
        
        # -------------------------
        # LEFT CHILD
        # -------------------------
        self.node_counter += 1
        left_child = BnPNode(
            node_id=self.node_counter,
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            path=parent_node.path + 'l'
        )
        
        from branching_constraints import SPPatternBranching
        
        left_constraint = SPPatternBranching(
            profile_k=k,
            pattern=pattern,
            direction='left',
            level=left_child.depth,
            floor_val=floor_val,
            ceil_val=ceil_val
        )
        
        left_child.branching_constraints = parent_node.branching_constraints.copy()
        left_child.branching_constraints.append(left_constraint)
        
        self._inherit_columns_from_parent(left_child, parent_node)
        
        # -------------------------
        # RIGHT CHILD
        # -------------------------
        self.node_counter += 1
        right_child = BnPNode(
            node_id=self.node_counter,
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            path=parent_node.path + 'r'
        )
        
        right_constraint = SPPatternBranching(
            profile_k=k,
            pattern=pattern,
            direction='right',
            level=right_child.depth,
            floor_val=floor_val,
            ceil_val=ceil_val
        )
        
        right_child.branching_constraints = parent_node.branching_constraints.copy()
        right_child.branching_constraints.append(right_constraint)
        
        self._inherit_columns_from_parent(right_child, parent_node)
        
        # Store nodes in the main dictionary first
        self.nodes[left_child.node_id] = left_child
        self.nodes[right_child.node_id] = right_child
        
        # -------------------------
        # SOLVE BOTH CHILDREN WITH FULL COLUMN GENERATION
        # -------------------------
        self.logger.info("\n" + "=" * 100)
        self.logger.info(" SOLVING BOTH CHILDREN IMMEDIATELY WITH FULL CG ".center(100, "="))
        self.logger.info("=" * 100)
        
        # Solve left child
        left_is_active, left_bound, left_lambdas = self._solve_and_evaluate_child_node(left_child)
        
        # Solve right child
        right_is_active, right_bound, right_lambdas = self._solve_and_evaluate_child_node(right_child)
        
        self.logger.info(f"\n  Child Node Evaluation Summary:")
        self.logger.info(f"    Left  (Node {left_child.node_id}): Bound={left_bound:.6f}, Active={left_is_active}, Status={left_child.status}")
        self.logger.info(f"    Right (Node {right_child.node_id}): Bound={right_bound:.6f}, Active={right_is_active}, Status={right_child.status}")
        
        # Add to open nodes ONLY if they are active (not fathomed)
        if self.search_strategy == 'bfs':
            if left_is_active:
                self.open_nodes.append((left_bound, left_child.node_id))
                self.logger.info(f"    ✓ Left child added to open_nodes with bound {left_bound:.6f}")
            if right_is_active:
                self.open_nodes.append((right_bound, right_child.node_id))
                self.logger.info(f"    ✓ Right child added to open_nodes with bound {right_bound:.6f}")
        else:  # DFS
            # Add right first, then left (so left is processed first in LIFO)
            if right_is_active:
                self.open_nodes.append(right_child.node_id)
                self.logger.info(f"    ✓ Right child added to open_nodes")
            if left_is_active:
                self.open_nodes.append(left_child.node_id)
                self.logger.info(f"    ✓ Left child added to open_nodes")
        
        parent_node.status = 'branched'
        
        self.logger.info(f"  Created left child:  Node {left_child.node_id} (depth {left_child.depth})")
        self.logger.info(f"  Created right child: Node {right_child.node_id} (depth {right_child.depth})")
        self.logger.info(f"{'=' * 100}\n")
        
        self.stats['nodes_branched'] += 1
        
        return left_child, right_child

    def _update_node_column_pool(self, node):
        """
        Update node's column pool with all columns from CG solver's global_solutions.

        This should be called after solving a node with CG to ensure the node
        has all generated columns in its pool.

        Args:
            node: BnPNode to update
        """
        self.logger.info(f"\n[Column Pool] Updating node {node.node_id} with generated columns...")

        initial_count = len(node.column_pool)

        # Iterate over all columns in global_solutions['x']
        # This is the authoritative source for all generated columns
        for (p, col_id) in self.cg_solver.global_solutions.get('x', {}).keys():
            # Skip if already in pool
            if (p, col_id) in node.column_pool:
                continue

            # Extract column data from global_solutions
            col_data = {
                'index': p,
                'column_id': col_id,
            }

            # Get x variables (assignments) - this is schedules_x
            # Format: {(p, j, t, itr): value}
            x_solution = self.cg_solver.global_solutions['x'][(p, col_id)]
            col_data['schedules_x'] = x_solution.copy()

            # Get LOS
            if (p, col_id) in self.cg_solver.global_solutions.get('LOS', {}):
                los_solution = self.cg_solver.global_solutions['LOS'][(p, col_id)]
                col_data['schedules_los'] = los_solution.copy()
            else:
                col_data['schedules_los'] = {}

            # Get other solution variables
            for var_name in ['y', 'z', 'S', 'l']:
                if (p, col_id) in self.cg_solver.global_solutions.get(var_name, {}):
                    col_data[f'{var_name}_data'] = self.cg_solver.global_solutions[var_name][(p, col_id)]

            if 'App' in self.cg_solver.global_solutions:
                if (p, col_id) in self.cg_solver.global_solutions['App']:
                    col_data['App_data'] = self.cg_solver.global_solutions['App'][(p, col_id)]

            # Add to column pool
            node.column_pool[(p, col_id)] = col_data

        final_count = len(node.column_pool)
        added_count = final_count - initial_count

        self.logger.info(f"[Column Pool] Updated: {initial_count} → {final_count} columns (+{added_count} new)")

        # Debug: Show some schedules_x info
        if added_count > 0:
            sample_key = list(node.column_pool.keys())[0]
            sample_col = node.column_pool[sample_key]
            self.logger.info(f"[Column Pool] Sample column {sample_key}:")
            self.logger.info(f"              schedules_x has {len(sample_col.get('schedules_x', {}))} entries")
            if sample_col.get('schedules_x'):
                first_schedule_key = list(sample_col['schedules_x'].keys())[0]
                self.logger.info(
                    f"              First entry: {first_schedule_key} = {sample_col['schedules_x'][first_schedule_key]}")

        # Show distribution
        col_per_profile = {}
        for (p, _) in node.column_pool.keys():
            col_per_profile[p] = col_per_profile.get(p, 0) + 1

        self.logger.info(f"[Column Pool] Distribution across profiles:")
        for p in sorted(col_per_profile.keys())[:5]:
            self.logger.info(f"  Profile {p}: {col_per_profile[p]} columns")
        if len(col_per_profile) > 5:
            self.logger.info(f"  ... and {len(col_per_profile) - 5} more profiles")

    def _finalize_and_print_results(self):
        """
        Finalize the Branch-and-Price solve and print results.

        Updates statistics and prints comprehensive results.
        """
        # Update total time
        self.stats['total_time'] = time.time() - self.start_time

        # Calculate final gap
        self.update_gap()

        # Print detailed results
        self._print_always("\n" + "=" * 100)
        self._print_always(" BRANCH-AND-PRICE RESULTS ".center(100, "="))
        self._print_always("=" * 100)

        # Termination status
        if not self.open_nodes:
            self.logger.info("[OK] Status: OPTIMAL (all nodes explored)")
        elif self.gap < 1e-4:
            self.logger.info(f"[OK] Status: OPTIMAL (gap < 0.01%)")
        else:
            self.logger.warning(f"⚠️  Status: INCOMPLETE (time/node limit reached)")

        # Bounds and gap
        self.logger.info("Objective Bounds:")
        self.logger.info(f"  Lower Bound (LP): {self.best_lp_bound:.6f}")
        if self.incumbent < float('inf'):
            self.logger.info(f"  Upper Bound (IP): {self.incumbent:.6f}")
            if self.gap < float('inf'):
                self.logger.info(f"  Gap:              {self.gap:.4%}")
            else:
                self.logger.info(f"  Gap:              ∞")
        else:
            self.logger.info(f"  Upper Bound (IP): None found")
            self.logger.info(f"  Gap:              ∞")

        # Node statistics
        self.logger.info("Node Statistics:")
        self.logger.info(f"  Total Nodes:      {self.stats['nodes_explored']}")
        self.logger.info(f"  Nodes Fathomed:   {self.stats['nodes_fathomed']}")
        self.logger.info(f"  Nodes Branched:   {self.stats['nodes_branched']}")
        self.logger.info(f"  Open Nodes:       {len(self.open_nodes)}")

        # Algorithm statistics
        self.logger.info("Algorithm Statistics:")
        self.logger.info(f"  Branching Strategy:   {self.branching_strategy.upper()}")
        self.logger.info(
            f"  Search Strategy:      {'Depth-First (DFS)' if self.search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
        self.logger.info(f"  Total CG Iterations:  {self.stats['total_cg_iterations']}")
        self.logger.info(f"  IP Solves:            {self.stats['ip_solves']}")
        self.logger.info(f"  Incumbent Updates:    {self.stats['incumbent_updates']}")
        self.logger.info(f"  Total Time:           {self.stats['total_time']:.2f}s")

        # Root node information
        self.logger.info("Root Node Information:")
        root = self.nodes[0]
        self.logger.info(f"  Status:           {root.status}")
        self.logger.info(f"  LP Bound:         {root.lp_bound:.6f}")
        self.logger.info(f"  Is Integral:      {root.is_integral}")
        if root.most_fractional_var:
            frac = root.most_fractional_var
            self.logger.info(f"  Most Frac Var:    {frac['var_name']} = {frac['value']:.6f}")

        # Tree structure (if nodes were explored)
        if self.stats['nodes_explored'] > 1:
            self.logger.info("Search Tree Structure:")
            self.logger.info(f"  Max Depth Reached: {max(node.depth for node in self.nodes.values())}")

            # Count nodes by status and fathom reason
            status_counts = {}
            fathom_reasons = {}

            for node in self.nodes.values():
                status_counts[node.status] = status_counts.get(node.status, 0) + 1

                if node.fathom_reason:
                    fathom_reasons[node.fathom_reason] = fathom_reasons.get(node.fathom_reason, 0) + 1

            for status, count in sorted(status_counts.items()):
                self.logger.info(f"  {status.capitalize():20}: {count}")

            if fathom_reasons:
                self.logger.info(f"\n  Fathoming Breakdown:")
                for reason, count in sorted(fathom_reasons.items()):
                    reason_display = reason.replace('_', ' ').title()
                    self.logger.info(f"    {reason_display:25}: {count}")

        # Detailed Processing Log
        if self.stats['node_processing_order']:
            # Find which node provided the incumbent
            incumbent_node_id = self._find_incumbent_node()
            
            # Build processing order string with incumbent node underlined
            order_parts = []
            for node_id in self.stats['node_processing_order']:
                if node_id == incumbent_node_id:
                    # Underline the incumbent node
                    order_parts.append(f"\033[4m{node_id}\033[0m")  # ANSI underline
                else:
                    order_parts.append(str(node_id))
            
            order_str = " -> ".join(order_parts)
            self._print_always(f"Processing Order: {order_str}")
            if incumbent_node_id in self.stats['node_processing_order']:
                self._print_always(f"                  (Node {incumbent_node_id} found the incumbent solution)\n")
            else:
                self._print_always("")

            if self.search_strategy == 'bfs' and self.stats['bfs_decision_log']:
                self._print_always("BFS Decision Breakdown:")
                for log in self.stats['bfs_decision_log']:
                    self._print_always(f"  Iteration {log['iteration']}:")

                    # Format the list of open nodes for printing
                    open_nodes_str = ", ".join(
                        [f"(Node {nid}, LP {b:.2f})" for b, nid in reversed(log['open_nodes_state'])])
                    self._print_always(f"    - Open Nodes (Ranked): [{open_nodes_str}]")
                    self._print_always(
                        f"    - Decision: Chose Node {log['chosen_node_id']} with the best LP bound of {log['chosen_node_bound']:.4f}.")
            self._print_always("-" * 100)

        # Solution quality
        if self.incumbent < float('inf') and self.incumbent_solution:
            self.logger.info("Best Solution Found:")
            self.logger.info(f"  Objective Value:  {self.incumbent:.6f}")
            self.logger.info(f"  Found at:         Node {self._find_incumbent_node()}")

            # Print some solution details if available
            if 'LOS' in self.incumbent_solution:
                los_values = [v for v in self.incumbent_solution['LOS'].values() if v > 0]
                if los_values:
                    self.logger.info(f"  Avg LOS:          {sum(los_values) / len(los_values):.2f}")
                    self.logger.info(f"  Max LOS:          {max(los_values)}")

        self.logger.info("=" * 100)

    def _find_incumbent_node(self):
        """
        Find which node produced the current incumbent.

        Returns:
            int: Node ID where incumbent was found, or 0 if unknown
        """
        # Search for integral node with matching objective
        for node_id, node in self.nodes.items():
            if node.is_integral and abs(node.lp_bound - self.incumbent) < 1e-5:
                return node_id

        # If not found in nodes, might be from heuristic
        return 0

    def visualize_tree(self, layout='hierarchical', detailed=False, academic=False,
                      save_path=None, dpi=300, **kwargs):
        """
        Visualize the Branch-and-Price search tree.

        Args:
            layout: 'hierarchical' or 'radial' (used for standard plot)
            detailed: If True, show detailed information
            academic: If True, use academic/thesis style (publication-ready)
            save_path: Path to save visualization (optional)
            dpi: Resolution for saved figure (default: 300)
            **kwargs: Additional arguments passed to plot function
                     For academic style: node_color, fathomed_color, integral_color, show_best_bound
                     For standard plot: figsize, show_bounds, show_edge_labels
        """
        from Utils.tree_visualization import BnPTreeVisualizer

        visualizer = BnPTreeVisualizer(self)

        if academic:
            visualizer.plot_academic_style(save_path=save_path, dpi=dpi, **kwargs)
        elif detailed:
            visualizer.plot_detailed(save_path=save_path, **kwargs)
        else:
            visualizer.plot(layout=layout, save_path=save_path, dpi=dpi, **kwargs)

        visualizer.print_tree_statistics()

    def export_tree_graphviz(self, filename='bnp_tree.dot'):
        """Export tree to Graphviz format."""
        from Utils.tree_visualization import BnPTreeVisualizer

        visualizer = BnPTreeVisualizer(self)
        visualizer.export_to_graphviz(filename)

    def export_tree_tikz(self, filename='bnp_tree.tex'):
        """
        Export tree as TikZ code for LaTeX integration.

        Perfect for including in thesis or papers.

        Args:
            filename: Output filename for TikZ code (default: 'bnp_tree.tex')
        """
        from Utils.tree_visualization import BnPTreeVisualizer

        visualizer = BnPTreeVisualizer(self)
        visualizer.export_tikz(filename)

    def _run_ip_heuristic(self, current_node_count):
        """
        Periodic IP heuristic: Solve RMP as IP without branching constraints.

        Based on Brunner (2010): Every N nodes, solve the RMP as IP with all
        generated columns but WITHOUT branching constraints. This enlarges the
        feasible region and may find better integer solutions.

        Args:
            current_node_count: Number of nodes explored so far

        Returns:
            bool: True if incumbent was improved
        """
        # Check if we should run the heuristic
        if self.ip_heuristic_frequency <= 0:
            return False

        if current_node_count % self.ip_heuristic_frequency != 0:
            return False

        print(f"\n{'─' * 100}")
        print(f" IP HEURISTIC (Node {current_node_count}) ".center(100, "─"))
        print(f"{'─' * 100}")
        print("Solving RMP as IP without branching constraints...")

        print("\n" + "╔" + "═" * 98 + "╗")
        print("║" + " IP HEURISTIC ".center(98) + "║")
        print("╠" + "═" * 98 + "╣")
        print(f"║ Triggered at iteration: {current_node_count}".ljust(99) + "║")
        print(f"║ Frequency: Every {self.ip_heuristic_frequency} nodes".ljust(99) + "║")
        print("╠" + "═" * 98 + "╣")
        print(f"║ 🔍 WHAT IS BEING SOLVED:".ljust(99) + "║")
        print(f"║    ➤ Uses GLOBAL master (not a specific node)".ljust(99) + "║")
        print(f"║    ➤ With ALL generated columns from entire search".ljust(99) + "║")
        print(f"║    ➤ WITHOUT any branching constraints".ljust(99) + "║")
        print(f"║    ➤ Lambda variables set to INTEGER".ljust(99) + "║")
        print("╠" + "═" * 98 + "╣")

        master = self.cg_solver.master

        # Count columns
        total_columns = len(master.lmbda)
        print(f"║ Total columns available: {total_columns}".ljust(99) + "║")
        print(f"║ Current incumbent:       {self.incumbent:.6f}".ljust(99) + "║")
        print("╠" + "═" * 98 + "╣")
        print(f"║ This is a RELAXATION (larger than any single node)".ljust(99) + "║")
        print("╚" + "═" * 98 + "╝\n")

        # Save original variable types and bounds
        original_vtypes = {}
        original_bounds = {}

        for var in master.lmbda.values():
            original_vtypes[var.VarName] = var.VType
            original_bounds[var.VarName] = (var.LB, var.UB)

            # Set to integer
            var.VType = gu.GRB.INTEGER

            # CRITICAL: Remove branching bounds to enlarge feasible region
            # This is the key difference from solving at a specific node
            var.LB = 0.0
            var.UB = gu.GRB.INFINITY

        master.Model.update()

        # Solve as IP with time limit
        master.Model.Params.OutputFlag = 0  # Silent
        master.Model.Params.TimeLimit = 60  # 1 minute time limit

        try:
            self.logger.info("  Optimizing...")
            ip_heur_start = time.time()
            master.Model.optimize()
            self.stats['time_in_ip_heuristic'] += time.time() - ip_heur_start

            improved = False

            if master.Model.status == gu.GRB.OPTIMAL:
                ip_obj = master.Model.objVal
                #master.Model.write(f'LPs/incumbent_{current_node_count}.lp')

                if ip_obj < self.incumbent - 1e-5:
                    # Found better solution!
                    old_incumbent = self.incumbent
                    self.incumbent = ip_obj
                    self.incumbent_solution = master.finalDicts(
                        self.cg_solver.global_solutions,
                        self.cg_solver.app_data, None
                    )
                    self.incumbent_node_id = 0 # Heuristic solution is not tied to a specific node
                    self.stats['integer_solutions_found'] += 1  # Track integer solution
                    self.stats['max_tree_depth'] = max(self.stats['max_tree_depth'], 0) # Heuristic is at depth 0 (global)
                    lambda_assignments = {}
                    for (p, a), var in master.lmbda.items():
                        if var.X > 1e-6:
                            lambda_assignments[(p, a)] = {'value': float(round(var.X)), 'obj': var.Obj}
                    self.incumbent_lambdas = lambda_assignments
                    self.stats['incumbent_updates'] += 1
                    # Track time to first incumbent
                    if self.stats['time_to_first_incumbent'] is None:
                        self.stats['time_to_first_incumbent'] = time.time() - self.start_time
                    self.update_gap()

                    self.logger.info(f"\n  [OK] IMPROVED INCUMBENT FOUND!")
                    self.logger.info(f"     Old incumbent: {old_incumbent:.6f}")
                    self.logger.info(f"     New incumbent: {self.incumbent:.6f}")
                    self.logger.info(f"     Improvement:   {old_incumbent - self.incumbent:.6f}")
                    self.logger.info(f"     New gap:       {self.gap:.4%}\n")
                    
                    # Save solution file
                    if master.Model.SolCount > 0:
                        sol_name = f"Node_{current_node_count}_heuristic.sol"
                        sol_path = os.path.join('sols/Integral', sol_name)
                        master.Model.write(sol_path)
                        self.logger.info(f"   [IP Save] Heuristic solution saved to {sol_path}")

                    print("\n" + "╔" + "═" * 98 + "╗")
                    print("║" + " IP HEURISTIC: IMPROVED INCUMBENT FOUND! ".center(98) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ Old Incumbent:  {old_incumbent:.6f}".ljust(99) + "║")
                    print(f"║ New Incumbent:  {self.incumbent:.6f}".ljust(99) + "║")
                    print(f"║ Improvement:    {old_incumbent - self.incumbent:.6f}".ljust(99) + "║")
                    print(f"║ New Gap:        {self.gap:.4%}".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ 🔍 CHECKING OPEN NODES FOR FATHOMING BY BOUND...".ljust(99) + "║")
                    print(f"║    Open nodes before: {len(self.open_nodes)}".ljust(99) + "║")
                    print("╚" + "═" * 98 + "╝\n")

                    # Fathom open nodes by bound
                    fathomed_count = self._fathom_by_bound()

                    print("\n" + "╔" + "═" * 98 + "╗")
                    print("║" + " FATHOMING BY BOUND RESULTS ".center(98) + "║")
                    print("╠" + "═" * 98 + "╣")
                    print(f"║ Nodes fathomed:    {fathomed_count}".ljust(99) + "║")
                    print(f"║ Open nodes after:  {len(self.open_nodes)}".ljust(99) + "║")
                    print("╠" + "═" * 98 + "╣")
                    if fathomed_count > 0:
                        print(f"║ [OK] {fathomed_count} nodes can no longer improve the incumbent".ljust(99) + "║")
                    if len(self.open_nodes) == 0:
                        print(f"║ 🎉 ALL NODES FATHOMED - OPTIMAL SOLUTION FOUND!".ljust(99) + "║")
                    print("╚" + "═" * 98 + "╝\n")

                    if fathomed_count > 0:
                        self.logger.info(f"  🔪 Fathomed {fathomed_count} open nodes by bound")

                    improved = True
                else:
                    self.logger.warning(f"  ⚠️  IP solution not improving: {ip_obj:.6f} >= {self.incumbent:.6f}")

            elif master.Model.status == gu.GRB.TIME_LIMIT:
                if master.Model.SolCount > 0:
                    ip_obj = master.Model.objVal
                    self.logger.info(f"  ⏱️  Time limit reached, best solution: {ip_obj:.6f}")

                    if ip_obj < self.incumbent - 1e-5:
                        old_incumbent = self.incumbent
                        self.incumbent = ip_obj
                        self.incumbent_solution = master.finalDicts(
                            self.cg_solver.global_solutions,
                            self.cg_solver.app_data, None
                        )
                        lambda_assignments = {}
                        for (p, a), var in master.lmbda.items():
                            if var.X > 1e-6:
                                lambda_assignments[(p, a)] = {'value': float(round(var.X)), 'obj': var.Obj}
                        self.incumbent_lambdas = lambda_assignments
                        
                        # Save solution file
                        if master.Model.SolCount > 0:
                            sol_name = f"Node_{current_node_count}_heuristic_timeout.sol"
                            sol_path = os.path.join('sols/Integral', sol_name)
                            master.Model.write(sol_path)
                            self.logger.info(f"   [IP Save] Heuristic solution (Time Limit) saved to {sol_path}")
                        self.stats['incumbent_updates'] += 1
                        self.update_gap()

                        self.logger.info(f"     Updated incumbent: {old_incumbent:.6f} → {self.incumbent:.6f}")

                        fathomed_count = self._fathom_by_bound()
                        if fathomed_count > 0:
                            self.logger.info(f"  🔪 Fathomed {fathomed_count} open nodes")

                        improved = True
                else:
                    self.logger.warning(f"  ⚠️  Time limit, no feasible solution found")
            else:
                self.logger.error(f"  ❌ IP solve unsuccessful (status={master.Model.status})")

        except Exception as e:
            self.logger.error(f"  ❌ Error during IP heuristic: {e}")
            
            # DEBUG MODE: Re-raise exception for debugging
            if self.debug_mode:
                self.logger.error("🐛 DEBUG_MODE enabled - re-raising exception!")
                # Still execute finally block to restore variables
                raise
            
            improved = False

        finally:
            # Restore original variable types and bounds
            for var in master.lmbda.values():
                var.VType = original_vtypes[var.VarName]
                var.LB = original_bounds[var.VarName][0]
                var.UB = original_bounds[var.VarName][1]

            master.Model.Params.OutputFlag = 0
            master.Model.Params.TimeLimit = gu.GRB.INFINITY
            master.Model.update()

        self.stats['ip_solves'] += 1
        self.logger.info(f"{'─' * 100}\n")

        return improved

    def _fathom_by_bound(self):
        """
        Fathom all open nodes whose LP bound is >= incumbent.

        Returns:
            int: Number of nodes fathomed
        """
        fathomed_count = 0
        nodes_to_remove = []

        # The items in open_nodes depend on the strategy
        if self.search_strategy == 'bfs':
            nodes_to_check = [node_id for _, node_id in self.open_nodes]
        else:  # dfs
            nodes_to_check = self.open_nodes

        print(f"\n  Checking {len(nodes_to_check)} open nodes for fathoming...")
        print(f"  Current incumbent: {self.incumbent:.6f}\n")

        for node_id in nodes_to_check:
            node = self.nodes[node_id]

            # Check if node's LP bound is worse than incumbent
            if node.lp_bound >= self.incumbent - 1e-5:
                node.status = 'fathomed'
                node.fathom_reason = 'bound_after_heuristic'
                self.stats['nodes_fathomed'] += 1
                nodes_to_remove.append(node_id)
                fathomed_count += 1

                gap_to_inc = node.lp_bound - self.incumbent
                print(f"  ❌ FATHOM Node {node_id}: LP={node.lp_bound:.6f} >= Inc={self.incumbent:.6f} (gap: +{gap_to_inc:.6f})")
                self.logger.info(f"     Fathomed node {node_id}: LP={node.lp_bound:.6f} >= Inc={self.incumbent:.6f}")
            else:
                gap_to_inc = self.incumbent - node.lp_bound
                print(f"  ✓ KEEP Node {node_id}: LP={node.lp_bound:.6f} < Inc={self.incumbent:.6f} (can improve by {gap_to_inc:.6f})")

        # Remove from open nodes
        if nodes_to_remove:
            if self.search_strategy == 'bfs':
                self.open_nodes = [(b, i) for b, i in self.open_nodes if i not in nodes_to_remove]
            else:  # dfs
                self.open_nodes = [i for i in self.open_nodes if i not in nodes_to_remove]

        return fathomed_count

    def _get_branching_profile(self, node):
        """
        Extract the branching profile from node's constraints.

        Returns:
            int or None: Profile that was branched on (n), or None if root
        """
        if not node.branching_constraints:
            return None

        # Get the most recent branching constraint (last one added)
        last_constraint = node.branching_constraints[-1]

        # Both SP and MP branching have a 'profile' attribute
        if hasattr(last_constraint, 'profile'):
            return last_constraint.profile

        return None

    # In branch_and_price.py

    def _check_and_fathom_parents(self, node_id):
        """
        Check if parent nodes can be fathomed after a child is fathomed.

        A parent can be fathomed if ALL its children are fathomed.
        This recursively propagates up the tree.

        Args:
            node_id: ID of the node that was just fathomed
        """
        node = self.nodes[node_id]

        if node.parent_id is None:
            return  # Root node has no parent

        parent = self.nodes[node.parent_id]

        # Only check if parent is 'branched' (not already fathomed)
        if parent.status != 'branched':
            return

        # Find all children of the parent
        children = [n for n in self.nodes.values() if n.parent_id == parent.node_id]

        # Check if ALL children are fathomed
        all_children_fathomed = all(child.status == 'fathomed' for child in children)

        if all_children_fathomed:
            # Determine best bound among children for fathom reason
            child_bounds = [child.lp_bound for child in children]
            best_child_bound = min(child_bounds) if child_bounds else float('inf')

            # Fathom parent
            parent.status = 'fathomed'
            parent.fathom_reason = 'all_children_fathomed'
            parent.lp_bound = best_child_bound  # Update with best child bound

            self.logger.info(f"\n  [OK] Parent Node {parent.node_id} fathomed: All children fathomed")
            self.logger.info(f"     Children: {[c.node_id for c in children]}")
            self.logger.info(f"     Best child bound: {best_child_bound:.6f}")

            # Recursively check grandparent
            self._check_and_fathom_parents(parent.node_id)

    # In branch_and_price.py

    def extract_optimal_schedules(self, include_all_patients=True):
        """
        Extract optimal schedules from the incumbent solution.

        Disaggregates profile-level solution to individual patient schedules.
        Based on Paper: "ex-post disaggregation step reconstructs recipient-level
        assignments from the profile-based solution"

        Args:
            include_all_patients: If True, include P_Pre and P_Post patients

        Returns:
            dict: {
                'patient_schedules': {patient_id: schedule_info},
                'objective_value': float,
                'total_los': int,
                'utilization': dict
            }
        """
        if self.incumbent_solution is None:
            self.logger.error("No incumbent solution available!")
            return None

        if self.verbose:
            print("\n" + "=" * 100)
            print(" EXTRACTING OPTIMAL SCHEDULES ".center(100, "="))
            print("=" * 100)

        # Find the node with the incumbent solution
        incumbent_node = self._find_incumbent_node()
        if incumbent_node == 0:
            incumbent_node = self._get_best_integral_node()

        node = self.nodes[incumbent_node]
        lambda_assignments = self.incumbent_lambdas

        # Filter out zero values and extract value
        if lambda_assignments:
            lambda_assignments = {k: v['value'] for k, v in lambda_assignments.items() if v['value'] > 1e-6}

        if self.verbose:
            print(f"\nExtracting from Node {incumbent_node}")
            print(f"  Objective Value: {self.incumbent:.6f}")
            print(f"  Status: {node.status}")
            print(f"  Assignments: {lambda_assignments}")

        # Disaggregate to individual patients
        patient_schedules = {}
        profile_counters = {}

        for (profile, col_id), count in sorted(lambda_assignments.items()):
            if self.verbose:
                print(f"\n  Profile {profile}, Column {col_id}: {count} patients")

            # Get schedule from column pool
            if (profile, col_id) not in node.column_pool:
                self.logger.warning(f"    ⚠️  Column ({profile},{col_id}) not in pool!")
                continue

            col_data = node.column_pool[(profile, col_id)]

            # Extract schedule information
            schedules_x = col_data.get('schedules_x', {})
            schedules_los = col_data.get('schedules_los', {})

            # Get LOS
            los_value = list(schedules_los.values())[0] if schedules_los else 0

            # Disaggregate: Assign this schedule to 'count' patients
            if profile not in profile_counters:
                profile_counters[profile] = 0

            for i in range(int(round(count))):
                # Create unique patient ID
                patient_id = f"P{profile}_{profile_counters[profile]}"
                profile_counters[profile] += 1

                # Extract therapist assignment
                assigned_therapist = None
                for (p, j, t, a), val in schedules_x.items():
                    if p == profile and val > 0.5:
                        assigned_therapist = j
                        break

                # Extract daily schedule
                daily_schedule = {}
                for (p, j, t, a), val in schedules_x.items():
                    if p == profile and val > 0.5:
                        if t not in daily_schedule:
                            daily_schedule[t] = []
                        daily_schedule[t].append({
                            'therapist': j,
                            'session': 'human'
                        })

                # Check for AI sessions
                if 'y_data' in col_data:
                    y_data = col_data['y_data']
                    for (p, d, _), val in y_data.items():
                        if p == profile and val > 0.5:
                            if d not in daily_schedule:
                                daily_schedule[d] = []
                            daily_schedule[d].append({
                                'therapist': None,
                                'session': 'AI'
                            })

                # Store patient schedule
                patient_schedules[patient_id] = {
                    'profile': profile,
                    'column': col_id,
                    'therapist': assigned_therapist,
                    'los': los_value,
                    'entry_day': self.cg_solver.Entry_agg[profile],
                    'required_sessions': self.cg_solver.Req_agg[profile],
                    'daily_schedule': daily_schedule,
                    'total_sessions': sum(len(sessions) for sessions in daily_schedule.values())
                }

                self.logger.info(f"    Patient {patient_id}: Therapist {assigned_therapist}, "
                                 f"LOS={los_value}, Sessions={patient_schedules[patient_id]['total_sessions']}")

        # Calculate statistics
        total_los = sum(s['los'] for s in patient_schedules.values())
        avg_los = total_los / len(patient_schedules) if patient_schedules else 0

        # Therapist utilization
        therapist_workload = {}
        for patient_info in patient_schedules.values():
            for day, sessions in patient_info['daily_schedule'].items():
                for session in sessions:
                    if session['therapist'] is not None:
                        t = session['therapist']
                        if t not in therapist_workload:
                            therapist_workload[t] = {}
                        if day not in therapist_workload[t]:
                            therapist_workload[t][day] = 0
                        therapist_workload[t][day] += 1

        # Summary
        self.logger.info("\n" + "=" * 100)
        self.logger.info(" OPTIMAL SOLUTION SUMMARY ".center(100, "="))
        self.logger.info("=" * 100)
        self.logger.info(f"\nPatients:")
        self.logger.info(f"  Total Patients: {len(patient_schedules)}")
        self.logger.info(f"  Total LOS: {total_los}")
        self.logger.info(f"  Average LOS: {avg_los:.2f}")

        self.logger.info(f"\nProfiles:")
        for profile, count in sorted(profile_counters.items()):
            self.logger.info(f"  Profile {profile}: {count} patients")

        self.logger.info(f"\nTherapist Utilization:")
        for t in sorted(therapist_workload.keys()):
            total_sessions = sum(therapist_workload[t].values())
            days_worked = len(therapist_workload[t])
            avg_daily = total_sessions / days_worked if days_worked > 0 else 0
            self.logger.info(f"  Therapist {t}: {total_sessions} sessions over {days_worked} days "
                             f"(avg {avg_daily:.1f}/day)")

        self.logger.info("=" * 100)

        if self.verbose:
            print(f"\nActive columns: {len(lambda_assignments)}")

            print(f"Total expected patients (Nr_agg): {sum(self.cg_solver.Nr_agg[k] for k in sorted(self.cg_solver.P_F + self.cg_solver.P_Post))}")
            print(f"Profiles in lambda_assignments: {set(p for p, _ in lambda_assignments.keys())}")
            print(f"P_Focus: {self.cg_solver.P_F}")
            print(f"P_Post: {self.cg_solver.P_Post}")
            print(f"P_Join: {self.cg_solver.P_Join}")

        return {
            'patient_schedules': patient_schedules,
            'objective_value': self.incumbent,
            'total_los': total_los,
            'avg_los': avg_los,
            'therapist_utilization': therapist_workload,
            'profile_distribution': profile_counters
        }

    def _get_best_integral_node(self):
        """
        Find the best integral node (lowest bound among integral nodes).

        Returns:
            int: Node ID of best integral node
        """
        integral_nodes = [
            (node.lp_bound, node.node_id)
            for node in self.nodes.values()
            if node.is_integral
        ]

        if not integral_nodes:
            return 0  # Return root if no integral node found

        # Return node with lowest bound
        return min(integral_nodes)[1]

    def print_detailed_schedule(self, patient_id, schedule_info):
        """
        Print a detailed schedule for a specific patient.

        Args:
            patient_id: Patient identifier
            schedule_info: Schedule information from extract_optimal_schedules
        """
        print("\n" + "=" * 80)
        print(f" SCHEDULE FOR {patient_id} ".center(80, "="))
        print("=" * 80)

        print(f"\nPatient Information:")
        print(f"  Profile: {schedule_info['profile']}")
        print(f"  Assigned Therapist: {schedule_info['therapist']}")
        print(f"  Entry Day: {schedule_info['entry_day']}")
        print(f"  Length of Stay: {schedule_info['los']} days")
        print(f"  Required Sessions: {schedule_info['required_sessions']}")
        print(f"  Total Sessions: {schedule_info['total_sessions']}")

        print(f"\nDaily Schedule:")
        print(f"  {'Day':<8} {'Therapist':<12} {'Type':<10}")
        print("  " + "-" * 40)

        for day in sorted(schedule_info['daily_schedule'].keys()):
            sessions = schedule_info['daily_schedule'][day]
            for session in sessions:
                therapist = session['therapist'] if session['therapist'] else "N/A"
                session_type = session['session']
                print(f"  {day:<8} {therapist:<12} {session_type:<10}")

        print("=" * 80)

    def export_schedules_to_csv(self, filename='optimal_schedules.csv'):
        """
        Export optimal schedules to CSV file.

        Args:
            filename: Output filename
        """
        import pandas as pd

        schedules = self.extract_optimal_schedules()
        if not schedules:
            return

        # Create rows for CSV
        rows = []
        for patient_id, info in schedules['patient_schedules'].items():
            for day, sessions in info['daily_schedule'].items():
                for session in sessions:
                    rows.append({
                        'patient_id': patient_id,
                        'profile': info['profile'],
                        'day': day,
                        'therapist': session['therapist'] if session['therapist'] else 'AI',
                        'session_type': session['session'],
                        'los': info['los'],
                        'entry_day': info['entry_day']
                    })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

        self.logger.info(f"\n[OK] Schedules exported to {filename}")
        self.logger.info(f"   Total rows: {len(df)}")


# ============================================================================
# WORKER THREAD CLASS FOR PARALLEL TREE EXPLORATION
# ============================================================================

class _NodeWorker:
    """
    Worker thread for processing nodes in parallel.
    
    Each worker continuously fetches nodes from the shared queue,
    solves them, checks for fathoming, and branches if necessary.
    """
    
    def __init__(self, worker_id, bnp_instance, shared_state):
        """
        Initialize worker.
        
        Args:
            worker_id: Unique worker identifier (0, 1, 2, ...)
            bnp_instance: Reference to BranchAndPrice instance
            shared_state: ThreadSafeSharedState instance
        """
        self.worker_id = worker_id
        self.bnp = bnp_instance
        self.shared = shared_state
        self.nodes_processed = 0
        
    def run(self):
        """Main worker loop."""
        logger = logging.getLogger(f"{__name__}.Worker{self.worker_id}")
        logger.info(f"Worker {self.worker_id} started")
        
        while self.shared.should_continue():
            # Get next node from queue
            node_id = self.shared.get_next_node()
            if node_id is None:
                # No more nodes or shutdown requested
                break
            
            try:
                # Process the node
                self._process_node(node_id, logger)
                self.nodes_processed += 1
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error on node {node_id}: {e}")
                if self.bnp.debug_mode:
                    raise
                # Mark node as fathomed due to error
                with self.shared.lock:
                    node = self.shared.nodes[node_id]
                    node.status = 'fathomed'
                    node.fathom_reason = 'error'
                    self.shared.increment_stat('nodes_fathomed')
        
        logger.info(f"Worker {self.worker_id} finished ({self.nodes_processed} nodes processed)")
    
    def _process_node(self, node_id, logger):
        """Process a single node: solve, fathom check, branch if needed."""
        node = self.shared.nodes[node_id]
        
        logger.info(f"[W{self.worker_id}] Processing Node {node_id} (depth={node.depth}, path='{node.path}')")
        
        # Solve node with Column Generation
        lp_bound, is_integral, most_frac_info, node_lambdas = self.bnp.solve_node_with_cg(
            node, max_cg_iterations=50
        )
        
        # Update shared statistics
        self.shared.increment_stat('nodes_explored')
        self.shared.add_to_stat_list('node_processing_order', node_id)
        
        # Update global lower bound
        self.shared.update_best_lb(lp_bound)
        
        logger.info(f"[W{self.worker_id}] Node {node_id} solved: LP={lp_bound:.6f}, Integral={is_integral}")
        
        # Check fathoming
        should_fathom = self._check_fathoming(node, lp_bound, is_integral, node_lambdas, logger)
        
        if should_fathom:
            self.shared.increment_stat('nodes_fathomed')
            logger.info(f"[W{self.worker_id}] Node {node_id} fathomed: {node.fathom_reason}")
            return
        
        # Not fathomed - need to branch
        logger.info(f"[W{self.worker_id}] Node {node_id} requires branching")
        self._branch_node(node, node_lambdas, logger)
        
    def _check_fathoming(self, node, lp_bound, is_integral, node_lambdas, logger):
        """
        Thread-safe fathoming check.
        
        Returns True if node should be fathomed.
        """
        # Check 1: Integral solution
        if is_integral:
            node.status = 'fathomed'
            node.fathom_reason = 'integral'
            
            # Try to update incumbent (thread-safe)
            if lp_bound < self.shared.incumbent:
                solution = self.bnp.cg_solver.master.finalDicts(
                    self.bnp.cg_solver.global_solutions,
                    self.bnp.cg_solver.app_data,
                    node_lambdas
                )
                lambdas_active = {k: v for k, v in node_lambdas.items() if v.get('value', 0) > 1e-6}
                
                updated = self.shared.try_update_incumbent(
                    lp_bound, solution, lambdas_active, node.node_id,
                    time_elapsed=time.time() - self.bnp.start_time
                )
                if updated:
                    logger.info(f"[W{self.worker_id}] [OK] NEW INCUMBENT from Node {node.node_id}: {lp_bound:.6f}")
            
            return True
        
        # Check 2: Infeasible
        if lp_bound == float('inf'):
            node.status = 'fathomed'
            node.fathom_reason = 'infeasible'
            return True
        
        # Check 3: Bound worse than incumbent (thread-safe read)
        current_incumbent = self.shared.incumbent
        if lp_bound >= current_incumbent - 1e-5:
            node.status = 'fathomed'
            node.fathom_reason = 'bound'
            return True
        
        # Cannot fathom
        return False
    
    def _branch_node(self, node, node_lambdas, logger):
        """
        Branch on node and add children to shared queue (thread-safe).
        """
        # Select branching candidate
        branching_type, branching_info = self.bnp.select_branching_candidate(node, node_lambdas)
        
        if not branching_type:
            logger.error(f"[W{self.worker_id}] No branching candidate for Node {node.node_id}!")
            node.status = 'fathomed'
            node.fathom_reason = 'no_branching_candidate'
            self.shared.increment_stat('nodes_fathomed')
            return
        
        # Create children (this accesses shared state via self.bnp.node_counter)
        # Need to lock node creation to avoid race conditions
        import time
        branching_start = time.time()
        
        with self.shared.lock:
            # Branch based on type
            if branching_type == 'mp':
                left_child, right_child = self.bnp.branch_on_mp_variable(node, branching_info)
            else:  # 'sp'
                left_child, right_child = self.bnp.branch_on_sp_pattern(node, branching_info)
                
                # Track pattern size
                pattern_size = branching_info.get('pattern_size', 1)
                self.shared.update_stat_dict('pattern_size_counts', pattern_size,
                                            self.shared.stats.get('pattern_size_counts', {}).get(pattern_size, 0) + 1)
            
            # Mark parent as branched
            node.status = 'branched'
            self.shared.increment_stat('nodes_branched')
            
            # Record branching time
            self.shared.add_timing('time_in_branching', time.time() - branching_start)
        
        logger.info(f"[W{self.worker_id}] Created children: Node {left_child.node_id} (L) & Node {right_child.node_id} (R)")
        
        # Solve both children immediately (current approach - eager solving)
        left_is_active, left_bound, left_lambdas = self._solve_and_evaluate_child(left_child, logger)
        right_is_active, right_bound, right_lambdas = self._solve_and_evaluate_child(right_child, logger)
        
        # Add active children to queue
        nodes_to_add = []
        if self.shared.search_strategy == 'bfs':
            if left_is_active:
                nodes_to_add.append((left_bound, left_child.node_id))
            if right_is_active:
                nodes_to_add.append((right_bound, right_child.node_id))
        else:  # DFS
            # Add right first, then left (so left is processed first in LIFO)
            if right_is_active:
                nodes_to_add.append(right_child.node_id)
            if left_is_active:
                nodes_to_add.append(left_child.node_id)
        
        if nodes_to_add:
            self.shared.add_nodes(nodes_to_add)
            logger.info(f"[W{self.worker_id}] Added {len(nodes_to_add)} child nodes to queue")
    
    def _solve_and_evaluate_child(self, child_node, logger):
        """
        Solve child node immediately and check if it should be added to queue.
        
        Returns:
            tuple: (is_active, lp_bound, node_lambdas)
        """
        logger.info(f"[W{self.worker_id}] Solving child Node {child_node.node_id}...")
        
        try:
            # Solve with CG
            lp_bound, is_integral, most_frac_info, node_lambdas = self.bnp.solve_node_with_cg(
                child_node, max_cg_iterations=50
            )
            
            # Update stats
            self.shared.increment_stat('nodes_explored')
            self.shared.add_to_stat_list('node_processing_order', child_node.node_id)
            self.shared.update_best_lb(lp_bound)
            
            # Check fathoming
            if self._check_fathoming(child_node, lp_bound, is_integral, node_lambdas, logger):
                self.shared.increment_stat('nodes_fathomed')
                logger.info(f"[W{self.worker_id}] Child Node {child_node.node_id} fathomed immediately")
                return False, lp_bound, node_lambdas
            
            # Not fathomed - mark as solved and active
            child_node.status = 'solved'
            logger.info(f"[W{self.worker_id}] Child Node {child_node.node_id} active (LP={lp_bound:.6f})")
            return True, lp_bound, node_lambdas
            
        except Exception as e:
            logger.error(f"[W{self.worker_id}] Error solving child {child_node.node_id}: {e}")
            if self.bnp.debug_mode:
                raise
            child_node.status = 'fathomed'
            child_node.fathom_reason = 'error'
            self.shared.increment_stat('nodes_fathomed')
            return False, float('inf'), {}
