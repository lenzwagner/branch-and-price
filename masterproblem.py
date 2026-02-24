import gurobipy as gu
from Utils.Generell.utils import *

class MasterProblem_d:
    def __init__(self, df, T_Max, Nr_agg, Req, pre_x, E_dict, verbose=False, deterministic=False, use_warmstart=True):
        self.P_Full = df['P_Full'].dropna().astype(int).unique().tolist()
        self.P_Pre = df['P_Pre'].dropna().astype(int).unique().tolist()
        self.P_Join = df['P_Join'].dropna().astype(int).unique().tolist()
        self.P_Focus = df['P_F'].dropna().astype(int).unique().tolist()
        self.P_Post = df['P_Post'].dropna().astype(int).unique().tolist()
        self.D = df['D_Ext'].dropna().astype(int).unique().tolist()
        self.T = df['T'].dropna().astype(int).unique().tolist()
        self.G = df['G'].dropna().astype(int).unique().tolist()
        self.A = [1]
        self._solve_counter = 0

        # Create Gurobi environment with suppressed output
        env = gu.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        self.Model = gu.Model("MasterProblem", env=env)
        self.Model.Params.Seed = 0  #
        self.deterministic = deterministic

        # Apply deterministic settings if requested
        if self.deterministic:
            self.Model.Params.Threads = 1  # Single thread for determinism
            self.Model.Params.Method = 2   # Barrier method (more deterministic than dual simplex)

        self.cons_p_max = {}
        self.cons_los = {}
        self.E = E_dict
        self.cons_lmbda = {}
        self.all_schedules = {}
        self.all_los = {}
        self.Nr = Nr_agg
        self.Req = Req
        self.output_len = 100
        self.T_max = T_Max
        self.pre_x = pre_x
        self.drop_threshold = 5
        self.column_pool = {}
        self.branching_bounds = {}
        self.all_patterns = {} # Stores path_pattern for reconstruction
        self.aggregated = defaultdict(int)
        self.verbose = verbose
        self.use_warmstart = use_warmstart
        for (p, t, d), value in self.pre_x.items():
            self.aggregated[(t, d)] += value

    def buildModel(self):
        self.genVars()
        self.genCons()
        self.genObj()
        self.Model.update()

    def genVars(self):
        self.lmbda = self.Model.addVars(self.P_Join, self.A, vtype=gu.GRB.INTEGER, name='lmbda')

    def genCons(self):
        for p in self.P_Join:
            self.cons_lmbda[p] = self.Model.addConstr(gu.quicksum(self.lmbda[p, a] for a in self.A) == self.Nr[p], name=f"lambda({p})")

        for t in self.T:
            for d in self.D:
                pre_load = sum(self.pre_x.get((p, t, d), 0) for p in self.P_Pre)
                self.cons_p_max[t, d] = self.Model.addConstr(gu.quicksum(self.lmbda[p, a] * self.all_schedules.get((p, t, d, a), 0) for p in self.P_Join for a in self.A) + pre_load <= self.T_max[t, d], name=f"p_max({t},{d})")


    def genObj(self):
        self.Model.setObjective(gu.quicksum(self.lmbda[p, a] for p in self.P_Focus for a in self.A), sense=gu.GRB.MINIMIZE)

    def getDuals(self):
        return {(t, d): self.cons_p_max[t, d].Pi for t in self.T for d in self.D}, {p: self.cons_lmbda[p].Pi for p in self.P_Join}

    def addSchedule(self, schedule):
        self.all_schedules.update(schedule)

    def addLOS(self, schedule):
        self.all_los.update(schedule)

    def startSol(self, schedule_x = None, schedule_los = None):
        for p in self.P_Join:
            if self.E[p] == 0:
                self.lmbda[p, 1].Obj = 0
            else:
                self.lmbda[p, 1].Obj = schedule_los.get(p, 0)
            for t, d in self.cons_p_max:
                value = schedule_x.get((p, t, d, 1), 0)
                self.Model.chgCoeff(self.cons_p_max[t, d], self.lmbda[p, 1], value)
        self.Model.update()

    def addLambdaVar(self, p, a, col, coef, pattern=None):
        new_col = gu.Column(col, self.Model.getConstrs())
        self.lmbda[p, a] = self.Model.addVar(obj=coef[0], vtype=gu.GRB.INTEGER, column=new_col, name=f"lmbda[{p},{a}]")
        self.A.append(a)
        if pattern is not None:
            self.all_patterns[(p, a)] = pattern
        self.Model.update()

    def finSol(self):
        all_integer, obj, most_frac_info = self.check_fractionality()
        self.Model.Params.OutputFlag = 1 if self.verbose else 0
        for var in self.lmbda.values():
            var.VType = gu.GRB.INTEGER
        self.Model.optimize()

        if self.Model.status == gu.GRB.INFEASIBLE:
            if self.verbose:
                print('\nThe following constraints and variables are in the IIS:')
                self.Model.computeIIS()
                for c in self.Model.getConstrs():
                    if c.IISConstr: print(f'\t{c.constrName}: {self.Model.getRow(c)} {c.Sense} {c.RHS}')
                for v in self.Model.getVars():
                    if v.IISLB: print(f'\t{v.varName} ≥ {v.LB}')
                    if v.IISUB: print(f'\t{v.varName} ≤ {v.UB}')

        return all_integer, obj, most_frac_info

    def solRelModel(self):
        self.Model.Params.OutputFlag = 0 if self.verbose else 0
        self._solve_counter += 1

        # Warm-Start Optimization (optional):
        # - First solve: Barrier (Method=2) - good for fresh LPs
        # - Subsequent solves: Dual Simplex (Method=1) - leverages basis warm-start
        if self._solve_counter == 1 and self.verbose:
            print(f"    [RMP] Warm-Start: {'ENABLED' if self.use_warmstart else 'DISABLED'}")
        
        if self.use_warmstart:
            if self._solve_counter == 1:
                self.Model.Params.Method = 2  # Barrier for first solve
            else:
                self.Model.Params.Method = 1  # Dual Simplex for warm-start
                self.Model.Params.LPWarmStart = 2  # Use provided start vectors
        else:
            self.Model.Params.Method = 2  # Always Barrier if warm-start disabled

        # Relax to LP and restore branching bounds
        for var in self.Model.getVars():
            var.VType = gu.GRB.CONTINUOUS

            # Restore branching bounds if they exist
            var_name = var.VarName
            if var_name in self.branching_bounds:
                var.LB = self.branching_bounds[var_name]['lb']
                var.UB = self.branching_bounds[var_name]['ub']
                if self._solve_counter == 2 and self.verbose:
                    print(f"    [Branching Bound] Restored {var_name}: "
                          f"LB={var.LB}, UB={var.UB}")
            else:
                var.LB = 0.0

        self.Model.optimize()
        if self.Model.status != gu.GRB.OPTIMAL:
            if self.verbose:
                print('\nThe following constraints and variables are in the IIS:')
                self.Model.computeIIS()
                for c in self.Model.getConstrs():
                    if c.IISConstr: print(f'\t{c.constrName}: {self.Model.getRow(c)} {c.Sense} {c.RHS}')
                for v in self.Model.getVars():
                    if v.IISLB: print(f'\t{v.varName} ≥ {v.LB}')
                    if v.IISUB: print(f'\t{v.varName} ≤ {v.UB}')

    def check_fractionality(self):
        """
        Check if all lambda variables are integer and find the most fractional solution.
        Tie-break: (1) smallest n, (2) smallest a

        Returns:
            tuple: (all_integer: bool, obj_val: float, most_frac_info: dict or None)
        """
        import math

        self.solRelModel()
        obj = self.Model.ObjVal

        all_integer = True
        max_fractionality = 0.0
        most_frac_info = None

        # Check all lambda variables

        for (n, a), var in self.lmbda.items():
            x_val = var.X

            # Calculate distances to floor and ceil
            floor_val = math.floor(x_val)
            ceil_val = math.ceil(x_val)
            dist_to_floor = x_val - floor_val
            dist_to_ceil = ceil_val - x_val

            frac_part = min(dist_to_floor, dist_to_ceil)

            if frac_part > 1e-8:
                all_integer = False

                is_new_most_frac = False

                if frac_part > max_fractionality + 1e-10:
                    is_new_most_frac = True
                elif abs(frac_part - max_fractionality) < 1e-10:
                    # Equal fractionality - apply tie-break
                    if most_frac_info is not None:
                        if n < most_frac_info['n']:
                            is_new_most_frac = True
                        elif n == most_frac_info['n'] and a < most_frac_info['a']:
                            is_new_most_frac = True
                    else:
                        is_new_most_frac = True

                if is_new_most_frac:
                    max_fractionality = frac_part
                    most_frac_info = {
                        'n': n,
                        'a': a,
                        'value': x_val,
                        'fractionality': frac_part,
                        'floor': floor_val,
                        'ceil': ceil_val,
                        'dist_to_floor': dist_to_floor,
                        'dist_to_ceil': dist_to_ceil,
                        'var_name': var.VarName
                    }

        # Print results
        if self.verbose:
            if all_integer:
                print("All lambda variables are integer.")
            else:
                print(f"Fractional solution detected!")
            if most_frac_info:
                print(f"\nMost fractional variable (tie-break: smallest n, then smallest a):")
                print(f"  Variable: lmbda[n={most_frac_info['n']}, a={most_frac_info['a']}]")
                print(f"  Value (X): {most_frac_info['value']:.6f}")
                print(f"  Fractionality: {most_frac_info['fractionality']:.6f}")
                print(f"  Floor: {most_frac_info['floor']}, Distance to floor: {most_frac_info['dist_to_floor']:.6f}")
                print(f"  Ceil:  {most_frac_info['ceil']}, Distance to ceil:  {most_frac_info['dist_to_ceil']:.6f}")

        return all_integer, obj, most_frac_info

    def finalDicts(self, sols_dict, app_data, lambda_list_cg = None):
        """
        Determine final dicts

        Args:
            sols_dict: Solution dict
            app_data: App_data
            lambda_list_cg: List with lambda variables
        """
        if lambda_list_cg is None:
            active_keys = []
            models = [self.Model]
            for model in models:
                for v in model.getVars():
                    if 'lmbda' in v.VarName and v.X > 0:
                        parts = v.VarName.split('[')[1].split(']')[0].split(',')
                        p = int(parts[0])
                        s = int(parts[1])
                        if p in self.P_Join:
                            solution_key = (p, s)
                            active_keys.append(solution_key)


        else:
            active_keys = [k for k, v in lambda_list_cg.items() if (v['value'] if isinstance(v, dict) else v) > 1e-6]

        if isinstance(app_data, (int, float)):
            active_solutions = {'x': {}, 'LOS': {}, 'y': {}, 'z': {}, 'S': {}, 'l': {}}
        else:
            active_solutions = {'x': {}, 'LOS': {}, 'y': {}, 'z': {}, 'App': {}, 'S': {}, 'l': {}}
        for key in active_keys:
            if key in sols_dict['x']:
                active_solutions['x'].update(sols_dict['x'].get(key, {}))
                active_solutions['LOS'].update(sols_dict['LOS'].get(key, {}))
                active_solutions['y'].update(sols_dict['y'].get(key, {}))
                active_solutions['z'].update(sols_dict['z'].get(key, {}))
                active_solutions['S'].update(sols_dict['S'].get(key, {}))
                active_solutions['l'].update(sols_dict['l'].get(key, {}))
                if not isinstance(app_data, (int, float)):
                    active_solutions['App'].update(sols_dict['App'].get(key, {}))
            
            # Reconstruct variables from pattern (AI usage y and Gaps)
            # This is critical for Labeling algorithm where y might not be in global_solutions
            if key in self.all_patterns and self.all_patterns[key] is not None:
                pat_info = self.all_patterns[key]
                p, col_id = key
                
                # Check format (dict vs list) - we standardized on dict in recent steps
                if isinstance(pat_info, dict):
                    pattern_vals = pat_info.get('path')
                    start_t = pat_info.get('start')
                else:
                    # Fallback or legacy (full list pattern without start?)
                    # If we don't have start, we can't map to time.
                    pattern_vals = None 
                    start_t = None
                
                if start_t is not None and pattern_vals is not None:
                    for i, val in enumerate(pattern_vals):
                        current_time = start_t + i
                        
                        # Interpretation:
                        # 0: AI Session -> y=1 (and x=0)
                        # 1: Human Session -> x=1 (already handled by schedules_x)
                        # 2: Gap -> y=0, x=0
                        
                        if val == 0: # AI
                            # Add to y solutions: (p, time, col_id) -> 1
                            # Match Subproblem.getVarSol format: (p, d, itr) -> val
                            active_solutions['y'][(p, current_time, col_id)] = 1.0
                            
        return active_solutions

    def set_branching_bound(self, var, bound_type, value):
        """
        Set a branching bound that will be preserved during LP relaxation.

        Args:
            var: Gurobi variable
            bound_type: 'lb' or 'ub'
            value: Bound value
        """
        var_name = var.VarName

        if var_name not in self.branching_bounds:
            self.branching_bounds[var_name] = {'lb': 0.0, 'ub': float('inf')}

        self.branching_bounds[var_name][bound_type] = value

        # Set the bound on the variable
        if bound_type == 'lb':
            var.LB = value
        else:
            var.UB = value

        print(f"    [Branching Bound] Set {var_name}.{bound_type.upper()} = {value}")