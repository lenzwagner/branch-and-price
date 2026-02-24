# Branch-and-Price for Hybrid Scheduling with Learning Effects

A sophisticated implementation of a Branch-and-Price algorithm for solving hybrid scheduling problems with learning curve effects in healthcare therapy scheduling.

## Overview

This project implements a Branch-and-Price framework that combines Column Generation with tree search to solve a complex scheduling problem where:
- Patients require multiple therapy sessions over time
- Therapists have limited capacity
- Learning effects improve therapy effectiveness over time
- Both human therapists and AI-assisted sessions are available
- Length of stay (LOS) optimization is critical

The implementation supports both classical Column Generation (root node only) and full Branch-and-Price tree search with multiple branching strategies.

## Key Features

### Core Algorithms
- **Column Generation**: Iterative solving of master and subproblems with pricing filters
- **Branch-and-Price**: Full tree exploration with DFS or BFS search strategies
- **Labeling Algorithm**: Dynamic programming approach for pricing subproblems with strict feasibility checks
  - State-based exploration with dominance rules
  - Handles rolling window constraints (MS/MIN_MS)
  - Validates start/end constraints and service targets
  - Supports timeout scenarios at planning horizon
- **Dual Stagnation Detection**: Early termination when LP bound stops improving
- **IP Heuristics**: Periodic integer program solving to find better incumbents

### Branching Strategies
1. **MP Variable Branching**: Branch on master problem variables (Lambda_{na})
   - Left branch: Upper bound on column usage
   - Right branch: Lower bound on column usage
   - No-good cuts prevent column regeneration

2. **SP Pattern Branching**: Branch on patterns P(k) ⊆ J × T_k (sets of resource pairs)
   - Hierarchical pattern-based branching on multiple (j,t) pairs simultaneously
   - Left branch: sum_{a ∈ A(k,P(k))} Lambda_{ka} <= floor(beta_P(k))
   - Right branch: sum_{a ∈ A(k,P(k))} Lambda_{ka} >= ceil(beta_P(k))
   - Subproblem constraints enforce pattern restrictions

### Learning Models
- **Exponential Learning**: Classic exponential learning curves
- **Sigmoid Learning**: S-shaped learning with inflection points
- **Linear Learning**: Linear improvement over sessions
- **Piecewise Linear (PWL)**: Tangent approximations for nonlinear curves

### Search Strategies
- **Depth-First Search (DFS)**: Memory efficient, explores deep branches first
- **Best-First Search (BFS)**: Best bound selection, faster convergence

## Project Structure

```
branch-and-price-hybrid-scheduling/
├── main.py                      # Entry point with configuration
├── branch_and_price.py          # Branch-and-Price algorithm (3994 lines)
├── CG.py                        # Column Generation solver (652 lines)
├── masterproblem.py             # Restricted Master Problem (RMP)
├── subproblem.py                # Pricing subproblems
├── label.py                     # Labeling algorithm for pricing (792 lines, DP-based)
├── bnp_node.py                  # Node representation for B&P tree
├── branching_constraints.py     # Branching constraint implementations
├── tree_visualization.py        # Tree visualization utilities
├── logging_config.py            # Multi-level logging configuration
│
├── Utils/
│   ├── Generell/
│   │   ├── instance_setup.py   # Data generation
│   │   ├── plots.py             # Visualization
│   │   └── utils.py             # Helper functions
│   ├── Pre_Patients/
│   │   └── pre_patients_heuristic.py  # Pre-patient scheduling
│   ├── compactmodel.py          # Compact MIP formulation (baseline)
│   ├── initial_cg_sol.py        # Initial CG solution
│   └── feasability_checker.py   # Solution validation
│
├── LPs/                         # LP/solution files
│   ├── MP/                      # Master problem files
│   └── SPs/                     # Subproblem files
│
├── logs/                        # Multi-level log files
│   ├── debug/                   # DEBUG level logs
│   ├── info/                    # INFO level logs
│   ├── warning/                 # WARNING level logs
│   └── error/                   # ERROR level logs
│
├── results/                     # Output files
└── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- Gurobi Optimizer (license required)
- 4+ CPU cores recommended for parallel pricing (labeling algorithm)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd branch-and-price-hybrid-scheduling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Gurobi license:
```bash
# Follow Gurobi installation instructions
# https://www.gurobi.com/documentation/
```

## Usage

### Basic Execution

Run the main script with default parameters:
```bash
python main.py
```

### Configuration

Edit `main.py` to customize parameters:

#### Instance Parameters
```python
seed = 13                    # Random seed
T = 3                        # Number of therapists
D_focus = 5                  # Focus days
pttr = 'medium'             # Patient-to-therapist ratio: 'low', 'medium', 'high'
```

#### Learning Parameters
```python
app_data = {
    'learn_type': ['lin'],      # Learning type: 'exp', 'sigmoid', 'lin', or numeric
    'theta_base': [0.02],       # Base effectiveness
    'lin_increase': [0.01],     # Linear increase rate
    'k_learn': [0.01],          # Learning rate (exp/sigmoid)
    'infl_point': [2],          # Inflection point (sigmoid)
    'MS': [5],                  # Maximum session window
    'MS_min': [2],              # Minimum sessions in window
    'W_on': [6],                # Work days per week
    'W_off': [1],               # Days off per week
    'daily': [4]                # Daily capacity per therapist
}
```

#### Algorithm Parameters
```python
# Column Generation
max_itr = 100                           # Maximum CG iterations
threshold = 1e-5                        # Convergence threshold
dual_improvement_iter = 20              # Max iterations without improvement
dual_stagnation_threshold = 1e-5        # Minimum relative improvement
pricing_filtering = True                # Enable pricing filter
learn_method = 'pwl'                    # Learning method: 'pwl', 'tangent', 'bilinear'

# Branch-and-Price
use_branch_and_price = True             # Enable B&P (vs. pure CG)
branching_strategy = 'sp'               # 'mp' or 'sp' branching
search_strategy = 'bfs'                 # 'dfs' or 'bfs'
ip_heuristic_frequency = 5              # IP heuristic every N nodes
early_incumbent_iteration = 1           # Compute incumbent at iteration N (0 = after CG)

# Labeling Algorithm (Pricing)
use_labeling = True                     # Use labeling algorithm for pricing (vs. Gurobi)
max_columns_per_iter = 10               # Max columns to return per recipient
use_parallel_pricing = True             # Enable parallel pricing
n_pricing_workers = 4                   # Number of parallel workers
```

#### Tree Visualization
```python
visualize_tree = False                  # Enable tree visualization
tree_layout = 'hierarchical'            # 'hierarchical' or 'radial'
detailed_tree = False                   # Show detailed node info
save_tree_path = 'bnp_tree.png'        # Save path
```

### Example: Column Generation Only

```python
use_branch_and_price = False
max_itr = 100
```

### Example: Branch-and-Price with BFS and Labeling

```python
use_branch_and_price = True
branching_strategy = 'sp'
search_strategy = 'bfs'
use_labeling = True
use_parallel_pricing = True
n_pricing_workers = 4
```

## Logging System

The project uses a sophisticated multi-level logging system with separate files for each log level.

### Log Levels

- **DEBUG**: Detailed algorithmic steps, state information
- **INFO**: General progress, worker selection, candidate workers
- **WARNING**: Potential issues, numerical warnings
- **ERROR**: Errors and failures
- **PRINT**: Terminal output only (via `logger.print()`)

### Log Files

```
logs/
├── debug/bnp_TIMESTAMP.log      # DEBUG messages only
├── info/bnp_TIMESTAMP.log       # INFO messages only
├── warning/bnp_TIMESTAMP.log    # WARNING messages only
└── error/bnp_TIMESTAMP.log      # ERROR messages only
```

Each log file contains only messages of its specific level (no duplication).

### Console Output

The console displays only `logger.print()` messages for clean terminal output:
- Critical progress updates
- Final results
- User-facing information

All other logging (DEBUG, INFO, WARNING, ERROR) goes exclusively to files.

### Example Log Usage

```python
from logging_config import get_logger

logger = get_logger(__name__)

# Terminal output (visible to user)
logger.print("Branch-and-Price completed!")

# File-only logging (not in terminal)
logger.debug("State exploration details...")
logger.info("Node 5 solved with bound 123.45")
logger.warning("Potential numerical instability")
logger.error("Pricing problem failed")
```

## Output

### Console Output
The algorithm prints only critical information via `logger.print()`:
- Setup phase completion
- Node processing milestones
- Final results and statistics

### Results Files

1. **LP Files**: `LPs/MP/LPs/master_node_*.lp`
   - Master problem formulations at each node

2. **Solution Files**: `LPs/MP/SOLs/master_node_*.sol`
   - LP solutions at each node

3. **Schedule Export**: `results/optimal_schedules.csv`
   - Detailed patient schedules with therapist assignments

4. **Tree Visualization**: `Pictures/Tree/tree_*.png`
   - Visual representation of the B&P tree

### Log Files
- `logs/debug/bnp_*.log`: Detailed algorithm steps
- `logs/info/bnp_*.log`: General progress information
- `logs/warning/bnp_*.log`: Warnings and potential issues
- `logs/error/bnp_*.log`: Errors and failures

## Algorithm Details

### Labeling Algorithm (Pricing Subproblems)

The labeling algorithm is a dynamic programming approach that solves the pricing problem for each recipient:

1. **Initialization**: Start with initial state (cost, progress, AI count, history)
2. **State Expansion**: For each time period, explore therapist and AI options
3. **Feasibility Checks**: 
   - Rolling window constraints (MS/MIN_MS)
   - Service target requirements
   - Timeout scenarios at horizon
4. **Dominance Pruning**:
   - Bucket dominance: Within same AI count and history
   - Global dominance: Across all buckets (optional)
5. **Lower Bound Pruning**: Prune states that cannot improve best solution
6. **Worker Dominance**: Pre-eliminate dominated workers before DP
7. **Terminal States**: Collect feasible schedules with negative reduced cost
8. **No-Good Cuts**: Enforce branching constraints via deviation vectors

Key features:
- Parallel processing with multiprocessing (optional)
- Returns multiple columns per recipient (up to max_columns_per_iter)
- Handles MP branching constraints (no-good cuts)
- Efficient state representation with tuples

### Column Generation Process

1. **Initialization**: Generate initial columns with heuristic
2. **Master Problem**: Solve RMP to get dual variables
3. **Pricing**: Solve subproblems (via labeling or Gurobi) to find negative reduced cost columns
4. **Pricing Filter**: Skip subproblems unlikely to produce improving columns
5. **Column Addition**: Add profitable columns to master
6. **Convergence Check**: Stop if no improving columns or stagnation detected
7. **Integrality**: Check if LP solution is integral

### Branch-and-Price Process

1. **Root Node**: Solve with Column Generation
2. **Initial Incumbent**: Solve RMP as IP (early or after CG)
3. **Node Selection**: Pick next node (DFS or BFS)
4. **Node Processing**:
   - Build master with inherited columns + branching constraints
   - Solve with CG at node
   - Check integrality and bounds
5. **Branching**: Select fractional variable and create child nodes
6. **Fathoming**: Eliminate nodes by bound, integrality, or infeasibility
7. **IP Heuristic**: Periodically solve RMP as IP without branching constraints
8. **Termination**: Stop when all nodes fathomed or time limit reached

### Performance Optimizations

- **Parallel Pricing**: Uses multiprocessing for labeling algorithm (4 workers default)
- **Pricing Filter**: Reduces subproblems solved by approximately 50-80%
- **Column Pool Management**: Efficient column inheritance in tree
- **Dual Stagnation Detection**: Early termination when LP stops improving
- **Solution Pool**: Generates multiple columns per subproblem (up to 10)
- **Worker Dominance**: Pre-eliminates dominated workers before DP
- **State Dominance**: Bucket and global dominance pruning

## Results Interpretation

### Branch-and-Price Statistics

```
Nodes Explored:    45        # Total nodes processed
Nodes Fathomed:    43        # Nodes eliminated
Nodes Branched:    2         # Nodes branched on
LP Bound (LB):     125.45    # Best LP relaxation bound (lower bound)
Incumbent (UB):    126.12    # Best integer solution (upper bound)
Gap:               0.53%     # Optimality gap
```

### Solution Quality

- **Gap < 1%**: High-quality solution, very close to optimal
- **Gap 1-5%**: Good solution, reasonable optimality guarantee
- **Gap > 5%**: May need more nodes or better branching strategy

### Performance Metrics

- **CG Iterations**: Fewer is better (indicates fast convergence)
- **Columns Added**: More columns = larger master problem
- **Subproblems Skipped**: Higher percentage = better pricing filter
- **IP Solves**: Frequency of heuristic runs
- **Pruning Statistics**: Lower bound pruning and state dominance effectiveness

## Troubleshooting

### Common Issues

1. **Gurobi License Error**
   - Ensure Gurobi is properly installed and licensed
   - Check license file location: `grbgetkey <license-key>`

2. **Memory Issues**
   - Reduce `max_nodes` parameter
   - Use DFS instead of BFS (less memory)
   - Reduce `D_focus` or `T` (smaller instances)
   - Disable parallel pricing: `use_parallel_pricing = False`

3. **Slow Performance**
   - Enable labeling algorithm: `use_labeling = True`
   - Enable parallel pricing: `use_parallel_pricing = True`
   - Enable pricing filter: `pricing_filtering = True`
   - Reduce `max_itr` (CG iterations)
   - Increase `dual_stagnation_threshold` for earlier termination

4. **No Improvement in Gap**
   - Increase `max_nodes` to explore more of tree
   - Try different `branching_strategy` ('mp' vs 'sp')
   - Adjust `ip_heuristic_frequency` (more frequent heuristics)
   - Increase `early_incumbent_iteration` for better initial bound

5. **Fractional Solutions at Root**
   - Normal for complex instances
   - Branch-and-Price will explore tree to find integer solution
   - Check that `use_branch_and_price = True`

6. **Parallel Pricing Issues**
   - Requires `use_labeling = True`
   - Check available CPU cores: `os.cpu_count()`
   - Reduce `n_pricing_workers` if system is overloaded

## Advanced Features

### Custom Branching Strategy

Modify `select_branching_candidate()` in `branch_and_price.py`:
```python
def select_branching_candidate(self, node, node_lambda):
    # Implement custom selection logic
    # Return branching_info dict
    pass
```

### Custom Learning Curves

Add new learning type in `subproblem.py`:
```python
def _add_custom_learning_constraints(self):
    # Define custom learning curve
    pass
```

### External Data Import

Replace `generate_patient_data_log()` in `main.py`:
```python
# Load from CSV/database instead of generation
Req, Entry, Max_t, P, D, ... = load_from_file('data.csv')
```

### Custom Logging Configuration

Modify `logging_config.py` for different log levels or formats:
```python
# Single file logging (all levels)
setup_logging(log_level='INFO', log_to_file=True, log_dir='logs')

# Multi-level logging (separate files)
setup_multi_level_logging(base_log_dir='logs', enable_console=True)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{branch-and-price-hybrid,
  title={Branch-and-Price for Hybrid Scheduling with Learning Effects},
  author={Wagner, Lorenz},
  year={2024}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [lorenz.wagner@uni-a.de](mailto:lorenz.wagner@uni-a.de)

## Acknowledgments

- Gurobi Optimization for the MIP solver
- Column Generation and Branch-and-Price methodology based on classical OR literature
- Learning curve models adapted from healthcare scheduling research
