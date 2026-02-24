import math

class BnPNode:
    """
    Represents a node in the Branch-and-Price search tree.

    This class stores all information that a node in the B&P tree requires:
    - Position in tree (ID, parent, depth)
    - LP bound and status
    - Branching constraints from root to this node
    - Column pool (available columns at this node)
    - Solutions (LP and IPs)

    Attributes:
        node_id: Unique node ID
        parent_id: Parent node ID (None for root)
        depth: Depth in search tree (root = 0)
        path: Path encoding (e.g., 'lrrl' for left-right-right-left)
        lp_bound: LP relaxation value at this node (AFTER full CG convergence)
        is_integral: Boolean - is the LP solution integral?
        branching_constraints: List of all branching decisions on path from root
        column_pool: Dict of available columns {(profile, col_id): col_data}
        status: Node status in Branch-and-Price tree
            - 'solved': Node has been fully solved with CG, ready to be branched
            - 'branched': Node has been selected and branched into children (closed)
            - 'fathomed': Node closed due to bound/infeasibility/integrality
            - 'integral': Node found integral solution (subtype of fathomed)
        fathom_reason: Reason for fathoming (if status='fathomed' or 'integral')
            - 'integral': LP solution is integral
            - 'bound': LP bound >= incumbent (no improvement possible)
            - 'infeasible': LP is infeasible
            - 'error': Error during solving
    """

    def __init__(self, node_id, parent_id=None, depth=0, path=''):
        # Tree position
        self.node_id = node_id
        self.parent_id = parent_id
        self.depth = depth
        self.path = path

        # Bounds and status
        self.lp_bound = float('inf')
        self.is_integral = False

        # Branching constraints
        self.branching_constraints = []

        # Column management
        self.column_pool = {}

        # Status tracking
        # NEW STATUS VALUES (corrected Branch-and-Price):
        # - 'solved': Node fully solved with CG, has valid LP bound, ready for branching
        # - 'branched': Node was selected and branched (closed)
        # - 'fathomed': Node closed (bound/infeasible/integral)
        # - 'integral': Node has integral solution (special case of fathomed)
        self.status = 'solved'  # Start as 'solved' after CG convergence
        self.fathom_reason = None  # 'integral', 'bound', 'infeasible', 'error'

        # Solutions
        self.master_solution = None  # LP solution (dict with variable values)
        self.most_fractional_var = None  # Info about most fractional variable

    def __str__(self):
        """Detailed string representation."""
        info = [
            f"Node {self.node_id}:",
            f"  Depth: {self.depth}",
            f"  Parent: {self.parent_id}",
            f"  Status: {self.status}",
            f"  LP Bound: {self.lp_bound:.6f}",
            f"  Is Integral: {self.is_integral}",
            f"  Branching Constraints: {len(self.branching_constraints)}",
            f"  Columns in Pool: {len(self.column_pool)}"
        ]

        if self.fathom_reason:
            info.append(f"  Fathom Reason: {self.fathom_reason}")
        return "\n".join(info)

    def __repr__(self):
        """String representation for debugging."""
        path_str = f"'{self.path}'" if self.path else "'root'"
        return (f"Node(id={self.node_id}, path={path_str}, depth={self.depth}, "
                f"status={self.status}, bound={self.lp_bound:.2f})")