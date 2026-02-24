"""
Thread-Safe Shared State Manager for Parallel Branch-and-Price

This module provides thread-safe wrappers for all shared state in the parallel
tree exploration of the Branch-and-Price algorithm.

Author: Branch-and-Price Optimization Team
Date: 2026-01-11
"""

import threading
import logging
from typing import Optional, List, Tuple, Dict, Any


class ThreadSafeSharedState:
    """
    Manages all shared state with thread-safe operations for parallel tree exploration.
    
    This class encapsulates all shared data structures that multiple worker threads
    need to access concurrently, using a re-entrant lock (RLock) to prevent race
    conditions.
    
    Attributes:
        lock: Re-entrant lock for thread-safe operations
        incumbent: Best integer solution found (upper bound)
        incumbent_solution: Solution dictionary for best incumbent
        incumbent_lambdas: Lambda values for best incumbent
        best_lp_bound: Best LP relaxation bound (lower bound)
        gap: Optimality gap (UB - LB) / UB
        search_strategy: 'dfs' or 'bfs'
        open_nodes: Queue of nodes to process (format depends on search_strategy)
        stats: Statistics dictionary
        shutdown_requested: Flag to signal worker shutdown
        nodes: Dictionary of all BnPNode objects {node_id -> BnPNode}
    """
    
    def __init__(self, search_strategy: str, initial_stats: dict, initial_nodes: dict):
        """
        Initialize thread-safe shared state.
        
        Args:
            search_strategy: 'dfs' or 'bfs'
            initial_stats: Initial statistics dictionary
            initial_nodes: Initial nodes dictionary {node_id -> BnPNode}
        """
        # Re-entrant lock allows the same thread to acquire the lock multiple times
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Bounds and solutions
        self.incumbent = float('inf')
        self.incumbent_solution = None
        self.incumbent_lambdas = None
        self.best_lp_bound = float('inf')
        self.gap = float('inf')
        
        # Node queue (different format for DFS vs BFS)
        self.search_strategy = search_strategy
        if search_strategy == 'bfs':
            self.open_nodes = []  # List of (bound, node_id) tuples
        else:  # dfs
            self.open_nodes = []  # List of node_ids (stack)
        
        # Statistics
        self.stats = initial_stats.copy()
        
        # Worker management
        self.shutdown_requested = False
        
        # Node storage (shared across all workers)
        self.nodes = initial_nodes  # Reference to BnP.nodes dict
        
    def get_next_node(self) -> Optional[int]:
        """
        Thread-safe node selection from the queue.
        
        For BFS: Selects node with lowest LP bound (best-first search)
        For DFS: Selects most recently added node (depth-first search)
        
        Returns:
            Node ID to process, or None if queue is empty or shutdown requested
        """
        with self.lock:
            if not self.open_nodes or self.shutdown_requested:
                return None
            
            if self.search_strategy == 'bfs':
                # Best-first: sort by bound (ascending) and select lowest
                # Tie-breaking: if bounds equal, select lower node_id (was created first)
                sorted_nodes = sorted(self.open_nodes, key=lambda x: (round(x[0], 4), x[1]))
                bound, node_id = sorted_nodes[0]
                self.open_nodes.remove((bound, node_id))
                
                self.logger.debug(f"[BFS] Selected Node {node_id} with bound {bound:.6f}")
                return node_id
            else:  # DFS
                node_id = self.open_nodes.pop()
                self.logger.debug(f"[DFS] Selected Node {node_id}")
                return node_id
    
    def add_nodes(self, nodes_to_add: List[Any]) -> None:
        """
        Thread-safe addition of nodes to the queue.
        
        Args:
            nodes_to_add: For BFS: list of (bound, node_id) tuples
                         For DFS: list of node_ids
        """
        with self.lock:
            if self.search_strategy == 'bfs':
                # nodes_to_add: list of (bound, node_id)
                self.open_nodes.extend(nodes_to_add)
                self.logger.debug(f"[BFS] Added {len(nodes_to_add)} nodes to queue")
            else:  # DFS
                # nodes_to_add: list of node_ids
                self.open_nodes.extend(nodes_to_add)
                self.logger.debug(f"[DFS] Added {len(nodes_to_add)} nodes to queue")
    
    def try_update_incumbent(
        self, 
        new_incumbent: float, 
        new_solution: Dict[str, Any], 
        new_lambdas: Dict[Tuple, Any],
        node_id: int,
        time_elapsed: float = None
    ) -> bool:
        """
        Thread-safe incumbent update.
        
        Only updates if new_incumbent is better than current incumbent.
        
        Args:
            new_incumbent: New incumbent value
            new_solution: New solution dictionary
            new_lambdas: New lambda values
            node_id: Node ID where incumbent was found
            time_elapsed: Time elapsed since start (seconds)
            
        Returns:
            True if incumbent was updated, False otherwise
        """
        with self.lock:
            if new_incumbent < self.incumbent - 1e-9:  # Tolerance for floating point
                old_incumbent = self.incumbent
                self.incumbent = new_incumbent
                self.incumbent_solution = new_solution
                self.incumbent_lambdas = new_lambdas
                self.stats['incumbent_updates'] += 1
                self.stats['incumbent_node_id'] = node_id
                
                # Update time to first incumbent if provided and not yet set
                if time_elapsed is not None and self.stats.get('time_to_first_incumbent') is None:
                    self.stats['time_to_first_incumbent'] = time_elapsed
                
                self._update_gap()
                
                self.logger.info(f"✅ NEW INCUMBENT: {new_incumbent:.6f} (previous: {old_incumbent:.6f}, gap: {self.gap:.4%})")
                return True
            return False
    
    def update_best_lb(self, new_lb: float) -> None:
        """
        Thread-safe lower bound update.
        
        Args:
            new_lb: New lower bound candidate
        """
        with self.lock:
            if new_lb < self.best_lp_bound:
                old_lb = self.best_lp_bound
                self.best_lp_bound = new_lb
                self._update_gap()
                self.logger.debug(f"Updated best LB: {old_lb:.6f} -> {new_lb:.6f}")
    
    def _update_gap(self) -> None:
        """
        Internal gap calculation (must be called with lock held).
        
        Gap = (UB - LB) / |UB|
        """
        if self.incumbent < float('inf') and self.best_lp_bound < float('inf'):
            if abs(self.incumbent) > 1e-10:
                self.gap = (self.incumbent - self.best_lp_bound) / abs(self.incumbent)
            else:
                self.gap = abs(self.incumbent - self.best_lp_bound)
        else:
            self.gap = float('inf')
    
    def should_continue(self) -> bool:
        """
        Check if workers should continue processing.
        
        Returns:
            True if there are open nodes and shutdown not requested
        """
        with self.lock:
            return bool(self.open_nodes) and not self.shutdown_requested
    
    def request_shutdown(self) -> None:
        """Signal all workers to stop gracefully."""
        with self.lock:
            self.shutdown_requested = True
            self.logger.info("Shutdown requested for all workers")
    
    def get_queue_size(self) -> int:
        """Thread-safe queue size query."""
        with self.lock:
            return len(self.open_nodes)
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of current state (thread-safe).
        
        Returns:
            Dictionary with current bounds, gap, queue size, etc.
        """
        with self.lock:
            return {
                'incumbent': self.incumbent,
                'best_lp_bound': self.best_lp_bound,
                'gap': self.gap,
                'queue_size': len(self.open_nodes),
                'nodes_explored': self.stats.get('nodes_explored', 0),
                'nodes_fathomed': self.stats.get('nodes_fathomed', 0),
                'nodes_branched': self.stats.get('nodes_branched', 0),
                'shutdown_requested': self.shutdown_requested
            }
    
    def increment_stat(self, stat_name: str, amount: int = 1) -> None:
        """
        Thread-safe statistics increment.
        
        Args:
            stat_name: Name of the statistic to increment
            amount: Amount to increment by (default: 1)
        """
        with self.lock:
            self.stats[stat_name] = self.stats.get(stat_name, 0) + amount
    
    def add_to_stat_list(self, stat_name: str, value: Any) -> None:
        """
        Thread-safe append to a statistics list.
        
        Args:
            stat_name: Name of the list statistic
            value: Value to append
        """
        with self.lock:
            if stat_name not in self.stats:
                self.stats[stat_name] = []
            self.stats[stat_name].append(value)
    
    def update_stat_dict(self, stat_name: str, key: Any, value: Any) -> None:
        """
        Thread-safe dictionary update in statistics.
        
        Args:
            stat_name: Name of the dict statistic
            key: Dictionary key
            value: Value to set
        """
        with self.lock:
            if stat_name not in self.stats:
                self.stats[stat_name] = {}
            self.stats[stat_name][key] = value
    
    def add_timing(self, timing_name: str, duration: float) -> None:
        """
        Thread-safe timing accumulation.
        
        Args:
            timing_name: Name of the timing metric (e.g., 'time_in_mp')
            duration: Duration to add in seconds
        """
        with self.lock:
            self.stats[timing_name] = self.stats.get(timing_name, 0.0) + duration
