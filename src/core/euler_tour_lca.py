"""
Euler Tour + Sparse Table (RMQ) O(1) Lowest Common Ancestor (LCA) Engine.

Reduces tree-based ancestor queries to a 1D Range Minimum Query (RMQ)
via Euler Tour traversal, solved in O(1) constant time after O(N log N)
preprocessing using a power-of-two Sparse Table.

Applicable to:
- Dialogue, conversation, and prompt branching trees (ChatGPT exports, conversational DAGs)
- Matrioshka polytope containment hierarchies
- Fossil and knowledge graph lineages
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import math


class EulerTourSparseTableLCA:
    """
    O(1) LCA Query Engine via Euler Tour and Sparse Table RMQ.
    
    Complexity:
    - Preprocessing: O(M log M) time, O(M log M) space, where M = 2N - 1
    - LCA Query: O(1) time
    - Distance Query: O(1) time
    - Ancestry Check: O(1) time
    """
    
    def __init__(self, tree: Optional[Dict[Any, List[Any]]] = None, root: Optional[Any] = None):
        """
        Args:
            tree: Adjacency list mapping parent node -> list of child nodes.
            root: Root node of the tree. If None, detected as node with in-degree 0.
        """
        self.tree: Dict[Any, List[Any]] = {}
        self.parent_map: Dict[Any, Optional[Any]] = {}
        self.euler_tour: List[Any] = []
        self.depth_tour: List[int] = []
        self.first_occurrence: Dict[Any, int] = {}
        self.node_depth: Dict[Any, int] = {}
        self.sparse_table: List[List[int]] = []
        self.log_table: List[int] = []
        self.root: Optional[Any] = root
        
        if tree:
            self.build_from_adjacency(tree, root)
            
    def build_from_adjacency(self, tree: Dict[Any, List[Any]], root: Optional[Any] = None) -> None:
        """Constructs the LCA engine from an adjacency mapping."""
        self.tree = {k: list(v) for k, v in tree.items()}
        self.parent_map = {}
        
        # Track all unique nodes
        all_children: Set[Any] = set()
        all_nodes: Set[Any] = set(self.tree.keys())
        
        for p, children in self.tree.items():
            for c in children:
                all_children.add(c)
                all_nodes.add(c)
                self.parent_map[c] = p
                
        # Determine root if not provided
        if root is not None:
            self.root = root
        else:
            candidates = list(all_nodes - all_children)
            if candidates:
                self.root = candidates[0]
            elif all_nodes:
                self.root = next(iter(all_nodes))
            else:
                self.root = None
                return
                
        self.parent_map[self.root] = None
        self._compute_euler_tour()
        self._build_sparse_table()

    @classmethod
    def from_chatgpt_mapping(cls, mapping: Dict[str, Any]) -> 'EulerTourSparseTableLCA':
        """
        Constructs LCA engine directly from a ChatGPT conversation mapping dictionary.
        
        Mapping format:
        {
            node_id: {
                "id": str,
                "parent": Optional[str],
                "children": List[str],
                "message": Optional[dict]
            }
        }
        """
        tree: Dict[str, List[str]] = {}
        root_candidate = None
        
        for nid, node in mapping.items():
            parent_id = node.get("parent")
            if parent_id is None or parent_id not in mapping:
                if root_candidate is None:
                    root_candidate = nid
            else:
                tree.setdefault(parent_id, []).append(nid)
                
            # Ensure leaf nodes exist in adjacency table
            if nid not in tree:
                tree[nid] = []
                
        instance = cls()
        instance.build_from_adjacency(tree, root=root_candidate)
        return instance

    def _compute_euler_tour(self) -> None:
        """Performs depth-first traversal to compute the Euler tour and depth arrays."""
        self.euler_tour = []
        self.depth_tour = []
        self.first_occurrence = {}
        self.node_depth = {}
        
        if self.root is None:
            return
            
        def dfs(node: Any, depth: int) -> None:
            self.node_depth[node] = depth
            idx = len(self.euler_tour)
            self.euler_tour.append(node)
            self.depth_tour.append(depth)
            
            if node not in self.first_occurrence:
                self.first_occurrence[node] = idx
                
            for child in self.tree.get(node, []):
                dfs(child, depth + 1)
                self.euler_tour.append(node)
                self.depth_tour.append(depth)
                
        dfs(self.root, 0)

    def _build_sparse_table(self) -> None:
        """Builds power-of-two Sparse Table on the depth tour array for O(1) RMQ."""
        m = len(self.depth_tour)
        if m == 0:
            return
            
        # Precompute floor log2 table
        self.log_table = [0] * (m + 1)
        for i in range(2, m + 1):
            self.log_table[i] = self.log_table[i // 2] + 1
            
        max_k = self.log_table[m] + 1
        # sparse_table[k][i] holds index of minimum element in depth_tour[i : i + 2^k]
        self.sparse_table = [[0] * m for _ in range(max_k)]
        
        # Base level k = 0
        for i in range(m):
            self.sparse_table[0][i] = i
            
        # Fill levels k >= 1
        k = 1
        while (1 << k) <= m:
            step = 1 << (k - 1)
            for i in range(m - (1 << k) + 1):
                idx1 = self.sparse_table[k - 1][i]
                idx2 = self.sparse_table[k - 1][i + step]
                if self.depth_tour[idx1] <= self.depth_tour[idx2]:
                    self.sparse_table[k][i] = idx1
                else:
                    self.sparse_table[k][i] = idx2
            k += 1

    def _rmq(self, left: int, right: int) -> int:
        """Returns the index of the minimum depth in depth_tour[left:right+1] in O(1) time."""
        length = right - left + 1
        k = self.log_table[length]
        idx1 = self.sparse_table[k][left]
        idx2 = self.sparse_table[k][right - (1 << k) + 1]
        return idx1 if self.depth_tour[idx1] <= self.depth_tour[idx2] else idx2

    def lca(self, u: Any, v: Any) -> Any:
        """
        Finds the Lowest Common Ancestor (LCA) of nodes u and v in O(1) constant time.
        
        Args:
            u: First node identifier
            v: Second node identifier
            
        Returns:
            The Lowest Common Ancestor node identifier, or None if either node is not in tree.
        """
        if u not in self.first_occurrence or v not in self.first_occurrence:
            return None
            
        if u == v:
            return u
            
        idx_u = self.first_occurrence[u]
        idx_v = self.first_occurrence[v]
        
        left = min(idx_u, idx_v)
        right = max(idx_u, idx_v)
        
        min_idx = self._rmq(left, right)
        return self.euler_tour[min_idx]

    def distance(self, u: Any, v: Any) -> Optional[int]:
        """
        Calculates the graph distance (number of edges) between u and v in O(1) time.
        dist(u, v) = depth(u) + depth(v) - 2 * depth(lca(u, v))
        """
        ancestor = self.lca(u, v)
        if ancestor is None:
            return None
        return self.node_depth[u] + self.node_depth[v] - 2 * self.node_depth[ancestor]

    def is_ancestor(self, ancestor: Any, descendant: Any) -> bool:
        """Checks if ancestor is an ancestor of descendant in O(1) time."""
        return self.lca(ancestor, descendant) == ancestor

    def path_to_root(self, node: Any) -> List[Any]:
        """Returns the path from the given node up to the root."""
        path = []
        curr = node
        while curr is not None:
            path.append(curr)
            curr = self.parent_map.get(curr)
        return path

    def find_divergence(self, branch_a: Any, branch_b: Any) -> Tuple[Any, List[Any], List[Any]]:
        """
        Finds the branching divergence point between two nodes in O(1) LCA lookup.
        
        Returns:
            Tuple of:
            - lca_node: The divergence root node
            - path_from_lca_to_a: Path of nodes from immediately below LCA down to branch_a
            - path_from_lca_to_b: Path of nodes from immediately below LCA down to branch_b
        """
        lca_node = self.lca(branch_a, branch_b)
        if lca_node is None:
            return (None, [], [])
            
        path_a = []
        curr = branch_a
        while curr is not None and curr != lca_node:
            path_a.append(curr)
            curr = self.parent_map.get(curr)
        path_a.reverse()
        
        path_b = []
        curr = branch_b
        while curr is not None and curr != lca_node:
            path_b.append(curr)
            curr = self.parent_map.get(curr)
        path_b.reverse()
        
        return (lca_node, path_a, path_b)
