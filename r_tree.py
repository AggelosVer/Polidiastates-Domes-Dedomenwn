import numpy as np
from typing import List, Tuple, Any, Optional, Union

class RTreeNode:
    def __init__(self, is_leaf: bool = True, parent: Optional['RTreeNode'] = None):
        self.is_leaf = is_leaf
        self.parent = parent
        self.entries = []

    def get_mbr(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.entries:
            return None
        min_coords = self.entries[0][0][0].copy()
        max_coords = self.entries[0][0][1].copy()
        for (m_min, m_max), _ in self.entries[1:]:
            min_coords = np.minimum(min_coords, m_min)
            max_coords = np.maximum(max_coords, m_max)
        return (min_coords, max_coords)

class RTree:
    def __init__(self, max_entries: int = 4, min_entries: int = 2, dimension: int = 5):
        self.max_entries = max_entries
        self.min_entries = min_entries
        self.dimension = dimension
        self.root = RTreeNode(is_leaf=True)

    def insert(self, point: List[float], value: Any):
        point = np.array(point, dtype=float)
        if len(point) != self.dimension:
            raise ValueError(f"Point must have {self.dimension} dimensions")
        mbr = (point, point)
        leaf = self._choose_leaf(self.root, mbr)
        leaf.entries.append((mbr, value))
        if len(leaf.entries) > self.max_entries:
            self._split_node(leaf)
        else:
            self._adjust_tree(leaf)

    def delete(self, point: List[float], value: Any = None) -> bool:
        point = np.array(point, dtype=float)
        mbr = (point, point)
        leaf = self._find_leaf(self.root, mbr, value)
        if leaf is None:
            return False
        idx_to_remove = -1
        for i, (e_mbr, e_val) in enumerate(leaf.entries):
            if np.array_equal(e_mbr[0], point) and (value is None or e_val == value):
                idx_to_remove = i
                break
        if idx_to_remove == -1:
            return False
        leaf.entries.pop(idx_to_remove)
        self._condense_tree(leaf)
        if not self.root.is_leaf and len(self.root.entries) == 1:
            self.root = self.root.entries[0][1]
            self.root.parent = None
        return True

    def update(self, old_point: List[float], old_value: Any, new_point: List[float], new_value: Any) -> bool:
        if self.delete(old_point, old_value):
            self.insert(new_point, new_value)
            return True
        return False

    def search(self, range_min: List[float], range_max: List[float]) -> List[Tuple[np.ndarray, Any]]:
        range_min = np.array(range_min, dtype=float)
        range_max = np.array(range_max, dtype=float)
        search_mbr = (range_min, range_max)
        results = []
        self._search_recursive(self.root, search_mbr, results)
        return results

    def _search_recursive(self, node: RTreeNode, search_mbr: Tuple[np.ndarray, np.ndarray], results: List):
        for mbr, child in node.entries:
            if self._intersects(mbr, search_mbr):
                if node.is_leaf:
                    results.append((mbr[0], child))
                else:
                    self._search_recursive(child, search_mbr, results)

    def _choose_leaf(self, node: RTreeNode, entry_mbr: Tuple[np.ndarray, np.ndarray]) -> RTreeNode:
        if node.is_leaf:
            return node
        best_child = None
        min_enlargement = float('inf')
        best_area = float('inf')
        for mbr, child in node.entries:
            enlargement = self._calc_enlargement(mbr, entry_mbr)
            area = self._calc_area(mbr)
            if enlargement < min_enlargement:
                min_enlargement = enlargement
                best_area = area
                best_child = child
            elif enlargement == min_enlargement:
                if area < best_area:
                    best_area = area
                    best_child = child
        return self._choose_leaf(best_child, entry_mbr)

    def _adjust_tree(self, node: RTreeNode, split_node: Optional[RTreeNode] = None):
        if node is None:
            return
        if node == self.root:
            if split_node:
                new_root = RTreeNode(is_leaf=False)
                new_root.entries.append((node.get_mbr(), node))
                new_root.entries.append((split_node.get_mbr(), split_node))
                node.parent = new_root
                split_node.parent = new_root
                self.root = new_root
            return
        parent = node.parent
        for i, (mbr, child) in enumerate(parent.entries):
            if child == node:
                parent.entries[i] = (node.get_mbr(), node)
                break
        if split_node:
            parent.entries.append((split_node.get_mbr(), split_node))
            split_node.parent = parent
            if len(parent.entries) > self.max_entries:
                self._split_node(parent)
            else:
                self._adjust_tree(parent)
        else:
            self._adjust_tree(parent)

    def _split_node(self, node: RTreeNode):
        seed1_idx, seed2_idx = self._pick_seeds(node.entries)
        entry1 = node.entries[seed1_idx]
        entry2 = node.entries[seed2_idx]
        new_node = RTreeNode(is_leaf=node.is_leaf, parent=node.parent)
        remaining_entries = [e for i, e in enumerate(node.entries) if i != seed1_idx and i != seed2_idx]
        node.entries = [entry1]
        new_node.entries = [entry2]
        while remaining_entries:
            if len(node.entries) + len(remaining_entries) == self.min_entries:
                node.entries.extend(remaining_entries)
                break
            if len(new_node.entries) + len(remaining_entries) == self.min_entries:
                new_node.entries.extend(remaining_entries)
                break
            best_idx = -1
            max_diff = -1
            preferred_group = 0
            group1_mbr = node.get_mbr()
            group2_mbr = new_node.get_mbr()
            for i, entry in enumerate(remaining_entries):
                d1 = self._calc_enlargement(group1_mbr, entry[0])
                d2 = self._calc_enlargement(group2_mbr, entry[0])
                diff = abs(d1 - d2)
                if diff > max_diff:
                    max_diff = diff
                    best_idx = i
                    preferred_group = 0 if d1 < d2 else 1
            entry = remaining_entries.pop(best_idx)
            if preferred_group == 0:
                node.entries.append(entry)
            else:
                new_node.entries.append(entry)
        self._adjust_tree(node, new_node)

    def _pick_seeds(self, entries: List) -> Tuple[int, int]:
        max_waste = -1
        seed1 = 0
        seed2 = 1
        n = len(entries)
        for i in range(n):
            for j in range(i + 1, n):
                mbr1 = entries[i][0]
                mbr2 = entries[j][0]
                combined = self._combine_mbr(mbr1, mbr2)
                waste = self._calc_area(combined) - self._calc_area(mbr1) - self._calc_area(mbr2)
                if waste > max_waste:
                    max_waste = waste
                    seed1 = i
                    seed2 = j
        return seed1, seed2

    def _condense_tree(self, node: RTreeNode):
        if node == self.root:
            return
        if len(node.entries) < self.min_entries:
            parent = node.parent
            idx_in_parent = -1
            for i, (_, child) in enumerate(parent.entries):
                if child == node:
                    idx_in_parent = i
                    break
            if idx_in_parent != -1:
                parent.entries.pop(idx_in_parent)
            orphaned_entries = self._collect_leaf_entries(node)
            for mbr, val in orphaned_entries:
                self.insert(mbr[0], val)
            self._condense_tree(parent)
        else:
            self._adjust_tree(node)

    def _collect_leaf_entries(self, node: RTreeNode) -> List:
        if node.is_leaf:
            return node.entries
        leaves = []
        for _, child in node.entries:
            leaves.extend(self._collect_leaf_entries(child))
        return leaves

    def _find_leaf(self, node: RTreeNode, entry_mbr: Tuple[np.ndarray, np.ndarray], value: Any) -> Optional[RTreeNode]:
        if node.is_leaf:
            for e_mbr, e_val in node.entries:
                if np.array_equal(e_mbr[0], entry_mbr[0]) and (value is None or e_val == value):
                    return node
            return None
        for mbr, child in node.entries:
            if self._intersects(mbr, entry_mbr):
                res = self._find_leaf(child, entry_mbr, value)
                if res:
                    return res
        return None

    def _calc_area(self, mbr: Tuple[np.ndarray, np.ndarray]) -> float:
        diff = mbr[1] - mbr[0]
        return np.prod(diff)

    def _calc_enlargement(self, current_mbr: Tuple[np.ndarray, np.ndarray], 
                          add_mbr: Tuple[np.ndarray, np.ndarray]) -> float:
        combined = self._combine_mbr(current_mbr, add_mbr)
        return self._calc_area(combined) - self._calc_area(current_mbr)

    def _combine_mbr(self, mbr1: Tuple[np.ndarray, np.ndarray], 
                     mbr2: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        min_c = np.minimum(mbr1[0], mbr2[0])
        max_c = np.maximum(mbr1[1], mbr2[1])
        return (min_c, max_c)

    def _intersects(self, mbr1: Tuple[np.ndarray, np.ndarray], 
                    mbr2: Tuple[np.ndarray, np.ndarray]) -> bool:
        disjoint = np.any(mbr1[1] < mbr2[0]) or np.any(mbr1[0] > mbr2[1])
        return not disjoint

if __name__ == "__main__":
    print("=== Testing R-Tree ===")
    rtree = RTree(max_entries=4, min_entries=2, dimension=2)
    points = [
        ([1, 1], "A"),
        ([2, 2], "B"),
        ([5, 5], "C"),
        ([6, 6], "D"),
        ([1, 6], "E"),
        ([6, 1], "F")
    ]
    print("Inserting points...")
    for p, v in points:
        rtree.insert(p, v)
        print(f"Inserted {v} at {p}")
    print("\nSearching MBR [[0,0], [3,3]] (Should find A, B):")
    results = rtree.search([0, 0], [3, 3])
    for p, v in results:
        print(f"  Found {v} at {p}")
    print("\nDeleting B...")
    rtree.delete([2, 2], "B")
    print("Searching MBR [[0,0], [3,3]] (Should find A only):")
    results = rtree.search([0, 0], [3, 3])
    for p, v in results:
        print(f"  Found {v} at {p}")
    print("\nUpdating A to [2.5, 2.5]...")
    rtree.update([1, 1], "A", [2.5, 2.5], "A_new")
    print("Searching MBR [[0,0], [3,3]] (Should find A_new):")
    results = rtree.search([0, 0], [3, 3])
    for p, v in results:
        print(f"  Found {v} at {p}")
