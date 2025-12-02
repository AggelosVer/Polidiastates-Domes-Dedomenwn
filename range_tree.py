import numpy as np
from typing import List, Tuple, Optional, Any
import heapq

class RangeTreeNode:

    def __init__(self, value=None, point=None, data=None, left=None, right=None, aux_tree=None):
        self.value = value  
        self.point = point
        self.data = data
        self.left = left
        self.right = right
        self.aux_tree = aux_tree  

class RangeTree:

    def __init__(self, points=None, data=None, dimension=None):
        self.root = None
        self.points = points
        self.data = data
        if points is not None and len(points) > 0:
            if dimension is None:
                self.dimension = len(points[0])
            else:
                self.dimension = dimension
                for point in points:
                    if len(point) != self.dimension:
                        raise ValueError(f"All points must have {self.dimension} dimensions")
            if self.dimension < 2 or self.dimension > 5:
                raise ValueError("Range Tree supports 2D-5D dimensions only")
            points_array = np.array(points)
            if data is None:
                data = [None] * len(points)
            self.root = self._build(points_array, data, 0)
        else:
            self.dimension = dimension if dimension is not None else 0
    
    def _build(self, points: np.ndarray, data: List[Any], dim: int) -> Optional[RangeTreeNode]:
        if len(points) == 0:
            return None
        if len(points) == 1:
            node = RangeTreeNode()
            node.point = points[0].copy()
            node.data = data[0]
            node.value = points[0][dim]
            if dim < self.dimension - 1:
                node.aux_tree = self._build(points, data, dim + 1)
            return node
        sorted_indices = np.argsort(points[:, dim])
        points_sorted = points[sorted_indices]
        data_sorted = [data[i] for i in sorted_indices]
        n = len(points_sorted)
        mid = n // 2
        node = RangeTreeNode()
        node.value = points_sorted[mid][dim]
        left_points = points_sorted[:mid]
        right_points = points_sorted[mid:]
        left_data = data_sorted[:mid]
        right_data = data_sorted[mid:]
        node.left = self._build(left_points, left_data, dim)
        node.right = self._build(right_points, right_data, dim)
        if dim < self.dimension - 1:
            node.aux_tree = self._build(points_sorted, data_sorted, dim + 1)
        return node
    
    def query(self, range_min: List[float], range_max: List[float]) -> List[Tuple[np.ndarray, Any]]:
        if self.root is None:
            return []
        if len(range_min) != self.dimension or len(range_max) != self.dimension:
            raise ValueError(f"Range must have {self.dimension} dimensions")
        results = []
        self._query(self.root, np.array(range_min), np.array(range_max), 0, results)
        return results
    
    def _query(self, node: Optional[RangeTreeNode], range_min: np.ndarray, 
               range_max: np.ndarray, dim: int, results: List[Tuple[np.ndarray, Any]]):
        if node is None:
            return
        if node.left is None and node.right is None:
            if node.point is not None:
                if self._point_in_range(node.point, range_min, range_max):
                    results.append((node.point.copy(), node.data))
            return
        if dim == self.dimension - 1:
            self._query_1d(node, range_min, range_max, dim, results)
            return
        min_val = range_min[dim]
        max_val = range_max[dim]
        split_node = self._find_split_node(node, min_val, max_val, dim)
        if split_node is None:
            return
        if split_node.left is None and split_node.right is None:
            if split_node.point is not None:
                if self._point_in_range(split_node.point, range_min, range_max):
                    results.append((split_node.point.copy(), split_node.data))
            return
        curr = split_node.left
        while curr is not None and (curr.left is not None or curr.right is not None):
            if min_val <= curr.value:
                if curr.right is not None and curr.right.aux_tree is not None:
                    self._query(curr.right.aux_tree, range_min, range_max, dim + 1, results)
                curr = curr.left
            else:
                curr = curr.right
        if curr is not None and curr.point is not None:
            if self._point_in_range(curr.point, range_min, range_max):
                results.append((curr.point.copy(), curr.data))
        curr = split_node.right
        while curr is not None and (curr.left is not None or curr.right is not None):
            if max_val >= curr.value:
                if curr.left is not None and curr.left.aux_tree is not None:
                    self._query(curr.left.aux_tree, range_min, range_max, dim + 1, results)
                curr = curr.right
            else:
                curr = curr.left
        if curr is not None and curr.point is not None:
            if self._point_in_range(curr.point, range_min, range_max):
                results.append((curr.point.copy(), curr.data))
    
    def _query_1d(self, node: RangeTreeNode, range_min: np.ndarray, 
                   range_max: np.ndarray, dim: int, results: List[Tuple[np.ndarray, Any]]):
        if node is None:
            return
        if node.left is None and node.right is None:
            if node.point is not None:
                if self._point_in_range(node.point, range_min, range_max):
                    results.append((node.point.copy(), node.data))
            return
        min_val = range_min[dim]
        max_val = range_max[dim]
        if max_val < node.value:
            self._query_1d(node.left, range_min, range_max, dim, results)
        elif min_val > node.value:
            self._query_1d(node.right, range_min, range_max, dim, results)
        else:
            self._query_1d(node.left, range_min, range_max, dim, results)
            self._query_1d(node.right, range_min, range_max, dim, results)
    
    def _find_split_node(self, node: RangeTreeNode, min_val: float, 
                         max_val: float, dim: int) -> Optional[RangeTreeNode]:
        curr = node
        while curr is not None and (curr.left is not None or curr.right is not None):
            if max_val < curr.value:
                curr = curr.left
            elif min_val > curr.value:
                curr = curr.right
            else:
                break
        return curr
    
    def _point_in_range(self, point: np.ndarray, range_min: np.ndarray, 
                        range_max: np.ndarray) -> bool:
        return np.all((point >= range_min) & (point <= range_max))
    
    def get_build_complexity(self) -> str:
        return f"O(n log^{self.dimension - 1} n)"
    
    def get_query_complexity(self) -> str:
        return f"O(log^{self.dimension} n + k) where k is the number of results"
    
    def get_space_complexity(self) -> str:
        return f"O(n log^{self.dimension - 1} n)"
    
    def size(self) -> int:
        return self._count_nodes(self.root)
    
    def _count_nodes(self, node: Optional[RangeTreeNode]) -> int:
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1 if node.point is not None else 0
        return self._count_nodes(node.left) + self._count_nodes(node.right)

    def knn_query(self, query_point: List[float], k: int = 1) -> List[Tuple[np.ndarray, Any]]:
        if self.root is None:
            return []
        query_point = np.array(query_point)
        if len(query_point) != self.dimension:
            raise ValueError(f"Query point must have {self.dimension} dimensions")
        heap = [] 
        self._knn_search(self.root, query_point, k, heap)
        results = []
        while heap:
            dist, point, data = heapq.heappop(heap)
            results.append((point, data))
        return results[::-1]

    def _knn_search(self, node: Optional[RangeTreeNode], query_point: np.ndarray, k: int, heap: List):
        if node is None:
            return
        if node.left is None and node.right is None:
            if node.point is not None:
                dist = np.linalg.norm(node.point - query_point)
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, node.point.copy(), node.data))
                elif dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist, node.point.copy(), node.data))
            return
        dim = 0
        dist_to_plane = query_point[dim] - node.value
        if dist_to_plane <= 0:
            first = node.left
            second = node.right
        else:
            first = node.right
            second = node.left
        self._knn_search(first, query_point, k, heap)
        if len(heap) < k or abs(dist_to_plane) < -heap[0][0]:
            self._knn_search(second, query_point, k, heap)

if __name__ == "__main__":
    print("=== Testing 2D Range Tree ===")
    points_2d = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2), (1, 5), (6, 8)]
    data_2d = ["A", "B", "C", "D", "E", "F", "G", "H"]
    tree_2d = RangeTree(points_2d, data_2d)
    print(f"Points: {points_2d}")
    print(f"Build complexity: {tree_2d.get_build_complexity()}")
    print(f"Query complexity: {tree_2d.get_query_complexity()}")
    print(f"Space complexity: {tree_2d.get_space_complexity()}")
    print(f"Tree size: {tree_2d.size()}")
    range_min_2d = [0, 0]
    range_max_2d = [6, 6]
    results_2d = tree_2d.query(range_min_2d, range_max_2d)
    print(f"\nRange query [{range_min_2d}, {range_max_2d}]:")
    for result in results_2d:
        print(f"  {tuple(result[0])} data={result[1]}")
    print("\n=== Testing 3D Range Tree ===")
    points_3d = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (2, 3, 4), (5, 6, 7)]
    tree_3d = RangeTree(points_3d)
    print(f"Points: {points_3d}")
    print(f"Build complexity: {tree_3d.get_build_complexity()}")
    print(f"Query complexity: {tree_3d.get_query_complexity()}")
    range_min_3d = [0, 0, 0]
    range_max_3d = [5, 5, 5]
    results_3d = tree_3d.query(range_min_3d, range_max_3d)
    print(f"\nRange query [{range_min_3d}, {range_max_3d}]:")
    for result in results_3d:
        print(f"  {tuple(result[0])}")
    print("\n=== Testing 5D Range Tree ===")
    points_5d = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9]
    ]
    tree_5d = RangeTree(points_5d)
    print(f"Points: {points_5d}")
    print(f"Build complexity: {tree_5d.get_build_complexity()}")
    print(f"Query complexity: {tree_5d.get_query_complexity()}")
    range_min_5d = [1, 1, 1, 1, 1]
    range_max_5d = [4, 4, 4, 4, 4]
    results_5d = tree_5d.query(range_min_5d, range_max_5d)
    print(f"\nRange query [{range_min_5d}, {range_max_5d}]:")
    for result in results_5d:
        print(f"  {tuple(result[0])}")
    print("\n=== Correctness Verification ===")
    def linear_scan(points, range_min, range_max):
        results = []
        for point in points:
            in_range = True
            for i in range(len(point)):
                if not (range_min[i] <= point[i] <= range_max[i]):
                    in_range = False
                    break
            if in_range:
                results.append(point)
        return results
    ls_results_2d = linear_scan(points_2d, range_min_2d, range_max_2d)
    rt_results_2d_sorted = sorted([tuple(x[0]) for x in results_2d])
    ls_results_2d_sorted = sorted([tuple(x) for x in ls_results_2d])
    print(f"2D Test: {'PASS' if rt_results_2d_sorted == ls_results_2d_sorted else 'FAIL'}")
    print(f"  Range Tree: {len(results_2d)} results")
    print(f"  Linear Scan: {len(ls_results_2d)} results")
    ls_results_5d = linear_scan(points_5d, range_min_5d, range_max_5d)
    rt_results_5d_sorted = sorted([tuple(x[0]) for x in results_5d])
    ls_results_5d_sorted = sorted([tuple(x) for x in ls_results_5d])
    print(f"\n5D Test: {'PASS' if rt_results_5d_sorted == ls_results_5d_sorted else 'FAIL'}")
    print(f"  Range Tree: {len(results_5d)} results")
    print(f"  Linear Scan: {len(ls_results_5d)} results")
    print("\n=== Testing kNN Search ===")
    query_pt = [4, 4]
    k = 3
    print(f"Query point: {query_pt}, k={k}")
    knn_results = tree_2d.knn_query(query_pt, k)
    print("kNN Results:")
    for p, d_val in knn_results:
        d = np.linalg.norm(np.array(p) - np.array(query_pt))
        print(f"  {tuple(p)}, data={d_val}, dist={d:.4f}")
    dists = []
    for i, p in enumerate(points_2d):
        d = np.linalg.norm(np.array(p) - np.array(query_pt))
        dists.append((d, p, data_2d[i]))
    dists.sort(key=lambda x: x[0])
    expected_knn = [(x[1], x[2]) for x in dists[:k]]
    print("Expected (Linear Scan):")
    for p, d_val in expected_knn:
        d = np.linalg.norm(np.array(p) - np.array(query_pt))
        print(f"  {tuple(p)}, data={d_val}, dist={d:.4f}")
    knn_tuples = [(tuple(x[0]), x[1]) for x in knn_results]
    expected_tuples = [(tuple(x[0]), x[1]) for x in expected_knn]
    print(f"kNN Test: {'PASS' if knn_tuples == expected_tuples else 'FAIL'}")
