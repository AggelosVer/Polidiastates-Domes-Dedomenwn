import numpy as np
from typing import List, Tuple, Optional, Any


class QuadTreeNode:
    def __init__(self, bounds: np.ndarray, k: int, capacity: int = 10):
        self.bounds = bounds
        self.k = k
        self.capacity = capacity
        self.points = []
        self.data = []
        self.children = None
        self.is_leaf = True
        
    def contains_point(self, point: np.ndarray) -> bool:
        return np.all((point >= self.bounds[:, 0]) & (point <= self.bounds[:, 1]))
    
    def subdivide(self):
        if not self.is_leaf:
            return
        
        self.is_leaf = False
        self.children = []
        
        center = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        
        for i in range(2 ** self.k):
            child_bounds = np.copy(self.bounds)
            for dim in range(self.k):
                bit = (i >> dim) & 1
                if bit == 0:
                    child_bounds[dim, 1] = center[dim]
                else:
                    child_bounds[dim, 0] = center[dim]
            
            child = QuadTreeNode(child_bounds, self.k, self.capacity)
            self.children.append(child)
    
    def get_child_index(self, point: np.ndarray) -> int:
        center = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        index = 0
        for dim in range(self.k):
            if point[dim] >= center[dim]:
                index |= (1 << dim)
        return index


class QuadTree:
    def __init__(self, bounds: np.ndarray, k: int = None, capacity: int = 10):
        if k is None:
            k = bounds.shape[0]
        
        if bounds.shape[1] != 2:
            raise ValueError("Bounds must be a (k, 2) array with [min, max] for each dimension")
        
        self.k = k
        self.capacity = capacity
        self.bounds = bounds
        self.root = QuadTreeNode(bounds, k, capacity)
        self.size = 0
    
    def build(self, points: np.ndarray, data: List[Any] = None):
        if points.shape[1] != self.k:
            raise ValueError(f"Points must have {self.k} dimensions")
        
        if data is None:
            data = [None] * len(points)
        
        if len(points) != len(data):
            raise ValueError("Points and data must have the same length")
        
        for i in range(len(points)):
            self.insert(points[i], data[i])
    
    def insert(self, point: np.ndarray, data: Any = None) -> bool:
        if point.shape[0] != self.k:
            raise ValueError(f"Point must have {self.k} dimensions")
        
        if not self.root.contains_point(point):
            return False
        
        return self._insert(self.root, point, data)
    
    def _insert(self, node: QuadTreeNode, point: np.ndarray, data: Any) -> bool:
        if not node.contains_point(point):
            return False
        
        if node.is_leaf:
            if len(node.points) < node.capacity:
                node.points.append(point.copy())
                node.data.append(data)
                self.size += 1
                return True
            else:
                node.subdivide()
                old_points = node.points.copy()
                old_data = node.data.copy()
                node.points = []
                node.data = []
                
                for i, p in enumerate(old_points):
                    child_idx = node.get_child_index(p)
                    if node.children[child_idx].contains_point(p):
                        node.children[child_idx].points.append(p)
                        node.children[child_idx].data.append(old_data[i])
                
                child_idx = node.get_child_index(point)
                if node.children[child_idx].contains_point(point):
                    return self._insert(node.children[child_idx], point, data)
                else:
                    return False
        
        child_idx = node.get_child_index(point)
        return self._insert(node.children[child_idx], point, data)
    
    def delete(self, point: np.ndarray, data: Any = None) -> bool:
        if point.shape[0] != self.k:
            raise ValueError(f"Point must have {self.k} dimensions")
        
        if not self.root.contains_point(point):
            return False
        
        result = self._delete(self.root, point, data)
        if result:
            self.size -= 1
        return result
    
    def _delete(self, node: QuadTreeNode, point: np.ndarray, data: Any) -> bool:
        if not node.contains_point(point):
            return False
        
        if node.is_leaf:
            for i in range(len(node.points)):
                if np.allclose(node.points[i], point):
                    if data is None or node.data[i] == data:
                        node.points.pop(i)
                        node.data.pop(i)
                        return True
            return False
        
        child_idx = node.get_child_index(point)
        return self._delete(node.children[child_idx], point, data)
    
    def range_query(self, query_bounds: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
        if query_bounds.shape != (self.k, 2):
            raise ValueError(f"Query bounds must be a ({self.k}, 2) array")
        
        points = []
        data = []
        self._range_query(self.root, query_bounds, points, data)
        
        if len(points) == 0:
            return np.array([]).reshape(0, self.k), []
        
        return np.array(points), data
    
    def _range_query(self, node: QuadTreeNode, query_bounds: np.ndarray, 
                     points: List, data: List):
        if not self._intersects(node.bounds, query_bounds):
            return
        
        if node.is_leaf:
            for i, point in enumerate(node.points):
                if self._point_in_bounds(point, query_bounds):
                    points.append(point.copy())
                    data.append(node.data[i])
        else:
            for child in node.children:
                self._range_query(child, query_bounds, points, data)
    
    def _intersects(self, bounds1: np.ndarray, bounds2: np.ndarray) -> bool:
        for dim in range(self.k):
            if bounds1[dim, 1] < bounds2[dim, 0] or bounds1[dim, 0] > bounds2[dim, 1]:
                return False
        return True
    
    def _point_in_bounds(self, point: np.ndarray, bounds: np.ndarray) -> bool:
        return np.all((point >= bounds[:, 0]) & (point <= bounds[:, 1]))
    
    def get_size(self) -> int:
        return self.size
    
    def get_height(self) -> int:
        return self._get_height(self.root)
    
    def _get_height(self, node: QuadTreeNode) -> int:
        if node.is_leaf:
            return 1
        return 1 + max(self._get_height(child) for child in node.children)

