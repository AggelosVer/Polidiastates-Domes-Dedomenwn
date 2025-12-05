
import numpy as np
from typing import List, Tuple, Any, Optional, Union
import heapq

from kd_tree import KDTree
from quadtree import QuadTree
from range_tree import RangeTree
from r_tree import RTree


class IndexManager:
    
    SUPPORTED_TYPES = ['kdtree', 'quadtree', 'rangetree', 'rtree']
    
    def __init__(self, index_type: str, dimension: int = 2, **kwargs):

        if index_type.lower() not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported index type '{index_type}'. "
                f"Supported types: {', '.join(self.SUPPORTED_TYPES)}"
            )
        
        self.index_type = index_type.lower()
        self.dimension = dimension
        self.kwargs = kwargs
        self._index = None
        self._bounds = kwargs.get('bounds', None)
        

        self._initialize_index()
    
    def _initialize_index(self):

        if self.index_type == 'kdtree':
            self._index = KDTree(k=self.dimension)
            
        elif self.index_type == 'quadtree':

            if self._bounds is not None:
                capacity = self.kwargs.get('capacity', 10)
                self._index = QuadTree(bounds=self._bounds, k=self.dimension, capacity=capacity)
            else:

                self._index = None
                
        elif self.index_type == 'rangetree':

            self._index = None
            
        elif self.index_type == 'rtree':
            max_entries = self.kwargs.get('max_entries', 4)
            min_entries = self.kwargs.get('min_entries', 2)
            self._index = RTree(max_entries=max_entries, min_entries=min_entries, dimension=self.dimension)
    
    def _calculate_bounds(self, points: np.ndarray) -> np.ndarray:

        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        margin = (max_coords - min_coords) * 0.01
        margin = np.where(margin == 0, 1.0, margin)  
        min_coords -= margin
        max_coords += margin
        return np.column_stack([min_coords, max_coords])
    
    def build(self, points: List, data: List = None) -> None:

        if len(points) == 0:
            return
        

        points_array = np.array(points, dtype=float)
        
        if points_array.ndim == 1:
            points_array = points_array.reshape(-1, 1)
        
        if points_array.shape[1] != self.dimension:
            raise ValueError(
                f"Points have {points_array.shape[1]} dimensions, "
                f"but index expects {self.dimension} dimensions"
            )
        
        if data is None:
            data = [None] * len(points)
        elif len(points) != len(data):
            raise ValueError("Points and data must have the same length")
        
        if self.index_type == 'kdtree':

            points_tuples = [tuple(p) for p in points_array]
            self._index.build(points_tuples, data)
            
        elif self.index_type == 'quadtree':

            if self._index is None:
                if self._bounds is None:
                    self._bounds = self._calculate_bounds(points_array)
                capacity = self.kwargs.get('capacity', 10)
                self._index = QuadTree(bounds=self._bounds, k=self.dimension, capacity=capacity)
            self._index.build(points_array, data)
            
        elif self.index_type == 'rangetree':

            self._index = RangeTree(points=points_array, data=data, dimension=self.dimension)
            
        elif self.index_type == 'rtree':

            for i, point in enumerate(points_array):
                self._index.insert(point.tolist(), data[i])
    
    def insert(self, point: List, data: Any = None) -> bool:

        point_array = np.array(point, dtype=float)
        
        if len(point_array) != self.dimension:
            raise ValueError(
                f"Point has {len(point_array)} dimensions, "
                f"but index expects {self.dimension} dimensions"
            )
        
        if self.index_type == 'kdtree':
            self._index.insert(tuple(point_array), data)
            return True
            
        elif self.index_type == 'quadtree':
            if self._index is None:
                raise RuntimeError("Quadtree not initialized. Call build() first.")
            return self._index.insert(point_array, data)
            
        elif self.index_type == 'rangetree':
            raise NotImplementedError(
                "Range tree does not support incremental insertion. "
                "Use build() to create a new tree."
            )
            
        elif self.index_type == 'rtree':
            self._index.insert(point_array.tolist(), data)
            return True
    
    def delete(self, point: List, data: Any = None) -> bool:

        point_array = np.array(point, dtype=float)
        
        if len(point_array) != self.dimension:
            raise ValueError(
                f"Point has {len(point_array)} dimensions, "
                f"but index expects {self.dimension} dimensions"
            )
        
        if self.index_type == 'kdtree':
            self._index.delete(tuple(point_array))
            return True
            
        elif self.index_type == 'quadtree':
            if self._index is None:
                raise RuntimeError("Quadtree not initialized. Call build() first.")
            return self._index.delete(point_array, data)
            
        elif self.index_type == 'rangetree':
            raise NotImplementedError(
                "Range tree does not support deletion. "
                "Rebuild the tree without the deleted point."
            )
            
        elif self.index_type == 'rtree':
            return self._index.delete(point_array.tolist(), data)
    
    def update(self, old_point: List, new_point: List, 
               old_data: Any = None, new_data: Any = None) -> bool:

        if new_data is None:
            new_data = old_data
        
        if self.index_type == 'kdtree':
            self._index.update(tuple(old_point), tuple(new_point))
            return True
            
        elif self.index_type == 'quadtree':
            if self._index is None:
                raise RuntimeError("Quadtree not initialized. Call build() first.")
            if self._index.delete(np.array(old_point, dtype=float), old_data):
                return self._index.insert(np.array(new_point, dtype=float), new_data)
            return False
            
        elif self.index_type == 'rangetree':
            raise NotImplementedError(
                "Range tree does not support updates. "
                "Rebuild the tree with updated points."
            )
            
        elif self.index_type == 'rtree':
            return self._index.update(old_point, old_data, new_point, new_data)
    
    def range_query(self, range_min: List, range_max: List) -> List[Tuple[List[float], Any]]:

        range_min_array = np.array(range_min, dtype=float)
        range_max_array = np.array(range_max, dtype=float)
        
        if len(range_min_array) != self.dimension or len(range_max_array) != self.dimension:
            raise ValueError(f"Range bounds must have {self.dimension} dimensions")
        
        if self.index_type == 'kdtree':
            results = self._index.range_query(tuple(range_min_array), tuple(range_max_array))

            return [(list(point), data) for point, data in results]
            
        elif self.index_type == 'quadtree':
            if self._index is None:
                return []
            query_bounds = np.column_stack([range_min_array, range_max_array])
            points, data = self._index.range_query(query_bounds)
            if len(points) == 0:
                return []
            return [(point.tolist(), d) for point, d in zip(points, data)]
            
        elif self.index_type == 'rangetree':
            if self._index is None:
                return []
            results = self._index.query(range_min_array.tolist(), range_max_array.tolist())

            return [(point.tolist() if isinstance(point, np.ndarray) else list(point), data) 
                    for point, data in results]
            
        elif self.index_type == 'rtree':
            results = self._index.search(range_min_array.tolist(), range_max_array.tolist())

            return [(point.tolist() if isinstance(point, np.ndarray) else list(point), data) 
                    for point, data in results]
    
    def knn_query(self, query_point: List, k: int = 1) -> List[Tuple[float, List[float], Any]]:

        query_array = np.array(query_point, dtype=float)
        
        if len(query_array) != self.dimension:
            raise ValueError(f"Query point must have {self.dimension} dimensions")
        
        if self.index_type == 'kdtree':
            results = self._index.knn_query(tuple(query_array), k)

            output = []
            for dist, point in results:

                exact_results = self._index.range_query(point, point)
                point_data = exact_results[0][1] if exact_results else None
                output.append((dist, list(point), point_data))
            return output
            
        elif self.index_type == 'quadtree':

            return self._brute_force_knn(query_array, k)
            
        elif self.index_type == 'rangetree':
            if self._index is None:
                return []
            results = self._index.knn_query(query_array.tolist(), k)

            output = []
            for point, data in results:
                point_array = np.array(point) if not isinstance(point, np.ndarray) else point
                dist = np.linalg.norm(point_array - query_array)
                output.append((dist, point_array.tolist(), data))
            return sorted(output, key=lambda x: x[0])
            
        elif self.index_type == 'rtree':

            return self._brute_force_knn(query_array, k)
    
    def _brute_force_knn(self, query_point: np.ndarray, k: int) -> List[Tuple[float, List[float], Any]]:


        if self.index_type == 'quadtree':
            if self._index is None:
                return []

            bounds = self._index.bounds
            range_min = bounds[:, 0]
            range_max = bounds[:, 1]
            all_points, all_data = self._index.range_query(bounds)
            if len(all_points) == 0:
                return []
            points_list = all_points
            data_list = all_data
            
        elif self.index_type == 'rtree':

            large_val = 1e10
            range_min = [-large_val] * self.dimension
            range_max = [large_val] * self.dimension
            results = self._index.search(range_min, range_max)
            if not results:
                return []
            points_list = [np.array(p) for p, _ in results]
            data_list = [d for _, d in results]
        else:
            return []
        
        

        all_results = []
        for point, data_val in zip(points_list, data_list):
            point_array = np.array(point) if not isinstance(point, np.ndarray) else point
            dist = float(np.linalg.norm(point_array - query_point))
            all_results.append((dist, point_array.tolist() if isinstance(point_array, np.ndarray) else list(point_array), data_val))
        

        all_results.sort(key=lambda x: x[0])
        return all_results[:k]
    
    def get_info(self) -> dict:

        info = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'initialized': self._index is not None
        }        

        if self.index_type == 'quadtree' and self._index is not None:
            info['size'] = self._index.get_size()
            info['height'] = self._index.get_height()
            info['bounds'] = self._bounds.tolist() if self._bounds is not None else None
            
        elif self.index_type == 'rangetree' and self._index is not None:
            info['size'] = self._index.size()
            
        elif self.index_type == 'rtree' and self._index is not None:
            info['height'] = self._index.get_height()
            stats = self._index.get_statistics()
            info['total_nodes'] = stats.get('total_nodes', 0)
            info['total_entries'] = stats.get('total_entries', 0)
        
        return info
    
    def __repr__(self) -> str:

        return (f"IndexManager(index_type='{self.index_type}', "
                f"dimension={self.dimension}, "
                f"initialized={self._index is not None})")


if __name__ == "__main__":

    print("=== IndexManager Demo ===\n")
    
    points_2d = [
        [2.0, 3.0],
        [5.0, 4.0],
        [9.0, 6.0],
        [4.0, 7.0],
        [8.0, 1.0],
        [7.0, 2.0]
    ]
    data_2d = ["A", "B", "C", "D", "E", "F"]
    

    for idx_type in ['kdtree', 'quadtree', 'rangetree', 'rtree']:
        print(f"\n--- Testing {idx_type.upper()} ---")
        

        manager = IndexManager(idx_type, dimension=2)
        manager.build(points_2d, data_2d)
        
        print(f"Built {manager}")
        print(f"Info: {manager.get_info()}")
        

        range_min = [0.0, 0.0]
        range_max = [6.0, 6.0]
        print(f"\nRange Query [{range_min}, {range_max}]:")
        results = manager.range_query(range_min, range_max)
        for point, data in results:
            print(f"  {point} -> {data}")
        

        query_pt = [9.0, 2.0]
        k = 3
        print(f"\nKNN Query (point={query_pt}, k={k}):")
        try:
            knn_results = manager.knn_query(query_pt, k)
            for dist, point, data in knn_results:
                print(f"  dist={dist:.3f}, point={point}, data={data}")
        except NotImplementedError as e:
            print(f"  {e}")
    
    print("\n=== Demo Complete ===")
