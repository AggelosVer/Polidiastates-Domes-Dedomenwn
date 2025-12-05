import numpy as np
import random
from kd_tree import KDTree
from quadtree import QuadTree
from range_tree import RangeTree
from r_tree import RTree


def generate_random_5d_points(n_points, bounds):
    points = []
    for _ in range(n_points):
        point = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(5)]
        points.append(tuple(point))
    return points


def compare_results(results_kd, results_quad, results_range, results_rtree, test_num):
    kd_points = set()
    for point, data in results_kd:
        kd_points.add(tuple(point) if isinstance(point, (list, np.ndarray)) else point)
    
    quad_points = set()
    if isinstance(results_quad[0], np.ndarray):
        for point in results_quad[0]:
            quad_points.add(tuple(point))
    else:
        for point in results_quad[0]:
            quad_points.add(tuple(point) if isinstance(point, (list, np.ndarray)) else point)
    
    range_points = set()
    for point, data in results_range:
        if isinstance(point, np.ndarray):
            range_points.add(tuple(point))
        else:
            range_points.add(tuple(point) if isinstance(point, list) else point)
    
    rtree_points = set()
    for point, data in results_rtree:
        if isinstance(point, np.ndarray):
            rtree_points.add(tuple(point))
        else:
            rtree_points.add(tuple(point) if isinstance(point, list) else point)
    
    all_match = (kd_points == quad_points == range_points == rtree_points)
    
    if not all_match:
        print(f"\nTest {test_num} FAILED - Results don't match!")
        print(f"  KD-Tree:     {len(kd_points)} points")
        print(f"  QuadTree:    {len(quad_points)} points")
        print(f"  RangeTree:   {len(range_points)} points")
        print(f"  R-Tree:      {len(rtree_points)} points")
        
        if kd_points != quad_points:
            only_kd = kd_points - quad_points
            only_quad = quad_points - kd_points
            if only_kd:
                print(f"  Only in KD-Tree: {len(only_kd)} points")
            if only_quad:
                print(f"  Only in QuadTree: {len(only_quad)} points")
        
        if kd_points != range_points:
            only_kd = kd_points - range_points
            only_range = range_points - kd_points
            if only_kd:
                print(f"  Only in KD-Tree: {len(only_kd)} points")
            if only_range:
                print(f"  Only in RangeTree: {len(only_range)} points")
        
        if kd_points != rtree_points:
            only_kd = kd_points - rtree_points
            only_rtree = rtree_points - kd_points
            if only_kd:
                print(f"  Only in KD-Tree: {len(only_kd)} points")
            if only_rtree:
                print(f"  Only in R-Tree: {len(only_rtree)} points")
        
        return False
    
    return True


def run_stress_test(n_tests=100, n_points=1000, query_size=0.1):
    bounds = [[0.0, 100.0] for _ in range(5)]
    
    passed = 0
    failed = 0
    
    for test_num in range(1, n_tests + 1):
        random.seed(test_num)
        np.random.seed(test_num)
        
        points = generate_random_5d_points(n_points, bounds)
        data = [f"data_{i}" for i in range(len(points))]
        
        kd_tree = KDTree(k=5)
        kd_tree.build(points, data)
        
        bounds_array = np.array(bounds)
        quad_tree = QuadTree(bounds_array, k=5, capacity=10)
        points_array = np.array(points)
        quad_tree.build(points_array, data)
        
        range_tree = RangeTree(points=points, data=data, dimension=5)
        
        r_tree = RTree(max_entries=10, min_entries=2, dimension=5)
        for point, d in zip(points, data):
            r_tree.insert(list(point), d)
        
        range_min = [random.uniform(bounds[i][0], bounds[i][1] * (1 - query_size)) for i in range(5)]
        range_max = [range_min[i] + random.uniform(bounds[i][1] * query_size * 0.5, bounds[i][1] * query_size) 
                     for i in range(5)]
        for i in range(5):
            range_max[i] = min(range_max[i], bounds[i][1])
        
        results_kd = kd_tree.range_query(range_min, range_max)
        
        query_bounds_quad = np.array([[range_min[i], range_max[i]] for i in range(5)])
        results_quad = quad_tree.range_query(query_bounds_quad)
        
        results_range = range_tree.query(range_min, range_max)
        
        results_rtree = r_tree.search(range_min, range_max)
        
        if compare_results(results_kd, results_quad, results_range, results_rtree, test_num):
            passed += 1
        else:
            failed += 1
    
    return passed == n_tests, passed, failed


if __name__ == "__main__":
    success, passed, failed = run_stress_test(n_tests=10, n_points=100, query_size=0.1)
    print(f"Tests: {passed + failed}, Passed: {passed}, Failed: {failed}")

