import time
import numpy as np
import pandas as pd
from extract_5d_vectors import main as extract_vectors
from range_tree import RangeTree

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

def main():
    print("Loading data...")

    vectors, _, _ = extract_vectors()

    
    points = vectors.tolist()
    print(f"Total points: {len(points)}")
    
    print("Building Range Tree...")
    start_time = time.time()
    rt = RangeTree(points)
    build_time = time.time() - start_time
    print(f"Range Tree built in {build_time:.4f} seconds")
    
    num_queries = 10
    print(f"\nRunning {num_queries} random queries...")
    
    min_vals = vectors.min(axis=0)
    max_vals = vectors.max(axis=0)
    
    total_query_time = 0
    total_linear_time = 0
    correct_count = 0
    
    for i in range(num_queries):
        r1 = np.random.uniform(min_vals, max_vals)
        r2 = np.random.uniform(min_vals, max_vals)
        range_min = np.minimum(r1, r2)
        range_max = np.maximum(r1, r2)
        
        start_q = time.time()
        rt_results = rt.query(range_min, range_max)
        query_time = time.time() - start_q
        total_query_time += query_time
        

        start_l = time.time()
        ls_results = linear_scan(points, range_min, range_max)
        linear_time = time.time() - start_l
        total_linear_time += linear_time
        

        rt_results_sorted = sorted([list(x) for x in rt_results])
        ls_results_sorted = sorted([list(x) for x in ls_results])
        
        if rt_results_sorted == ls_results_sorted:
            correct_count += 1
        else:
            print(f"Query {i+1} FAILED")
            print(f"Range: {range_min} - {range_max}")
            print(f"RT count: {len(rt_results)}, LS count: {len(ls_results)}")

            
    print(f"\nCorrectness: {correct_count}/{num_queries}")
    print(f"Average Query Time (Range Tree): {total_query_time/num_queries:.6f} sec")
    print(f"Average Query Time (Linear Scan): {total_linear_time/num_queries:.6f} sec")
    print(f"Speedup: {(total_linear_time/num_queries) / (total_query_time/num_queries):.2f}x")

if __name__ == "__main__":
    main()
