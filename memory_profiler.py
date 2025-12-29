import sys
import tracemalloc
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import time

from kd_tree import KDTree
from quadtree import QuadTree
from range_tree import RangeTree
from r_tree import RTree
from project1_loader import load_and_process_data
from extract_5d_vectors import extract_5d_vectors


def get_size_deep(obj, seen=None):
    """
    Recursively calculate the deep size of a Python object.
    This includes nested objects, attributes, and collections.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([get_size_deep(v, seen) for v in obj.values()])
        size += sum([get_size_deep(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size_deep(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum([get_size_deep(i, seen) for i in obj])
        except TypeError:
            pass
    
    return size


def measure_memory_basic(index_structure, name: str) -> Dict[str, Any]:
    """
    Measure memory using sys.getsizeof for shallow and deep sizes.
    """
    shallow_size = sys.getsizeof(index_structure)
    deep_size = get_size_deep(index_structure)
    
    return {
        'name': name,
        'shallow_bytes': shallow_size,
        'deep_bytes': deep_size,
        'shallow_kb': shallow_size / 1024,
        'shallow_mb': shallow_size / (1024 * 1024),
        'deep_kb': deep_size / 1024,
        'deep_mb': deep_size / (1024 * 1024)
    }


def measure_memory_tracemalloc(build_function, name: str) -> Dict[str, Any]:
    """
    Measure memory using tracemalloc to track actual memory allocations.
    
    Args:
        build_function: A callable that builds the index structure
        name: Name of the index structure
    
    Returns:
        Dictionary containing memory measurements
    """
    tracemalloc.start()
    
    snapshot_before = tracemalloc.take_snapshot()
    
    start_time = time.time()
    index_structure = build_function()
    build_time = time.time() - start_time
    
    snapshot_after = tracemalloc.take_snapshot()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    
    return {
        'name': name,
        'current_bytes': current,
        'peak_bytes': peak,
        'allocated_bytes': total_allocated,
        'current_kb': current / 1024,
        'current_mb': current / (1024 * 1024),
        'peak_kb': peak / 1024,
        'peak_mb': peak / (1024 * 1024),
        'allocated_kb': total_allocated / 1024,
        'allocated_mb': total_allocated / (1024 * 1024),
        'build_time_seconds': build_time
    }, index_structure


def build_kdtree(points: np.ndarray, data: List) -> KDTree:
    """Build KD-Tree index."""
    tree = KDTree(k=points.shape[1])
    tree.build(points.tolist(), data)
    return tree


def build_quadtree(points: np.ndarray, data: List) -> QuadTree:
    """Build QuadTree index."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    bounds = np.column_stack([mins, maxs])
    
    tree = QuadTree(bounds=bounds, k=points.shape[1], capacity=10)
    tree.build(points, data)
    return tree


def build_rangetree(points: np.ndarray, data: List) -> RangeTree:
    """Build Range Tree index."""
    tree = RangeTree(points=points.tolist(), data=data, dimension=points.shape[1])
    return tree


def build_rtree(points: np.ndarray, data: List) -> RTree:
    """Build R-Tree index."""
    tree = RTree(max_entries=10, min_entries=3, dimension=points.shape[1])
    for i, point in enumerate(points):
        tree.insert(point.tolist(), data[i])
    return tree


def profile_all_indexes(points: np.ndarray, data: List, dataset_size: int) -> pd.DataFrame:
    """
    Profile memory usage of all index structures.
    
    Args:
        points: NumPy array of data points
        data: List of associated data values
        dataset_size: Number of points to use for profiling
    
    Returns:
        DataFrame with profiling results
    """
    print(f"\n{'='*80}")
    print(f"MEMORY PROFILING FOR {dataset_size} DATA POINTS")
    print(f"{'='*80}\n")
    
    subset_points = points[:dataset_size]
    subset_data = data[:dataset_size]
    
    results = []
    
    index_builders = [
        ('KD-Tree', lambda: build_kdtree(subset_points, subset_data)),
        ('QuadTree', lambda: build_quadtree(subset_points, subset_data)),
        ('Range Tree', lambda: build_rangetree(subset_points, subset_data)),
        ('R-Tree', lambda: build_rtree(subset_points, subset_data))
    ]
    
    for name, builder in index_builders:
        print(f"Profiling {name}...")
        
        tracemalloc_result, index_obj = measure_memory_tracemalloc(builder, name)
        
        basic_result = measure_memory_basic(index_obj, name)
        
        combined_result = {
            'Index Structure': name,
            'Dataset Size': dataset_size,
            'Build Time (s)': f"{tracemalloc_result['build_time_seconds']:.4f}",
            'sys.getsizeof Deep (MB)': f"{basic_result['deep_mb']:.4f}",
            'sys.getsizeof Deep (KB)': f"{basic_result['deep_kb']:.2f}",
            'sys.getsizeof Shallow (KB)': f"{basic_result['shallow_kb']:.2f}",
            'tracemalloc Current (MB)': f"{tracemalloc_result['current_mb']:.4f}",
            'tracemalloc Peak (MB)': f"{tracemalloc_result['peak_mb']:.4f}",
            'tracemalloc Allocated (MB)': f"{tracemalloc_result['allocated_mb']:.4f}",
            'tracemalloc Current (KB)': f"{tracemalloc_result['current_kb']:.2f}",
            'tracemalloc Peak (KB)': f"{tracemalloc_result['peak_kb']:.2f}",
        }
        
        results.append(combined_result)
        print(f"  ✓ {name} completed: {basic_result['deep_mb']:.2f} MB (deep), "
              f"{tracemalloc_result['peak_mb']:.2f} MB (peak)\n")
    
    return pd.DataFrame(results)


def print_comparison_table(df: pd.DataFrame):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("MEMORY USAGE COMPARISON")
    print(f"{'='*80}\n")
    
    display_cols = [
        'Index Structure',
        'Dataset Size',
        'Build Time (s)',
        'sys.getsizeof Deep (MB)',
        'tracemalloc Peak (MB)'
    ]
    
    print(df[display_cols].to_string(index=False))
    print(f"\n{'='*80}\n")


def main():
    """Main function to run memory profiling."""
    
    print("Loading movie dataset...")
    FILE_PATH = 'data_movies_clean.csv'
    df = load_and_process_data(FILE_PATH, apply_filter=True)
    
    if df is None or df.empty:
        print("Error: Could not load data.")
        return
    
    print(f"Loaded {len(df)} movies")
    
    vectors, reference_df, vector_df = extract_5d_vectors(df)
    print(f"Extracted {len(vectors)} 5D vectors")
    
    data_values = reference_df['id'].tolist() if 'id' in reference_df.columns else list(range(len(vectors)))
    
    # Dynamically determine dataset sizes based on available data
    total_vectors = len(vectors)
    if total_vectors >= 2000:
        dataset_sizes = [100, 500, 1000, 2000]
    elif total_vectors >= 1000:
        dataset_sizes = [100, 500, 1000]
    elif total_vectors >= 500:
        dataset_sizes = [100, 500]
    elif total_vectors >= 100:
        dataset_sizes = [50, 100]
    elif total_vectors >= 50:
        dataset_sizes = [20, total_vectors]
    else:
        dataset_sizes = [total_vectors]
    
    # Ensure we don't exceed available data
    dataset_sizes = [size for size in dataset_sizes if size <= total_vectors]
    
    if not dataset_sizes:
        dataset_sizes = [total_vectors]
    
    print(f"\nTesting with dataset sizes: {dataset_sizes}\n")
    
    all_results = []
    
    for size in dataset_sizes:
        result_df = profile_all_indexes(vectors, data_values, size)
        all_results.append(result_df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print_comparison_table(combined_df)
    
    output_file = 'memory_profiling_results.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    for size in dataset_sizes:
        subset = combined_df[combined_df['Dataset Size'] == size]
        print(f"\nDataset Size: {size} points")
        print("-" * 80)
        for _, row in subset.iterrows():
            print(f"  {row['Index Structure']:15s}: "
                  f"{row['sys.getsizeof Deep (MB)']:>8s} MB (deep), "
                  f"{row['tracemalloc Peak (MB)']:>8s} MB (peak), "
                  f"{row['Build Time (s)']:>8s}s build")
    
    print("\n" + "="*80)
    print("MEMORY EFFICIENCY RANKING (by sys.getsizeof Deep)")
    print("="*80 + "\n")
    
    for size in dataset_sizes:
        subset = combined_df[combined_df['Dataset Size'] == size].copy()
        subset['deep_mb_num'] = subset['sys.getsizeof Deep (MB)'].astype(float)
        subset = subset.sort_values('deep_mb_num')
        
        print(f"\nDataset Size: {size} points")
        print("-" * 80)
        for rank, (_, row) in enumerate(subset.iterrows(), 1):
            print(f"  {rank}. {row['Index Structure']:15s}: {row['sys.getsizeof Deep (MB)']:>8s} MB")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
