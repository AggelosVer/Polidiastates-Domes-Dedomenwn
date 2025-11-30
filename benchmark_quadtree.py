import numpy as np
import time
import matplotlib.pyplot as plt
from quadtree import QuadTree
from typing import List, Tuple, Dict
import pandas as pd


class QuadTreeBenchmark:
    
    def __init__(self, k: int = 5, capacity: int = 10):
        self.k = k
        self.capacity = capacity
        self.results = []
        
    def generate_random_data(self, size: int, seed: int = 42) -> np.ndarray:
        np.random.seed(seed)
        return np.random.rand(size, self.k)
    
    def generate_query_bounds(self, data_bounds: np.ndarray, 
                              query_size_ratio: float = 0.1) -> np.ndarray:
        query_bounds = np.zeros((self.k, 2))
        
        for dim in range(self.k):
            range_size = data_bounds[dim, 1] - data_bounds[dim, 0]
            query_range = range_size * query_size_ratio
            
            start = np.random.uniform(data_bounds[dim, 0], 
                                     data_bounds[dim, 1] - query_range)
            query_bounds[dim, 0] = start
            query_bounds[dim, 1] = start + query_range
        
        return query_bounds
    
    def build_quadtree(self, points: np.ndarray) -> Tuple[QuadTree, float]:
        data_bounds = np.column_stack([
            points.min(axis=0),
            points.max(axis=0)
        ])
        
        margin = 0.01
        data_bounds[:, 0] -= margin
        data_bounds[:, 1] += margin
        
        start_time = time.perf_counter()
        qtree = QuadTree(data_bounds, k=self.k, capacity=self.capacity)
        qtree.build(points)
        build_time = time.perf_counter() - start_time
        
        return qtree, build_time
    
    def benchmark_range_queries(self, qtree: QuadTree, data_bounds: np.ndarray,
                                num_queries: int = 100,
                                query_size_ratio: float = 0.1) -> Dict:
        query_times = []
        result_counts = []
        
        for _ in range(num_queries):
            query_bounds = self.generate_query_bounds(data_bounds, query_size_ratio)
            
            start_time = time.perf_counter()
            results, _ = qtree.range_query(query_bounds)
            query_time = time.perf_counter() - start_time
            
            query_times.append(query_time * 1000)
            result_counts.append(len(results))
        
        return {
            'mean_time_ms': np.mean(query_times),
            'median_time_ms': np.median(query_times),
            'std_time_ms': np.std(query_times),
            'min_time_ms': np.min(query_times),
            'max_time_ms': np.max(query_times),
            'mean_results': np.mean(result_counts),
            'total_queries': num_queries
        }
    
    def run_benchmark(self, dataset_sizes: List[int], 
                     num_queries: int = 100,
                     query_size_ratios: List[float] = [0.05, 0.1, 0.2]) -> pd.DataFrame:
        
        print("="*80)
        print("QUADTREE RANGE QUERY BENCHMARK")
        print("="*80)
        print(f"Dimensions (k): {self.k}")
        print(f"Node Capacity: {self.capacity}")
        print(f"Queries per test: {num_queries}")
        print(f"Query size ratios: {query_size_ratios}")
        print("="*80)
        
        for size in dataset_sizes:
            print(f"\n[Dataset Size: {size:,}]")
            
            points = self.generate_random_data(size)
            
            print(f"  Building quadtree...")
            qtree, build_time = self.build_quadtree(points)
            
            data_bounds = np.column_stack([
                points.min(axis=0),
                points.max(axis=0)
            ])
            
            print(f"  Build time: {build_time:.4f}s")
            print(f"  Tree height: {qtree.get_height()}")
            print(f"  Tree size: {qtree.get_size()}")
            
            for ratio in query_size_ratios:
                print(f"\n  Query size ratio: {ratio:.2%}")
                
                metrics = self.benchmark_range_queries(
                    qtree, data_bounds, num_queries, ratio
                )
                
                result = {
                    'dataset_size': size,
                    'build_time_s': build_time,
                    'tree_height': qtree.get_height(),
                    'query_size_ratio': ratio,
                    'num_queries': num_queries,
                    **metrics
                }
                
                self.results.append(result)
                
                print(f"    Mean query time: {metrics['mean_time_ms']:.4f} ms")
                print(f"    Median query time: {metrics['median_time_ms']:.4f} ms")
                print(f"    Std dev: {metrics['std_time_ms']:.4f} ms")
                print(f"    Range: [{metrics['min_time_ms']:.4f}, {metrics['max_time_ms']:.4f}] ms")
                print(f"    Avg results returned: {metrics['mean_results']:.1f}")
        
        df_results = pd.DataFrame(self.results)
        return df_results
    
    def plot_results(self, df: pd.DataFrame, save_path: str = 'quadtree_benchmark.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('QuadTree Range Query Performance Benchmark', fontsize=16, fontweight='bold')
        
        ratios = df['query_size_ratio'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))
        
        ax1 = axes[0, 0]
        for i, ratio in enumerate(ratios):
            data = df[df['query_size_ratio'] == ratio]
            ax1.plot(data['dataset_size'], data['mean_time_ms'], 
                    marker='o', label=f'Ratio {ratio:.2%}', color=colors[i], linewidth=2)
        ax1.set_xlabel('Dataset Size', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Mean Query Time (ms)', fontsize=11, fontweight='bold')
        ax1.set_title('Query Time vs Dataset Size', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        ax2 = axes[0, 1]
        ax2.plot(df['dataset_size'].unique(), 
                df.groupby('dataset_size')['build_time_s'].first(),
                marker='s', color='coral', linewidth=2, markersize=8)
        ax2.set_xlabel('Dataset Size', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Build Time (seconds)', fontsize=11, fontweight='bold')
        ax2.set_title('Tree Construction Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        ax3 = axes[1, 0]
        ax3.plot(df['dataset_size'].unique(), 
                df.groupby('dataset_size')['tree_height'].first(),
                marker='^', color='green', linewidth=2, markersize=8)
        ax3.set_xlabel('Dataset Size', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Tree Height', fontsize=11, fontweight='bold')
        ax3.set_title('Tree Height vs Dataset Size', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        ax4 = axes[1, 1]
        for i, ratio in enumerate(ratios):
            data = df[df['query_size_ratio'] == ratio]
            ax4.plot(data['dataset_size'], data['mean_results'], 
                    marker='d', label=f'Ratio {ratio:.2%}', color=colors[i], linewidth=2)
        ax4.set_xlabel('Dataset Size', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Average Results Returned', fontsize=11, fontweight='bold')
        ax4.set_title('Query Selectivity', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close()
    
    def save_results(self, df: pd.DataFrame, csv_path: str = 'quadtree_benchmark_results.csv'):
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    def print_summary(self, df: pd.DataFrame):
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        print("\nQuery Time Statistics by Dataset Size:")
        print("-"*80)
        summary = df.groupby('dataset_size').agg({
            'mean_time_ms': ['mean', 'min', 'max'],
            'build_time_s': 'first',
            'tree_height': 'first'
        }).round(4)
        print(summary)
        
        print("\n" + "="*80)


def main():
    dataset_sizes = [1000, 5000, 10000, 50000, 100000, 200000, 500000]
    
    benchmark = QuadTreeBenchmark(k=5, capacity=10)
    
    df_results = benchmark.run_benchmark(
        dataset_sizes=dataset_sizes,
        num_queries=100,
        query_size_ratios=[0.05, 0.1, 0.2]
    )
    
    benchmark.print_summary(df_results)
    
    benchmark.save_results(df_results, 'quadtree_benchmark_results.csv')
    
    benchmark.plot_results(df_results, 'quadtree_benchmark.png')
    
    print("\n Benchmark completed successfully!")


if __name__ == "__main__":
    main()
