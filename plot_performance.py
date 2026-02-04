import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from project1_loader import load_and_process_data
from extract_5d_vectors import extract_5d_vectors
from preprocess_text import clean_and_tokenize
from kd_tree import KDTree
from quadtree import QuadTree
from range_tree import RangeTree
from r_tree import RTree
from lsh import MinHashLSH

def benchmark_varying_size(max_size=5000, step=500):
    print("=" * 80)
    print("BENCHMARKING INDEX PERFORMANCE WITH VARYING DATA SIZE")
    print("=" * 80)
    
    print("\nLoading dataset...")
    df = load_and_process_data('data_movies_clean.csv', apply_filter=False, normalize=False)
    if df is None:
        print("Failed to load data")
        return None
    

    df = df[df['production_company_names'].notna() & (df['production_company_names'] != '')]
    
    if len(df) > max_size:
        df = df.sample(n=max_size, random_state=42)
    
    vectors, _, vector_df = extract_5d_vectors(df)
    doc_ids = vector_df.index.tolist()
    
    text_tokens = {}
    for idx in doc_ids:
        if idx in df.index:
            text_raw = df.loc[idx, 'production_company_names']
            text_tokens[idx] = clean_and_tokenize(text_raw)
    
    mins = np.min(vectors, axis=0)
    maxs = np.max(vectors, axis=0)
    bounds = np.column_stack((mins, maxs + 0.001))
    
    sizes = list(range(step, len(vectors) + 1, step))
    if sizes[-1] != len(vectors):
        sizes.append(len(vectors))
    
    results = {
        'sizes': sizes,
        'kd_tree': {'build': [], 'query': []},
        'quadtree': {'build': [], 'query': []},
        'range_tree': {'build': [], 'query': []},
        'r_tree': {'build': [], 'query': []},
        'lsh': {'build': []}
    }
    
    q_min = [mins[0], mins[1], mins[2], mins[3], mins[4]]
    q_max = [maxs[0], maxs[1], maxs[2], maxs[3], maxs[4]]
    
    print(f"\nTesting sizes: {sizes}")
    
    for size in sizes:
        print(f"\n{'='*80}")
        print(f"Testing with {size} points...")
        print(f"{'='*80}")
        
        v = vectors[:size]
        d = doc_ids[:size]
        t = {k: v for k, v in text_tokens.items() if k in d}
        

        print("  Building KD-Tree...", end=" ")
        start = time.time()
        kd = KDTree(k=5)
        kd.build(v.tolist(), d)
        kd_build = time.time() - start
        
        start = time.time()
        kd.range_query(q_min, q_max)
        kd_query = time.time() - start
        
        results['kd_tree']['build'].append(kd_build)
        results['kd_tree']['query'].append(kd_query)
        print(f"Build: {kd_build:.4f}s, Query: {kd_query:.4f}s")
        

        print("  Building QuadTree...", end=" ")
        b = np.column_stack((mins, maxs + 0.001))
        start = time.time()
        qt = QuadTree(b, k=5, capacity=20)
        qt.build(v, d)
        qt_build = time.time() - start
        
        q_bounds = np.column_stack((q_min, q_max))
        start = time.time()
        qt.range_query(q_bounds)
        qt_query = time.time() - start
        
        results['quadtree']['build'].append(qt_build)
        results['quadtree']['query'].append(qt_query)
        print(f"Build: {qt_build:.4f}s, Query: {qt_query:.4f}s")
        

        print("  Building Range Tree...", end=" ")
        start = time.time()
        rt = RangeTree(v.tolist(), d)
        rt_build = time.time() - start
        
        start = time.time()
        rt.query(q_min, q_max)
        rt_query = time.time() - start
        
        results['range_tree']['build'].append(rt_build)
        results['range_tree']['query'].append(rt_query)
        print(f"Build: {rt_build:.4f}s, Query: {rt_query:.4f}s")
        

        if size <= 2000:
            print("  Building R-Tree...", end=" ")
            start = time.time()
            r_tree = RTree(max_entries=4, min_entries=2, dimension=5)
            for i in range(len(v)):
                r_tree.insert(v[i], d[i])
            r_build = time.time() - start
            
            start = time.time()
            r_tree.search(q_min, q_max)
            r_query = time.time() - start
            
            results['r_tree']['build'].append(r_build)
            results['r_tree']['query'].append(r_query)
            print(f"Build: {r_build:.4f}s, Query: {r_query:.4f}s")
        else:
            results['r_tree']['build'].append(None)
            results['r_tree']['query'].append(None)
            print("Skipped (too slow for large datasets)")
        

        print("  Building LSH...", end=" ")
        start = time.time()
        lsh = MinHashLSH(num_perm=128, threshold=0.5)
        for doc_id, tokens in t.items():
            sig = lsh.compute_signature(tokens)
            lsh.add(doc_id, sig)
        lsh_build = time.time() - start
        
        results['lsh']['build'].append(lsh_build)
        print(f"Build: {lsh_build:.4f}s")
    
    return results

def plot_build_times(results, log_scale=False, save_path='build_times.png'):
    sizes = results['sizes']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(sizes, results['kd_tree']['build'], marker='o', label='KD-Tree', linewidth=2)
    ax.plot(sizes, results['quadtree']['build'], marker='s', label='QuadTree', linewidth=2)
    ax.plot(sizes, results['range_tree']['build'], marker='^', label='Range Tree', linewidth=2)
    
    r_sizes = [s for i, s in enumerate(sizes) if results['r_tree']['build'][i] is not None]
    r_times = [t for t in results['r_tree']['build'] if t is not None]
    if r_times:
        ax.plot(r_sizes, r_times, marker='d', label='R-Tree', linewidth=2)
    
    ax.plot(sizes, results['lsh']['build'], marker='x', label='LSH', linewidth=2, linestyle='--')
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Build Time (seconds, log scale)', fontsize=12)
    else:
        ax.set_ylabel('Build Time (seconds)', fontsize=12)
    
    ax.set_xlabel('Dataset Size (number of points)', fontsize=12)
    ax.set_title('Index Build Time Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_query_times(results, log_scale=False, save_path='query_times.png'):
    sizes = results['sizes']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(sizes, results['kd_tree']['query'], marker='o', label='KD-Tree', linewidth=2)
    ax.plot(sizes, results['quadtree']['query'], marker='s', label='QuadTree', linewidth=2)
    ax.plot(sizes, results['range_tree']['query'], marker='^', label='Range Tree', linewidth=2)
    
    r_sizes = [s for i, s in enumerate(sizes) if results['r_tree']['query'][i] is not None]
    r_times = [t for t in results['r_tree']['query'] if t is not None]
    if r_times:
        ax.plot(r_sizes, r_times, marker='d', label='R-Tree', linewidth=2)
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Query Time (seconds, log scale)', fontsize=12)
    else:
        ax.set_ylabel('Query Time (seconds)', fontsize=12)
    
    ax.set_xlabel('Dataset Size (number of points)', fontsize=12)
    ax.set_title('Range Query Time Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_combined_comparison(results, save_path='combined_comparison.png'):
    sizes = results['sizes']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(sizes, results['kd_tree']['build'], marker='o', label='KD-Tree', linewidth=2)
    ax1.plot(sizes, results['quadtree']['build'], marker='s', label='QuadTree', linewidth=2)
    ax1.plot(sizes, results['range_tree']['build'], marker='^', label='Range Tree', linewidth=2)
    
    r_sizes = [s for i, s in enumerate(sizes) if results['r_tree']['build'][i] is not None]
    r_times = [t for t in results['r_tree']['build'] if t is not None]
    if r_times:
        ax1.plot(r_sizes, r_times, marker='d', label='R-Tree', linewidth=2)
    
    ax1.set_xlabel('Dataset Size', fontsize=11)
    ax1.set_ylabel('Build Time (seconds)', fontsize=11)
    ax1.set_title('Build Time Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(sizes, results['kd_tree']['query'], marker='o', label='KD-Tree', linewidth=2)
    ax2.plot(sizes, results['quadtree']['query'], marker='s', label='QuadTree', linewidth=2)
    ax2.plot(sizes, results['range_tree']['query'], marker='^', label='Range Tree', linewidth=2)
    
    r_sizes_q = [s for i, s in enumerate(sizes) if results['r_tree']['query'][i] is not None]
    r_times_q = [t for t in results['r_tree']['query'] if t is not None]
    if r_times_q:
        ax2.plot(r_sizes_q, r_times_q, marker='d', label='R-Tree', linewidth=2)
    
    ax2.set_xlabel('Dataset Size', fontsize=11)
    ax2.set_ylabel('Query Time (seconds)', fontsize=11)
    ax2.set_title('Query Time Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_bar_comparison(results, save_path='bar_comparison.png'):
    last_size = results['sizes'][-1]
    schemes = ['KD-Tree', 'QuadTree', 'Range Tree', 'R-Tree']
    
    build_times = [
        results['kd_tree']['build'][-1],
        results['quadtree']['build'][-1],
        results['range_tree']['build'][-1],
        results['r_tree']['build'][-1] if results['r_tree']['build'][-1] is not None else 0
    ]
    
    query_times = [
        results['kd_tree']['query'][-1],
        results['quadtree']['query'][-1],
        results['range_tree']['query'][-1],
        results['r_tree']['query'][-1] if results['r_tree']['query'][-1] is not None else 0
    ]
    
    x = np.arange(len(schemes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, build_times, width, label='Build Time', alpha=0.8)
    bars2 = ax.bar(x + width/2, query_times, width, label='Query Time', alpha=0.8)
    
    ax.set_xlabel('Index Scheme', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Build vs Query Time (n={last_size})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  TIME COMPLEXITY PLOTTING")
    print("  Comparing: KD-Tree, QuadTree, Range Tree, R-Tree + LSH")
    print("=" * 80)
    
    output_dir = "benchmark_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    results = benchmark_varying_size(max_size=3000, step=300)
    
    if results:
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80 + "\n")
        
        plot_build_times(results, log_scale=False, save_path=os.path.join(output_dir, 'build_times_linear.png'))
        plot_build_times(results, log_scale=True, save_path=os.path.join(output_dir, 'build_times_log.png'))
        
        plot_query_times(results, log_scale=False, save_path=os.path.join(output_dir, 'query_times_linear.png'))
        plot_query_times(results, log_scale=True, save_path=os.path.join(output_dir, 'query_times_log.png'))
        
        plot_combined_comparison(results, save_path=os.path.join(output_dir, 'combined_comparison.png'))
        
        plot_bar_comparison(results, save_path=os.path.join(output_dir, 'bar_comparison.png'))
        
        print("\n" + "=" * 80)
        print("PLOTTING COMPLETE")
        print("=" * 80)
        print("\nGenerated files in " + output_dir + ":")
        print(f"  - build_times_linear.png")
        print(f"  - build_times_log.png")
        print(f"  - query_times_linear.png")
        print(f"  - query_times_log.png")
        print(f"  - combined_comparison.png")
        print(f"  - bar_comparison.png")
