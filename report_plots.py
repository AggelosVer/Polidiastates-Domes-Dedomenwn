
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Any, Optional
import seaborn as sns


# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_query_time_comparison(stats_df: pd.DataFrame, save_path: str) -> str:

    if stats_df.empty:
        return ""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    schemes = stats_df['Scheme'].tolist()
    total_times = stats_df['Avg Total Time (s)'].tolist()
    tree_times = stats_df['Avg Tree Time (s)'].tolist()
    lsh_times = stats_df['Avg LSH Time (s)'].tolist()
    
    x = np.arange(len(schemes))
    width = 0.25
    
    bars1 = ax.bar(x - width, total_times, width, label='Total Time', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x, tree_times, width, label='Tree Time', alpha=0.8, color='#2ecc71')
    bars3 = ax.bar(x + width, lsh_times, width, label='LSH Time', alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('Index Scheme', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Average Query Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_build_time_comparison(performance_data: Dict[str, Any], save_path: str) -> str:

    return ""


def plot_memory_usage(memory_df: pd.DataFrame, save_path: str) -> str:

    if memory_df.empty:
        return ""
    
    # Get unique dataset sizes
    sizes = sorted(memory_df['Dataset Size'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Memory usage by index structure
    ax1 = axes[0]
    for structure in memory_df['Index Structure'].unique():
        subset = memory_df[memory_df['Index Structure'] == structure]
        
        # Convert to numeric if needed
        x_vals = subset['Dataset Size'].values
        y_vals = pd.to_numeric(subset['sys.getsizeof Deep (MB)'], errors='coerce').values
        
        ax1.plot(x_vals, y_vals, marker='o', label=structure, linewidth=2)
    
    ax1.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('Memory Usage vs Dataset Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Use log scale to show all data points clearly
    
    # Plot 2: Build time vs memory usage
    ax2 = axes[1]
    for structure in memory_df['Index Structure'].unique():
        subset = memory_df[memory_df['Index Structure'] == structure]
        
        x_vals = pd.to_numeric(subset['sys.getsizeof Deep (MB)'], errors='coerce').values
        y_vals = pd.to_numeric(subset['Build Time (s)'], errors='coerce').values
        
        ax2.scatter(x_vals, y_vals, label=structure, s=100, alpha=0.7)
    
    ax2.set_xlabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Build Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Build Time vs Memory Usage', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_result_count_distribution(stats_df: pd.DataFrame, save_path: str) -> str:

    if stats_df.empty:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    schemes = stats_df['Scheme'].tolist()
    tree_counts = stats_df['Avg Tree Count'].tolist()
    final_counts = stats_df['Avg Final Count'].tolist()
    
    x = np.arange(len(schemes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, tree_counts, width, label='Avg Tree Results', alpha=0.8, color='#9b59b6')
    bars2 = ax.bar(x + width/2, final_counts, width, label='Avg Final Results (after LSH)', alpha=0.8, color='#f39c12')
    
    ax.set_xlabel('Index Scheme', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Result Count', fontsize=12, fontweight='bold')
    ax.set_title('Result Count Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_performance_heatmap(performance_data: Dict[str, Any], save_path: str) -> str:

    queries = performance_data.get('queries', [])
    if not queries:
        return ""
    
    # Extract scheme names
    scheme_names = list(queries[0].get('schemes', {}).keys())
    query_ids = [q['query_id'] for q in queries]
    
    # Create matrix of total times
    time_matrix = []
    for query in queries:
        row = []
        for scheme in scheme_names:
            time_val = query['schemes'].get(scheme, {}).get('total_time', 0)
            row.append(time_val)
        time_matrix.append(row)
    
    time_matrix = np.array(time_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(time_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(scheme_names)))
    ax.set_yticks(np.arange(len(query_ids)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scheme_names], rotation=45, ha='right')
    ax.set_yticklabels([f'Query {qid}' for qid in query_ids])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Time (s)', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i in range(len(query_ids)):
        for j in range(len(scheme_names)):
            text = ax.text(j, i, f'{time_matrix[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Query Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Index Scheme', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_speedup_comparison(speedup_df: pd.DataFrame, save_path: str) -> str:

    if speedup_df.empty:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    schemes = speedup_df['Scheme'].tolist()
    
    # Find speedup column (it varies based on baseline)
    speedup_col = [col for col in speedup_df.columns if 'Speedup vs' in col]
    if not speedup_col:
        return ""
    
    speedup_col = speedup_col[0]
    speedups = speedup_df[speedup_col].tolist()
    
    colors = ['#2ecc71' if s >= 1 else '#e74c3c' for s in speedups]
    
    bars = ax.bar(schemes, speedups, color=colors, alpha=0.8, edgecolor='black')
    
    # Add horizontal line at y=1 (no speedup)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Index Scheme', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Ratio', fontsize=12, fontweight='bold')
    ax.set_title(f'{speedup_col}', fontsize=14, fontweight='bold')
    ax.set_xticklabels(schemes, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}x', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def save_plots_for_report(performance_data: Dict[str, Any], 
                          stats_df: pd.DataFrame,
                          memory_df: pd.DataFrame,
                          speedup_df: pd.DataFrame,
                          output_dir: str) -> Dict[str, str]:

    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = {}
    
    print("Generating plots...")
    
    # Query time comparison
    path = plot_query_time_comparison(stats_df, os.path.join(output_dir, 'query_time_comparison.png'))
    if path:
        plot_paths['query_time_comparison'] = path
        print(f"  ✓ Query time comparison")
    
    # Result count distribution
    path = plot_result_count_distribution(stats_df, os.path.join(output_dir, 'result_count_distribution.png'))
    if path:
        plot_paths['result_count_distribution'] = path
        print(f"  ✓ Result count distribution")
    
    # Performance heatmap
    path = plot_performance_heatmap(performance_data, os.path.join(output_dir, 'performance_heatmap.png'))
    if path:
        plot_paths['performance_heatmap'] = path
        print(f"  ✓ Performance heatmap")
    
    # Speedup comparison
    if not speedup_df.empty:
        path = plot_speedup_comparison(speedup_df, os.path.join(output_dir, 'speedup_comparison.png'))
        if path:
            plot_paths['speedup_comparison'] = path
            print(f"  ✓ Speedup comparison")
    
    # Memory usage
    if not memory_df.empty:
        path = plot_memory_usage(memory_df, os.path.join(output_dir, 'memory_usage.png'))
        if path:
            plot_paths['memory_usage'] = path
            print(f"   Memory usage")
    
    print(f"\nGenerated {len(plot_paths)} plots")
    
    return plot_paths
