import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def compute_query_statistics(performance_data: Dict[str, Any]) -> pd.DataFrame:

    scheme_metrics = defaultdict(lambda: {
        'total_times': [],
        'tree_times': [],
        'lsh_times': [],
        'tree_counts': [],
        'final_counts': []
    })
    
    for query in performance_data.get('queries', []):
        for scheme_name, metrics in query.get('schemes', {}).items():
            scheme_metrics[scheme_name]['total_times'].append(metrics.get('total_time', 0))
            scheme_metrics[scheme_name]['tree_times'].append(metrics.get('tree_time', 0))
            scheme_metrics[scheme_name]['lsh_times'].append(metrics.get('lsh_time', 0))
            scheme_metrics[scheme_name]['tree_counts'].append(metrics.get('tree_count', 0))
            scheme_metrics[scheme_name]['final_counts'].append(metrics.get('final_count', 0))
    
    stats_data = []
    for scheme_name, metrics in scheme_metrics.items():
        stats_data.append({
            'Scheme': scheme_name.replace('_', ' ').title(),
            'Avg Total Time (s)': np.mean(metrics['total_times']),
            'Std Total Time (s)': np.std(metrics['total_times']),
            'Min Total Time (s)': np.min(metrics['total_times']),
            'Max Total Time (s)': np.max(metrics['total_times']),
            'Avg Tree Time (s)': np.mean(metrics['tree_times']),
            'Avg LSH Time (s)': np.mean(metrics['lsh_times']),
            'Avg Tree Count': np.mean(metrics['tree_counts']),
            'Avg Final Count': np.mean(metrics['final_counts']),
            'Median Total Time (s)': np.median(metrics['total_times']),
        })
    
    return pd.DataFrame(stats_data)


def compute_scheme_comparison(performance_data: Dict[str, Any]) -> Dict[str, Any]:

    stats_df = compute_query_statistics(performance_data)
    
    if stats_df.empty:
        return {}
    
    comparison = {
        'fastest_avg': stats_df.loc[stats_df['Avg Total Time (s)'].idxmin(), 'Scheme'],
        'fastest_avg_time': stats_df['Avg Total Time (s)'].min(),
        'slowest_avg': stats_df.loc[stats_df['Avg Total Time (s)'].idxmax(), 'Scheme'],
        'slowest_avg_time': stats_df['Avg Total Time (s)'].max(),
        'most_consistent': stats_df.loc[stats_df['Std Total Time (s)'].idxmin(), 'Scheme'],
        'most_consistent_std': stats_df['Std Total Time (s)'].min(),
        'most_results': stats_df.loc[stats_df['Avg Final Count'].idxmax(), 'Scheme'],
        'most_results_count': stats_df['Avg Final Count'].max(),
    }
    
    return comparison


def identify_best_performer(stats_df: pd.DataFrame, metric: str = 'Avg Total Time (s)') -> Tuple[str, float]:

    if stats_df.empty:
        return ("N/A", 0.0)
    
    if 'Time' in metric or 'Std' in metric:
        idx = stats_df[metric].idxmin()
    else:
        idx = stats_df[metric].idxmax()
    
    return stats_df.loc[idx, 'Scheme'], stats_df.loc[idx, metric]


def compute_speedup_ratios(stats_df: pd.DataFrame, baseline_scheme: str = 'R Tree') -> pd.DataFrame:

    if stats_df.empty:
        return pd.DataFrame()
    
    baseline_row = stats_df[stats_df['Scheme'] == baseline_scheme]
    if baseline_row.empty:
        baseline_row = stats_df.iloc[0:1]
        baseline_scheme = baseline_row['Scheme'].values[0]
    
    baseline_time = baseline_row['Avg Total Time (s)'].values[0]
    
    speedup_data = []
    for _, row in stats_df.iterrows():
        speedup = baseline_time / row['Avg Total Time (s)'] if row['Avg Total Time (s)'] > 0 else 0
        speedup_data.append({
            'Scheme': row['Scheme'],
            'Avg Total Time (s)': row['Avg Total Time (s)'],
            f'Speedup vs {baseline_scheme}': speedup,
            'Percentage Improvement': (speedup - 1) * 100 if speedup > 0 else 0
        })
    
    return pd.DataFrame(speedup_data)


def generate_summary_statistics(performance_data: Dict[str, Any]) -> Dict[str, Any]:

    stats_df = compute_query_statistics(performance_data)
    comparison = compute_scheme_comparison(performance_data)
    
    num_queries = len(performance_data.get('queries', []))
    
    summary = {
        'num_queries': num_queries,
        'statistics_table': stats_df,
        'comparison': comparison,
        'schemes_tested': stats_df['Scheme'].tolist() if not stats_df.empty else [],
    }
    
    if not stats_df.empty:
        speedup_df = compute_speedup_ratios(stats_df)
        summary['speedup_table'] = speedup_df
    
    return summary


def analyze_memory_results(memory_csv_path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(memory_csv_path)
        
        # Convert string columns to numeric
        numeric_cols = ['sys.getsizeof Deep (MB)', 'tracemalloc Peak (MB)', 'Build Time (s)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading memory results: {e}")
        return pd.DataFrame()


def compute_memory_statistics(memory_df: pd.DataFrame) -> Dict[str, Any]:

    if memory_df.empty:
        return {}
    
    stats = {}
    
    # Group by dataset size
    for size in memory_df['Dataset Size'].unique():
        subset = memory_df[memory_df['Dataset Size'] == size]
        
        stats[f'size_{size}'] = {
            'most_efficient': subset.loc[subset['sys.getsizeof Deep (MB)'].idxmin(), 'Index Structure'],
            'least_efficient': subset.loc[subset['sys.getsizeof Deep (MB)'].idxmax(), 'Index Structure'],
            'avg_memory_mb': subset['sys.getsizeof Deep (MB)'].mean(),
            'total_memory_mb': subset['sys.getsizeof Deep (MB)'].sum(),
        }
    
    return stats


def format_time(seconds: float) -> str:
 
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.4f} s"


def format_memory(mb: float) -> str:

    if mb < 1:
        return f"{mb * 1024:.2f} KB"
    elif mb < 1024:
        return f"{mb:.2f} MB"
    else:
        return f"{mb / 1024:.2f} GB"
