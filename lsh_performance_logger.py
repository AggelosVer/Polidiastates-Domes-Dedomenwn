import numpy as np
import pandas as pd
import time
from typing import Dict, List
from lsh import MinHashLSH
from preprocess_text import clean_and_tokenize


def measure_lsh_performance(
    df: pd.DataFrame,
    text_attribute: str = 'production_company_names',
    dataset_size: int = 10000,
    num_queries: int = 100,
    num_perm: int = 128,
    threshold: float = 0.5,
    random_seed: int = 42
) -> Dict[str, float]:
    np.random.seed(random_seed)
    
    valid_docs = df[df[text_attribute].notna()].copy()
    if len(valid_docs) == 0:
        raise ValueError(f"No documents with valid '{text_attribute}' found")
    
    dataset_size = min(dataset_size, len(valid_docs))
    sampled_df = valid_docs.sample(n=dataset_size, random_state=random_seed)
    doc_ids = sampled_df.index.tolist()
    
    doc_tokens = {}
    for doc_id in doc_ids:
        text = sampled_df.loc[doc_id, text_attribute]
        tokens = clean_and_tokenize(text)
        if tokens:
            doc_tokens[doc_id] = tokens
    
    doc_ids = list(doc_tokens.keys())
    
    if len(doc_ids) < 2:
        raise ValueError("Need at least 2 documents with valid tokens")
    
    lsh = MinHashLSH(num_perm=num_perm, threshold=threshold)
    
    signature_creation_times = []
    signatures = {}
    
    for doc_id in doc_ids:
        tokens = doc_tokens[doc_id]
        start_sig = time.time()
        sig = lsh.compute_signature(tokens)
        signature_creation_times.append(time.time() - start_sig)
        signatures[doc_id] = sig
    
    total_sig_time = sum(signature_creation_times)
    avg_sig_time = np.mean(signature_creation_times)
    
    start_index = time.time()
    for doc_id in doc_ids:
        lsh.add(doc_id, signatures[doc_id])
    indexing_time = time.time() - start_index
    
    query_times = []
    num_queries = min(num_queries, len(doc_ids))
    query_ids = np.random.choice(doc_ids, size=num_queries, replace=False)
    
    for query_id in query_ids:
        query_sig = signatures[query_id]
        start_query = time.time()
        lsh.query(query_sig)
        query_times.append(time.time() - start_query)
    
    avg_query_time = np.mean(query_times)
    total_query_time = sum(query_times)
    
    dataset_size = len(doc_ids)
    
    return {
        'dataset_size': dataset_size,
        'total_signature_creation_time': total_sig_time,
        'avg_signature_creation_time': avg_sig_time,
        'signature_throughput': dataset_size / total_sig_time if total_sig_time > 0 else 0,
        'indexing_time': indexing_time,
        'indexing_throughput': dataset_size / indexing_time if indexing_time > 0 else 0,
        'total_query_time': total_query_time,
        'avg_query_time': avg_query_time,
        'query_throughput': num_queries / total_query_time if total_query_time > 0 else 0,
        'num_queries': num_queries
    }


def run_performance_benchmark(
    df: pd.DataFrame,
    text_attribute: str = 'production_company_names',
    dataset_sizes: List[int] = [10000, 50000, 100000, 200000],
    num_queries: int = 100,
    num_perm: int = 128,
    threshold: float = 0.5,
    random_seed: int = 42
) -> pd.DataFrame:
    results = []
    
    for size in dataset_sizes:
        try:
            perf = measure_lsh_performance(
                df=df,
                text_attribute=text_attribute,
                dataset_size=size,
                num_queries=num_queries,
                num_perm=num_perm,
                threshold=threshold,
                random_seed=random_seed
            )
            results.append(perf)
        except Exception as e:
            print(f"Error for size {size}: {e}")
            continue
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    from project1_loader import load_and_process_data
    
    df = load_and_process_data('data_movies_clean.csv', apply_filter=False, normalize=False)
    
    if df is None:
        exit(1)
    
    if 'id' in df.columns:
        df.set_index('id', inplace=True)
    elif df.index.name is None:
        df.index.name = 'movie_id'
    
    results_df = run_performance_benchmark(
        df=df,
        text_attribute='production_company_names',
        dataset_sizes=[10000, 50000, 100000, 200000],
        num_queries=100,
        num_perm=128,
        threshold=0.5,
        random_seed=42
    )
    
    print("\n" + "="*120)
    print("LSH PERFORMANCE RESULTS")
    print("="*120)
    if not results_df.empty:
        print(f"{'Size':<10} {'Sig Time (s)':<15} {'Sig Throughput':<18} {'Index Time (s)':<15} "
              f"{'Index Throughput':<18} {'Query Time (s)':<15} {'Query Throughput':<18}")
        print("-"*120)
        for _, row in results_df.iterrows():
            print(f"{int(row['dataset_size']):<10} {row['total_signature_creation_time']:<15.6f} "
                  f"{row['signature_throughput']:<18.2f} {row['indexing_time']:<15.6f} "
                  f"{row['indexing_throughput']:<18.2f} {row['total_query_time']:<15.6f} "
                  f"{row['query_throughput']:<18.2f}")
    print("="*120)

