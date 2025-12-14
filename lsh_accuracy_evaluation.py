import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
import random
from lsh import MinHashLSH
from preprocess_text import clean_and_tokenize


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_ground_truth_similarities(
    doc_ids: List[int],
    doc_tokens: Dict[int, List[str]],
    similarity_threshold: float = 0.3
) -> Dict[Tuple[int, int], float]:
    ground_truth = {}
    doc_sets = {doc_id: set(tokens) for doc_id, tokens in doc_tokens.items()}
    
    for i, doc_id1 in enumerate(doc_ids):
        tokens1 = doc_sets[doc_id1]
        for doc_id2 in doc_ids[i+1:]:
            tokens2 = doc_sets[doc_id2]
            similarity = jaccard_similarity(tokens1, tokens2)
            if similarity >= similarity_threshold:
                ground_truth[(doc_id1, doc_id2)] = similarity
                ground_truth[(doc_id2, doc_id1)] = similarity
    
    return ground_truth


def get_ground_truth_similar_docs(
    query_id: int,
    doc_ids: List[int],
    ground_truth: Dict[Tuple[int, int], float],
    min_similarity: float = 0.3
) -> Set[int]:
    similar_docs = set()
    for doc_id in doc_ids:
        if doc_id == query_id:
            continue
        pair = (query_id, doc_id)
        if pair in ground_truth and ground_truth[pair] >= min_similarity:
            similar_docs.add(doc_id)
    return similar_docs


def evaluate_lsh_accuracy(
    df: pd.DataFrame,
    text_attribute: str = 'production_company_names',
    sample_size: int = 100,
    num_queries: int = 10,
    num_perm: int = 128,
    threshold: float = 0.5,
    similarity_threshold: float = 0.3,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, float]:
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    valid_docs = df[df[text_attribute].notna()].copy()
    if len(valid_docs) == 0:
        raise ValueError(f"No documents with valid '{text_attribute}' found")
    
    sample_size = min(sample_size, len(valid_docs))
    sampled_df = valid_docs.sample(n=sample_size, random_state=random_seed)
    doc_ids = sampled_df.index.tolist()
    
    doc_tokens = {}
    for doc_id in doc_ids:
        text = sampled_df.loc[doc_id, text_attribute]
        tokens = clean_and_tokenize(text)
        if tokens:
            doc_tokens[doc_id] = tokens
    
    doc_ids = list(doc_tokens.keys())
    
    if len(doc_ids) < 2:
        raise ValueError("Need at least 2 documents with valid tokens for evaluation")
    
    ground_truth = compute_ground_truth_similarities(doc_ids, doc_tokens, similarity_threshold)
    
    lsh = MinHashLSH(num_perm=num_perm, threshold=threshold)
    signatures = {}
    
    for doc_id in doc_ids:
        tokens = doc_tokens[doc_id]
        sig = lsh.compute_signature(tokens)
        signatures[doc_id] = sig
        lsh.add(doc_id, sig)
    
    query_ids = random.sample(doc_ids, min(num_queries, len(doc_ids)))
    
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for query_id in query_ids:
        query_sig = signatures[query_id]
        lsh_candidates = lsh.query(query_sig)
        lsh_candidates.discard(query_id)
        
        ground_truth_similar = get_ground_truth_similar_docs(
            query_id, doc_ids, ground_truth, similarity_threshold
        )
        
        tp = len(lsh_candidates & ground_truth_similar)
        fp = len(lsh_candidates - ground_truth_similar)
        fn = len(ground_truth_similar - lsh_candidates)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    macro_precision = np.mean(all_precisions)
    macro_recall = np.mean(all_recalls)
    macro_f1 = np.mean(all_f1_scores)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    results = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'avg_lsh_candidates': np.mean([len(lsh.query(signatures[q])) - 1 for q in query_ids]),
        'avg_ground_truth_similar': np.mean([len(get_ground_truth_similar_docs(q, doc_ids, ground_truth, similarity_threshold)) for q in query_ids]),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'num_queries': num_queries
    }
    
    if verbose:
        print(f"Macro Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
        print(f"Micro Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1: {micro_f1:.4f}")
    
    return results


if __name__ == "__main__":
    from project1_loader import load_and_process_data
    
    df = load_and_process_data('data_movies_clean.csv', apply_filter=False, normalize=False)
    
    if df is None:
        exit(1)
    
    if 'id' in df.columns:
        df.set_index('id', inplace=True)
    elif df.index.name is None:
        df.index.name = 'movie_id'
    
    results = evaluate_lsh_accuracy(
        df=df,
        text_attribute='production_company_names',
        sample_size=200,
        num_queries=10,
        num_perm=128,
        threshold=0.5,
        similarity_threshold=0.3,
        random_seed=42
    )

