import numpy as np
import hashlib
from typing import List, Set, Dict, Any, Tuple

class MinHashLSH:
    def __init__(self, num_perm: int = 128, threshold: float = 0.5):
        self.num_perm = num_perm
        self.threshold = threshold
        self.p = 2**61 - 1
        np.random.seed(42)
        self.a = np.random.randint(1, self.p, size=num_perm, dtype=np.int64)
        self.b = np.random.randint(0, self.p, size=num_perm, dtype=np.int64)
        self.b_bands, self.r_rows = self._get_optimal_params(threshold, num_perm)
        print(f"LSH Initialized: {num_perm} perms, threshold={threshold:.2f} -> b={self.b_bands}, r={self.r_rows}")
        self.buckets = [{} for _ in range(self.b_bands)]
        self.signatures = {}

    def _get_optimal_params(self, threshold: float, num_perm: int) -> Tuple[int, int]:
        best_b, best_r = 1, num_perm
        min_error = float('inf')
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                t = (1/b)**(1/r)
                error = abs(t - threshold)
                if error < min_error:
                    min_error = error
                    best_b = b
                    best_r = r
        return best_b, best_r

    def compute_signature(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.full(self.num_perm, self.p, dtype=np.int64)
        token_hashes = np.array([int(hashlib.sha1(t.encode('utf-8')).hexdigest(), 16) % self.p 
                                 for t in tokens], dtype=np.int64)
        signature = np.full(self.num_perm, self.p, dtype=np.int64)
        a_col = self.a.reshape(-1, 1)
        b_col = self.b.reshape(-1, 1)
        t_row = token_hashes.reshape(1, -1)
        hashes = (a_col * t_row + b_col) % self.p
        signature = np.min(hashes, axis=1)
        return signature

    def add(self, doc_id: Any, signature: np.ndarray):
        self.signatures[doc_id] = signature
        for i in range(self.b_bands):
            start = i * self.r_rows
            end = (i + 1) * self.r_rows
            band_sig = tuple(signature[start:end])
            if band_sig not in self.buckets[i]:
                self.buckets[i][band_sig] = []
            self.buckets[i][band_sig].append(doc_id)

    def query(self, signature: np.ndarray) -> Set[Any]:
        candidates = set()
        for i in range(self.b_bands):
            start = i * self.r_rows
            end = (i + 1) * self.r_rows
            band_sig = tuple(signature[start:end])
            if band_sig in self.buckets[i]:
                candidates.update(self.buckets[i][band_sig])
        return candidates
        
    def get_jaccard_sim(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        return np.mean(sig1 == sig2)

def find_top_n_similar_movies(
    filtered_movie_ids: List[int],
    df,
    text_attribute: str = 'production_company_names',
    query_movie_id: int = None,
    top_n: int = 10,
    num_perm: int = 128,
    threshold: float = 0.5
) -> List[Tuple[int, float, List[str]]]:

    import pandas as pd
    import ast
    import re
    
    def clean_and_tokenize(text):

        if pd.isna(text):
            return []
        try:

            if isinstance(text, str) and text.strip().startswith('[') and text.strip().endswith(']'):
                items = ast.literal_eval(text)
            else:
                items = [text]
        except (ValueError, SyntaxError):
            items = [text]
        
        if not isinstance(items, list):
            items = [str(items)]
        
        all_tokens = set()
        for item in items:
            if not isinstance(item, str):
                continue
            item_lower = item.lower()
            item_clean = re.sub(r'[^\w\s]', '', item_lower)
            tokens = item_clean.split()
            all_tokens.update(tokens)
        
        return list(all_tokens)
    

    if text_attribute not in df.columns:
        raise ValueError(f"Column '{text_attribute}' not found in DataFrame")
    
    if not filtered_movie_ids:
        raise ValueError("filtered_movie_ids list cannot be empty")
    
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    
    if query_movie_id is None:
        query_movie_id = filtered_movie_ids[0]
    

    if query_movie_id not in filtered_movie_ids:
        raise ValueError(f"query_movie_id {query_movie_id} not in filtered_movie_ids")
    
    print(f"Building LSH index for {len(filtered_movie_ids)} movies...")
    print(f"Text attribute: '{text_attribute}'")
    print(f"Query movie ID: {query_movie_id}")
    

    lsh = MinHashLSH(num_perm=num_perm, threshold=threshold)  

    movie_tokens = {}
    signatures = {}
    
    for movie_id in filtered_movie_ids:

        if movie_id not in df.index:
            print(f"Warning: Movie ID {movie_id} not found in dataset, skipping")
            continue
        
        row = df.loc[movie_id]
        text_raw = row.get(text_attribute, '')
        
        if pd.isna(text_raw) or text_raw == '':
            text_raw = ''
        

        tokens = clean_and_tokenize(text_raw)
        movie_tokens[movie_id] = tokens
        

        sig = lsh.compute_signature(tokens)
        signatures[movie_id] = sig
        

        lsh.add(movie_id, sig)
    
    print(f"Indexed {len(signatures)} movies")
    

    if query_movie_id not in signatures:
        raise ValueError(f"Query movie {query_movie_id} has no valid signature")
    
    query_sig = signatures[query_movie_id]
    query_tokens = movie_tokens[query_movie_id]
    
    print(f"Query movie tokens: {query_tokens}")
    

    candidates = lsh.query(query_sig)
    print(f"LSH returned {len(candidates)} candidates")
    

    results = []
    for movie_id in candidates:
        if movie_id == query_movie_id:

            continue
        
        if movie_id in signatures:
            similarity = lsh.get_jaccard_sim(query_sig, signatures[movie_id])
            tokens = movie_tokens.get(movie_id, [])
            results.append((movie_id, similarity, tokens))
    

    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:top_n]
    
    print(f"\nTop {len(top_results)} similar movies:")
    for rank, (movie_id, sim, tokens) in enumerate(top_results, 1):
        print(f"  {rank}. Movie ID {movie_id}: similarity={sim:.4f}, tokens={tokens[:5]}...")
    
    return top_results


if __name__ == "__main__":
    print("=== Testing MinHash LSH ===")
    lsh = MinHashLSH(num_perm=128, threshold=0.5)
    doc1 = ["action", "adventure", "sci-fi"]
    doc2 = ["action", "adventure", "romance"]
    doc3 = ["comedy", "drama", "romance"]
    doc4 = ["action", "adventure", "sci-fi", "thriller"]
    sig1 = lsh.compute_signature(doc1)
    sig2 = lsh.compute_signature(doc2)
    sig3 = lsh.compute_signature(doc3)
    sig4 = lsh.compute_signature(doc4)
    lsh.add("doc1", sig1)
    lsh.add("doc2", sig2)
    lsh.add("doc3", sig3)
    print(f"Doc1: {doc1}")
    print(f"Doc2: {doc2}")
    print(f"Doc3: {doc3}")
    print(f"Doc4: {doc4}")
    print(f"\nSim(Doc1, Doc2) Est: {lsh.get_jaccard_sim(sig1, sig2):.4f}")
    print(f"Sim(Doc1, Doc3) Est: {lsh.get_jaccard_sim(sig1, sig3):.4f}")
    print(f"Sim(Doc1, Doc4) Est: {lsh.get_jaccard_sim(sig1, sig4):.4f}")
    print("\nQuerying for Doc4 (should be similar to Doc1):")
    candidates = lsh.query(sig4)
    print(f"Candidates: {candidates}")
    
    print("\n\n=== Testing find_top_n_similar_movies ===")
    try:
        import pandas as pd
        from project1_loader import load_and_process_data
        

        df = load_and_process_data('data_movies_clean.csv', apply_filter=False, normalize=False)
        
        if df is not None and len(df) > 0:

            sample_ids = df.index[:100].tolist()
            

            results = find_top_n_similar_movies(
                filtered_movie_ids=sample_ids,
                df=df,
                text_attribute='production_company_names',
                query_movie_id=sample_ids[0],
                top_n=5,
                num_perm=128,
                threshold=0.5
            )
            
            print(f"\nFound {len(results)} similar movies")
        else:
            print("Could not load data for extended testing")
    except Exception as e:
        print(f"Extended test skipped: {e}")
