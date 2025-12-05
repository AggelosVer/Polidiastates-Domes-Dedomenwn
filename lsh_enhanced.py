import numpy as np
import hashlib
import re
from typing import List, Set, Dict, Any, Tuple
from collections import defaultdict


class LSHBanding:

    
    def __init__(self, num_perm: int = 128, num_bands: int = None, rows_per_band: int = None, threshold: float = 0.5):

        self.num_perm = num_perm
        self.threshold = threshold
        
        if num_bands is not None and rows_per_band is not None:
            if num_bands * rows_per_band != num_perm:
                raise ValueError(f"num_bands ({num_bands}) × rows_per_band ({rows_per_band}) must equal num_perm ({num_perm})")
            self.b_bands = num_bands
            self.r_rows = rows_per_band
        else:
            self.b_bands, self.r_rows = self._get_optimal_params(threshold, num_perm)
        
        self.p = 2**61 - 1
        
        np.random.seed(42)
        self.a = np.random.randint(1, self.p, size=num_perm, dtype=np.int64)
        self.b = np.random.randint(0, self.p, size=num_perm, dtype=np.int64)
        
        self.buckets = [defaultdict(list) for _ in range(self.b_bands)]
        
        self.signatures = {}
        
        self.indexed_texts = {}
        
        print(f"LSH Banding: {num_perm} perms, threshold={threshold:.2f} → b={self.b_bands} bands, r={self.r_rows} rows/band")
        print(f"  Estimated similarity threshold: {self._estimated_threshold():.4f}")
    
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
    
    def _estimated_threshold(self) -> float:

        return (1 / self.b_bands) ** (1 / self.r_rows)
    
    def preprocess_text(self, text: str, method: str = 'words') -> List[str]:

        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        if method == 'words':
            tokens = text.split()
        elif method == 'shingles':
            k = 3
            tokens = [text[i:i+k] for i in range(len(text) - k + 1)]
        elif method == 'word_shingles':
            words = text.split()
            k = 2
            tokens = [' '.join(words[i:i+k]) for i in range(len(words) - k + 1)]
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        return [t for t in tokens if t.strip()]
    
    def compute_signature(self, tokens: List[str]) -> np.ndarray:

        if not tokens:
            return np.full(self.num_perm, self.p, dtype=np.int64)
        
        token_hashes = np.array([
            int(hashlib.sha1(t.encode('utf-8')).hexdigest(), 16) % self.p 
            for t in tokens
        ], dtype=np.int64)
        
        signature = np.full(self.num_perm, self.p, dtype=np.int64)
        
        a_col = self.a.reshape(-1, 1)
        b_col = self.b.reshape(-1, 1)
        t_row = token_hashes.reshape(1, -1)
        
        hashes = (a_col * t_row + b_col) % self.p
        signature = np.min(hashes, axis=1)
        
        return signature
    
    def index_text(self, doc_id: Any, text: str, method: str = 'words') -> np.ndarray:

        tokens = self.preprocess_text(text, method)
        
        signature = self.compute_signature(tokens)
        
        self.add(doc_id, signature)
        
        self.indexed_texts[doc_id] = text
        
        return signature
    
    def add(self, doc_id: Any, signature: np.ndarray) -> None:

        self.signatures[doc_id] = signature
        
        for band_idx in range(self.b_bands):
            start = band_idx * self.r_rows
            end = (band_idx + 1) * self.r_rows
            band_sig = tuple(signature[start:end])
            self.buckets[band_idx][band_sig].append(doc_id)
    
    def query_similar(self, query_text: str, method: str = 'words', 
                     top_k: int = None, min_similarity: float = None) -> List[Tuple[Any, float, str]]:

        tokens = self.preprocess_text(query_text, method)
        
        query_sig = self.compute_signature(tokens)
        
        candidates = self.query(query_sig)
        
        results = []
        for doc_id in candidates:
            if doc_id in self.signatures:
                similarity = self.get_jaccard_sim(query_sig, self.signatures[doc_id])
                
                if min_similarity is None or similarity >= min_similarity:
                    text = self.indexed_texts.get(doc_id, "")
                    results.append((doc_id, similarity, text))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def query(self, signature: np.ndarray) -> Set[Any]:

        candidates = set()
        for band_idx in range(self.b_bands):
            start = band_idx * self.r_rows
            end = (band_idx + 1) * self.r_rows
            band_sig = tuple(signature[start:end])
            
            if band_sig in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_sig])
        
        return candidates
    
    def get_jaccard_sim(self, sig1: np.ndarray, sig2: np.ndarray) -> float:

        return float(np.mean(sig1 == sig2))
    
    def get_stats(self) -> Dict[str, Any]:

        total_buckets = sum(len(bucket) for bucket in self.buckets)
        avg_bucket_size = np.mean([len(docs) for bucket in self.buckets for docs in bucket.values()]) if total_buckets > 0 else 0
        
        return {
            'num_documents': len(self.signatures),
            'num_permutations': self.num_perm,
            'num_bands': self.b_bands,
            'rows_per_band': self.r_rows,
            'threshold': self.threshold,
            'estimated_threshold': self._estimated_threshold(),
            'total_buckets': total_buckets,
            'avg_bucket_size': avg_bucket_size
        }
    
    def clear(self) -> None:

        self.buckets = [defaultdict(list) for _ in range(self.b_bands)]
        self.signatures = {}
        self.indexed_texts = {}


if __name__ == "__main__":
    print("=" * 70)
    print("LSH BANDING TECHNIQUE DEMONSTRATION")
    print("=" * 70)
    
    lsh = LSHBanding(num_perm=128, threshold=0.5)
    
    print("\n1. INDEXING TEXT ENTRIES")
    print("-" * 70)
    
    documents = {
        "movie1": "action adventure sci-fi thriller space exploration alien",
        "movie2": "action adventure sci-fi fantasy epic journey quest",
        "movie3": "romantic comedy drama love story wedding",
        "movie4": "horror thriller suspense mystery investigation crime",
        "movie5": "action adventure thriller heist crime investigation",
        "movie6": "sci-fi thriller dystopian future technology AI",
        "movie7": "romantic drama love emotional relationships family"
    }
    
    for doc_id, text in documents.items():
        lsh.index_text(doc_id, text, method='words')
        print(f"  Indexed: {doc_id} - '{text[:50]}...'")
    
    print("\n2. QUERYING SIMILAR ITEMS")
    print("-" * 70)
    
    queries = [
        "action adventure space aliens sci-fi",
        "romantic love story drama",
        "thriller crime mystery"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = lsh.query_similar(query, method='words', top_k=3, min_similarity=0.2)
        
        if results:
            print(f"  Found {len(results)} similar documents:")
            for rank, (doc_id, similarity, text) in enumerate(results, 1):
                print(f"    {rank}. {doc_id} (similarity: {similarity:.3f})")
                print(f"       '{text[:60]}...'")
        else:
            print("  No similar documents found")
    
    print("\n3. LSH STATISTICS")
    print("-" * 70)
    stats = lsh.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n4. TESTING DIFFERENT BAND CONFIGURATIONS")
    print("-" * 70)
    
    configs = [
        (128, 64, 2),  # 64 bands, 2 rows each
        (128, 32, 4),  # 32 bands, 4 rows each  
        (128, 16, 8),  # 16 bands, 8 rows each
    ]
    
    for num_perm, b, r in configs:
        lsh_test = LSHBanding(num_perm=num_perm, num_bands=b, rows_per_band=r)
        threshold = lsh_test._estimated_threshold()
        print(f"  Config: b={b}, r={r} → threshold ≈ {threshold:.4f}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
