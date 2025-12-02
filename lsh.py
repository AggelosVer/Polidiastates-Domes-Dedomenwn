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
