import numpy as np
import hashlib
from typing import List, Set, Union
from preprocess_text import clean_and_tokenize


class MinHashSignatureGenerator:
    def __init__(self, num_hash_functions: int = 128, seed: int = 42):
        self.num_hash_functions = num_hash_functions
        self.seed = seed
        self.p = 2**61 - 1
        np.random.seed(seed)
        self.a = np.random.randint(1, self.p, size=num_hash_functions, dtype=np.int64)
        self.b = np.random.randint(0, self.p, size=num_hash_functions, dtype=np.int64)
    
    def _hash_token(self, token: str) -> int:
        token_bytes = token.encode('utf-8')
        hash_value = int(hashlib.sha1(token_bytes).hexdigest(), 16)
        return hash_value % self.p
    
    def compute_signature(self, tokens: Union[List[str], Set[str], str]) -> np.ndarray:
        if isinstance(tokens, str):
            tokens = clean_and_tokenize(tokens)
        
        if not tokens:
            return np.full(self.num_hash_functions, self.p, dtype=np.int64)
        
        if isinstance(tokens, set):
            tokens = list(tokens)
        
        token_hashes = np.array([self._hash_token(t) for t in tokens], dtype=np.int64)
        
        signature = np.full(self.num_hash_functions, self.p, dtype=np.int64)
        
        a_col = self.a.reshape(-1, 1)
        b_col = self.b.reshape(-1, 1)
        t_row = token_hashes.reshape(1, -1)
        
        hashes = (a_col * t_row + b_col) % self.p
        signature = np.min(hashes, axis=1)
        
        return signature
    
    def compute_signatures_batch(self, token_lists: List[Union[List[str], Set[str], str]]) -> np.ndarray:
        signatures = []
        for tokens in token_lists:
            sig = self.compute_signature(tokens)
            signatures.append(sig)
        return np.array(signatures)
    
    def estimate_jaccard_similarity(self, signature1: np.ndarray, signature2: np.ndarray) -> float:
        if signature1.shape != signature2.shape:
            raise ValueError("Signatures must have the same length")
        return np.mean(signature1 == signature2)
    
    def get_num_hash_functions(self) -> int:
        return self.num_hash_functions
    
    def set_num_hash_functions(self, num_hash_functions: int):
        self.num_hash_functions = num_hash_functions
        np.random.seed(self.seed)
        self.a = np.random.randint(1, self.p, size=num_hash_functions, dtype=np.int64)
        self.b = np.random.randint(0, self.p, size=num_hash_functions, dtype=np.int64)


if __name__ == "__main__":
    gen = MinHashSignatureGenerator(num_hash_functions=128)
    
    text1 = "['Warner Bros.', 'DC Entertainment']"
    text2 = "['Warner Bros.', 'Legendary Pictures']"
    text3 = "['Disney', 'Marvel Studios']"
    
    tokens1 = clean_and_tokenize(text1)
    tokens2 = clean_and_tokenize(text2)
    tokens3 = clean_and_tokenize(text3)
    
    sig1 = gen.compute_signature(tokens1)
    sig2 = gen.compute_signature(tokens2)
    sig3 = gen.compute_signature(tokens3)
    
    gen.set_num_hash_functions(256)
    sig1_new = gen.compute_signature(tokens1)
    sig2_new = gen.compute_signature(tokens2)

