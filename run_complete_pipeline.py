
import subprocess
import sys
import time

print("=" * 70)
print("  COMPLETE PROJECT PIPELINE")
print("=" * 70)


scripts = [
    ('save_preprocessed.py', 'Step 1: Preprocessing and filtering movie data'),
    ('extract_5d_vectors.py', 'Step 2: Extracting 5D vectors for indexing'),
    ('memory_profiler.py', 'Step 3: Memory profiling all index structures'),
    ('comprehensive_evaluation.py', 'Step 4: Running comprehensive evaluation'),
]

total_start = time.time()

for script, description in scripts:
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  Running: {script}")
    print("=" * 70)
    
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n ERROR: {script} failed with code {result.returncode}")
        sys.exit(1)
    
    print(f"\n {script} completed in {elapsed:.1f}s")

total_elapsed = time.time() - total_start

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE!")
print("=" * 70)
print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
print("\nGenerated files:")
print("  movies_preprocessed.csv       - Filtered dataset")
print("  movie_5d_vectors.npy          - 5D vectors (binary)")
print("  movie_5d_vectors.csv          - 5D vectors (CSV)")
print("  memory_profiling_results.csv  - Memory benchmarks")
print("  evaluation_results/           - Query results")
print("=" * 70)
