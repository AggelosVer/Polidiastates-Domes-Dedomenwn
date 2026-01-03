import subprocess
import sys

# Run scripts in order
scripts = [
    'extract_5d_vectors.py',
    'benchmark_runner.py',
    'memory_profiler.py',
    'comprehensive_evaluation.py'
]

for script in scripts:
    subprocess.run([sys.executable, script])
