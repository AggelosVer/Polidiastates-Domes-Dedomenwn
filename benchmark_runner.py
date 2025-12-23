import time
import numpy as np
import pandas as pd
from index_manager import IndexManager

class BenchmarkRunner:
    def __init__(self, output_file='benchmark_results.csv'):
        self.sizes = [10000, 50000, 100000]
        self.index_types = ['kdtree', 'quadtree', 'rangetree', 'rtree']
        self.dimension = 5
        self.output_file = output_file
        self.results = []

    def generate_data(self, size):
        return np.random.random((size, self.dimension))

    def run(self):
        for size in self.sizes:
            print(f"Benchmarking dataset size: {size}")
            data = self.generate_data(size)
            points = [tuple(x) for x in data]
            payloads = list(range(size))

            for index_type in self.index_types:
                if index_type == 'rangetree' and size > 10000:
                    continue
                print(f"  Testing {index_type}...")
                
                manager = IndexManager(index_type, dimension=self.dimension)
                
                start = time.time()
                manager.build(points, payloads)
                build_time = time.time() - start

                insert_time = -1
                try:
                    insert_points = self.generate_data(1000)
                    start = time.time()
                    for p in insert_points:
                        manager.insert(tuple(p), 0)
                    insert_time = time.time() - start
                except Exception:
                    pass

                range_query_time = -1
                try:
                    start = time.time()
                    for _ in range(100):
                        r_min = np.random.random(self.dimension) * 0.5
                        r_max = r_min + 0.2
                        manager.range_query(list(r_min), list(r_max))
                    range_query_time = time.time() - start
                except Exception:
                    pass

                knn_time = -1
                try:
                    start = time.time()
                    for _ in range(100):
                        q = np.random.random(self.dimension)
                        manager.knn_query(list(q), k=10)
                    knn_time = time.time() - start
                except Exception:
                    pass

                delete_time = -1
                try:
                    delete_indices = np.random.choice(len(points), 1000, replace=False)
                    to_delete = [points[i] for i in delete_indices]
                    start = time.time()
                    for p in to_delete:
                        manager.delete(p)
                    delete_time = time.time() - start
                except Exception:
                    pass

                update_time = -1
                try:
                    update_indices = np.random.choice(len(points), 1000, replace=False)
                    to_update = [points[i] for i in update_indices]
                    new_points = self.generate_data(1000)
                    start = time.time()
                    for i, p in enumerate(to_update):
                        manager.update(p, tuple(new_points[i]))
                    update_time = time.time() - start
                except Exception:
                    pass

                self.results.append({
                    'size': size,
                    'type': index_type,
                    'build_time': build_time,
                    'insert_time': insert_time,
                    'delete_time': delete_time,
                    'update_time': update_time,
                    'range_query_time': range_query_time,
                    'knn_time': knn_time
                })

        df = pd.DataFrame(self.results)
        df.to_csv(self.output_file, index=False)
        print(f"Results saved to {self.output_file}")
        print(df)

if __name__ == '__main__':
    runner = BenchmarkRunner()
    runner.run()
