# Experimental Results Report
**Generated:** 2026-01-02 23:50:21
**Evaluation System:** Multi-dimensional Indexing + LSH
---
## Executive Summary
This report summarizes the performance evaluation of 4 index structures across 5 queries. The evaluation combines multi-dimensional spatial indexing with Locality-Sensitive Hashing (LSH) for similarity-based filtering.

**Key Findings:**
- **Fastest Scheme:** Kd Tree (120.00 μs avg)
- **Most Consistent:** R Tree (σ = 0.000000s)
- **Most Results:** Kd Tree (1.0 avg results)

---
## Performance Metrics
### Query Performance Statistics
| Scheme     |   Avg Total Time (s) |   Std Total Time (s) |   Min Total Time (s) |   Max Total Time (s) |   Avg Tree Time (s) |   Avg LSH Time (s) |   Avg Tree Count |   Avg Final Count |   Median Total Time (s) |
|:-----------|---------------------:|---------------------:|---------------------:|---------------------:|--------------------:|-------------------:|-----------------:|------------------:|------------------------:|
| Kd Tree    |              0.00012 |                4e-05 |               0.0001 |               0.0002 |             0       |             0.0001 |               31 |                 1 |                  0.0001 |
| Quadtree   |              0.00034 |                8e-05 |               0.0003 |               0.0005 |             0.00024 |             0.0001 |               31 |                 1 |                  0.0003 |
| Range Tree |              0.00022 |                4e-05 |               0.0002 |               0.0003 |             0.00016 |             0.0001 |               31 |                 1 |                  0.0002 |
| R Tree     |              0.0003  |                0     |               0.0003 |               0.0003 |             0.00026 |             0.0001 |               31 |                 1 |                  0.0003 |

![Query Time Comparison](artifacts\query_time_comparison.png)
*Figure 1: Average query time breakdown by index scheme*

### Speedup Analysis
| Scheme     |   Avg Total Time (s) |   Speedup vs R Tree |   Percentage Improvement |
|:-----------|---------------------:|--------------------:|-------------------------:|
| Kd Tree    |              0.00012 |            2.5      |                 150      |
| Quadtree   |              0.00034 |            0.882353 |                 -11.7647 |
| Range Tree |              0.00022 |            1.36364  |                  36.3636 |
| R Tree     |              0.0003  |            1        |                   0      |

![Speedup Comparison](artifacts\speedup_comparison.png)
*Figure 2: Speedup ratios relative to baseline*

---
## Result Distribution
This section shows how many results each index structure returns before and after LSH filtering.

![Result Count Distribution](artifacts\result_count_distribution.png)
*Figure 3: Average result counts from tree queries vs final LSH-filtered results*

---
## Query-by-Query Performance

![Performance Heatmap](artifacts\performance_heatmap.png)
*Figure 4: Performance heatmap showing query times across all queries and schemes*

### Individual Query Results
|   Query ID | Title                                | Kd Tree   | Quadtree   | Range Tree   | R Tree   |
|-----------:|:-------------------------------------|:----------|:-----------|:-------------|:---------|
|     606425 | Rodney King                          | 0.0002s   | 0.0005s    | 0.0002s      | 0.0003s  |
|     308586 | Ice Cube: The Making of a Don        | 0.0001s   | 0.0003s    | 0.0003s      | 0.0003s  |
|     297637 | Pass The Mic!                        | 0.0001s   | 0.0003s    | 0.0002s      | 0.0003s  |
|     620126 | Russia 1917: Countdown to Revolution | 0.0001s   | 0.0003s    | 0.0002s      | 0.0003s  |
|     485204 | The Evil Dead Inbred Rednecks        | 0.0001s   | 0.0003s    | 0.0002s      | 0.0003s  |

---
## Memory Analysis
Memory profiling results for different index structures and dataset sizes.

### Memory Usage Summary
| Index Structure   |   Dataset Size |   Build Time (s) |   sys.getsizeof Deep (MB) |   sys.getsizeof Deep (KB) |   sys.getsizeof Shallow (KB) |   tracemalloc Current (MB) |   tracemalloc Peak (MB) |   tracemalloc Allocated (MB) |   tracemalloc Current (KB) |   tracemalloc Peak (KB) |
|:------------------|---------------:|-----------------:|--------------------------:|--------------------------:|-----------------------------:|---------------------------:|------------------------:|-----------------------------:|---------------------------:|------------------------:|
| KD-Tree           |             45 |           0.0003 |                    0.0181 |                     18.58 |                         0.05 |                     0.0159 |                  0.0161 |                       0.0152 |                      16.31 |                   16.51 |
| QuadTree          |             45 |           0.0034 |                    0.0331 |                     33.86 |                         0.05 |                     0.0248 |                  0.0252 |                       0.0241 |                      25.35 |                   25.83 |
| Range Tree        |             45 |           0.2309 |                    7.1931 |                   7365.71 |                         0.05 |                     4.315  |                  4.3417 |                       4.3143 |                    4418.52 |                 4445.91 |
| R-Tree            |             45 |           0.0318 |                    0.0292 |                     29.95 |                         0.05 |                     0.0149 |                  0.0163 |                       0.0144 |                      15.23 |                   16.73 |

![Memory Usage](artifacts\memory_usage.png)
*Figure 5: Memory consumption and build time analysis*

### Memory Efficiency Rankings

**Dataset Size: 45 points**
- Most Efficient: KD-Tree
- Least Efficient: Range Tree
- Average Memory: 1.82 MB

---
## Discussion
### Performance Insights
The evaluation reveals that **Kd Tree** achieves the best average query performance, while **Quadtree** shows the slowest average times. In terms of consistency, **R Tree** demonstrates the lowest standard deviation, indicating more predictable performance across different queries.

### LSH Impact
The combination of spatial indexing with LSH provides a two-phase filtering approach:
1. **Spatial filtering** narrows down candidates based on 5D feature similarity
2. **LSH filtering** refines results based on textual similarity

This hybrid approach balances precision and recall while maintaining query efficiency.

### Memory Considerations
Memory profiling shows varying space requirements across index structures. The choice of index should consider both query performance and memory constraints based on the specific deployment environment.

### Trade-offs
- **Speed vs Memory**: Faster indexes may require more memory
- **Build Time vs Query Time**: Some structures have longer build times but faster queries
- **Precision vs Recall**: LSH threshold affects the balance between result quality and quantity

---
## Methodology
### Index Structures Evaluated
- **Kd Tree**: Combined with LSH for similarity filtering
- **Quadtree**: Combined with LSH for similarity filtering
- **Range Tree**: Combined with LSH for similarity filtering
- **R Tree**: Combined with LSH for similarity filtering

### Evaluation Process
1. **Data Filtering**: Movies filtered by release year (2000-2020), popularity (3-6), vote average (3-5), runtime (30-60 min)
2. **5D Vector Extraction**: Features extracted from movie metadata
3. **Index Construction**: All index structures built on the same dataset
4. **LSH Setup**: MinHash LSH with 128 permutations, threshold 0.5
5. **Query Execution**: Range queries on 5D space + LSH similarity filtering
6. **Metrics Collection**: Total time, tree time, LSH time, result counts

---
## Conclusion
This comprehensive evaluation demonstrates the effectiveness of combining multi-dimensional indexing with LSH for similarity search in movie datasets. **Kd Tree** emerges as the recommended choice for query-intensive workloads where performance is critical. 

The results provide valuable insights for selecting appropriate index structures based on specific application requirements, dataset characteristics, and performance constraints.

---

*Report generated by Statistical Report Generator v1.0*
