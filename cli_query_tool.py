#!/usr/bin/env python3

import argparse
import sys
import json
from typing import Tuple, List
from query_pipeline import MovieQueryPipeline

TEXTUAL_ATTRIBUTES = {
    'production_company_names': 'Production Company Names',
    'genre_names': 'Genre Names'
}

NUMERIC_ATTRIBUTES = {
    'release_year': 'Release Year',
    'popularity': 'Popularity Score',
    'vote_average': 'Average Vote Rating',
    'runtime': 'Runtime (minutes)',
    'budget': 'Budget (USD)',
    'revenue': 'Revenue (USD)',
    'vote_count': 'Number of Votes'
}

DEFAULT_SPATIAL_ATTRIBUTES = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']

INDEX_TYPES = {
    'kd': 'kdtree',
    'quad': 'quadtree',
    'range': 'rangetree',
    'rtree': 'rtree'
}


def parse_numeric_filter(filter_str: str) -> Tuple[str, float, float]:
    try:
        parts = filter_str.split(':')
        if len(parts) < 2:
            raise ValueError("Filter must be in format: attribute:min:max or attribute:value1,value2")
        
        attr = parts[0].strip()
        
        if attr == 'origin_country':
            countries = [c.strip() for c in ':'.join(parts[1:]).split(',')]
            return attr, 0.0, ','.join(countries)
        elif attr == 'original_language':
            lang = parts[1].strip()
            return attr, 0.0, lang
        
        if len(parts) != 3:
            raise ValueError("Numeric filter must be in format: attribute:min:max")
        
        min_val = float(parts[1].strip())
        max_val = float(parts[2].strip())
        
        if attr not in NUMERIC_ATTRIBUTES:
            raise ValueError(f"Unknown attribute '{attr}'. Available: {', '.join(NUMERIC_ATTRIBUTES.keys())}")
        
        if min_val > max_val:
            raise ValueError(f"Min value ({min_val}) must be <= max value ({max_val})")
        
        return attr, min_val, max_val
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parse_range_query(range_str: str) -> Tuple[float, float]:
    try:
        parts = range_str.split(':')
        if len(parts) != 2:
            raise ValueError("Range must be in format: min:max")
        
        min_val = float(parts[0].strip())
        max_val = float(parts[1].strip())
        
        if min_val > max_val:
            raise ValueError(f"Min value ({min_val}) must be <= max value ({max_val})")
        
        return min_val, max_val
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Multi-dimensional Data Indexing and Similarity Query Processing CLI'
    )
    
    parser.add_argument(
        '--index', '-i',
        type=str,
        choices=['kd', 'quad', 'range', 'rtree'],
        required=True,
        help='Index type: kd (k-d tree), quad (quad tree), range (range tree), rtree (R-tree)'
    )
    
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        required=True,
        help='Number of top similar results to return (N)'
    )
    
    parser.add_argument(
        '--text-attr', '-t',
        type=str,
        choices=list(TEXTUAL_ATTRIBUTES.keys()),
        required=True,
        help=f'Textual attribute for LSH similarity: {", ".join(TEXTUAL_ATTRIBUTES.keys())}'
    )
    
    parser.add_argument(
        '--data-file', '-f',
        type=str,
        default='data_movies_clean.csv',
        help='Path to the movies dataset CSV file (default: data_movies_clean.csv)'
    )
    
    parser.add_argument(
        '--sample-size', '-s',
        type=int,
        default=None,
        help='Limit dataset to first N movies for faster testing (optional)'
    )
    
    parser.add_argument(
        '--filter', '-F',
        type=parse_numeric_filter,
        action='append',
        default=[],
        help='Filter in format attribute:min:max (numeric) or attribute:value1,value2 (special). '
             f'Available numeric: {", ".join(NUMERIC_ATTRIBUTES.keys())}. '
             'Special: origin_country:US,GB or original_language:en (can be used multiple times)'
    )
    
    parser.add_argument(
        '--range-query', '-r',
        type=str,
        default=None,
        help='Range query on spatial index. Format: min1:max1,min2:max2,... (5 values for 5D)'
    )
    
    parser.add_argument(
        '--knn-query', '-k',
        type=str,
        default=None,
        help='kNN query point on spatial index. Format: val1,val2,val3,val4,val5 (5 values for 5D)'
    )
    
    parser.add_argument(
        '--knn-k',
        type=int,
        default=50,
        help='Number of nearest neighbors for kNN query (default: 50)'
    )
    
    parser.add_argument(
        '--query-movie-id',
        type=int,
        default=None,
        help='Specific movie ID to use as query for LSH similarity (default: first movie in filtered set)'
    )
    
    parser.add_argument(
        '--lsh-threshold',
        type=float,
        default=0.5,
        help='LSH similarity threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--lsh-permutations',
        type=int,
        default=128,
        help='Number of MinHash permutations for LSH (default: 128)'
    )
    
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize numeric attributes (default: True)'
    )
    
    parser.add_argument(
        '--no-normalize',
        dest='normalize',
        action='store_false',
        help='Disable normalization of numeric attributes'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for JSON results (optional)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--rtree-max-entries',
        type=int,
        default=10,
        help='R-tree max entries per node (default: 10)'
    )
    
    parser.add_argument(
        '--rtree-min-entries',
        type=int,
        default=4,
        help='R-tree min entries per node (default: 4)'
    )
    
    parser.add_argument(
        '--quadtree-capacity',
        type=int,
        default=10,
        help='Quad tree capacity per node (default: 10)'
    )
    
    return parser


def parse_range_query_string(range_str: str, dimension: int = 5) -> Tuple[List[float], List[float]]:
    parts = [s.strip() for s in range_str.split(',')]
    if len(parts) != dimension:
        raise ValueError(f"Range query must have {dimension} values, got {len(parts)}")
    
    range_min = []
    range_max = []
    for part in parts:
        min_val, max_val = parse_range_query(part)
        range_min.append(min_val)
        range_max.append(max_val)
    
    return range_min, range_max


def parse_knn_query_string(knn_str: str, dimension: int = 5) -> List[float]:
    parts = [s.strip() for s in knn_str.split(',')]
    if len(parts) != dimension:
        raise ValueError(f"kNN query must have {dimension} values, got {len(parts)}")
    
    return [float(p) for p in parts]


def apply_special_filters(df, filters: List[Tuple[str, float, float]]):
    import pandas as pd
    
    filtered_df = df.copy()
    
    for attr, min_val, max_val in filters:
        if attr == 'origin_country':
            countries = set(max_val.split(','))
            def contains_country(country_str, countries_set):
                if pd.isna(country_str):
                    return False
                if isinstance(country_str, str):
                    country_str = country_str.strip("[]'\"")
                    country_list = [c.strip().strip("'\"") for c in country_str.split(',')]
                    return any(c in countries_set for c in country_list)
                elif isinstance(country_str, list):
                    return any(c in countries_set for c in country_str)
                return False
            
            if 'origin_country' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['origin_country'].apply(lambda x: contains_country(x, countries))
                ]
        
        elif attr == 'original_language':
            lang = max_val.strip()
            if 'original_language' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['original_language'] == lang]
    
    return filtered_df


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if args.range_query and args.knn_query:
        print("Error: Cannot specify both --range-query and --knn-query", file=sys.stderr)
        sys.exit(1)
    
    index_type = INDEX_TYPES[args.index]
    
    index_kwargs = {}
    if index_type == 'rtree':
        index_kwargs['max_entries'] = args.rtree_max_entries
        index_kwargs['min_entries'] = args.rtree_min_entries
    elif index_type == 'quadtree':
        index_kwargs['capacity'] = args.quadtree_capacity
    
    try:
        pipeline = MovieQueryPipeline(
            csv_path=args.data_file,
            index_type=index_type,
            dimension=5,
            text_attribute=args.text_attr,
            lsh_num_perm=args.lsh_permutations,
            lsh_threshold=args.lsh_threshold,
            **index_kwargs
        )
        
        if not pipeline.load_data(apply_filter=False, normalize=args.normalize):
            print("Error: Failed to load data", file=sys.stderr)
            sys.exit(1)
        
        if args.sample_size and args.sample_size > 0:
            pipeline.df = pipeline.df.head(args.sample_size)
        
        special_filters = [(a, m, M) for a, m, M in args.filter if a in ['origin_country', 'original_language']]
        if special_filters:
            pipeline.df = apply_special_filters(pipeline.df, special_filters)
        
        if not pipeline.build_spatial_index(DEFAULT_SPATIAL_ATTRIBUTES):
            print("Error: Failed to build spatial index", file=sys.stderr)
            sys.exit(1)
        
        numeric_constraints = {}
        for attr, min_val, max_val in args.filter:
            if attr not in ['origin_country', 'original_language']:
                numeric_constraints[attr] = (min_val, max_val)
        
        query_type = 'range'
        range_min = None
        range_max = None
        knn_query_point = None
        
        if args.range_query:
            try:
                range_min, range_max = parse_range_query_string(args.range_query, dimension=5)
                query_type = 'range'
            except Exception as e:
                print(f"Error parsing range query: {e}", file=sys.stderr)
                sys.exit(1)
        elif args.knn_query:
            try:
                knn_query_point = parse_knn_query_string(args.knn_query, dimension=5)
                query_type = 'knn'
            except Exception as e:
                print(f"Error parsing kNN query: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            if args.normalize:
                range_min = [0.0] * 5
                range_max = [1.0] * 5
            else:
                df = pipeline.df
                for i, attr in enumerate(DEFAULT_SPATIAL_ATTRIBUTES):
                    if attr in df.columns:
                        if range_min is None:
                            range_min = []
                            range_max = []
                        range_min.append(float(df[attr].min()))
                        range_max.append(float(df[attr].max()))
                if range_min is None:
                    range_min = [0.0] * 5
                    range_max = [1.0] * 5
        
        results = pipeline.end_to_end_query(
            numeric_constraints=numeric_constraints if numeric_constraints else None,
            query_type=query_type,
            range_min=range_min,
            range_max=range_max,
            knn_query_point=knn_query_point,
            knn_k=args.knn_k,
            query_movie_id=args.query_movie_id,
            top_n=args.top_n
        )
        
        if not results:
            print("No results found matching the criteria.")
            sys.exit(0)
        
        output_results = []
        for rank, (movie_id, similarity, info) in enumerate(results, 1):
            result_dict = {
                'rank': rank,
                'movie_id': int(movie_id),
                'similarity': float(similarity),
                'title': info.get('title', 'Unknown'),
                'release_year': int(info.get('release_year', 0)) if info.get('release_year') != 'Unknown' else None,
                'vote_average': float(info.get('vote_average', 0)) if info.get('vote_average') != 'Unknown' else None,
                'popularity': float(info.get('popularity', 0)) if info.get('popularity') != 'Unknown' else None,
                'point': info.get('point', []),
                'tokens': info.get('tokens', [])
            }
            output_results.append(result_dict)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'query_params': {
                        'index_type': args.index,
                        'top_n': args.top_n,
                        'text_attribute': args.text_attr,
                        'filters': [{'attr': a, 'min': m, 'max': M} for a, m, M in args.filter],
                        'range_query': args.range_query,
                        'knn_query': args.knn_query,
                        'knn_k': args.knn_k
                    },
                    'results': output_results
                }, f, indent=2)
        
        print(f"Found {len(results)} results")
        if output_results:
            avg_sim = sum(r['similarity'] for r in output_results) / len(output_results)
            print(f"Average similarity: {avg_sim:.4f}")
        
    except KeyboardInterrupt:
        print("\nQuery interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
