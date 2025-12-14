import json
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd


def format_query_results_to_json(
    query_results: List[Tuple[int, float, List[str]]],
    df: Optional[pd.DataFrame] = None,
    query_movie_id: Optional[int] = None,
    include_titles: bool = True,
    include_tokens: bool = False,
    output_file: Optional[str] = None,
    pretty_print: bool = True
) -> str:
    output = {
        "query": {},
        "results": [],
        "metadata": {
            "total_results": len(query_results),
            "include_titles": include_titles,
            "include_tokens": include_tokens
        }
    }
    
    if query_movie_id is not None:
        output["query"]["movie_id"] = int(query_movie_id)
        if df is not None and query_movie_id in df.index and 'title' in df.columns:
            output["query"]["title"] = str(df.loc[query_movie_id, 'title'])
    
    for rank, (movie_id, similarity, tokens) in enumerate(query_results, start=1):
        result_entry = {
            "rank": rank,
            "movie_id": int(movie_id),
            "similarity": round(float(similarity), 4)
        }
        
        if include_titles and df is not None:
            if movie_id in df.index and 'title' in df.columns:
                result_entry["title"] = str(df.loc[movie_id, 'title'])
            else:
                result_entry["title"] = "Unknown"
        
        if include_tokens and tokens:
            result_entry["tokens"] = tokens
        
        output["results"].append(result_entry)
    
    if pretty_print:
        json_str = json.dumps(output, indent=2, ensure_ascii=False)
    else:
        json_str = json.dumps(output, ensure_ascii=False)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"JSON output saved to: {output_file}")
    
    return json_str


def format_simple_results_to_json(
    movie_ids: List[int],
    similarities: List[float],
    titles: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    pretty_print: bool = True
) -> str:
    if len(movie_ids) !=  len(similarities):
        raise ValueError("movie_ids and similarities must have the same length")
    
    if titles is not None and len(titles) != len(movie_ids):
        raise ValueError("titles must have the same length as movie_ids")
    
    output = {
        "results": [],
        "metadata": {
            "total_results": len(movie_ids),
            "has_titles": titles is not None
        }
    }
    
    for rank, (movie_id, similarity) in enumerate(zip(movie_ids, similarities), start=1):
        result_entry = {
            "rank": rank,
            "movie_id": int(movie_id),
            "similarity": round(float(similarity), 4)
        }
        
        if titles is not None:
            result_entry["title"] = str(titles[rank - 1])
        
        output["results"].append(result_entry)
    
    if pretty_print:
        json_str = json.dumps(output, indent=2, ensure_ascii=False)
    else:
        json_str = json.dumps(output, ensure_ascii=False)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"JSON output saved to: {output_file}")
    
    return json_str


def batch_format_queries_to_json(
    queries_results: Dict[int, List[Tuple[int, float, List[str]]]],
    df: Optional[pd.DataFrame] = None,
    include_titles: bool = True,
    include_tokens: bool = False,
    output_file: Optional[str] = None,
    pretty_print: bool = True
) -> str:
    output = {
        "queries": [],
        "metadata": {
            "total_queries": len(queries_results),
            "include_titles": include_titles,
            "include_tokens": include_tokens
        }
    }
    
    for query_id, results in queries_results.items():
        query_entry = {
            "query_movie_id": int(query_id),
            "results": []
        }
        
        if df is not None and query_id in df.index and 'title' in df.columns:
            query_entry["query_title"] = str(df.loc[query_id, 'title'])
        
        for rank, (movie_id, similarity, tokens) in enumerate(results, start=1):
            result_entry = {
                "rank": rank,
                "movie_id": int(movie_id),
                "similarity": round(float(similarity), 4)
            }
            
            if include_titles and df is not None:
                if movie_id in df.index and 'title' in df.columns:
                    result_entry["title"] = str(df.loc[movie_id, 'title'])
                else:
                    result_entry["title"] = "Unknown"
            
            if include_tokens and tokens:
                result_entry["tokens"] = tokens
            
            query_entry["results"].append(result_entry)
        
        query_entry["total_results"] = len(results)
        output["queries"].append(query_entry)
    
    if pretty_print:
        json_str = json.dumps(output, indent=2, ensure_ascii=False)
    else:
        json_str = json.dumps(output, ensure_ascii=False)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"Batch JSON output saved to: {output_file}")
    
    return json_str


if __name__ == "__main__":
    print("=== Testing JSON Formatter ===\n")
    
    test_results = [
        (123, 0.8547, ['action', 'adventure', 'sci-fi']),
        (456, 0.7234, ['action', 'thriller']),
        (789, 0.6891, ['adventure', 'sci-fi']),
        (321, 0.5432, ['action', 'drama'])
    ]
    
    print("1. Basic formatting with query ID:")
    json_output = format_query_results_to_json(
        test_results,
        query_movie_id=100,
        include_titles=False,
        include_tokens=True
    )
    print(json_output)
    
    print("\n2. Simple format test:")
    ids = [123, 456, 789]
    sims = [0.85, 0.72, 0.65]
    titles = ["The Matrix", "Inception", "Interstellar"]
    json_output = format_simple_results_to_json(ids, sims, titles)
    print(json_output)
    
    print("\n3. Batch queries test:")
    batch_queries = {
        100: test_results[:2],
        200: test_results[2:]
    }
    json_output = batch_format_queries_to_json(
        batch_queries,
        include_titles=False,
        include_tokens=True
    )
    print(json_output)
    
    print("\n=== Integration test with real data ===")
    try:
        from project1_loader import load_and_process_data
        from lsh import find_top_n_similar_movies
        
        df = load_and_process_data('data_movies_clean.csv', apply_filter=False, normalize=False)
        
        if df is not None and len(df) > 0:
            sample_ids = df.index[:50].tolist()
            
            results = find_top_n_similar_movies(
                filtered_movie_ids=sample_ids,
                df=df,
                text_attribute='production_company_names',
                query_movie_id=sample_ids[0],
                top_n=5,
                num_perm=128,
                threshold=0.5
            )
            
            print("\n4. Real query results with titles:")
            json_output = format_query_results_to_json(
                results,
                df=df,
                query_movie_id=sample_ids[0],
                include_titles=True,
                include_tokens=False,
                output_file="query_results.json"
            )
            print(json_output)
            
        else:
            print("Could not load data for integration test")
    except Exception as e:
        print(f"Integration test skipped: {e}")
