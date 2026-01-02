

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from report_statistics import (
    compute_query_statistics,
    compute_scheme_comparison,
    generate_summary_statistics,
    analyze_memory_results,
    compute_memory_statistics,
    format_time,
    format_memory
)
from report_plots import save_plots_for_report


class ReportGenerator:

    
    def __init__(self, 
                 results_dir: str = 'evaluation_results',
                 memory_csv: str = 'memory_profiling_results.csv',
                 output_dir: str = 'reports'):

        self.results_dir = results_dir
        self.memory_csv = memory_csv
        self.output_dir = output_dir
        self.artifacts_dir = os.path.join(output_dir, 'artifacts')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
    
    def load_evaluation_results(self) -> Dict[str, Any]:

        summary_file = os.path.join(self.results_dir, 'performance_summary.json')
        
        if not os.path.exists(summary_file):
            print(f"Warning: {summary_file} not found")
            return {}
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def load_memory_results(self) -> pd.DataFrame:

        return analyze_memory_results(self.memory_csv)
    
    def build_markdown_report(self, 
                              performance_data: Dict[str, Any],
                              summary_stats: Dict[str, Any],
                              memory_df: pd.DataFrame,
                              plot_paths: Dict[str, str]) -> str:

        md = []
        
        # Title and metadata
        md.append("# Experimental Results Report")
        md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"\n**Evaluation System:** Multi-dimensional Indexing + LSH")
        md.append("\n---\n")
        
        # Executive Summary
        md.append("## Executive Summary\n")
        md.append(f"This report summarizes the performance evaluation of {len(summary_stats.get('schemes_tested', []))} ")
        md.append(f"index structures across {summary_stats.get('num_queries', 0)} queries. ")
        md.append("The evaluation combines multi-dimensional spatial indexing with Locality-Sensitive Hashing (LSH) ")
        md.append("for similarity-based filtering.\n")
        
        comparison = summary_stats.get('comparison', {})
        if comparison:
            md.append(f"\n**Key Findings:**\n")
            md.append(f"- **Fastest Scheme:** {comparison.get('fastest_avg', 'N/A')} ")
            md.append(f"({format_time(comparison.get('fastest_avg_time', 0))} avg)\n")
            md.append(f"- **Most Consistent:** {comparison.get('most_consistent', 'N/A')} ")
            md.append(f"(σ = {comparison.get('most_consistent_std', 0):.6f}s)\n")
            md.append(f"- **Most Results:** {comparison.get('most_results', 'N/A')} ")
            md.append(f"({comparison.get('most_results_count', 0):.1f} avg results)\n")
        
        md.append("\n---\n")
        
        # Performance Metrics
        md.append("## Performance Metrics\n")
        md.append("### Query Performance Statistics\n")
        
        stats_df = summary_stats.get('statistics_table')
        if stats_df is not None and not stats_df.empty:
            md.append(self._dataframe_to_markdown(stats_df))
            md.append("\n")
        
        # Embed query time comparison plot
        if 'query_time_comparison' in plot_paths:
            rel_path = os.path.relpath(plot_paths['query_time_comparison'], self.output_dir)
            md.append(f"\n![Query Time Comparison]({rel_path})\n")
            md.append("*Figure 1: Average query time breakdown by index scheme*\n")
        
        # Speedup analysis
        speedup_df = summary_stats.get('speedup_table')
        if speedup_df is not None and not speedup_df.empty:
            md.append("\n### Speedup Analysis\n")
            md.append(self._dataframe_to_markdown(speedup_df))
            md.append("\n")
            
            if 'speedup_comparison' in plot_paths:
                rel_path = os.path.relpath(plot_paths['speedup_comparison'], self.output_dir)
                md.append(f"\n![Speedup Comparison]({rel_path})\n")
                md.append("*Figure 2: Speedup ratios relative to baseline*\n")
        
        md.append("\n---\n")
        
        # Result Distribution
        md.append("## Result Distribution\n")
        md.append("This section shows how many results each index structure returns ")
        md.append("before and after LSH filtering.\n")
        
        if 'result_count_distribution' in plot_paths:
            rel_path = os.path.relpath(plot_paths['result_count_distribution'], self.output_dir)
            md.append(f"\n![Result Count Distribution]({rel_path})\n")
            md.append("*Figure 3: Average result counts from tree queries vs final LSH-filtered results*\n")
        
        md.append("\n---\n")
        
        # Performance Heatmap
        md.append("## Query-by-Query Performance\n")
        
        if 'performance_heatmap' in plot_paths:
            rel_path = os.path.relpath(plot_paths['performance_heatmap'], self.output_dir)
            md.append(f"\n![Performance Heatmap]({rel_path})\n")
            md.append("*Figure 4: Performance heatmap showing query times across all queries and schemes*\n")
        
        # Query details table
        md.append("\n### Individual Query Results\n")
        queries = performance_data.get('queries', [])
        if queries:
            query_table_data = []
            for query in queries:
                row = {
                    'Query ID': query.get('query_id', 'N/A'),
                    'Title': query.get('title', 'Unknown')[:40] + '...' if len(query.get('title', '')) > 40 else query.get('title', 'Unknown')
                }
                
                # Add scheme times
                for scheme_name, metrics in query.get('schemes', {}).items():
                    row[scheme_name.replace('_', ' ').title()] = f"{metrics.get('total_time', 0):.4f}s"
                
                query_table_data.append(row)
            
            query_df = pd.DataFrame(query_table_data)
            md.append(self._dataframe_to_markdown(query_df))
            md.append("\n")
        
        md.append("\n---\n")
        
        # Memory Analysis
        if not memory_df.empty:
            md.append("## Memory Analysis\n")
            md.append("Memory profiling results for different index structures and dataset sizes.\n")
            
            md.append("\n### Memory Usage Summary\n")
            md.append(self._dataframe_to_markdown(memory_df))
            md.append("\n")
            
            if 'memory_usage' in plot_paths:
                rel_path = os.path.relpath(plot_paths['memory_usage'], self.output_dir)
                md.append(f"\n![Memory Usage]({rel_path})\n")
                md.append("*Figure 5: Memory consumption and build time analysis*\n")
            
            # Memory statistics
            mem_stats = compute_memory_statistics(memory_df)
            if mem_stats:
                md.append("\n### Memory Efficiency Rankings\n")
                for size_key, stats in mem_stats.items():
                    size = size_key.replace('size_', '')
                    md.append(f"\n**Dataset Size: {size} points**\n")
                    md.append(f"- Most Efficient: {stats.get('most_efficient', 'N/A')}\n")
                    md.append(f"- Least Efficient: {stats.get('least_efficient', 'N/A')}\n")
                    md.append(f"- Average Memory: {format_memory(stats.get('avg_memory_mb', 0))}\n")
            
            md.append("\n---\n")
        
        # Discussion
        md.append("## Discussion\n")
        md.append(self._generate_discussion(summary_stats, comparison, memory_df))
        
        md.append("\n---\n")
        
        # Methodology
        md.append("## Methodology\n")
        md.append("### Index Structures Evaluated\n")
        for scheme in summary_stats.get('schemes_tested', []):
            md.append(f"- **{scheme}**: Combined with LSH for similarity filtering\n")
        
        md.append("\n### Evaluation Process\n")
        md.append("1. **Data Filtering**: Movies filtered by release year (2000-2020), ")
        md.append("popularity (3-6), vote average (3-5), runtime (30-60 min)\n")
        md.append("2. **5D Vector Extraction**: Features extracted from movie metadata\n")
        md.append("3. **Index Construction**: All index structures built on the same dataset\n")
        md.append("4. **LSH Setup**: MinHash LSH with 128 permutations, threshold 0.5\n")
        md.append("5. **Query Execution**: Range queries on 5D space + LSH similarity filtering\n")
        md.append("6. **Metrics Collection**: Total time, tree time, LSH time, result counts\n")
        
        md.append("\n---\n")
        
        # Conclusion
        md.append("## Conclusion\n")
        md.append(self._generate_conclusion(comparison))
        
        md.append("\n---\n")
        md.append(f"\n*Report generated by Statistical Report Generator v1.0*\n")
        
        return ''.join(md)
    
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:

        return df.to_markdown(index=False)
    
    def _generate_discussion(self, 
                            summary_stats: Dict[str, Any],
                            comparison: Dict[str, Any],
                            memory_df: pd.DataFrame) -> str:

        discussion = []
        
        discussion.append("### Performance Insights\n")
        
        if comparison:
            fastest = comparison.get('fastest_avg', 'N/A')
            slowest = comparison.get('slowest_avg', 'N/A')
            
            discussion.append(f"The evaluation reveals that **{fastest}** achieves the best average ")
            discussion.append(f"query performance, while **{slowest}** shows the slowest average times. ")
            
            most_consistent = comparison.get('most_consistent', 'N/A')
            discussion.append(f"In terms of consistency, **{most_consistent}** demonstrates the lowest ")
            discussion.append("standard deviation, indicating more predictable performance across different queries.\n")
        
        discussion.append("\n### LSH Impact\n")
        discussion.append("The combination of spatial indexing with LSH provides a two-phase filtering approach:\n")
        discussion.append("1. **Spatial filtering** narrows down candidates based on 5D feature similarity\n")
        discussion.append("2. **LSH filtering** refines results based on textual similarity\n")
        discussion.append("\nThis hybrid approach balances precision and recall while maintaining query efficiency.\n")
        
        if not memory_df.empty:
            discussion.append("\n### Memory Considerations\n")
            discussion.append("Memory profiling shows varying space requirements across index structures. ")
            discussion.append("The choice of index should consider both query performance and memory constraints ")
            discussion.append("based on the specific deployment environment.\n")
        
        discussion.append("\n### Trade-offs\n")
        discussion.append("- **Speed vs Memory**: Faster indexes may require more memory\n")
        discussion.append("- **Build Time vs Query Time**: Some structures have longer build times but faster queries\n")
        discussion.append("- **Precision vs Recall**: LSH threshold affects the balance between result quality and quantity\n")
        
        return ''.join(discussion)
    
    def _generate_conclusion(self, comparison: Dict[str, Any]) -> str:

        conclusion = []
        
        conclusion.append("This comprehensive evaluation demonstrates the effectiveness of combining ")
        conclusion.append("multi-dimensional indexing with LSH for similarity search in movie datasets. ")
        
        if comparison:
            fastest = comparison.get('fastest_avg', 'N/A')
            conclusion.append(f"**{fastest}** emerges as the recommended choice for query-intensive workloads ")
            conclusion.append("where performance is critical. ")
        
        conclusion.append("\n\nThe results provide valuable insights for selecting appropriate index structures ")
        conclusion.append("based on specific application requirements, dataset characteristics, and ")
        conclusion.append("performance constraints.\n")
        
        return ''.join(conclusion)
    
    def export_to_pdf(self, markdown_content: str, output_path: str) -> bool:

        try:
            import markdown
            from weasyprint import HTML, CSS
            
            # Convert markdown to HTML
            html_content = markdown.markdown(
                markdown_content,
                extensions=['tables', 'fenced_code', 'codehilite']
            )
            
            # Add CSS styling
            css_style = """
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }
                h3 { color: #7f8c8d; margin-top: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th { background-color: #3498db; color: white; padding: 12px; text-align: left; }
                td { border: 1px solid #ddd; padding: 10px; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
                code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
                strong { color: #2c3e50; }
            </style>
            """
            
            full_html = f"<html><head>{css_style}</head><body>{html_content}</body></html>"
            
            # Convert to PDF
            HTML(string=full_html).write_pdf(output_path)
            
            return True
        except ImportError:
            print("Warning: PDF export requires 'markdown' and 'weasyprint' packages")
            print("Install with: pip install markdown weasyprint")
            return False
        except Exception as e:
            print(f"Error exporting to PDF: {e}")
            return False
    
    def generate_report(self, 
                       output_filename: str = 'experimental_report',
                       export_pdf: bool = True) -> Dict[str, str]:
        print("=" * 80)
        print("STATISTICAL REPORT GENERATOR")
        print("=" * 80)
        
        # Load data
        print("\n1. Loading evaluation results...")
        performance_data = self.load_evaluation_results()
        
        if not performance_data:
            print("   ❌ No evaluation results found")
            return {}
        
        print(f"   ✓ Loaded {len(performance_data.get('queries', []))} queries")
        
        print("\n2. Loading memory profiling results...")
        memory_df = self.load_memory_results()
        
        if not memory_df.empty:
            print(f"   ✓ Loaded memory data for {len(memory_df)} configurations")
        else:
            print("   ⚠ No memory profiling data found")
        
        # Compute statistics
        print("\n3. Computing statistics...")
        summary_stats = generate_summary_statistics(performance_data)
        print(f"   ✓ Analyzed {len(summary_stats.get('schemes_tested', []))} schemes")
        
        # Generate plots
        print("\n4. Generating visualizations...")
        stats_df = summary_stats.get('statistics_table', pd.DataFrame())
        speedup_df = summary_stats.get('speedup_table', pd.DataFrame())
        
        plot_paths = save_plots_for_report(
            performance_data,
            stats_df,
            memory_df,
            speedup_df,
            self.artifacts_dir
        )
        
        # Build markdown report
        print("\n5. Building markdown report...")
        markdown_content = self.build_markdown_report(
            performance_data,
            summary_stats,
            memory_df,
            plot_paths
        )
        
        # Save markdown
        md_path = os.path.join(self.output_dir, f'{output_filename}.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"   ✓ Saved markdown report: {md_path}")
        
        output_files = {'markdown': md_path}
        
        # Export to PDF
        if export_pdf:
            print("\n6. Exporting to PDF...")
            pdf_path = os.path.join(self.output_dir, f'{output_filename}.pdf')
            
            if self.export_to_pdf(markdown_content, pdf_path):
                print(f"   ✓ Saved PDF report: {pdf_path}")
                output_files['pdf'] = pdf_path
            else:
                print("   ⚠ PDF export skipped (dependencies not available)")
        
        print("\n" + "=" * 80)
        print("REPORT GENERATION COMPLETE")
        print("=" * 80)
        print(f"\nGenerated files:")
        for file_type, path in output_files.items():
            print(f"  - {file_type.upper()}: {path}")
        
        print(f"\nPlots saved to: {self.artifacts_dir}")
        
        return output_files


def main():
    """Main entry point for report generation."""
    generator = ReportGenerator(
        results_dir='evaluation_results',
        memory_csv='memory_profiling_results.csv',
        output_dir='reports'
    )
    
    output_files = generator.generate_report(
        output_filename='experimental_report',
        export_pdf=True
    )
    
    if output_files:
        print("\n Report generation successful!")
    else:
        print("\n Report generation failed!")


if __name__ == "__main__":
    main()
