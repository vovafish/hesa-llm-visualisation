# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Natural Language Query Processor for HESA Data

# This module processes natural language queries about HESA data and returns
# structured results that can be used for visualization and reporting.
# """

# import os
# import sys
# import json
# import argparse
# import logging
# from pathlib import Path
# import pandas as pd
# import re
# from typing import Dict, List, Any, Optional, Tuple, Union

# # Add the project root to the Python path
# project_root = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(project_root))

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler(project_root / "logs" / "query_processor.log", mode='a')
#     ]
# )
# logger = logging.getLogger("query_processor")

# # Create logs directory if it doesn't exist
# (project_root / "logs").mkdir(parents=True, exist_ok=True)


# class QueryProcessor:
#     """
#     A class for processing natural language queries about HESA data.
    
#     This class provides methods to:
#     - Parse and understand natural language queries
#     - Identify data sources and filtering criteria
#     - Determine appropriate visualizations
#     - Return structured results
#     """
    
#     def __init__(self, data_dir: Optional[str] = None):
#         """
#         Initialize the query processor.
        
#         Args:
#             data_dir: Path to the directory containing HESA data files (default: data/cleaned_files)
#         """
#         if data_dir:
#             self.data_dir = Path(data_dir)
#         else:
#             self.data_dir = project_root / "data" / "cleaned_files"
        
#         # Check if data directory exists
#         if not self.data_dir.exists():
#             logger.warning(f"Data directory does not exist: {self.data_dir}")
#             logger.info("Creating data directory")
#             self.data_dir.mkdir(parents=True, exist_ok=True)
        
#         # Load available data files
#         self.data_files = self._get_available_data_files()
#         logger.info(f"Found {len(self.data_files)} data files")
        
#         # Initialize keyword mappings for query parsing
#         self._init_keyword_mappings()
    
#     def _get_available_data_files(self) -> List[Dict[str, Any]]:
#         """Get a list of available data files with basic metadata."""
#         data_files = []
        
#         for file_path in self.data_dir.glob("*.csv"):
#             try:
#                 # Read a small sample to get column names and basic info
#                 df_sample = pd.read_csv(file_path, nrows=5)
                
#                 data_files.append({
#                     "name": file_path.name,
#                     "path": str(file_path),
#                     "columns": list(df_sample.columns),
#                     "size_bytes": file_path.stat().st_size,
#                     "modified_time": file_path.stat().st_mtime
#                 })
#             except Exception as e:
#                 logger.warning(f"Error reading file {file_path.name}: {str(e)}")
        
#         return data_files
    
#     def _init_keyword_mappings(self):
#         """Initialize keyword mappings for query parsing."""
#         # Data categories
#         self.category_keywords = {
#             "student": ["student", "students", "enrollment", "enrolment", "admission"],
#             "performance": ["performance", "grades", "results", "achievement", "outcomes"],
#             "demographic": ["demographic", "gender", "ethnicity", "age", "socioeconomic", "nationality"],
#             "finance": ["finance", "financial", "funding", "budget", "costs", "fees", "tuition"],
#             "staff": ["staff", "faculty", "teachers", "professors", "employees"],
#             "institution": ["institution", "university", "college", "campus", "school"]
#         }
        
#         # Time-related terms
#         self.time_keywords = {
#             "year": ["year", "annual", "yearly", "academic year"],
#             "semester": ["semester", "term", "period"],
#             "trend": ["trend", "over time", "historical", "changes", "evolution"]
#         }
        
#         # Analysis types
#         self.analysis_keywords = {
#             "comparison": ["compare", "comparison", "versus", "against", "compared to"],
#             "distribution": ["distribution", "spread", "range", "breakdown"],
#             "correlation": ["correlation", "relationship", "connection", "link", "associated"],
#             "proportion": ["proportion", "percentage", "ratio", "share", "portion"],
#             "ranking": ["ranking", "rank", "top", "bottom", "best", "worst", "highest", "lowest"]
#         }
        
#         # Visualization types
#         self.viz_keywords = {
#             "bar_chart": ["bar chart", "bar graph", "column chart", "bar"],
#             "line_chart": ["line chart", "line graph", "trend line", "line"],
#             "pie_chart": ["pie chart", "pie", "circular chart"],
#             "scatter_plot": ["scatter plot", "scatter", "scatter graph", "correlation plot"],
#             "heatmap": ["heatmap", "heat map", "correlation matrix"],
#             "map": ["map", "geographic", "geospatial", "regional"],
#             "table": ["table", "tabular", "grid", "raw data"]
#         }
    
#     def process_query(self, query: str) -> Dict[str, Any]:
#         """
#         Process a natural language query about HESA data.
        
#         Args:
#             query: The natural language query string
            
#         Returns:
#             Dictionary containing structured results based on the query
#         """
#         logger.info(f"Processing query: {query}")
        
#         # Initialize result structure
#         result = {
#             "query": query,
#             "query_type": None,  # What kind of question is being asked
#             "data_category": None,  # What category of data is being asked about
#             "time_frame": None,  # What time period is relevant
#             "analysis_type": None,  # What type of analysis is being requested
#             "entities": [],  # Specific entities mentioned (institutions, subjects, etc.)
#             "suggested_visualizations": [],  # Recommended visualization types
#             "data_sources": [],  # Recommended data files to use
#             "filters": {},  # Suggested filters to apply
#             "response": None  # Natural language response to the query
#         }
        
#         # Parse query to identify key elements
#         self._identify_data_category(query, result)
#         self._identify_time_frame(query, result)
#         self._identify_analysis_type(query, result)
#         self._extract_entities(query, result)
        
#         # Determine query type
#         self._determine_query_type(query, result)
        
#         # Suggest data sources
#         self._suggest_data_sources(result)
        
#         # Suggest visualizations
#         self._suggest_visualizations(result)
        
#         # Generate a natural language response
#         self._generate_response(result)
        
#         return result
    
#     def _identify_data_category(self, query: str, result: Dict[str, Any]):
#         """Identify the data category being asked about."""
#         query_lower = query.lower()
        
#         for category, keywords in self.category_keywords.items():
#             for keyword in keywords:
#                 if keyword in query_lower:
#                     result["data_category"] = category
#                     logger.info(f"Identified data category: {category}")
#                     return
        
#         logger.info("No specific data category identified")
    
#     def _identify_time_frame(self, query: str, result: Dict[str, Any]):
#         """Identify the time frame being asked about."""
#         query_lower = query.lower()
        
#         # Look for specific years
#         year_pattern = r"\b(20\d{2})\b"
#         years = re.findall(year_pattern, query_lower)
        
#         if years:
#             result["time_frame"] = {
#                 "type": "specific_years",
#                 "years": [int(year) for year in years]
#             }
#             logger.info(f"Identified specific years: {years}")
#             return
        
#         # Look for year ranges
#         year_range_pattern = r"\b(20\d{2})[-\s]+(?:to|through|and|\s)[-\s]*(20\d{2})\b"
#         year_ranges = re.findall(year_range_pattern, query_lower)
        
#         if year_ranges:
#             start_year, end_year = year_ranges[0]
#             result["time_frame"] = {
#                 "type": "year_range",
#                 "start_year": int(start_year),
#                 "end_year": int(end_year)
#             }
#             logger.info(f"Identified year range: {start_year} to {end_year}")
#             return
        
#         # Look for relative time frames
#         if "last 5 years" in query_lower or "past 5 years" in query_lower:
#             result["time_frame"] = {
#                 "type": "relative",
#                 "period": "last_5_years"
#             }
#             logger.info("Identified time frame: last 5 years")
#         elif "last 10 years" in query_lower or "past 10 years" in query_lower:
#             result["time_frame"] = {
#                 "type": "relative",
#                 "period": "last_10_years"
#             }
#             logger.info("Identified time frame: last 10 years")
#         elif "last year" in query_lower or "past year" in query_lower:
#             result["time_frame"] = {
#                 "type": "relative",
#                 "period": "last_year"
#             }
#             logger.info("Identified time frame: last year")
#         elif "trend" in query_lower or "over time" in query_lower or "historical" in query_lower:
#             result["time_frame"] = {
#                 "type": "trend",
#                 "period": "all_available"
#             }
#             logger.info("Identified time frame: trend over all available time")
    
#     def _identify_analysis_type(self, query: str, result: Dict[str, Any]):
#         """Identify the analysis type being requested."""
#         query_lower = query.lower()
        
#         for analysis_type, keywords in self.analysis_keywords.items():
#             for keyword in keywords:
#                 if keyword in query_lower:
#                     result["analysis_type"] = analysis_type
#                     logger.info(f"Identified analysis type: {analysis_type}")
#                     return
        
#         logger.info("No specific analysis type identified")
    
#     def _extract_entities(self, query: str, result: Dict[str, Any]):
#         """Extract specific entities mentioned in the query."""
#         # This is a simplified entity extraction
#         # In a real implementation, this would use a more sophisticated NLP approach
        
#         # Look for quoted terms which might be specific entities
#         quoted_pattern = r'"([^"]+)"'
#         quoted_entities = re.findall(quoted_pattern, query)
        
#         if quoted_entities:
#             result["entities"] = quoted_entities
#             logger.info(f"Extracted quoted entities: {quoted_entities}")
    
#     def _determine_query_type(self, query: str, result: Dict[str, Any]):
#         """Determine the type of query being asked."""
#         query_lower = query.lower()
        
#         # Look for question patterns
#         if query_lower.startswith("what") or query_lower.startswith("which"):
#             result["query_type"] = "factual"
#             logger.info("Identified query type: factual")
#         elif query_lower.startswith("how many") or query_lower.startswith("how much"):
#             result["query_type"] = "quantitative"
#             logger.info("Identified query type: quantitative")
#         elif query_lower.startswith("why") or query_lower.startswith("how"):
#             result["query_type"] = "explanatory"
#             logger.info("Identified query type: explanatory")
#         elif query_lower.startswith("show") or query_lower.startswith("display"):
#             result["query_type"] = "visualization"
#             logger.info("Identified query type: visualization")
#         elif "compare" in query_lower or "comparison" in query_lower:
#             result["query_type"] = "comparative"
#             logger.info("Identified query type: comparative")
#         elif "trend" in query_lower or "over time" in query_lower:
#             result["query_type"] = "trend"
#             logger.info("Identified query type: trend")
#         else:
#             result["query_type"] = "general"
#             logger.info("Identified query type: general")
    
#     def _suggest_data_sources(self, result: Dict[str, Any]):
#         """Suggest data sources based on the query analysis."""
#         # Start with an empty list
#         suggested_sources = []
        
#         # If we have a data category, look for matching files
#         if result["data_category"]:
#             category = result["data_category"]
            
#             for file_info in self.data_files:
#                 file_name = file_info["name"].lower()
                
#                 # Check if the category appears in the file name
#                 if category in file_name:
#                     suggested_sources.append(file_info)
#                     continue
                
#                 # If not in filename, check column names for relevance
#                 columns = [col.lower() for col in file_info["columns"]]
#                 relevant_columns = [col for col in columns if category in col]
                
#                 if relevant_columns:
#                     file_info["relevant_columns"] = relevant_columns
#                     suggested_sources.append(file_info)
        
#         # If we didn't find any sources by category, suggest based on other criteria
#         if not suggested_sources and result["analysis_type"] == "correlation":
#             # For correlation analysis, suggest pairs of files that might be related
#             # This is a simplified implementation - a real system would be more sophisticated
#             if len(self.data_files) >= 2:
#                 suggested_sources = self.data_files[:2]
#                 result["note"] = "Suggesting potentially related data sources for correlation analysis"
        
#         # If we still don't have suggestions, include all available files
#         if not suggested_sources:
#             suggested_sources = self.data_files
#             result["note"] = "No specific data sources identified; including all available files"
        
#         # Add to result
#         result["data_sources"] = suggested_sources
#         logger.info(f"Suggested {len(suggested_sources)} data sources")
    
#     def _suggest_visualizations(self, result: Dict[str, Any]):
#         """Suggest visualizations based on the query analysis."""
#         suggested_viz = []
        
#         # Based on query type
#         if result["query_type"] == "trend":
#             suggested_viz.append({
#                 "type": "line_chart",
#                 "suitability": "high",
#                 "reason": "Line charts are excellent for showing trends over time"
#             })
#         elif result["query_type"] == "comparative":
#             suggested_viz.append({
#                 "type": "bar_chart",
#                 "suitability": "high",
#                 "reason": "Bar charts are well-suited for comparisons between categories"
#             })
#             suggested_viz.append({
#                 "type": "radar_chart",
#                 "suitability": "medium",
#                 "reason": "Radar charts can display multiple variables for comparison"
#             })
#         elif result["query_type"] == "quantitative":
#             suggested_viz.append({
#                 "type": "bar_chart",
#                 "suitability": "high",
#                 "reason": "Bar charts effectively show quantities by category"
#             })
        
#         # Based on analysis type
#         if result["analysis_type"] == "distribution":
#             suggested_viz.append({
#                 "type": "histogram",
#                 "suitability": "high",
#                 "reason": "Histograms show the distribution of values"
#             })
#             suggested_viz.append({
#                 "type": "box_plot",
#                 "suitability": "high",
#                 "reason": "Box plots show statistical distribution of values"
#             })
#         elif result["analysis_type"] == "proportion":
#             suggested_viz.append({
#                 "type": "pie_chart",
#                 "suitability": "high",
#                 "reason": "Pie charts show proportions of a whole"
#             })
#             suggested_viz.append({
#                 "type": "stacked_bar_chart",
#                 "suitability": "medium",
#                 "reason": "Stacked bar charts show proportions and totals"
#             })
#         elif result["analysis_type"] == "correlation":
#             suggested_viz.append({
#                 "type": "scatter_plot",
#                 "suitability": "high",
#                 "reason": "Scatter plots show relationships between variables"
#             })
#             suggested_viz.append({
#                 "type": "heatmap",
#                 "suitability": "high",
#                 "reason": "Heatmaps show correlation matrices between multiple variables"
#             })
        
#         # Deduplication and prioritization
#         seen_types = set()
#         filtered_viz = []
        
#         for viz in suggested_viz:
#             if viz["type"] not in seen_types:
#                 seen_types.add(viz["type"])
#                 filtered_viz.append(viz)
        
#         # Add to result
#         result["suggested_visualizations"] = filtered_viz
#         logger.info(f"Suggested {len(filtered_viz)} visualizations")
    
#     def _generate_response(self, result: Dict[str, Any]):
#         """Generate a natural language response to the query."""
#         # This is a simple template-based response generator
#         # A real implementation would use a more sophisticated NLG approach
        
#         response_parts = []
        
#         # Introduction based on query type
#         if result["query_type"] == "factual":
#             response_parts.append(f"Based on your question about {result['data_category'] or 'HESA data'}, ")
#         elif result["query_type"] == "quantitative":
#             response_parts.append(f"To answer your question about the number or amount of {result['data_category'] or 'items'}, ")
#         elif result["query_type"] == "comparative":
#             response_parts.append(f"For your comparison of {result['data_category'] or 'data'}, ")
#         elif result["query_type"] == "trend":
#             response_parts.append(f"To show the trends in {result['data_category'] or 'the data'}, ")
#         elif result["query_type"] == "visualization":
#             response_parts.append(f"To visualize {result['data_category'] or 'the data'} you requested, ")
#         else:
#             response_parts.append("Based on your query, ")
        
#         # Data sources
#         if result["data_sources"]:
#             if len(result["data_sources"]) == 1:
#                 source_name = result["data_sources"][0]["name"]
#                 response_parts.append(f"I'll use the '{source_name}' dataset. ")
#             else:
#                 source_names = [s["name"] for s in result["data_sources"][:3]]
#                 if len(result["data_sources"]) > 3:
#                     source_text = f"{', '.join(source_names[:2])}, and other relevant datasets"
#                 else:
#                     source_text = f"{', '.join(source_names[:-1])}, and {source_names[-1]}"
#                 response_parts.append(f"I'll analyze data from {source_text}. ")
#         else:
#             response_parts.append("I couldn't find appropriate data sources for your query. ")
        
#         # Analysis approach
#         if result["analysis_type"] and result["data_category"]:
#             response_parts.append(f"I'll perform a {result['analysis_type']} analysis of {result['data_category']} data. ")
#         elif result["analysis_type"]:
#             response_parts.append(f"I'll perform a {result['analysis_type']} analysis. ")
        
#         # Time frame
#         if result["time_frame"]:
#             time_type = result["time_frame"].get("type")
#             if time_type == "specific_years":
#                 years_text = ", ".join([str(y) for y in result["time_frame"]["years"]])
#                 response_parts.append(f"Focusing on the year(s): {years_text}. ")
#             elif time_type == "year_range":
#                 start = result["time_frame"]["start_year"]
#                 end = result["time_frame"]["end_year"]
#                 response_parts.append(f"Covering the period from {start} to {end}. ")
#             elif time_type == "trend":
#                 response_parts.append("Showing the trend over all available time periods. ")
#             elif time_type == "relative":
#                 period = result["time_frame"]["period"]
#                 if period == "last_5_years":
#                     response_parts.append("Looking at the last 5 years. ")
#                 elif period == "last_10_years":
#                     response_parts.append("Analyzing the past decade. ")
#                 elif period == "last_year":
#                     response_parts.append("Focusing on the most recent year. ")
        
#         # Visualization suggestions
#         if result["suggested_visualizations"]:
#             viz_types = [v["type"].replace("_", " ") for v in result["suggested_visualizations"]]
#             if len(viz_types) == 1:
#                 response_parts.append(f"I recommend a {viz_types[0]} for visualizing this data. ")
#             else:
#                 response_parts.append(f"I suggest using a {viz_types[0]} or {viz_types[1]} to visualize this data. ")
        
#         # Combine parts into final response
#         result["response"] = "".join(response_parts)
#         logger.info(f"Generated response: {result['response']}")


# def main():
#     """
#     Main function to run the query processor from the command line.
#     """
#     parser = argparse.ArgumentParser(description="HESA Natural Language Query Processor")
#     parser.add_argument("--query", type=str, required=True, help="Natural language query to process")
#     parser.add_argument("--data-dir", type=str, help="Directory containing data files (default: data/cleaned_files)")
#     parser.add_argument("--output", type=str, help="Output file for the result (in JSON format)")
#     args = parser.parse_args()
    
#     # Create query processor
#     processor = QueryProcessor(data_dir=args.data_dir)
    
#     # Process query
#     print(f"\nProcessing query: {args.query}")
#     print("=" * 60)
    
#     result = processor.process_query(args.query)
    
#     # Print response
#     print("\nResponse:")
#     print("-" * 60)
#     print(result["response"])
    
#     # Print visualization suggestions
#     if result["suggested_visualizations"]:
#         print("\nSuggested Visualizations:")
#         print("-" * 60)
#         for viz in result["suggested_visualizations"]:
#             print(f"- {viz['type'].replace('_', ' ').title()}: {viz['reason']}")
    
#     # Print data sources
#     if result["data_sources"]:
#         print("\nRecommended Data Sources:")
#         print("-" * 60)
#         for source in result["data_sources"]:
#             print(f"- {source['name']}")
    
#     # Save output if requested
#     if args.output:
#         with open(args.output, 'w') as f:
#             json.dump(result, f, indent=2)
#         print(f"\nResult saved to: {args.output}")
    
#     print("\nQuery processing complete!")
#     return 0


# if __name__ == "__main__":
#     sys.exit(main()) 