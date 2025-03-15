from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from .llm_utils import generate_response
from .data_processing import CSVProcessor
from .utils.query_processor import parse_llm_response, apply_data_operations
from .utils.chart_generator import generate_chart
from .visualization.chart_generator import ChartGenerator
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for thread safety
import matplotlib.pyplot as plt
import io
import csv
from datetime import datetime
from pathlib import Path
from django.conf import settings
import os
import re
from .data_processor import transform_chart_data, prepare_chart_data
import seaborn as sns
import base64
import logging
from .data_processing.storage.storage_service import StorageService

# Custom JSON encoder to handle NaN values and other numeric types
class NumericEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super(NumericEncoder, self).default(obj)

# Custom JsonResponse that uses our encoder
def CustomJsonResponse(data, **kwargs):
    return JsonResponse(data, encoder=NumericEncoder, **kwargs)

def documentation(request):
    """Render the documentation page."""
    return render(request, 'documentation.html')

def dashboard(request):
    """Render the main dashboard page."""
    import random
    # Add a random number for cache busting
    random_cache_buster = random.randint(10000, 99999)
    return render(request, 'dashboard.html', {'random_cache_buster': random_cache_buster})

@require_http_methods(["POST"])
def process_query(request):
    """Process natural language query and return visualization data."""
    try:
        # Get query from request
        query = request.POST.get('query')
        if not query:
            return CustomJsonResponse({'error': 'No query provided'}, status=400)

        # Log the incoming query
        logger = logging.getLogger(__name__)
        logger.info(f"Processing query: {query}")

        # For testing, return a simple response
        return CustomJsonResponse({
            'status': 'success',
            'query': query,
            'chart': {
                'type': 'line', 
                'data': [
                    {'year': '2021', 'enrollment': 1500},
                    {'year': '2022', 'enrollment': 1700}
                ]
            },
            'query_interpretation': {
                'metrics': ['enrollment'],
                'time_period': {'start': '2021', 'end': '2022'},
                'institutions': ['All'],
                'comparison_type': 'trend',
                'visualization': {'type': 'line', 'options': {}}
            }
        })
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in process_query view: {str(e)}")
        return CustomJsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def download_data(request, format):
    """Download processed data in various formats."""
    try:
        # Get the latest processed data
        csv_processor = CSVProcessor()
        data = csv_processor.get_latest_processed_data()
        
        if data is None:
            return CustomJsonResponse({'error': 'No data available'}, status=404)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            # Generate CSV file
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="hesa_data_{timestamp}.csv"'
            
            data.to_csv(response, index=False)
            return response
            
        elif format == 'excel':
            # Generate Excel file
            buffer = io.BytesIO()
            data.to_excel(buffer, index=False)
            buffer.seek(0)
            
            response = HttpResponse(buffer.read(),
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = f'attachment; filename="hesa_data_{timestamp}.xlsx"'
            return response
            
        elif format == 'pdf':
            # Generate PDF with data visualization
            buffer = io.BytesIO()
            
            # Create PDF with matplotlib
            plt.figure(figsize=(10, 6))
            if 'date' in data.columns:
                plt.plot(data['date'], data['value'])
            else:
                plt.bar(range(len(data)), data[data.columns[0]])
            plt.title('HESA Data Visualization')
            plt.tight_layout()
            
            # Save to buffer
            plt.savefig(buffer, format='pdf')
            buffer.seek(0)
            plt.close()
            
            response = HttpResponse(buffer.read(), content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="hesa_data_{timestamp}.pdf"'
            return response
            
        else:
            return CustomJsonResponse({'error': 'Invalid format'}, status=400)
            
    except Exception as e:
        return CustomJsonResponse({'error': str(e)}, status=500)

def generate_summary(data):
    """Generate summary statistics from processed data."""
    try:
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        
        summary = {
            'stats': {
                'Total Records': len(data),
                'Time Period': f"{data.index.min()} - {data.index.max()}" if isinstance(data.index, pd.DatetimeIndex) else None,
            }
        }
        
        # Calculate basic statistics for numeric columns
        if not numeric_cols.empty:
            stats = data[numeric_cols].agg(['mean', 'min', 'max']).round(2)
            for col in numeric_cols:
                summary['stats'][f'{col} (avg)'] = float(stats.loc['mean', col])
                summary['stats'][f'{col} (range)'] = f"{stats.loc['min', col]} - {stats.loc['max', col]}"
        
        # Generate insights
        insights = []
        for col in numeric_cols:
            mean_val = data[col].mean()
            max_val = data[col].max()
            min_val = data[col].min()
            
            insights.append(f"Average {col}: {mean_val:.2f}")
            if max_val > mean_val * 1.5:
                insights.append(f"Notable high value in {col}: {max_val:.2f}")
            if min_val < mean_val * 0.5:
                insights.append(f"Notable low value in {col}: {min_val:.2f}")
        
        summary['insights'] = insights[:5]  # Limit to top 5 insights
        
        return summary
        
    except Exception as e:
        return {
            'stats': {'error': str(e)},
            'insights': ['Unable to generate insights due to an error']
        }

def test_charts(request):
    return render(request, 'visualization/test_charts.html')

def query_builder(request):
    """Render the query builder interface."""
    return render(request, 'visualization/query_builder.html')

def get_chart_data(request, chart_type):
    try:
        # Path to the CSV file
        csv_path = os.path.join(settings.BASE_DIR, 'data', 'cleaned_files', 'chart-1_cleaned.csv')
        
        # Read the CSV file with pandas
        try:
            # First read the metadata to determine where data starts
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                
            # Find where the actual data starts (after metadata)
            data_start = 0
            for i, line in enumerate(lines):
                if 'Academic Year' in line or 'Level of study' in line:
                    data_start = i
                    break
            
            # Read the CSV file starting from the data
            df = pd.read_csv(csv_path, skiprows=data_start)
            
            # Clean up column names
            df.columns = df.columns.str.strip()
            
            # Ensure required columns exist
            if 'Level of study' not in df.columns:
                raise ValueError("Column 'Level of study' not found in the data")
            if 'Academic Year' not in df.columns:
                raise ValueError("Column 'Academic Year' not found in the data")
            if 'Number' not in df.columns:
                raise ValueError("Column 'Number' not found in the data")
            
            # Convert Number to numeric, replacing any non-numeric values with NaN
            df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
            
            # Drop any rows with missing values
            df = df.dropna(subset=['Number', 'Level of study', 'Academic Year'])
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
        
        # Create figure (using Agg backend)
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'line':
            # Group by years and plot each level of study
            for level in df['Level of study'].unique():
                level_data = df[df['Level of study'] == level]
                plt.plot(level_data['Academic Year'], level_data['Number'], label=level, marker='o')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.title('Students by Level of Study Over Time')
            
        elif chart_type == 'bar':
            # Group by Level of study and sum the numbers
            data = df.groupby('Level of study')['Number'].sum().sort_values(ascending=False)
            plt.bar(range(len(data)), data.values)
            plt.xticks(range(len(data)), data.index, rotation=45, ha='right')
            plt.title('Total Students by Level of Study')
            
        elif chart_type == 'pie':
            # Group by Level of study and sum the numbers
            data = df.groupby('Level of study')['Number'].sum()
            plt.pie(data.values, labels=data.index, autopct='%1.1f%%')
            plt.title('Distribution of Students by Level of Study')
            
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        # Encode
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        
        return CustomJsonResponse({
            'data': {
                'image': f'data:image/png;base64,{graphic}'
            }
        })
    
    except Exception as e:
        return CustomJsonResponse({'error': str(e)}, status=500)

@require_http_methods(["POST"])
def process_hesa_query(request):
    """Process a query about HESA data and return matching data from CSV files."""
    try:
        # Get query and chart type from request
        query = request.POST.get('query')
        chart_type = request.POST.get('chart_type', 'line')
        
        # Log the request details
        logger = logging.getLogger(__name__)
        logger.info(f"Processing HESA query: {query}")
        logger.info(f"Chart type: {chart_type}")
        
        if not query:
            logger.warning("No query provided in request")
            return CustomJsonResponse({'status': 'error', 'error': 'No query provided'}, status=400)
        
        # Parse the query to extract components
        logger.info("Parsing query to extract components")
        query_info = parse_hesa_query(query)
        
        if not query_info:
            logger.warning(f"Failed to parse query: {query}")
            return CustomJsonResponse({
                'status': 'error', 
                'error': 'Could not parse the query. Please ensure it contains file pattern, HE provider, and year.'
            }, status=400)
        
        # Store the original query text for display in the UI
        query_info['original_query'] = query
        
        logger.info(f"Query parsed successfully: {query_info}")
        
        # Find relevant CSV files
        logger.info(f"Looking for CSV files matching pattern: {query_info['file_pattern']}")
        file_matches = find_relevant_csv_files(query_info['file_pattern'])
        
        if not file_matches:
            logger.warning(f"No CSV files found matching pattern: {query_info['file_pattern']}")
            return CustomJsonResponse({
                'status': 'error', 
                'error': f"No CSV files found matching pattern: {query_info['file_pattern']}"
            }, status=404)
        
        logger.info(f"Found {len(file_matches)} matching CSV files")
        
        # Filter files by requested years
        requested_years = query_info['years']
        year_filtered_matches = []
        
        # First, try to find exact year matches
        for year in requested_years:
            exact_year_matches = [
                m for m in file_matches 
                if m.get('year') and m.get('year') == str(year)
            ]
            
            if exact_year_matches:
                # Add exact year matches to our filtered list
                for match in exact_year_matches:
                    if match not in year_filtered_matches:
                        year_filtered_matches.append(match)
                        logger.info(f"Exact year match found for {year}: {match['file_name']}")
        
        # If no exact matches, try a more lenient approach (check if year appears in filename)
        if not year_filtered_matches:
            logger.info("No exact year matches found, trying filename pattern matching")
            
            for year in requested_years:
                # Look for patterns like "2020&21", "2020/21", "2020-21"
                year_patterns = [
                    f"{year}",
                    f"{year}&{str(int(year) + 1)[2:]}",
                    f"{year}-{str(int(year) + 1)[2:]}",
                    f"{year}/{str(int(year) + 1)[2:]}",
                    f"{year}_{str(int(year) + 1)[2:]}",
                    f"{year}-{int(year) + 1}"
                ]
                
                for match in file_matches:
                    for pattern in year_patterns:
                        if pattern in match['file_name']:
                            if match not in year_filtered_matches:
                                year_filtered_matches.append(match)
                                logger.info(f"Year pattern match found for {year}: {match['file_name']} (pattern: {pattern})")
                            break
        
        # If we have year-filtered matches, use those instead of all matches
        if year_filtered_matches:
            logger.info(f"Using {len(year_filtered_matches)} year-filtered matches instead of all {len(file_matches)} matches")
            file_matches = year_filtered_matches
        else:
            logger.warning(f"⚠️ No files found matching requested years: {requested_years}. Using all matching files instead.")
            
        # Re-sort by score since we might have re-ordered things
        file_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Check if we have multiple potential matches with similar scores
        multiple_matches = False
        
        if len(file_matches) > 1:
            # Check if top matches have similar scores (within 2 points)
            top_score = file_matches[0]['score']
            similar_matches = [m for m in file_matches if top_score - m['score'] <= 2]
            
            if len(similar_matches) > 1:
                multiple_matches = True
                logger.info(f"Multiple files with similar scores found: {len(similar_matches)}")
                
                # Group similar files by their base title pattern (excluding the year)
                # This will combine files that represent the same dataset but for different years
                
                # Helper function to extract base title (without year)
                def extract_base_title(filename):
                    # Remove year pattern from filename
                    base_title = re.sub(r'\d{4}[\&\-_\/]\d{2}', 'YEAR', filename)
                    return base_title
                
                # Group files by their base title
                grouped_files = {}
                for match in similar_matches:
                    file_name = match['file_name']
                    base_title = extract_base_title(file_name)
                    
                    if base_title not in grouped_files:
                        grouped_files[base_title] = []
                    
                    grouped_files[base_title].append(match)
                
                logger.info(f"Grouped files into {len(grouped_files)} distinct dataset types")
                
                # Process preview data for each group
                all_file_results = []
                
                for base_title, files in grouped_files.items():
                    logger.info(f"Processing group: {base_title} with {len(files)} files")
                    
                    # Sort files within group by year
                    files.sort(key=lambda x: x.get('year', '0000'))
                    
                    # Get a representative file for structure
                    representative_file = files[0]
                    
                    # Create a combined preview for all files in this group
                    combined_preview_data = None
                    all_file_paths = []
                    all_file_names = []
                    all_years = []
                    
                    # Process each file in the group
                    for file_match in files:
                        file_path = file_match['file_path']
                        file_name = file_match['file_name']
                        file_year = file_match.get('year', 'Unknown')
                        
                        # Only include files for the requested years
                        if file_year != 'Unknown' and file_year not in query_info['years']:
                            logger.info(f"Skipping file {file_name} with year {file_year} not in requested years {query_info['years']}")
                            continue
                        
                        all_file_paths.append(file_path)
                        all_file_names.append(file_name)
                        all_years.append(file_year)
                        
                        # Get individual file preview
                        preview_data = extract_provider_data_preview(file_path, query_info['he_providers'], max_rows=3, requested_years=query_info['years'])
                        
                        if preview_data:
                            if not combined_preview_data:
                                # First file in group, use as base
                                combined_preview_data = preview_data
                            else:
                                # Add data from this file to the combined data
                                combined_preview_data['data'].extend(preview_data['data'])
                    
                    if combined_preview_data:
                        # Create a unique ID for this group
                        group_id = hash(base_title) % 10000000  # Use modulo to keep it a reasonable size
                        
                        # Combine all file paths with a separator for later processing
                        joined_file_paths = "||".join(all_file_paths)
                        
                        # Clean up the base title for display
                        display_title = base_title.replace('YEAR', '').strip()
                        
                        # Sort the combined data by year
                        combined_preview_data['data'].sort(key=lambda x: x.get('Year', '0000'))
                        
                        # Add to results
                        # Limit total preview data to 3 rows across all files
                        has_more_rows = False
                        if len(combined_preview_data['data']) > 3:
                            # Set flag indicating there are more rows than shown
                            has_more_rows = True
                            # Truncate data to only 3 rows
                            combined_preview_data['data'] = combined_preview_data['data'][:3]
                            
                        all_file_results.append({
                            'file_info': {
                                'group_title': display_title,
                                'file_names': all_file_names,
                                'file_paths': joined_file_paths,
                                'years': all_years,
                                'match_score': representative_file['score'],
                                'matched_terms': representative_file['matched_terms']
                            },
                            'columns': combined_preview_data['columns'],
                            'data': combined_preview_data['data'],
                            'has_more_rows': has_more_rows,
                            'file_id': group_id
                        })
                
                # If we have multiple groups with preview data, return them
                if all_file_results:
                    logger.info("Returning preview data for multiple grouped dataset types")
                    return CustomJsonResponse({
                        'status': 'success',
                        'query_info': query_info,
                        'multiple_matches': True,
                        'preview_results': all_file_results
                    })
        
        # Otherwise proceed with single file processing as before
        # Track all files used and data extracted
        all_file_info = []
        all_data = []
        all_columns = None
        
        # Process each year in the query
        for year in query_info['years']:
            # Find the specific file for the current year
            logger.info(f"Looking for file matching year: {year}")
            
            # Try to find an exact match for this year
            year_matches = [m for m in file_matches if m.get('year') and m.get('year') == str(year)]
            
            # If no exact match, try pattern matching
            if not year_matches:
                year_patterns = [
                    f"{year}",
                    f"{year}&{str(int(year) + 1)[2:]}",
                    f"{year}-{str(int(year) + 1)[2:]}",
                    f"{year}/{str(int(year) + 1)[2:]}",
                    f"{year}_{str(int(year) + 1)[2:]}",
                    f"{year}-{int(year) + 1}"
                ]
                
                for match in file_matches:
                    for pattern in year_patterns:
                        if pattern in match['file_name']:
                            year_matches.append(match)
                            break
            
            if not year_matches:
                logger.warning(f"No file found for year: {year}")
                continue  # Skip this year but continue with others
            
            # Use the highest-scoring file for this year
            target_file = max(year_matches, key=lambda x: x['score'])
            
            # Extract the file_path from the file_match object
            file_path = target_file['file_path']
            file_name = target_file['file_name']
            
            logger.info(f"Found file for requested year {year}: {file_path}")
            
            # Check if a cleaned version exists
            cleaned_file_path = Path(settings.BASE_DIR) / 'data' / 'cleaned_files' / file_name
            
            if cleaned_file_path.exists():
                # Add a very noticeable log message
                logger.info("="*50)
                logger.info(f"USING CLEANED FILE FOR YEAR {year}: {file_name}")
                logger.info("="*50)
            else:
                logger.warning(f"⚠️ No cleaned version found for year {year}, will process raw file: {file_name}")
            
            # Extract data for the specified HE providers
            logger.info(f"Extracting data for HE providers: {query_info['he_providers']} for year {year}")
            result = extract_provider_data(file_path, query_info['he_providers'])
            
            if not result:
                logger.warning(f"No data found for providers: {query_info['he_providers']} in year {year}")
                continue  # Skip this year but continue with others
            
            logger.info(f"Data extracted successfully for year {year}: {len(result['data'])} rows found")
            
            # Store the columns from the first successful result
            if all_columns is None:
                all_columns = result['columns']
            
            # Get the file year (if available)
            file_year = target_file.get('year', year)
            
            # Add year information to each row
            for row in result['data']:
                row['Year'] = file_year  # Use the actual file year, not the requested year
                all_data.append(row)
            
            # Store file info
            all_file_info.append({
                'year': file_year,  # Use the actual file year
                'raw_file': file_path,
                'using_cleaned_file': cleaned_file_path.exists(),
                'cleaned_file_path': str(cleaned_file_path) if cleaned_file_path.exists() else None,
                'file_name': file_name
            })
        
        # Check if we found any data
        if not all_data:
            logger.warning(f"No data found for any year in range: {query_info['years']}")
            return CustomJsonResponse({
                'status': 'error', 
                'error': f"No data found for any year in range: {query_info['years']}"
            }, status=404)
        
        logger.info(f"Combined data for all years: {len(all_data)} rows")
        
        # Add 'Year' to columns if it's not already there
        if all_columns and 'Year' not in all_columns:
            all_columns.append('Year')
        
        # Convert data to chart format if needed
        chart_data = None
        if chart_type == 'line':
            logger.info("Preparing data for line chart")
            chart_data = prepare_chart_data_from_result({'columns': all_columns, 'data': all_data})
        
        # Return the results
        logger.info("Returning results to client")
        return CustomJsonResponse({
            'status': 'success',
            'query_info': query_info,
            'file_info': all_file_info,
            'columns': all_columns,
            'data': all_data,
            'chart_data': chart_data
        })
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in process_hesa_query view: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return CustomJsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)

@require_http_methods(["POST"])
def select_file_source(request):
    """Process request to select a specific file source after preview."""
    try:
        # Get query and file ID from request
        query = request.POST.get('query')
        file_id = request.POST.get('file_id')
        chart_type = request.POST.get('chart_type', 'line')
        
        logger = logging.getLogger(__name__)
        logger.info(f"Processing file selection - Query: {query}, File ID: {file_id}")
        
        if not query or not file_id:
            return CustomJsonResponse({
                'status': 'error', 
                'error': 'Missing query or file ID'
            }, status=400)
        
        # Parse the query
        query_info = parse_hesa_query(query)
        if not query_info:
            return CustomJsonResponse({
                'status': 'error', 
                'error': 'Could not parse the query'
            }, status=400)
        
        # Store the original query text
        query_info['original_query'] = query
        
        # Find all matching files again
        file_matches = find_relevant_csv_files(query_info['file_pattern'])
        if not file_matches:
            return CustomJsonResponse({
                'status': 'error', 
                'error': 'No matching files found'
            }, status=404)
        
        # Check if we're dealing with a group ID or a single file ID
        # First, try to find a single file match
        selected_file = None
        for match in file_matches:
            if hash(match['file_path']) % 10000000 == int(file_id):
                selected_file = match
                break
        
        # If no single file match, look for a group match
        if not selected_file:
            # Helper function to extract base title (without year)
            def extract_base_title(filename):
                # Remove year pattern from filename
                base_title = re.sub(r'\d{4}[\&\-_\/]\d{2}', 'YEAR', filename)
                return base_title
            
            # Group files by their base title
            grouped_files = {}
            for match in file_matches:
                file_name = match['file_name']
                base_title = extract_base_title(file_name)
                
                if base_title not in grouped_files:
                    grouped_files[base_title] = []
                
                grouped_files[base_title].append(match)
            
            # Check each group to see if its hash matches the file_id
            for base_title, files in grouped_files.items():
                if hash(base_title) % 10000000 == int(file_id):
                    logger.info(f"Found matching group: {base_title} with {len(files)} files")
                    
                    # Process all files in this group
                    all_file_info = []
                    all_data = []
                    all_columns = None
                    
                    # Sort files by year
                    files.sort(key=lambda x: x.get('year', '0000'))
                    
                    # Collect all files matching the requested years
                    matching_files = []
                    for match in files:
                        file_year = match.get('year')
                        if file_year and file_year in query_info['years']:
                            matching_files.append(match)
                    
                    # If no files match the requested years, use the original files list
                    if not matching_files:
                        logger.warning(f"No files match the requested years {query_info['years']}. Using all available files.")
                        matching_files = files
                    
                    # Process each file in the matching set
                    for file_match in matching_files:
                        file_path = file_match['file_path']
                        file_name = file_match['file_name']
                        file_year = file_match.get('year')
                        
                        logger.info(f"Processing file from group: {file_name}")
                        
                        # Check if a cleaned version exists
                        cleaned_file_path = Path(settings.BASE_DIR) / 'data' / 'cleaned_files' / file_name
                        using_cleaned_file = cleaned_file_path.exists()
                        
                        # Extract data for the specified HE providers
                        result = extract_provider_data(file_path, query_info['he_providers'])
                        
                        if result:
                            # Store the columns from the first successful result
                            if all_columns is None:
                                all_columns = result['columns']
                            
                            # If we don't have a file year, try to extract it from the filename
                            if not file_year:
                                year_match = re.search(r'(\d{4})[\&\-_\/](\d{2})', file_name)
                                if year_match:
                                    file_year = year_match.group(1)
                                    logger.info(f"Extracted year from filename: {file_year}")
                            
                            # Add year information to each row
                            for row in result['data']:
                                row['Year'] = file_year
                                all_data.append(row)
                            
                            # Store file info
                            all_file_info.append({
                                'year': file_year,
                                'raw_file': file_path,
                                'using_cleaned_file': using_cleaned_file,
                                'cleaned_file_path': str(cleaned_file_path) if using_cleaned_file else None,
                                'file_name': file_name
                            })
                    
                    # Check if we found any data
                    if not all_data:
                        return CustomJsonResponse({
                            'status': 'error', 
                            'error': f"No data found for providers: {query_info['he_providers']} in selected files"
                        }, status=404)
                    
                    # Add 'Year' to columns if not already there
                    if all_columns and 'Year' not in all_columns:
                        all_columns.append('Year')
                    
                    # Convert data to chart format if needed
                    chart_data = None
                    if chart_type == 'line':
                        logger.info("Preparing data for line chart")
                        chart_data = prepare_chart_data_from_result({'columns': all_columns, 'data': all_data})
                    
                    # Return the results
                    return CustomJsonResponse({
                        'status': 'success',
                        'query_info': query_info,
                        'file_info': all_file_info,  # Now returns an array of file info objects
                        'columns': all_columns,
                        'data': all_data,
                        'chart_data': chart_data
                    })
            
            # If we get here, no matching group was found
            return CustomJsonResponse({
                'status': 'error', 
                'error': 'Selected dataset type not found'
            }, status=404)
        
        # If we found a single file match, process it as before
        logger.info(f"Found selected file: {selected_file['file_name']}")
        
        # Process the selected file
        file_path = selected_file['file_path']
        file_name = selected_file['file_name']
        file_year = selected_file.get('year')
        
        # Log the file info for debugging
        logger.info("\n" + "*" * 80 + "    ")
        logger.info(f"PROCESSING FILE: {file_name}")
        logger.info("*" * 80)
        
        # Check if a cleaned version exists
        cleaned_file_path = Path(settings.BASE_DIR) / 'data' / 'cleaned_files' / file_name
        using_cleaned_file = cleaned_file_path.exists()
        
        # Extract data for the specified HE providers
        logger.info(f"Processing file: {file_path} for providers: {query_info['he_providers']}")
        result = extract_provider_data(file_path, query_info['he_providers'])
        
        if not result:
            return CustomJsonResponse({
                'status': 'error', 
                'error': f"No data found for providers: {query_info['he_providers']} in file: {file_name}"
            }, status=404)
        
        # If we don't have a file year from the file matching process, try to extract it from the filename
        if not file_year:
            # Try to extract year from the filename
            year_match = re.search(r'(\d{4})[\&\-_\/](\d{2})', file_name)
            if year_match:
                file_year = year_match.group(1)
                logger.info(f"Extracted year from filename: {file_year}")
            else:
                # Default to the first year from the query if we can't extract it
                file_year = query_info['years'][0] if query_info['years'] else None
                logger.info(f"Using default year from query: {file_year}")
        
        # Add year information to each row
        for row in result['data']:
            row['Year'] = file_year
        
        # Add 'Year' to columns if not already there
        columns = result['columns']
        if 'Year' not in columns:
            columns.append('Year')
        
        # Convert data to chart format if needed
        chart_data = None
        if chart_type == 'line':
            logger.info("Preparing data for line chart")
            chart_data = prepare_chart_data_from_result(result)
        
        # Return the results with a single file in the file_info array for consistency
        return CustomJsonResponse({
            'status': 'success',
            'query_info': query_info,
            'file_info': [{
                'year': file_year,
                'raw_file': file_path,
                'using_cleaned_file': using_cleaned_file,
                'cleaned_file_path': str(cleaned_file_path) if using_cleaned_file else None,
                'file_name': file_name
            }],
            'columns': columns,
            'data': result['data'],
            'chart_data': chart_data
        })
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in select_file_source view: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return CustomJsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)

def parse_hesa_query(query):
    """
    Parse a natural language query to extract:
    - File pattern (keywords from the file name)
    - HE provider name(s) - can handle multiple providers separated by "and"
    - Year or year range (e.g., 2015, 2015-2016, from 2015 to 2016)
    """
    logger = logging.getLogger(__name__)
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Extract file pattern keywords - these are more flexible now
    file_pattern_keywords = []
    
    # Common patterns in file names - expanded to include variations
    patterns = [
        "full-time", "full time", "fulltime",
        "student enrolments", "student enrollments", "enrolments", "enrollments",  
        "term-time accommodation", "term time", "accommodation",
        "he provider", "provider", "university",
        "he qualifiers", "qualifiers", "qualification",
        "level of qualification", "classification", "first degrees",
        "fte", "full-time equivalence", "cost centre", "level of study",
        "distance learning", "placement marker", "subject of study", "itt students"
    ]
    
    for pattern in patterns:
        if pattern.lower() in query_lower:
            file_pattern_keywords.append(pattern)
    
    logger.info(f"Extracted file pattern keywords: {file_pattern_keywords}")
    
    # Extract HE provider(s) (university name(s))
    providers = []
    
    # Try to extract provider section using regex patterns
    provider_section_match = None
    
    # Try pattern for year range: "for X in 2015-2016" or "for X in 2015 to 2016"
    if not provider_section_match:
        provider_section_match = re.search(r'for\s+(.*?)\s+in\s+\d{4}\s*[\-–—to]\s*\d{4}', query)
    
    # Try standard pattern: "for X in 2015"
    if not provider_section_match:
        provider_section_match = re.search(r'for\s+(.*?)\s+in\s+\d{4}', query)
    
    if provider_section_match:
        provider_section = provider_section_match.group(1).strip()
        logger.info(f"Found provider section: {provider_section}")
        
        # Split by "and" to get multiple providers
        if " and " in provider_section:
            provider_parts = provider_section.split(" and ")
            for part in provider_parts:
                providers.append(part.strip())
            logger.info(f"Extracted multiple HE providers: {providers}")
        else:
            # Single provider
            providers.append(provider_section)
            logger.info(f"Extracted single HE provider: {providers[0]}")
    else:
        logger.warning("Failed to extract HE provider(s) from query")
    
    # Extract year or year range
    years = []
    
    # Try to extract year range first: 2015-2016 format
    year_range_match = re.search(r'in\s+(\d{4})\s*[\-–—]\s*(\d{4})', query)
    if year_range_match:
        start_year = year_range_match.group(1)
        end_year = year_range_match.group(2)
        logger.info(f"Extracted year range: {start_year} to {end_year}")
        
        # Add all years in the range
        for year in range(int(start_year), int(end_year) + 1):
            years.append(str(year))
        
        logger.info(f"Years in range: {years}")
    
    # Try "from X to Y" format
    elif re.search(r'in\s+\d{4}\s+to\s+\d{4}', query):
        year_range_match = re.search(r'in\s+(\d{4})\s+to\s+(\d{4})', query)
        if year_range_match:
            start_year = year_range_match.group(1)
            end_year = year_range_match.group(2)
            logger.info(f"Extracted year range (from-to format): {start_year} to {end_year}")
            
            # Add all years in the range
            for year in range(int(start_year), int(end_year) + 1):
                years.append(str(year))
            
            logger.info(f"Years in range: {years}")
    
    # If no range found, try single year
    elif not years:
        year_match = re.search(r'in\s+(\d{4})', query)
        if year_match:
            years.append(year_match.group(1))
            logger.info(f"Extracted single year: {years[0]}")
    
    if not years:
        logger.warning("Failed to extract year or year range from query")
    
    # If we couldn't extract essential components, return None
    if not file_pattern_keywords or not providers or not years:
        logger.warning("Missing essential components from query")
        if not file_pattern_keywords:
            logger.warning("Missing file pattern keywords")
        if not providers:
            logger.warning("Missing HE provider(s)")
        if not years:
            logger.warning("Missing year or year range")
        return None
    
    return {
        'file_pattern': ' '.join(file_pattern_keywords),
        'he_providers': providers,
        'years': years  # Now returns a list of years
    }

def find_relevant_csv_files(file_pattern):
    """Find CSV files that match the given pattern using metadata from cleaned files."""
    logger = logging.getLogger(__name__)
    
    # Get path to cleaned_files directory (use this instead of raw_files)
    cleaned_files_dir = Path(settings.BASE_DIR) / 'data' / 'cleaned_files'
    
    logger.info(f"Searching for CSV files in: {cleaned_files_dir}")
    
    # Check if directory exists
    if not cleaned_files_dir.exists():
        logger.error(f"Directory does not exist: {cleaned_files_dir}")
        return []
    
    # List all CSV files in the directory
    all_csv_files = list(cleaned_files_dir.glob('*.csv'))
    logger.info(f"Found {len(all_csv_files)} CSV files in directory")
    
    # Extract keywords from file pattern
    pattern_keywords = [keyword.lower() for keyword in file_pattern.lower().split()]
    
    # Find all CSV files that match based on metadata
    matching_files_with_scores = []
    
    for file_path in all_csv_files:
        try:
            # Read just the first line of the file to get metadata
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            
            # Default year extraction from filename
            file_year = None
            # Try to extract year from filename using regex
            year_match = re.search(r'(\d{4})[\&\-_\/](\d{2})', file_path.name)
            if year_match:
                file_year = year_match.group(1)  # Extract the first year
                logger.info(f"Extracted year from filename: {file_year} for {file_path.name}")
            
            # Check if it contains our metadata format
            if first_line.startswith('#METADATA:'):
                # Extract the JSON metadata
                metadata_json = first_line[9:].strip()  # Skip the #METADATA: prefix
                try:
                    # Try to parse the JSON, handling potential issues
                    try:
                        metadata = json.loads(metadata_json)
                    except json.JSONDecodeError as json_error:
                        # If there's an error, try to clean the JSON string
                        logger.warning(f"Initial JSON parse error for {file_path.name}: {str(json_error)}")
                        # Try removing any trailing characters that might be causing issues
                        metadata_json = re.sub(r'\s+[^\{\}\"]+$', '', metadata_json)
                        metadata = json.loads(metadata_json)
                    
                    title = metadata.get('title', '').lower()
                    keywords = [k.lower() for k in metadata.get('keywords', [])]
                    
                    # Try to get year from metadata if not already extracted from filename
                    if not file_year and 'academic_year' in metadata:
                        academic_year = metadata.get('academic_year', '')
                        # Extract the first year from formats like "2015/16" or "2015-16"
                        year_match = re.search(r'(\d{4})', academic_year)
                        if year_match:
                            file_year = year_match.group(1)
                            logger.info(f"Extracted year from metadata: {file_year} for {file_path.name}")
                    
                    # Count keyword matches
                    keyword_matches = 0
                    matched_terms = []
                    
                    # Check title for keyword matches
                    title_matches = sum(1 for keyword in pattern_keywords if keyword in title)
                    if title_matches > 0:
                        keyword_matches += title_matches
                        matched_terms.append(f"title contains {title_matches} keywords")
                    
                    # Check keywords for matches
                    expanded_keywords = []
                    for k in keywords:
                        expanded_keywords.extend(k.split())
                    
                    keyword_match_count = 0
                    for pattern_kw in pattern_keywords:
                        if any(pattern_kw in ex_kw for ex_kw in expanded_keywords):
                            keyword_match_count += 1
                    
                    if keyword_match_count > 0:
                        keyword_matches += keyword_match_count
                        matched_terms.append(f"keywords match {keyword_match_count} terms")
                    
                    # If we have matches, add this file to our results
                    if keyword_matches > 0:
                        matching_files_with_scores.append({
                            'file_path': str(file_path),
                            'score': keyword_matches,
                            'matched_terms': matched_terms,
                            'file_name': file_path.name,
                            'year': file_year  # Add the extracted year
                        })
                        logger.info(f"File matched by metadata ({keyword_matches} keywords): {file_path} (Year: {file_year})")
                
                except Exception as e:
                    logger.warning(f"Invalid metadata JSON in file: {file_path}")
                    logger.warning(f"Error: {str(e)}")
                    # Fall back to filename matching if metadata parsing fails
                    _check_filename_match_with_score(file_path, pattern_keywords, matching_files_with_scores)
                    # Add year information from filename if we fell back to filename matching
                    if matching_files_with_scores and matching_files_with_scores[-1]['file_path'] == str(file_path):
                        matching_files_with_scores[-1]['year'] = file_year
            else:
                # If no metadata, fall back to filename matching
                logger.info(f"No metadata found in file, falling back to filename matching: {file_path}")
                _check_filename_match_with_score(file_path, pattern_keywords, matching_files_with_scores)
                # Add year information from filename if we have a match
                if matching_files_with_scores and matching_files_with_scores[-1]['file_path'] == str(file_path):
                    matching_files_with_scores[-1]['year'] = file_year
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            # Continue with the next file
    
    # Sort by score (highest first)
    matching_files_with_scores.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Found {len(matching_files_with_scores)} matching files")
    
    return matching_files_with_scores

def _check_filename_match_with_score(file_path, pattern_keywords, matching_files_with_scores):
    """Helper function to check if a filename matches the pattern keywords and add with score."""
    logger = logging.getLogger(__name__)
    filename = file_path.name.lower()
    
    # Count how many keywords match in the filename
    keyword_matches = 0
    matched_terms = []
    
    for keyword in pattern_keywords:
        # Check for variations to improve matching
        if keyword in filename:
            keyword_matches += 1
            matched_terms.append(keyword)
        elif keyword == 'enrollment' and 'enrolment' in filename:
            keyword_matches += 1  # Handle UK/US spelling differences
            matched_terms.append(f"{keyword}→enrolment")
        elif keyword == 'enrollments' and 'enrolments' in filename:
            keyword_matches += 1
            matched_terms.append(f"{keyword}→enrolments")
        elif keyword == 'term-time' and 'term time' in filename:
            keyword_matches += 1
            matched_terms.append(f"{keyword}→term time")
    
    # If at least half of the keywords match, consider it a match
    match_threshold = max(1, len(pattern_keywords) // 2)
    if keyword_matches >= match_threshold:
        matching_files_with_scores.append({
            'file_path': str(file_path),
            'score': keyword_matches,
            'matched_terms': matched_terms,
            'file_name': file_path.name
        })
        logger.info(f"File matched by filename ({keyword_matches} keywords): {file_path}")

def find_file_for_year(file_matches, year):
    """Find the specific file for the requested year using metadata if available."""
    logger = logging.getLogger(__name__)
    
    # Try to find an exact match in metadata first
    for file_match in file_matches:
        file_path = file_match['file_path']
        try:
            # Read metadata from the first line
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            
            if first_line.startswith('#METADATA:'):
                try:
                    metadata_json = first_line[9:].strip()
                    # Try to parse the JSON, handling potential issues
                    try:
                        metadata = json.loads(metadata_json)
                    except json.JSONDecodeError as json_error:
                        # If there's an error, try to clean the JSON string
                        logger.warning(f"Initial JSON parse error in find_file_for_year for {file_path}: {str(json_error)}")
                        # Try removing any trailing characters that might be causing issues
                        metadata_json = re.sub(r'\s+[^\{\}\"]+$', '', metadata_json)
                        metadata = json.loads(metadata_json)
                    
                    # Check if academic_year matches
                    academic_year = metadata.get('academic_year', '')
                    if academic_year:
                        year_match = False
                        
                        # Check different formats for academic year in metadata
                        if str(year) == academic_year:
                            year_match = True
                        elif f"{year}" in academic_year:
                            year_match = True
                        elif f"{year}/{str(int(year) + 1)[2:]}" in academic_year:  # 2015/16
                            year_match = True
                        elif f"{year}-{str(int(year) + 1)[2:]}" in academic_year:  # 2015-16
                            year_match = True
                        
                        if year_match:
                            logger.info(f"Found file for year {year} using metadata: {file_path}")
                            return file_match
                except Exception as e:
                    # If JSON parsing fails, try to extract the academic_year using regex
                    try:
                        year_match = re.search(r'"academic_year":\s*"([^"]+)"', first_line)
                        if year_match:
                            academic_year = year_match.group(1)
                            # Check if year matches any of our formats
                            if str(year) == academic_year or f"{year}" in academic_year:
                                logger.info(f"Found file for year {year} using regex-extracted academic_year: {file_path}")
                                return file_match
                    except Exception as extract_error:
                        logger.warning(f"Error extracting academic_year with regex: {str(extract_error)}")
        
        except Exception as e:
            logger.warning(f"Error reading metadata from {file_path}: {str(e)}")
    
    # If no match found using metadata, fall back to filename matching
    logger.info("Falling back to filename matching for year")
    
    # Different formats for academic year
    academic_year_formats = [
        f"{year}",                   # Just the year
        f"{year}&{str(int(year) + 1)[2:]}",  # 2015&16 format
        f"{year}-{str(int(year) + 1)[2:]}",  # 2015-16 format
        f"{year}_{str(int(year) + 1)[2:]}",  # 2015_16 format
        f"{year}/{str(int(year) + 1)[2:]}",  # 2015/16 format
        f"{year}-{int(year) + 1}",   # 2015-2016 format
    ]
    
    logger.info(f"Looking for files with year patterns: {academic_year_formats}")
    
    for file_match in file_matches:
        file_path = file_match['file_path']
        for year_format in academic_year_formats:
            if year_format in file_path:
                logger.info(f"Found matching year format '{year_format}' in {file_path}")
                return file_match
    
    return None

def extract_provider_data(file_path, he_providers):
    """Extract data for the specified HE providers from the CSV file."""
    try:
        logger = logging.getLogger(__name__)
        
        # Get just the filename without the path
        file_name = Path(file_path).name
        
        # Add prominent log message showing the filename being used
        logger.info("\n" + "*"*80)
        logger.info(f"PROCESSING FILE: {file_name}")
        logger.info("*"*80 + "\n")
        
        logger.info(f"Processing file: {file_path} for providers: {he_providers}")
        
        # Check if a cleaned version of this file exists
        file_name = Path(file_path).name
        cleaned_file_path = Path(settings.BASE_DIR) / 'data' / 'cleaned_files' / file_name
        
        if cleaned_file_path.exists():
            logger.info(f"Using cleaned version of file: {cleaned_file_path}")
            try:
                # Check if first line is metadata
                with open(cleaned_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                
                # If first line is metadata, skip it when reading with pandas
                skiprows = 1 if first_line.startswith('#METADATA:') else 0
                
                # Try reading the cleaned file directly
                df = pd.read_csv(cleaned_file_path, skiprows=skiprows)
                logger.info(f"Successfully read cleaned file with shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error reading cleaned file: {str(e)}")
                # Fall back to processing the raw file
                logger.info("Falling back to raw file processing")
                return process_raw_file(file_path, he_providers)
        else:
            logger.info(f"No cleaned version found for {file_name}, processing raw file")
            return process_raw_file(file_path, he_providers)
        
        # Look for the HE provider column in the cleaned file
        he_provider_col = None
        possible_provider_columns = df.columns
        
        for col_name in possible_provider_columns:
            col_lower = str(col_name).lower()
            if ('he provider' in col_lower or 'provider' in col_lower or 
                'institution' in col_lower or 'university' in col_lower):
                he_provider_col = col_name
                logger.info(f"Found provider column: {he_provider_col}")
                break
        
        if not he_provider_col:
            logger.error("No HE provider column found in CSV file")
            return None
        
        # Clean the data
        df = df.apply(lambda x: x.str.strip() if hasattr(x, 'str') and x.dtype == "object" else x)
        
        # Find matching rows for all providers
        all_provider_rows = []
        
        for he_provider in he_providers:
            logger.info(f"Looking for data for provider: {he_provider}")
            
            # Normalize the provider name for case-insensitive comparison
            norm_provider = he_provider.lower().replace('university of ', '').replace('the ', '')
            
            # First, try an exact match
            logger.info(f"Looking for exact match with provider: {he_provider}")
            provider_rows = df[df[he_provider_col].astype(str).str.lower() == he_provider.lower()]
            
            # If no exact match, try with "The" prefix
            if provider_rows.empty and not he_provider.lower().startswith('the '):
                logger.info("Trying with 'The' prefix")
                provider_rows = df[df[he_provider_col].astype(str).str.lower() == f"the {he_provider.lower()}"]
            
            # If still no match, try without "University of" prefix
            if provider_rows.empty and 'university of' in he_provider.lower():
                logger.info("Trying without 'University of' prefix")
                shorter_name = he_provider.lower().replace('university of ', '')
                provider_rows = df[df[he_provider_col].astype(str).str.lower().str.contains(shorter_name, na=False)]
            
            # If still empty, try more flexible matching
            if provider_rows.empty:
                logger.info("Using more flexible matching approach")
                for _, row in df.iterrows():
                    provider_name = row[he_provider_col]
                    if provider_name and not pd.isna(provider_name):
                        provider_lower = str(provider_name).lower()
                        # Check if the key part of the university name is in the provider
                        if (norm_provider in provider_lower or 
                            any(word in provider_lower for word in norm_provider.split() if len(word) > 3)):
                            logger.info(f"Found provider with keyword match: {provider_name}")
                            provider_rows = df[df[he_provider_col] == provider_name]
                            break
            
            if not provider_rows.empty:
                logger.info(f"Found {len(provider_rows)} rows for provider: {he_provider}")
                all_provider_rows.append(provider_rows)
            else:
                logger.warning(f"No data found for provider: {he_provider}")
        
        if not all_provider_rows:
            logger.error(f"No data found for any of the providers: {he_providers}")
            return None
        
        # Combine all provider rows
        combined_rows = pd.concat(all_provider_rows)
        logger.info(f"Combined data for all providers: {len(combined_rows)} rows")
        
        # Replace NaN values with None before converting to dict - this is key to fix the JSON issue
        combined_rows = combined_rows.replace({np.nan: None})
        
        # Convert the rows to a dict for the response
        provider_data = combined_rows.to_dict('records')
        
        # Return columns and data
        return {
            'columns': df.columns.tolist(),
            'data': provider_data
        }
        
    except Exception as e:
        logger.error(f"Error extracting provider data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_raw_file(file_path, he_providers):
    """Process the raw file if no cleaned version is available."""
    try:
        logger = logging.getLogger(__name__)
        
        # Read the CSV file contents to manually determine where data starts
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        
        # Check if first line is metadata and skip it if needed
        skip_first_line = False
        if lines and lines[0].strip().startswith('#METADATA:'):
            logger.info(f"Found metadata in first line, will skip for data processing")
            skip_first_line = True
            # Adjust lines to skip metadata
            lines = lines[1:]
        
        # Log the first few lines to help with debugging
        logger.info(f"First 10 lines of the file (after metadata if present):")
        for i, line in enumerate(lines[:10]):
            logger.info(f"Line {i}: {line.strip()}")
        
        # Find header row by looking for a line with HE provider or similar in it
        header_row = -1
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Look for actual header row with column headers
            if ('"he provider"' in line_lower or 'he provider' in line_lower or
                '"ukprn"' in line_lower or 'ukprn' in line_lower):
                header_row = i
                logger.info(f"Found header row at line {i}: {line.strip()}")
                break
        
        if header_row == -1:
            logger.error("Could not find header row with 'HE provider' or 'UKPRN' column")
            # Try a different approach - look for lines with many commas
            for i, line in enumerate(lines):
                if i > 5 and line.count(',') >= 3:  # Assume header has multiple columns
                    header_row = i
                    logger.info(f"Found potential header row at line {i}: {line.strip()}")
                    break
        
        if header_row == -1:
            logger.error("Could not find header row, defaulting to line 15")
            header_row = 15  # Common for HESA files
        
        # The data starts in the line after the header
        data_start_row = header_row + 1
        
        # Custom read approach:
        # 1. Read only the header row to get column names
        try:
            # Adjust skiprows to account for metadata if present
            skiprows_header = header_row + (1 if skip_first_line else 0)
            header_df = pd.read_csv(file_path, skiprows=skiprows_header, nrows=1)
            column_names = header_df.columns
            logger.info(f"Column names from header: {column_names}")
        except Exception as e:
            logger.error(f"Error reading header: {str(e)}")
            # Try manual parsing
            header_line = lines[header_row].strip()
            import csv
            try:
                reader = csv.reader([header_line])
                column_names = next(reader)
                logger.info(f"Manually parsed column names: {column_names}")
            except:
                column_names = header_line.split(',')
                logger.info(f"Fallback column names: {column_names}")
        
        # 2. Read the data starting from the data row
        try:
            # Adjust skiprows to account for metadata if present
            skiprows_data = data_start_row + (1 if skip_first_line else 0)
            df = pd.read_csv(file_path, skiprows=skiprows_data, names=column_names)
            logger.info(f"Read data with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            # Try with different approach
            try:
                skiprows_range = list(range(0, data_start_row))
                if skip_first_line:
                    skiprows_range.insert(0, 0)  # Add line 0 (metadata) to skip list
                df = pd.read_csv(file_path, skiprows=skiprows_range, names=column_names)
                logger.info(f"Alternative parsing successful with shape: {df.shape}")
            except Exception as e2:
                logger.error(f"Alternative parsing also failed: {str(e2)}")
                
                # Last resort: manual parsing
                logger.info("Trying manual CSV parsing")
                data_rows = []
                for i in range(data_start_row, len(lines)):
                    try:
                        # Simple CSV parsing, handling quoted values
                        row_values = []
                        line = lines[i].strip()
                        in_quotes = False
                        current_value = ''
                        
                        for char in line:
                            if char == '"':
                                in_quotes = not in_quotes
                            elif char == ',' and not in_quotes:
                                row_values.append(current_value.strip('"'))
                                current_value = ''
                            else:
                                current_value += char
                                
                        # Add the last value
                        if current_value:
                            row_values.append(current_value.strip('"'))
                            
                        # Make sure we have the right number of columns
                        if len(row_values) == len(column_names):
                            data_rows.append(dict(zip(column_names, row_values)))
                    except Exception as e3:
                        logger.warning(f"Skipping problematic row {i}: {str(e3)}")
                
                if not data_rows:
                    logger.error("Manual parsing produced no valid rows")
                    return None
                    
                df = pd.DataFrame(data_rows)
                logger.info(f"Manual parsing successful, created dataframe with shape: {df.shape}")
        
        # Look for the HE provider column
        he_provider_col = None
        possible_provider_columns = column_names
        
        for col_name in possible_provider_columns:
            col_lower = str(col_name).lower()
            if ('he provider' in col_lower or 'provider' in col_lower or 
                'institution' in col_lower or 'university' in col_lower):
                he_provider_col = col_name
                logger.info(f"Found provider column: {he_provider_col}")
                break
        
        if not he_provider_col:
            logger.error("No HE provider column found in CSV file")
            return None
        
        # Find matching rows for all providers
        all_provider_rows = []
        
        for he_provider in he_providers:
            logger.info(f"Looking for data for provider: {he_provider}")
            
            # Normalize the provider name for case-insensitive comparison
            norm_provider = he_provider.lower().replace('university of ', '').replace('the ', '')
            
            try:
                # First, try an exact match
                logger.info(f"Looking for exact match with provider: {he_provider}")
                provider_rows = df[df[he_provider_col].astype(str).str.lower() == he_provider.lower()]
                
                # If no exact match, try with "The" prefix
                if provider_rows.empty and not he_provider.lower().startswith('the '):
                    logger.info("Trying with 'The' prefix")
                    provider_rows = df[df[he_provider_col].astype(str).str.lower() == f"the {he_provider.lower()}"]
                
                # If still no match, try without "University of" prefix
                if provider_rows.empty and 'university of' in he_provider.lower():
                    logger.info("Trying without 'University of' prefix")
                    shorter_name = he_provider.lower().replace('university of ', '')
                    provider_rows = df[df[he_provider_col].astype(str).str.lower().str.contains(shorter_name, na=False)]
                
                # If still empty, try more flexible matching
                if provider_rows.empty:
                    logger.info("Using more flexible matching approach")
                    for _, row in df.iterrows():
                        provider_name = row[he_provider_col]
                        if provider_name and not pd.isna(provider_name):
                            provider_lower = str(provider_name).lower()
                            # Check if the key part of the university name is in the provider
                            if (norm_provider in provider_lower or 
                                any(word in provider_lower for word in norm_provider.split() if len(word) > 3)):
                                logger.info(f"Found provider with keyword match: {provider_name}")
                                provider_rows = df[df[he_provider_col] == provider_name]
                                break
            
                if not provider_rows.empty:
                    logger.info(f"Found {len(provider_rows)} rows for provider: {he_provider}")
                    all_provider_rows.append(provider_rows)
                else:
                    logger.warning(f"No data found for provider: {he_provider}")
                    
            except Exception as e:
                logger.error(f"Error finding provider {he_provider}: {str(e)}")
        
        if not all_provider_rows:
            logger.error(f"No data found for any of the providers: {he_providers}")
            return None
        
        # Combine all provider rows
        combined_rows = pd.concat(all_provider_rows)
        logger.info(f"Combined data for all providers: {len(combined_rows)} rows")
        
        # Replace NaN values with None before converting to dict - this is key to fix the JSON issue
        combined_rows = combined_rows.replace({np.nan: None})
        
        # Convert the rows to a dict for the response
        provider_data = combined_rows.to_dict('records')
        
        # Return columns and data
        return {
            'columns': df.columns.tolist(),
            'data': provider_data
        }
        
    except Exception as e:
        logger.error(f"Error processing raw file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def prepare_chart_data_from_result(result):
    """
    Prepare chart data from the query result for use with Chart.js.
    Now handles multiple rows for different providers and years.
    """
    if not result or not result['data'] or not result['columns']:
        return None
    
    data = result['data']
    columns = result['columns']
    
    # Determine which column has the provider name
    provider_col = None
    for col in columns:
        if 'provider' in col.lower() or 'institution' in col.lower() or 'university' in col.lower():
            provider_col = col
            break
    
    if not provider_col:
        provider_col = columns[0]  # Fallback to first column
    
    # Get unique providers and years
    providers = []
    years = []
    
    for row in data:
        if provider_col in row and row[provider_col] not in providers:
            providers.append(row[provider_col])
        
        if 'Year' in row and row['Year'] not in years:
            years.append(row['Year'])
    
    # Sort years chronologically
    years.sort()
    
    # Find numeric columns (excluding Year and provider columns)
    numeric_columns = []
    for col in columns:
        if col != provider_col and col != 'Year':
            try:
                # Check if at least one row has a numeric value for this column
                for row in data:
                    if col in row:
                        value = row[col]
                        if isinstance(value, str):
                            value = value.replace(',', '')  # Remove commas from numbers like "1,234"
                        float(value)  # Try to convert to float
                        numeric_columns.append(col)
                        break
            except (ValueError, TypeError):
                # Skip non-numeric columns
                continue
    
    # If no numeric columns found (unlikely), return None
    if not numeric_columns:
        return None
    
    # Create datasets for Chart.js
    # Each provider will have its own dataset for each numeric column
    datasets = []
    
    # First, use the first numeric column for all providers
    main_metric = numeric_columns[0]
    
    for provider in providers:
        dataset = {
            'label': f"{provider} - {main_metric}",
            'data': [],
            'borderColor': get_random_color(),
            'backgroundColor': 'rgba(0, 0, 0, 0.1)',
            'fill': False
        }
        
        # Collect data points for each year
        for year in years:
            value_found = False
            
            # Find the matching row for this provider and year
            for row in data:
                if row.get(provider_col) == provider and row.get('Year') == year and main_metric in row:
                    try:
                        value = row[main_metric]
                        if value is None:
                            # Handle None values by using null (will be converted to 'null' in JSON)
                            dataset['data'].append(None)
                        elif isinstance(value, str):
                            value = value.replace(',', '')  # Remove commas
                            dataset['data'].append(float(value))
                        else:
                            # Handle potential NaN values
                            if pd.isna(value):  # This checks for np.nan, pd.NA, etc.
                                dataset['data'].append(None)
                            else:
                                dataset['data'].append(float(value))
                        value_found = True
                        break
                    except (ValueError, TypeError):
                        # If conversion fails, add null
                        dataset['data'].append(None)
                        value_found = True
                        break
            
            # If no value found for this year, add null for continuity
            if not value_found:
                dataset['data'].append(None)
        
        datasets.append(dataset)
    
    # Create the final result
    chart_data = {
        'labels': years,
        'datasets': datasets
    }
    
    # Convert any remaining NaN values to None for proper JSON serialization
    # This is a safety step in case any NaN values slipped through
    return json.loads(json.dumps(chart_data, cls=NumericEncoder))

def get_random_color():
    """Generate a random RGB color."""
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'rgb({r}, {g}, {b})'

def extract_provider_data_preview(file_path, he_providers, max_rows=3, requested_years=None):
    """
    Extract a preview of data for the specified HE providers from the CSV file.
    Limited to a maximum number of rows for quick preview display.
    """
    try:
        logger = logging.getLogger(__name__)
        
        # Get just the filename without the path
        file_name = Path(file_path).name
        
        logger.info(f"Processing preview for file: {file_path} for providers: {he_providers}")
        
        # Check if file exists
        if not Path(file_path).exists():
            logger.error(f"File does not exist: {file_path}")
            return None
        
        # Try to extract year from the filename
        file_year = None
        year_match = re.search(r'(\d{4})[\&\-_\/](\d{2})', file_name)
        if year_match:
            file_year = year_match.group(1)
            logger.info(f"Extracted year from filename for preview: {file_year}")
        
        # Read the CSV file
        try:
            # Check if first line is metadata
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            
            # If first line is metadata, skip it when reading with pandas
            skiprows = 1 if first_line.startswith('#METADATA:') else 0
            
            # Try to extract year from metadata if we couldn't from filename
            if skiprows == 1 and not file_year:
                try:
                    if first_line.startswith('#METADATA:'):
                        metadata_json = first_line[9:].strip()
                        metadata = json.loads(metadata_json)
                        if 'academic_year' in metadata:
                            year_match = re.search(r'(\d{4})', metadata['academic_year'])
                            if year_match:
                                file_year = year_match.group(1)
                                logger.info(f"Extracted year from metadata for preview: {file_year}")
                except Exception as e:
                    logger.warning(f"Could not extract year from metadata: {str(e)}")
            
            # Read the file
            df = pd.read_csv(file_path, skiprows=skiprows)
            logger.info(f"Successfully read file with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return None
        
        # If we have requested years, check if this file's year is in the requested years
        if requested_years and file_year:
            if file_year not in requested_years:
                logger.info(f"File year {file_year} not in requested years {requested_years}, skipping")
                return None
        
        # Look for the HE provider column
        he_provider_col = None
        possible_provider_columns = df.columns
        
        for col_name in possible_provider_columns:
            col_lower = str(col_name).lower()
            if ('he provider' in col_lower or 'provider' in col_lower or 
                'institution' in col_lower or 'university' in col_lower):
                he_provider_col = col_name
                logger.info(f"Found provider column: {he_provider_col}")
                break
        
        if not he_provider_col:
            logger.error("No HE provider column found in CSV file")
            return None
        
        # Look for data for each provider
        all_provider_rows = []
        
        for provider in he_providers:
            logger.info(f"Looking for data for provider: {provider}")
            
            # Try exact match first
            logger.info(f"Looking for exact match with provider: {provider}      ")
            provider_rows = df[df[he_provider_col] == provider]
            
            # If no results, try with "The" prefix
            if len(provider_rows) == 0 and not provider.lower().startswith('the '):
                logger.info(f"Trying with 'The' prefix")
                provider_rows = df[df[he_provider_col] == f"The {provider}"]
            
            # If still no results, try case-insensitive contains
            if len(provider_rows) == 0:
                logger.info(f"Trying contains match")
                # Convert to string first to handle non-string columns
                mask = df[he_provider_col].astype(str).str.lower().str.contains(provider.lower())
                provider_rows = df[mask]
            
            if len(provider_rows) > 0:
                logger.info(f"Found {len(provider_rows)} rows for provider: {provider}")
                
                # Limit to max_rows
                provider_rows = provider_rows.head(max_rows)
                
                # Convert to dict for output
                for _, row in provider_rows.iterrows():
                    all_provider_rows.append(row.to_dict())
            else:
                logger.warning(f"No data found for provider: {provider}")
        
        # Add year to all rows if we have it
        if file_year:
            for row in all_provider_rows:
                row['Year'] = file_year
        
        # Return data with columns
        if all_provider_rows:
            # Get columns from the first row
            columns = list(all_provider_rows[0].keys())
            
            # Add Year to columns if it's in the data but not in columns
            if file_year and 'Year' not in columns:
                columns.append('Year')
            
            logger.info(f"Preview data ready with {len(all_provider_rows)} rows and {len(columns)} columns")
            return {
                'columns': columns,
                'data': all_provider_rows
            }
        else:
            logger.warning("No data found for any provider")
            return None
        
    except Exception as e:
        logger.error(f"Error extracting provider data preview: {str(e)}")
        return None