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
from .data_processing.csv_processor import CLEANED_FILES_DIR, RAW_FILES_DIR  # Import the constants

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

@require_http_methods(["GET", "POST"])
def process_hesa_query(request):
    """
    Process a HESA data query and return matching CSV files.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get query parameters
        if request.method == 'GET':
            query = request.GET.get('query', '')
            chart_type = request.GET.get('chart_type', 'bar')
            max_matches = int(request.GET.get('max_matches', 3))
            institution = request.GET.get('institution', '')
            start_year = request.GET.get('start_year', '')
            end_year = request.GET.get('end_year', '')
        else:  # POST
            query = request.POST.get('query', '')
            chart_type = request.POST.get('chart_type', 'bar')
            max_matches = int(request.POST.get('max_matches', 3))
            institution = request.POST.get('institution', '')
            start_year = request.POST.get('start_year', '')
            end_year = request.POST.get('end_year', '')
        
        logger.info(f"Processing HESA query: '{query}', chart_type: {chart_type}, max_matches: {max_matches}")
        logger.info(f"Institution: '{institution}', start_year: {start_year}, end_year: {end_year}")
        
        # Process the query to extract relevant information
        file_pattern, keywords, stopwords = parse_hesa_query(query)
        
        logger.info(f"Query parsed: file_pattern: '{file_pattern}', keywords: {keywords}")
        logger.info(f"Stopwords removed: {stopwords}")
        
        # Get expanded keywords with synonyms
        synonyms = {
            'student': ['students', 'learner', 'learners'],
            'enrollment': ['enrolment', 'enrolments', 'enrollments', 'enroll', 'enrol'],
            'accommodation': ['housing', 'residence', 'living', 'accommodation'],
            'university': ['universities', 'college', 'colleges'],
            'graduate': ['graduates', 'graduation'],
            'undergraduate': ['undergraduates'],
            'postgraduate': ['postgraduates'],
            'degree': ['degrees', 'qualification', 'qualifications'],
            'term-time': ['term time', 'termtime'],
            'full-time': ['full time', 'fulltime'],
            'part-time': ['part time', 'parttime']
        }
        
        # Expand keywords with synonyms
        expanded_keywords = set()
        added_synonyms = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Add the original keyword
            expanded_keywords.add(keyword_lower)
            
            # Add synonyms for this keyword if available
            for base_word, syn_list in synonyms.items():
                if keyword_lower == base_word or keyword_lower in syn_list:
                    # Add base word if it's not already the keyword
                    if base_word != keyword_lower and base_word not in expanded_keywords:
                        expanded_keywords.add(base_word)
                        added_synonyms.append(base_word)
                    
                    # Add all synonyms
                    for syn in syn_list:
                        if syn != keyword_lower and syn not in expanded_keywords:
                            expanded_keywords.add(syn)
                            added_synonyms.append(syn)
        
        logger.info(f"Added synonyms: {added_synonyms}")
        logger.info(f"Expanded keywords: {expanded_keywords}")
        
        # Handle years
        years = []
        if start_year:
            try:
                start = int(start_year)
                
                if end_year:
                    end = int(end_year)
                    
                    # Validate years
                    if start > end:
                        return CustomJsonResponse({
                            'status': 'error',
                            'message': 'Start year cannot be greater than end year'
                        }, status=400)
                    
                    # Generate list of years in YYYY/YY format
                    for year in range(start, end + 1):
                        academic_year = f"{year}/{str(year+1)[2:4]}"
                        years.append(academic_year)
                        
                    logger.info(f"Year range: {start}-{end}, expanded to: {years}")
                else:
                    # Single year provided - convert to academic year format
                    academic_year = f"{start}/{str(start+1)[2:4]}"
                    years.append(academic_year)
                    logger.info(f"Single year: {start}, converted to academic year: {academic_year}")
            except ValueError:
                logger.warning(f"Invalid year format: start_year={start_year}, end_year={end_year}")
                return CustomJsonResponse({
                    'status': 'error',
                    'message': 'Invalid year format'
                }, status=400)
        
        # Find relevant CSV files based on the query
        file_matches = find_relevant_csv_files(file_pattern, list(expanded_keywords), years if years else None)
        
        # Group files by title
        file_groups = {}
        for file_match in file_matches:
            title = file_match['title']
            if title not in file_groups:
                file_groups[title] = {
                    'title': title,
                    'files': [],
                    'score': file_match['score'],
                    'percentage': file_match['percentage'],
                    'matched_keywords': file_match['matched_keywords'],
                    'available_years': set(),
                    'missing_years': []
                }
            
            file_groups[title]['files'].append(file_match)
            if file_match['year']:
                file_groups[title]['available_years'].add(file_match['year'])
        
        # Convert to list and sort by score
        grouped_matches = list(file_groups.values())
        for group in grouped_matches:
            group['available_years'] = sorted(list(group['available_years']))
            
            # Check for missing years if years were requested
            if years:
                group['missing_years'] = [year for year in years if year not in group['available_years']]
        
        # Sort by score (descending)
        grouped_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to max_matches
        grouped_matches = grouped_matches[:max_matches]
        
        # Generate preview data
        preview_data = []
        for i, group in enumerate(grouped_matches):
            # Create a unique identifier for this group
            group_id = f"group_{i+1}_{group['score']}"
            
            # Generate file previews
            file_previews = []
            for file_match in group['files']:
                file_path = file_match['file_path']
                
                # Generate preview data for this file
                preview = get_csv_preview(file_path, institution=institution)
                if preview:
                    preview['year'] = file_match['year']
                    preview['file_name'] = file_match['file_name']
                    file_previews.append(preview)
            
            # Create a summary of this group
            group_summary = {
                'group_id': group_id,
                'title': group['title'],
                'score': group['score'],
                'match_percentage': str(group['percentage']),
                'matched_keywords': group['matched_keywords'],
                'available_years': group['available_years'],
                'missing_years': group['missing_years'],
                'file_count': len(group['files']),
                'file_previews': file_previews
            }
            
            preview_data.append(group_summary)
        
        # Return the results
        response_data = {
            'status': 'success',
            'query_info': {
                'original_query': query,
                'file_pattern': file_pattern,
                'removed_words': stopwords,
                'years': years,
                'added_synonyms': added_synonyms
            },
            'preview_data': preview_data
        }
        
        return CustomJsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error processing HESA query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Provide detailed error information for debugging
        error_details = {
            'error_message': str(e),
            'query': query if 'query' in locals() else 'unknown',
            'institution': institution if 'institution' in locals() else 'unknown',
            'years': years if 'years' in locals() else []
        }
        
        # Add traceback for debugging if in DEBUG mode
        if settings.DEBUG:
            tb = traceback.format_exc()
            last_5_lines = '\n'.join(tb.split('\n')[-5:])
            error_details['traceback'] = last_5_lines
        
        error_response = {
            'status': 'error',
            'message': f"Failed to process query: {str(e)}",
            'details': error_details,
            'suggestions': [
                "Check if the requested academic years exist in the dataset",
                "Try broadening your search terms",
                "Try a different institution name"
            ]
        }
        
        return CustomJsonResponse(error_response, status=500)

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

@require_http_methods(["POST"])
def select_file_source(request):
    """Handle selection of a file source (dataset) for visualization."""
    logger = logging.getLogger(__name__)
    logger.info(f"Selecting file source: {request.POST}")
    
    try:
        # Extract parameters
        group_id = request.POST.get('group_id', '')
        query = request.POST.get('query', '')
        institution = request.POST.get('institution', '')
        start_year = request.POST.get('start_year', '')
        end_year = request.POST.get('end_year', '')
        file_id = request.POST.get('file_id', '')  # Optional, for selecting a specific file within a group
        
        logger.info(f"Group ID: {group_id}")
        logger.info(f"Query: {query}")
        logger.info(f"Institution: {institution}")
        logger.info(f"Years: {start_year} - {end_year}")
        logger.info(f"File ID: {file_id}")
        
        if not group_id:
            return JsonResponse({
                'error': 'No dataset group selected',
                'status': 'error'
            }, status=400)
        
        # Parse the query to extract keywords and other parameters
        query_info = parse_hesa_query(query)
        logger.info(f"Parsed query: {query_info}")
        
        # Extract file pattern keywords and expand them
        file_pattern_keywords = query_info.get('file_pattern_keywords', [])
        expanded_keywords = query_info.get('expanded_keywords', [])
        he_providers = query_info.get('he_providers', [])
        years = query_info.get('years', [])
        
        # Add institution to HE providers if specified
        if institution and institution.strip():
            logger.info(f"Adding institution to HE providers: {institution}")
            he_providers.append(institution.strip())
        
        # Handle year range
        if start_year or end_year:
            try:
                start = int(start_year) if start_year else 0
                end = int(end_year) if end_year else 9999
                
                # Generate a list of years in the range as strings
                if start > 0 and end < 9999:
                    years = [str(y) for y in range(start, end + 1)]
                elif start > 0:
                    years = [str(y) for y in range(start, datetime.now().year + 1)]
                elif end < 9999:
                    years = [str(y) for y in range(2000, end + 1)]  # Assuming no data before 2000
                
                logger.info(f"Expanded years range: {years}")
            except ValueError:
                logger.warning(f"Invalid year format: {start_year} - {end_year}")
        
        # Get file pattern for searching
        file_pattern = ' '.join(file_pattern_keywords)
        
        # Find all matching files again
        data_files = find_relevant_csv_files(file_pattern, expanded_keywords, years)
        logger.info(f"Found {len(data_files)} matching files")
        
        if not data_files:
            return JsonResponse({
                'error': 'No matching files found',
                'status': 'error'
            }, status=404)
        
        # Group files by title to find the selected group
        file_groups = {}
        
        for file_match in data_files:
            file_name = file_match['file_name']
            file_path = file_match['file_path']
            file_year = file_match.get('year')
            file_score = file_match.get('score', 0)
            file_title = file_match.get('title', '')
            file_matched_keywords = file_match.get('matched_keywords', [])
            
            # Create a normalized title as the group key - removing the year
            normalized_title = file_title.strip() if file_title else re.sub(r'\s*\d{4}[\&\-_\/](\d{2}|\d{4})', ' YEAR', file_name)
            
            if normalized_title not in file_groups:
                file_groups[normalized_title] = {
                    'title': normalized_title,
                    'files': [],
                    'score': 0,
                    'percentage': 0,
                    'matched_keywords': set(),
                    'available_years': set()
                }
            
            # Add the file to its group
            file_groups[normalized_title]['files'].append(file_match)
            
            # Update the group's score with the highest file score
            if file_score > file_groups[normalized_title]['score']:
                file_groups[normalized_title]['score'] = file_score
                file_groups[normalized_title]['percentage'] = file_match.get('percentage', 0)
            
            # Add matched keywords and year to the group
            file_groups[normalized_title]['matched_keywords'].update(file_matched_keywords)
            if file_year:
                file_groups[normalized_title]['available_years'].add(file_year)
        
        # Find the selected group
        selected_group = None
        for i, (title, group) in enumerate(file_groups.items()):
            curr_group_id = f"group_{i+1}_{group['score']}"
            if curr_group_id == group_id:
                selected_group = group
                break
        
        if not selected_group:
            return JsonResponse({
                'error': 'Selected dataset group not found',
                'status': 'error'
            }, status=404)
        
        logger.info(f"Selected group: {selected_group['title']} with {len(selected_group['files'])} files")
        
        # Prepare data for each file in the group
        files_data = []
        
        for file_info in selected_group['files']:
            file_path = file_info['file_path']
            file_name = file_info['file_name']
            file_year = file_info.get('year', 'Unknown')
            
            logger.info(f"Processing file data for: {file_name} (Year: {file_year})")
            
            # Extract data for the requested institution
            if institution:
                file_data = extract_provider_data(file_path, [institution], use_substring_match=True)
            else:
                # If no institution specified, get all data
                file_data = extract_provider_data(file_path, ["All"], use_substring_match=True)
            
            if file_data:
                files_data.append({
                    'file_name': file_name,
                    'year': file_year,
                    'columns': file_data.get('columns', []),
                    'data': file_data.get('data', [])
                })
            else:
                logger.warning(f"No data extracted for file: {file_name}")
                
        if not files_data:
            return JsonResponse({
                'error': 'No data found for the selected files',
                'status': 'error'
            }, status=404)
        
        logger.info(f"Returning data for {len(files_data)} files")
        
        # Return the full data for all files in the selected group
        return JsonResponse({
            'status': 'success',
            'query_info': query_info,
            'institution': institution,
            'years': years,
            'group_title': selected_group['title'],
            'files_data': files_data,
            'message': f'Loaded data from {len(files_data)} files'
        })
        
    except Exception as e:
        import traceback
        logger.error(f"Error selecting file source: {str(e)}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        return JsonResponse({
            'error': f'Error selecting file source: {str(e)}',
            'status': 'error'
        }, status=500)

def extract_data_from_csv(file_path, institution=None):
    """Extract data from a CSV file, with optional institution filtering."""
    logger = logging.getLogger(__name__)
    
    try:
        # Read the CSV file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip metadata line if present
            first_line = f.readline()
            if first_line.startswith('#METADATA:'):
                next_line = f.readline()
            else:
                next_line = first_line
                
            # Reset file pointer if not a metadata line
            if not first_line.startswith('#METADATA:'):
                f.seek(0)
                
            # Read CSV data
            csv_reader = csv.reader(f)
            
            # Get column headers
            if first_line.startswith('#METADATA:'):
                headers = next_line.strip().split(',')
            else:
                headers = next(csv_reader)
                
            # Prepare data container
            data = []
            total_rows = 0
            matched_rows = 0
            
            # Process rows
            for row in csv_reader:
                total_rows += 1
                
                # If row is shorter than headers, extend it
                if len(row) < len(headers):
                    row.extend([''] * (len(headers) - len(row)))
                
                # Filter by institution if provided
                if institution and len(row) > 0:
                    # The institution column is typically the first column
                    if not row[0].lower().find(institution.lower()) >= 0:
                        continue
                
                matched_rows += 1
                
                # Only add up to max_rows
                if len(data) < max_rows:
                    data.append(row)
            
            # Create result
            result = {
                'columns': headers,
                'data': data,
                'total_rows': total_rows,
                'matched_rows': matched_rows,
                'has_more': matched_rows > len(data)
            }
            
            return result
        
    except Exception as e:
        logger.error(f"Error extracting preview from {file_path}: {str(e)}")
        return {
            'columns': ['Error'],
            'data': [[f"Could not read file: {str(e)}"]],
            'error': str(e)
        }

def parse_hesa_query(query):
    """
    Parse a HESA query to extract relevant information.
    Returns:
        file_pattern: A pattern to match file names
        keywords: A list of keywords from the query
        stopwords: A list of words that were removed from the query
    """
    logger = logging.getLogger(__name__)
    
    # Convert to lowercase
    query = query.lower()
    
    # Define stopwords to remove
    stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                 'where', 'when', 'how', 'of', 'for', 'with', 'by', 'from', 'to', 'in',
                 'on', 'at', 'me', 'you', 'we', 'they', 'he', 'she', 'it', 'this', 'that',
                 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 
                 'would', 'should', 'shall', 'might', 'may', 'about', 'find', 'show', 'get',
                 'all', 'any', 'each']
    
    # Extract file pattern (if any)
    file_pattern = ""
    if "file:" in query:
        pattern_match = re.search(r"file:\s*([^\s,]+)", query)
        if pattern_match:
            file_pattern = pattern_match.group(1)
            query = query.replace(pattern_match.group(0), "")
    
    # Find and preserve hyphenated terms like "term-time" and "full-time"
    hyphenated_terms = re.findall(r'\b\w+(?:-\w+)+\b', query)
    
    # Temporarily replace hyphenated terms with placeholders
    hyphen_placeholders = {}
    for i, term in enumerate(hyphenated_terms):
        placeholder = f"HYPHEN_TERM_{i}"
        hyphen_placeholders[placeholder] = term
        query = query.replace(term, placeholder)
    
    # Split query into words
    words = re.findall(r'\b\w+\b', query)
    
    # Restore hyphenated terms
    for i, word in enumerate(words):
        if word in hyphen_placeholders:
            words[i] = hyphen_placeholders[word]
    
    # Filter out stopwords and short words
    removed_words = []
    keywords = []
    for word in words:
        # Skip short words (except if they're years) and numeric-only terms
        if (len(word) <= 2 and not re.match(r'^\d{2,4}$', word)) or word.isdigit():
            removed_words.append(word)
            continue
            
        # Skip stopwords
        if word in stopwords:
            removed_words.append(word)
            continue
            
        keywords.append(word)
    
    logger.info(f"Parsed query into keywords: {keywords}")
    logger.info(f"Removed stopwords: {removed_words}")
    
    return file_pattern, keywords, removed_words

def extract_provider_data(file_path, he_providers, use_substring_match=False):
    """
    Extract the full dataset for the specified HE providers from the CSV file.
    
    Parameters:
    - file_path: Path to the CSV file
    - he_providers: List or comma-separated string of HE provider names to filter by
    - use_substring_match: Use substring matching for institution names instead of exact matches
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Skip the metadata line
        # Read the CSV file with pandas
        df = pd.read_csv(file_path, skiprows=1)
        
        # Check if we have any data
        if df.empty:
            logger.warning(f"No data found in file: {file_path}")
            return None
            
        # Identify provider column
        provider_column = None
        for col in df.columns:
            if 'provider' in col.lower() or 'institution' in col.lower():
                provider_column = col
                break
        
        if not provider_column:
            logger.warning(f"No provider column found in file: {file_path}")
            return None
        
        # Process provider list from string if needed
        if isinstance(he_providers, str) and ',' in he_providers:
            he_providers = [p.strip() for p in he_providers.split(',') if p.strip()]
            logger.info(f"Split provider string into list: {he_providers}")
        
        # Filter by HE providers
        if he_providers and he_providers[0] != "All":
            if use_substring_match:
                # Use substring matching - check if any of the providers contains the search term
                mask = df[provider_column].apply(
                    lambda x: any(provider.lower() in str(x).lower() 
                                  for provider in he_providers)
                )
                filtered_df = df[mask]
            else:
                # Use exact matching (case-insensitive)
                filtered_df = df[df[provider_column].str.lower().isin([p.lower() for p in he_providers])]
            
            if filtered_df.empty:
                logger.warning(f"No matching providers found in file: {file_path}")
                # Try more flexible matching if exact matching fails
                if not use_substring_match:
                    logger.info("Trying substring matching for providers")
                    mask = df[provider_column].apply(
                        lambda x: any(provider.lower() in str(x).lower() 
                                      for provider in he_providers)
                    )
                    filtered_df = df[mask]
                    if filtered_df.empty:
                        logger.warning("No matches found even with substring matching")
                        return None
        else:
            filtered_df = df
            
        # If no matches, return None
        if filtered_df.empty:
            logger.warning(f"No matching data after filtering in file: {file_path}")
            return None
                    
        # Convert to list format
        columns = filtered_df.columns.tolist()
        data = filtered_df.values.tolist()
        
        return {
            'columns': columns,
            'data': data
        }
        
    except Exception as e:
        logger.error(f"Error extracting provider data from {file_path}: {str(e)}")
        return None

def extract_provider_data_preview(file_path, he_providers, max_rows=10, requested_years=None, use_substring_match=False):
    """
    Extract a preview of data for the specified HE providers from the CSV file.
    Returns a preview with just the top few rows for display, along with total row count.
    
    Parameters:
    - file_path: Path to the CSV file
    - he_providers: List or comma-separated string of HE provider names to filter by
    - max_rows: Maximum number of rows to include in the preview (default 10)
    - requested_years: Optional list of years to filter by
    - use_substring_match: Use substring matching for institution names instead of exact matches
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Skip the metadata line
        # Read the CSV file with pandas
        df = pd.read_csv(file_path, skiprows=1)
        
        # Check if we have any data
        if df.empty:
            logger.warning(f"No data found in file: {file_path}")
            return None
        
        # Identify provider column
        provider_column = None
        for col in df.columns:
            if 'provider' in col.lower() or 'institution' in col.lower():
                provider_column = col
                break
        
        if not provider_column:
            logger.warning(f"No provider column found in file: {file_path}")
            return None
        
        # Process provider list from string if needed
        if isinstance(he_providers, str) and ',' in he_providers:
            he_providers = [p.strip() for p in he_providers.split(',') if p.strip()]
            logger.info(f"Split provider string into list: {he_providers}")
        
        # Filter by HE providers
        if he_providers and he_providers[0] != "All":
            if use_substring_match:
                # Use substring matching - check if any of the providers contains the search term
                mask = df[provider_column].apply(
                    lambda x: any(provider.lower() in str(x).lower() 
                                  for provider in he_providers)
                )
                filtered_df = df[mask]
            else:
                # Use exact matching (case-insensitive)
                filtered_df = df[df[provider_column].str.lower().isin([p.lower() for p in he_providers])]
                
            if filtered_df.empty:
                logger.warning(f"No matching providers found in file: {file_path}")
                # Try more flexible matching if exact matching fails
                if not use_substring_match:
                    logger.info("Trying substring matching for providers")
                    mask = df[provider_column].apply(
                        lambda x: any(provider.lower() in str(x).lower() 
                                      for provider in he_providers)
                    )
                    filtered_df = df[mask]
                    if filtered_df.empty:
                        logger.warning("No matches found even with substring matching")
                        return None
        else:
            filtered_df = df
            
        # Filter by years if requested and if we have a year column
        if requested_years and year_column:
            # Convert requested years to lowercase for comparison
            requested_years_lower = [y.lower() for y in requested_years]
            # Filter rows where the year column contains any of the requested years
            year_mask = filtered_df[year_column].apply(
                lambda x: any(year in str(x).lower() for year in requested_years_lower)
            )
            filtered_df = filtered_df[year_mask]
            
        # If still no matches, return None
        if filtered_df.empty:
            logger.warning(f"No matching data after filtering in file: {file_path}")
            return None
        
        # Get the total number of rows before limiting
        total_row_count = len(filtered_df)
        
        # Limit to top rows for preview
        preview_data = filtered_df.head(max_rows)
        
        # Convert to list format
        columns = preview_data.columns.tolist()
        data = preview_data.values.tolist()
        
        return {
            'columns': columns,
            'data': data,
            'total_row_count': total_row_count  # Include the total row count
        }
        
    except Exception as e:
        logger.error(f"Error extracting provider data preview from {file_path}: {str(e)}")
        return None

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

def get_csv_preview(file_path, max_rows=5, institution=None):
    """
    Extract a preview of CSV data, optionally filtered by institution.
    
    Args:
        file_path: Path to the CSV file
        max_rows: Maximum number of rows to return
        institution: Optional institution name(s) to filter by (comma-separated list for multiple)
        
    Returns:
        Dictionary with columns and preview data
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Read the CSV file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip metadata line if present
            first_line = f.readline()
            if first_line.startswith('#METADATA:'):
                next_line = f.readline()
            else:
                next_line = first_line
                
            # Reset file pointer if not a metadata line
            if not first_line.startswith('#METADATA:'):
                f.seek(0)
                
            # Read CSV data
            csv_reader = csv.reader(f)
            
            # Get column headers
            if first_line.startswith('#METADATA:'):
                headers = next_line.strip().split(',')
            else:
                headers = next(csv_reader)
                
            # Prepare data container
            data = []
            total_rows = 0
            matched_rows = 0
            
            # Split institution names if provided
            institutions = []
            if institution:
                institutions = [inst.strip().lower() for inst in institution.split(',') if inst.strip()]
                logger.info(f"Filtering by institutions: {institutions}")
            
            # Process rows
            for row in csv_reader:
                total_rows += 1
                
                # If row is shorter than headers, extend it
                if len(row) < len(headers):
                    row.extend([''] * (len(headers) - len(row)))
                
                # Filter by institution if provided
                if institutions and len(row) > 0:
                    # Check if any of the provided institutions match
                    institution_matches = False
                    for inst in institutions:
                        if row[0].lower().find(inst) >= 0:
                            institution_matches = True
                            break
                    
                    if not institution_matches:
                        continue
                
                matched_rows += 1
                
                # Only add up to max_rows
                if len(data) < max_rows:
                    data.append(row)
            
            # Create result
            result = {
                'columns': headers,
                'data': data,
                'total_rows': total_rows,
                'matched_rows': matched_rows,
                'has_more': matched_rows > len(data)
            }
            
            return result
        
    except Exception as e:
        logger.error(f"Error extracting preview from {file_path}: {str(e)}")
        return {
            'columns': ['Error'],
            'data': [[f"Could not read file: {str(e)}"]],
            'error': str(e)
        }

def find_relevant_csv_files(file_pattern, keywords, requested_years=None):
    """
    Find CSV files that match the given pattern and keywords.
    Returns a list of relevant files with scores based on keyword matches.
    Uses Density-Based Weighting to prioritize files with focused matches.
    """
    logger = logging.getLogger(__name__)
    
    # List all CSV files in the directory
    all_csv_files = list(CLEANED_FILES_DIR.glob('*.csv'))
    logger.info(f"Found {len(all_csv_files)} CSV files in directory")
    
    # Log all CSV files for debugging when there's a year filter issue
    if requested_years:
        logger.info(f"Filtering files for years: {requested_years}")
        sample_files = [f.name for f in all_csv_files[:5]]
        logger.info(f"Sample files in directory: {sample_files}")
        
        # Log files containing year strings in their names for debugging
        year_pattern_files = []
        for file in all_csv_files:
            for year_str in requested_years:
                year_simple = year_str.split('/')[0]  # Get the first part (e.g., "2015" from "2015/16")
                if year_simple in file.name:
                    year_pattern_files.append(file.name)
                    break
        
        logger.info(f"Files with requested years in filename: {year_pattern_files[:10]}")
    else:
        logger.info("No year filtering applied, considering all files")
    
    # Define synonyms dictionary for keyword expansion
    synonyms = {
        'student': ['students', 'learner', 'learners'],
        'enrollment': ['enrolment', 'enrolments', 'enrollments', 'enroll', 'enrol'],
        'accommodation': ['housing', 'residence', 'living', 'accommodation'],
        'university': ['universities', 'college', 'colleges'],
        'graduate': ['graduates', 'graduation'],
        'undergraduate': ['undergraduates'],
        'postgraduate': ['postgraduates'],
        'degree': ['degrees', 'qualification', 'qualifications'],
        'term-time': ['term time', 'termtime'],
        'full-time': ['full time', 'fulltime'],
        'part-time': ['part time', 'parttime']
    }
    
    # Expand keywords with synonyms
    expanded_keywords = set()
    for keyword in keywords:
        keyword_lower = keyword.lower()
        # Add the original keyword
        expanded_keywords.add(keyword_lower)
        
        # Add synonyms for this keyword if available
        for base_word, syn_list in synonyms.items():
            if keyword_lower == base_word or keyword_lower in syn_list:
                expanded_keywords.add(base_word)
                expanded_keywords.update(syn_list)
    
    logger.info(f"Original keywords: {keywords}")
    logger.info(f"Expanded keywords with synonyms: {expanded_keywords}")
    
    # Create a list to store matching files and their scores
    file_matches = []
    
    # Track academic years found in files for better error messages
    academic_years_found = set()
    
    # Process each CSV file
    for file_path in all_csv_files:
        try:
            file_name = file_path.name
            logger.debug(f"Checking file: {file_name}")
            
            # Try to extract metadata from the first line
            metadata = {}
            title = ""
            academic_year = ""
            keywords_title = []
            keywords_columns = []
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                
                # Check if the line starts with #METADATA:
                if first_line.startswith('#METADATA:'):
                    try:
                        # Extract the JSON part
                        metadata_str = first_line[len('#METADATA:'):]
                        metadata = json.loads(metadata_str)
                        
                        # Extract relevant fields
                        title = metadata.get('title', '')
                        academic_year = metadata.get('academic_year', '')
                        keywords_title = metadata.get('keywords_title', [])
                        keywords_columns = metadata.get('keywords_columns', [])
                        
                        # Track all academic years found
                        if academic_year:
                            academic_years_found.add(academic_year)
                        
                        logger.debug(f"Extracted metadata from {file_name}:")
                        logger.debug(f"  Title: [{title}]")
                        logger.debug(f"  Academic Year: [{academic_year}]")
                        logger.debug(f"  Title keywords: {keywords_title}")
                        logger.debug(f"  Column keywords: {keywords_columns}")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing metadata JSON in {file_name}: {str(e)}")
                        # Continue to fallback method
                    
                # If metadata parsing failed or no structured metadata found, 
                # try to read the file line by line to find metadata
                if not title or not academic_year:
                    # Reopen the file and try to extract metadata from unstructured content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line.strip() for line in f.readlines()[:20]]
                        
                    # Look for title
                    for line in lines:
                        if 'Title:' in line:
                            # Extract text after 'Title:'
                            title_match = re.search(r'Title:,\s*\"?(.*?)\"?$', line)
                            if title_match:
                                title = title_match.group(1).strip('\"')
                                # If title contains "Table XX - ", extract only what comes after
                                table_match = re.search(r'Table \d+ - (.*)', title)
                                if table_match:
                                    title = table_match.group(1).strip()
                        
                        # Look for academic year
                        year_match = re.search(r',\s*Academic year\s*,\s*(20\d{2}/\d{2})', line)
                        if year_match:
                            academic_year = year_match.group(1).strip()
                            academic_years_found.add(academic_year)
                            logger.debug(f"Found academic year: [{academic_year}]")
            
            # Skip this file if the year doesn't match requested years
            if requested_years and (not academic_year or academic_year not in requested_years):
                logger.debug(f"Skipping {file_name} as year {academic_year} does not match requested years: {requested_years}")
                continue
            
            # If no keywords extracted from metadata, derive them from the title
            if not keywords_title and title:
                title_lower = title.lower()
                # Extract words from title, including hyphenated terms
                title_keywords = re.findall(r'\b\w+(?:-\w+)*\b', title_lower)
                # Filter out short words
                keywords_title = [w for w in title_keywords if len(w) > 2]
                logger.debug(f"Derived title keywords from title: {keywords_title}")
            
            # Track which specific keywords were matched
            matched_keywords = set()
            
            # Count matches in title and title keywords
            title_matches = 0
            title_lower = title.lower()
            
            # First check for matches in the full title
            for query_keyword in expanded_keywords:
                query_keyword_lower = query_keyword.lower()
                # Check if keyword is in the complete title
                if query_keyword_lower in title_lower:
                    title_matches += 1
                    matched_keywords.add(query_keyword)
                    logger.debug(f"  Matched '{query_keyword}' in title: {title}")
            
            # Then check for matches in individual title keywords
            for query_keyword in expanded_keywords:
                query_keyword_lower = query_keyword.lower()
                found_in_title_keywords = False
                
                for title_keyword in keywords_title:
                    title_keyword_lower = title_keyword.lower()
                    # Check for substring match in either direction
                    if (query_keyword_lower in title_keyword_lower or 
                        title_keyword_lower in query_keyword_lower):
                        # Only count this match if we didn't already count it in the full title
                        if query_keyword_lower not in matched_keywords:
                            title_matches += 1
                            matched_keywords.add(query_keyword)
                            found_in_title_keywords = True
                            logger.debug(f"  Matched '{query_keyword}' with title keyword '{title_keyword}'")
                            break
                
                if found_in_title_keywords:
                    continue
            
            # Count matches in column keywords
            column_matches = 0
            for query_keyword in expanded_keywords:
                query_keyword_lower = query_keyword.lower()
                # Skip if already matched in title
                if query_keyword_lower in matched_keywords:
                    continue
                    
                for column_keyword in keywords_columns:
                    column_keyword_lower = column_keyword.lower()
                    # Check for substring match in either direction
                    if (query_keyword_lower in column_keyword_lower or 
                        column_keyword_lower in query_keyword_lower):
                        column_matches += 1
                        matched_keywords.add(query_keyword)
                        logger.debug(f"  Matched '{query_keyword}' with column keyword '{column_keyword}'")
                        break
            
            # Calculate density scores (with protection against division by zero)
            title_keywords_count = max(len(keywords_title), 1)
            column_keywords_count = max(len(keywords_columns), 1)
            
            title_density = title_matches / title_keywords_count
            column_density = column_matches / column_keywords_count
            
            # Apply density-based weighting with scaling factors
            title_score = 2 * title_density * 10  # Title matches are twice as important
            column_score = 1 * column_density * 10
            
            # Total score is the sum of all components
            total_score = title_score + column_score
            
            # Calculate a percentage match based on the density
            max_possible_score = 2 * 10 + 1 * 10  # Maximum possible score with perfect density
            percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
            
            # Add all files with year matches to results
            # If we have keyword matches, include actual matched keywords
            # If no keyword matches but year matches, show "year match only"
            match_info = {
                'file_path': str(file_path),
                'file_name': file_name,
                'title': title,
                'year': academic_year,
                'score': total_score,
                'percentage': round(percentage, 2),
                'matched_keywords': list(matched_keywords) if matched_keywords else ['year match only'],
                'title_matches': title_matches,
                'title_keywords_total': len(keywords_title),
                'title_density': round(title_density, 3) if title_keywords_count > 0 else 0,
                'title_score': round(title_score, 2),
                'column_matches': column_matches,
                'column_keywords_total': len(keywords_columns),
                'column_density': round(column_density, 3) if column_keywords_count > 0 else 0,
                'column_score': round(column_score, 2)
            }
            file_matches.append(match_info)
            
            if matched_keywords:
                logger.info(f"Keyword match found for {file_name} with score {total_score:.2f} ({percentage:.2f}%)")
                logger.info(f"  Title density: {title_density:.3f} ({title_matches}/{len(keywords_title)}) → {title_score:.2f} points")
                logger.info(f"  Column density: {column_density:.3f} ({column_matches}/{len(keywords_columns)}) → {column_score:.2f} points")
                logger.info(f"  Matched keywords: {matched_keywords}")
            else:
                logger.info(f"Year match only for {file_name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Sort matches by score (highest first)
    file_matches.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Found {len(file_matches)} matches in total")
    
    # Log helpful information if no matches but years were specified
    if not file_matches and requested_years:
        logger.warning(f"No matches found with requested years: {requested_years}")
        logger.warning(f"Academic years found in files: {academic_years_found}")
        
        # Check if requested years exist in the available years
        for year in requested_years:
            if year in academic_years_found:
                logger.warning(f"Year {year} exists but no matching keywords were found")
            else:
                logger.warning(f"Year {year} does not exist in any of the files")
    
    return file_matches