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
from django.views.decorators.csrf import csrf_exempt
import re
import json
import glob
from .gemini_client import GeminiClient
from .mock_ai_client import MockAIClient
import fnmatch
from collections import defaultdict

# Mission groups for institution filtering
MISSION_GROUPS = {
    "Russell Group": [
        "University of Birmingham", "University of Bristol", "University of Cambridge",
        "Cardiff University", "Durham University", "University of Edinburgh",
        "University of Exeter", "University of Glasgow", "Imperial College London",
        "King's College London", "University of Leeds", "University of Liverpool",
        "London School of Economics and Political Science", "University of Manchester",
        "Newcastle University", "University of Nottingham", "University of Oxford",
        "Queen Mary University of London", "Queen's University Belfast",
        "University of Sheffield", "University of Southampton", "University College London",
        "University of Warwick", "University of York"
    ],
    "Million+": [
        "University of Bolton", "University of Central Lancashire", "Coventry University",
        "De Montfort University", "University of Derby", "University of East London",
        "University of Greenwich", "University of Hertfordshire", "University of Lincoln",
        "Liverpool John Moores University", "Manchester Metropolitan University",
        "Middlesex University", "University of Northampton", "University of South Wales",
        "University of West London", "University of the West of England, Bristol",
        "University of Wolverhampton"
    ],
    "University Alliance": [
        "Anglia Ruskin University", "Birmingham City University", "Coventry University",
        "De Montfort University", "University of Derby", "University of East London",
        "University of Hertfordshire", "University of Lincoln", "Liverpool John Moores University",
        "Manchester Metropolitan University", "Middlesex University", "University of Northampton",
        "University of South Wales", "University of West London", "University of the West of England, Bristol",
        "University of Worcester"
    ]
}

# University of Leicester - always included
LEICESTER = "University of Leicester"

# Function to normalize institution names for better matching
def normalize_institution_name(name):
    """
    Normalize institution names by converting to lowercase and removing common prefixes.
    """
    name = name.lower().strip()
    if name.startswith('the '):
        name = name[4:].strip()
    return name

# Function to check if institution matches any target name
def match_institution(institution_name, target_names):
    """
    Check if institution_name matches any of the target names using substring matching.
    Returns True if there's a match, False otherwise.
    """
    if not institution_name or not target_names:
        return False
    
    # Convert institution name to lowercase for case-insensitive matching
    normalized_institution = institution_name.lower().strip()
    # Remove 'the ' from the beginning if present
    if normalized_institution.startswith('the '):
        normalized_institution = normalized_institution[4:].strip()
    
    for target in target_names:
        # Skip empty targets
        if not target:
            continue
            
        # Convert target to lowercase for case-insensitive matching
        normalized_target = target.lower().strip()
        if normalized_target.startswith('the '):
            normalized_target = normalized_target[4:].strip()
        
        # Check for substring match in either direction
        if normalized_institution in normalized_target or normalized_target in normalized_institution:
            return True
    
    return False

# Configuration constants
MAX_PREVIEW_ROWS = getattr(settings, 'DASHBOARD_SETTINGS', {}).get('MAX_PREVIEW_ROWS', 3)  # Get from settings with fallback

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

def ai_dashboard(request):
    """Render the AI-powered dashboard page."""
    import random
    # Add a random number for cache busting
    random_cache_buster = random.randint(10000, 99999)
    return render(request, 'ai_dashboard.html', {'random_cache_buster': random_cache_buster})

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
            mission_group = request.GET.get('mission_group', '')
        else:  # POST
            query = request.POST.get('query', '')
            chart_type = request.POST.get('chart_type', 'bar')
            max_matches = int(request.POST.get('max_matches', 3))
            institution = request.POST.get('institution', '')
            start_year = request.POST.get('start_year', '')
            end_year = request.POST.get('end_year', '')
            mission_group = request.POST.get('mission_group', '')
        
        logger.info(f"Processing HESA query: '{query}', chart_type: {chart_type}, max_matches: {max_matches}")
        logger.info(f"Institution: '{institution}', start_year: {start_year}, end_year: {end_year}")
        logger.info(f"CLEANED_FILES_DIR: {CLEANED_FILES_DIR}")
        logger.info(f"Path exists: {CLEANED_FILES_DIR.exists()}")
        
        if mission_group:
            logger.info(f"Mission group: {mission_group}")
        
        # Process the query to extract relevant information
        query_info = parse_hesa_query(query)
        
        logger.info(f"Query parsed: {query_info}")
        
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
        
        for keyword in query_info['keywords']:
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
        
        # If no CSV files are in the directory, return a clear error
        all_csv_files = list(CLEANED_FILES_DIR.glob('*.csv'))
        if not all_csv_files:
            logger.error(f"No CSV files found in CLEANED_FILES_DIR: {CLEANED_FILES_DIR}")
            return CustomJsonResponse({
                'status': 'error',
                'message': f'No CSV files found in data directory. Please ensure the data directory is properly set up at {CLEANED_FILES_DIR}'
            }, status=404)
        
        # Find relevant CSV files based on the query
        file_matches = find_relevant_csv_files(query_info['file_pattern'], list(expanded_keywords), years if years else None)
        
        # If no matches found, return appropriate message
        if not file_matches:
            return CustomJsonResponse({
                'status': 'no_matches',
                'message': 'No matching datasets found for your query. Try different keywords or broaden your search.',
                'query_info': query_info
            }, status=200)
        
        # Limit to max_matches
        file_matches = file_matches[:max_matches]
        
        # Process the grouped matches to create preview data
        preview_data = []
        for i, group in enumerate(file_matches):
            # Create a unique identifier for this group
            group_id = f"group_{i+1}_{group['score']}"
            
            # Generate file previews
            file_previews = []
            for file_match in group['files']:
                file_path = file_match['file_path']
                
                # Get the filename from the path
                file_name = os.path.basename(file_path)
                
                # Generate preview data for this file, filtered by mission group if provided
                preview = get_csv_preview(
                    file_path, 
                    institution=institution,
                    mission_group=mission_group
                )
                if preview:
                    preview['year'] = file_match.get('year', 'Unknown')
                    preview['file_name'] = file_name
                    file_previews.append(preview)
            
            # Create a summary of this group
            group_summary = {
                'group_id': group_id,
                'title': group['title'],
                'score': float(group['score']),
                'match_percentage': group.get('match_percentage', group.get('percentage', 0)),
                'matched_keywords': group['matched_keywords'],
                'available_years': group.get('available_years', group.get('years', [])),
                'missing_years': [year for year in years if year not in group.get('available_years', group.get('years', []))] if years else [],
                'file_count': len(group['files']),
                'file_previews': file_previews
            }
            
            preview_data.append(group_summary)
        
        # Return the results
        response_data = {
            'status': 'success',
            'query_info': query_info,
            'max_preview_rows': MAX_PREVIEW_ROWS,
            'preview_data': preview_data,
            'mission_group': mission_group  # Include mission group in response
        }
        
        return CustomJsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error processing HESA query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Check if the error might be related to the directory
        try:
            if not CLEANED_FILES_DIR.exists():
                error_message = f"Data directory not found: {CLEANED_FILES_DIR}"
            else:
                error_message = f"Error processing query: {str(e)}"
        except:
            error_message = f"Error processing query: {str(e)}"
        
        return CustomJsonResponse({
            'status': 'error',
            'message': error_message,
            'suggestions': [
                "Check if your data files are in the correct location",
                "Try broadening your search terms",
                "Try a different institution name"
            ]
        }, status=500)

@csrf_exempt
def select_file_source(request):
    """
    View to handle selection of a file source for visualization
    Shows all rows from all files with the same title
    """
    logger = logging.getLogger(__name__)
    
    try:
        if request.method == 'POST':
            # Parse parameters from the request
            data = json.loads(request.body.decode('utf-8'))
            query = data.get('query', '')
            institution = data.get('institution', '')
            mission_group = data.get('missionGroup', '')
            start_year = data.get('startYear', '')
            end_year = data.get('endYear', '')
            group_id = data.get('fileId', '')  # This is the group_id from the client
            
            logger.info(f"Selecting file source: {group_id}")
            logger.info(f"Query: {query}")
            logger.info(f"Institution: {institution}")
            logger.info(f"Mission Group: {mission_group}")
            logger.info(f"Years: {start_year} - {end_year}")
            logger.info(f"CLEANED_FILES_DIR: {CLEANED_FILES_DIR}")

            # Parse the query to extract keywords
            query_info = parse_hesa_query(query)
            keywords = query_info.get('keywords', [])
            file_pattern = query_info.get('file_pattern', '')
            removed_words = query_info.get('removed_words', [])
            logger.info(f"Parsed query into keywords: {keywords}")
            logger.info(f"Removed stopwords: {removed_words}")
            
            # Find all CSV files
            all_files = list(CLEANED_FILES_DIR.glob('*.csv'))
            logger.info(f"Found {len(all_files)} potential CSV files")
            
            # Get years to search if provided
            years_to_search = None
            if start_year:
                if end_year:
                    years_to_search = []
                    try:
                        for year in range(int(start_year), int(end_year) + 1):
                            academic_year = f"{year}/{str(year+1)[2:4]}"
                            years_to_search.append(academic_year)
                    except ValueError:
                        logger.warning(f"Invalid year format: {start_year}-{end_year}")
                else:
                    academic_year = f"{start_year}/{str(int(start_year) + 1)[2:4]}"
                    years_to_search = [academic_year]
            
            # Extract parts from the group_id (format varies: group_N or group_N_score)
            parts = group_id.split('_')
            group_number = None
            score = None
            
            if len(parts) >= 2:
                try:
                    group_number = int(parts[1])  # Extract the group number
                    if len(parts) >= 3:
                        score = parts[2]  # Extract the score if available
                    logger.info(f"Looking for files in group {group_number} with score {score}")
                except ValueError:
                    logger.warning(f"Could not parse group number from {group_id}")
            
            if group_number is None:
                logger.warning(f"Invalid group ID format: {group_id}")
                return JsonResponse({
                    'error': 'Invalid group ID format',
                    'success': False
                }, status=400)
            
            # Try to find the matching group based on group number
            csv_matches = find_relevant_csv_files(file_pattern, keywords, years_to_search)
            
            target_title = None
            csv_files = []
            
            # Find the group that matches our group number
            if csv_matches and 0 < group_number <= len(csv_matches):
                matching_group = csv_matches[group_number - 1]  # Convert to 0-based index
                target_title = matching_group['title']
                logger.info(f"Found matching group {group_number} with title: {target_title}")
                            
                            # Directly use the file paths from the matching group
                csv_files = [(file_match['file_path'], file_match.get('academic_year', file_match.get('year', 'Unknown'))) 
                                         for file_match in matching_group['files']]
            
            # If we still don't have a target title, try to find by title from metadata
            if not csv_files and not target_title:
                for file_path in all_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            first_line = f.readline().strip()
                            if first_line.startswith('#METADATA:'):
                                metadata_str = first_line[len('#METADATA:'):]
                                try:
                                    metadata = json.loads(metadata_str)
                                    metadata_group_id = metadata.get('group_id', '')
                                    
                                    # Check various ways the group might be identified
                                    if (group_id == metadata_group_id or 
                                        f"group_{group_number}" in metadata_group_id):
                                        target_title = metadata.get('title', '')
                                        logger.info(f"Found target title by group number in metadata: {target_title}")
                                        break
                                except json.JSONDecodeError:
                                    logger.warning(f"Invalid metadata JSON in {file_path}")
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {str(e)}")
            
            # Direct approach - try some common defaults based on group number
            if not target_title:
                # These are common dataset titles that we can fall back to
                default_titles = {
                    1: "HE student enrolments by HE provider",
                    2: "UK permanent address HE students by HE provider and permanent address",
                    3: "HE qualifiers by HE provider and level of qualification obtained"
                }
                
                if group_number in default_titles:
                    target_title = default_titles[group_number]
                    logger.info(f"Using default title for group {group_number}: {target_title}")
                else:
                    # Last attempt - use any file with a title containing keywords
                    if keywords:
                        for file_path in all_files:
                            try:
                                file_name = os.path.basename(str(file_path))
                                # Check if any keyword is in the filename
                                if any(keyword.lower() in file_name.lower() for keyword in keywords):
                                    # Construct a title from the filename
                                    target_title = file_name.replace('.csv', '').replace('_', ' ').title()
                                    logger.info(f"Using filename-derived title as fallback: {target_title}")
                                    break
                            except Exception:
                                continue
            
            if not target_title:
                logger.warning(f"No matching title found for group ID: {group_id}")
                return JsonResponse({
                    'error': f'No files found for the selected dataset',
                    'success': False
                }, status=404)
            
            # Find all files with matching title
            if not csv_files:
                for file_path in all_files:
                    file_name = os.path.basename(str(file_path))
                    
                    # First try using metadata if available
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            first_line = f.readline().strip()
                            if first_line.startswith('#METADATA:'):
                                metadata_str = first_line[len('#METADATA:'):]
                                try:
                                    metadata = json.loads(metadata_str)
                                    file_title = metadata.get('title', '')
                                    academic_year = metadata.get('academic_year', '')
                                    
                                        # Case-insensitive substring matching in both directions
                                    if (file_title.lower() == target_title.lower() or 
                                        target_title.lower() in file_title.lower() or 
                                        file_title.lower() in target_title.lower()):
                                        csv_files.append((str(file_path), academic_year or 'Unknown'))
                                except json.JSONDecodeError:
                                    # Fall back to using filename
                                    derived_title = file_name.replace('.csv', '').replace('_', ' ').title()
                                    # Extract year from filename if possible
                                    year_match = re.search(r'(20\d{2})', file_name)
                                    if year_match:
                                        year = year_match.group(1)
                                        academic_year = f"{year}/{str(int(year)+1)[2:4]}"
                                    else:
                                        academic_year = 'Unknown'
                                        
                                    # Check if derived title matches target
                                    if (derived_title.lower() == target_title.lower() or 
                                        target_title.lower() in derived_title.lower() or 
                                        derived_title.lower() in target_title.lower()):
                                        csv_files.append((str(file_path), academic_year))
                    except Exception as e:
                        logger.warning(f"Error reading metadata from {file_path}: {str(e)}")
                        # Fall back to filename matching
                        derived_title = file_name.replace('.csv', '').replace('_', ' ').title()
                        # Extract year from filename if possible
                        year_match = re.search(r'(20\d{2})', file_name)
                        academic_year = f"{year_match.group(1)}/{str(int(year_match.group(1))+1)[2:]}" if year_match else 'Unknown'
                        
                        # Check if derived title matches target
                        if (derived_title.lower() == target_title.lower() or 
                            target_title.lower() in derived_title.lower() or 
                            derived_title.lower() in target_title.lower()):
                            csv_files.append((str(file_path), academic_year))
            
            if not csv_files:
                logger.warning(f"No CSV files found for title: {target_title}")
                return JsonResponse({
                    'error': f'No files found for the selected dataset',
                    'success': False
                }, status=404)
            
            # Filter files by year if requested
            if start_year:
                academic_year = f"{start_year}/{str(int(start_year) + 1)[-2:]}"
                logger.info(f"Filtering files by academic year: {academic_year}")
                year_filtered_files = [(path, year) for path, year in csv_files if academic_year.lower() in year.lower()]
                if year_filtered_files:
                    csv_files = year_filtered_files
                    logger.info(f"Found {len(csv_files)} files for requested academic year")
            
            # Sort files by academic year
            csv_files.sort(key=lambda x: x[1])
            logger.info(f"Found {len(csv_files)} files with title similar to '{target_title}'")
            
            # Process each file to extract data - no row limit to ensure all data is shown
            all_columns = []
            all_data = []
            all_years = []
            total_rows = 0
            
            for csv_path, year in csv_files:
                logger.info(f"Processing file: {csv_path} for year {year}")
                
                # Use get_csv_preview with mission group support
                if mission_group:
                    # Use mission group filtering
                    data = get_csv_preview(
                        str(csv_path),
                        max_rows=None,  # No limit - display all rows
                        mission_group=mission_group
                    )
                else:
                    # Use institution filtering
                    data = get_csv_preview(
                        str(csv_path),
                        max_rows=None,  # No limit - display all rows
                        institution=institution
                    )
                
                if data and 'columns' in data and 'data' in data:
                    if not all_columns:
                        all_columns = data['columns']
                    
                    # Add this file's data to the combined dataset
                    all_data.extend(data['data'])
                    # Add year to each row
                    all_years.extend([year] * len(data['data']))
                    total_rows += data.get('matched_rows', 0)
                    
                    logger.info(f"Added {len(data['data'])} rows from {csv_path}")
                else:
                    logger.warning(f"No data returned for {csv_path}")
            
            logger.info(f"Total combined rows: {len(all_data)} from {len(csv_files)} files")
            
            # Return the combined data to the client
            return JsonResponse({
                'success': True,
                'columns': all_columns,
                'data': all_data,
                'years': all_years,
                'total_row_count': total_rows,
                'file_count': len(csv_files),
                'dataset_title': target_title,
                'mission_group': mission_group
            })
        
        return JsonResponse({
            'error': 'Invalid request method',
            'success': False
        }, status=400)
        
    except Exception as e:
        logger.error(f"Error selecting file source: {str(e)}")
        import traceback
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        return JsonResponse({
            'error': f'Error selecting file source: {str(e)}',
            'success': False
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
        A dictionary containing:
        - file_pattern: A pattern to match file names
        - keywords: A list of keywords from the query
        - removed_words: A list of words that were removed from the query
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
    
    # Return a dictionary instead of a tuple
    return {
        'file_pattern': file_pattern,
        'keywords': keywords,
        'removed_words': removed_words,
        'file_pattern_keywords': [], # Empty by default
        'expanded_keywords': keywords, # Same as keywords by default
        'he_providers': [], # Empty by default
        'years': [] # Empty by default
    }

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
                    
        # Convert to list format - return ALL rows, not just a preview
        columns = filtered_df.columns.tolist()
        data = filtered_df.values.tolist()
        
        return {
            'columns': columns,
            'data': data,
            'total_rows': len(data)
        }
        
    except Exception as e:
        logger.error(f"Error extracting provider data from {file_path}: {str(e)}")
        return None

def extract_provider_data_preview(file_path, provider_names=None, max_rows=None, use_substring_match=False):
    """
    Extract data from a CSV file with optional filtering by institution
    This version is optimized for displaying complete data with no row limits
    
    Args:
        file_path: Path to the CSV file
        provider_names: List of provider (institution) names to filter by
        max_rows: Maximum number of rows to return (set to None for all rows)
        use_substring_match: Whether to use substring matching for provider names
        
    Returns:
        dict with:
            columns: List of column headers
            data: List of rows (each row is a list of values)
            total_rows: Total number of rows in the file
            matched_rows: Number of rows matching the filter
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting complete data from {file_path} for providers: {provider_names}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read the first line to check for metadata
            first_line = f.readline().strip()
            if first_line.startswith('#METADATA:'):
                # Skip the metadata line
                first_line = f.readline().strip()
                
            # Reset file pointer if we don't have metadata
            if not first_line.startswith('#METADATA:'):
                f.seek(0)
                
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            
            # Find the provider/institution column index
            provider_col_idx = None
            for i, header in enumerate(headers):
                if 'provider' in header.lower() or 'institution' in header.lower() or 'he provider' in header.lower():
                    provider_col_idx = i
                    break
                    
            if provider_col_idx is None:
                logger.warning(f"No provider column found in {file_path}")
                return None
                
            # Process all rows - no pagination
            rows = []
            total_rows = 0
            matched_rows = 0
            
            for row in csv_reader:
                total_rows += 1
                
                # Check if this row matches any of the requested providers
                if provider_names and provider_names != ["All"]:
                    provider_matches = False
                    provider_value = row[provider_col_idx] if provider_col_idx < len(row) else ""
                    
                    for provider in provider_names:
                        if use_substring_match:
                            # Use substring match (case-insensitive)
                            if provider.lower() in provider_value.lower():
                                provider_matches = True
                                break
                        else:
                            # Use exact match (case-insensitive)
                            if provider.lower() == provider_value.lower():
                                provider_matches = True
                                break
                                
                    if not provider_matches:
                        continue
                        
                matched_rows += 1
                rows.append(row)
                
                # We're not limiting rows here - always return all matching rows
                # only temporarily limit if really needed for initial processing
                if max_rows and matched_rows >= max_rows and max_rows > 0:
                    logger.warning(f"Reached limit of {max_rows} rows, but continuing to process all data")
                    
            logger.info(f"Processed {matched_rows} matched rows out of {total_rows} total rows")
            
            return {
                'columns': headers,
                'data': rows,
                'total_rows': total_rows,
                'matched_rows': matched_rows
            }
            
    except Exception as e:
        logger.error(f"Error extracting data from CSV: {str(e)}")
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

def get_csv_preview(file_path, max_rows=MAX_PREVIEW_ROWS, institution=None, mission_group=None):
    """
    Get a preview of a CSV file.
    Can filter by institution name or mission group.
    
    Args:
        file_path: Path to the CSV file
        max_rows: Maximum number of rows to include in the preview (None for all rows)
        institution: Institution name(s) to filter by (can be a comma-separated list)
        mission_group: Mission group to filter by
        
    Returns:
        dict: Preview data including columns and rows
    """
    import csv
    import re
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Generating preview for {file_path}, max_rows={max_rows}, institution={institution}")
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        # Read CSV file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip metadata line if present
            first_line = f.readline()
            if first_line.startswith('#METADATA:'):
                # Move cursor back to beginning
                f.seek(0)
                # Skip the first line
                next(f)
            else:
                # If no metadata, reset file pointer
                f.seek(0)
                
            # Read CSV data
            csv_reader = csv.reader(f)
            headers = next(csv_reader)  # Column headers
            
            # Determine the institution column index
            institution_col_idx = -1
            mission_group_col_idx = -1
            
            # Try to find the institution and mission group columns
            for i, header in enumerate(headers):
                header_lower = header.lower()
                if 'provider' in header_lower or 'institution' in header_lower or 'university' in header_lower:
                    institution_col_idx = i
                if 'mission' in header_lower and 'group' in header_lower:
                    mission_group_col_idx = i
            
            # Parse institution names
            institution_names = []
            if institution:
                # Parse comma-separated list of institutions
                institution_names = [name.strip() for name in institution.split(',') if name.strip()]
                logger.info(f"Filtering by institutions: {institution_names}")
            
            # Different logic for preview mode vs. dataset selection mode
            is_preview_mode = max_rows is not None and max_rows > 0
            
            # Read all rows from CSV
            all_rows = list(csv_reader)
            
            # For Preview Mode (limit to exactly 3 rows)
            if is_preview_mode:
                logger.info("Using Preview Mode logic")
                
                # Ensure we find "The University of Leicester" and other matching institutions
                leicester_row = None
                matching_rows = []
                
                for row in all_rows:
                    # Skip empty rows or incorrectly formatted rows
                    if not row or len(row) != len(headers) or institution_col_idx >= len(row):
                        continue
                    
                    institution_name = row[institution_col_idx]
                    
                    # Check for Leicester first
                    if "university of leicester" in institution_name.lower():
                        leicester_row = row
                    elif institution_names:
                        # Check for other institution matches
                        for name in institution_names:
                            if name.lower() in institution_name.lower():
                                matching_rows.append(row)
                                break
                
                # Prepare final data set (up to 3 rows)
                data = []
                
                # Add Leicester row if found
                if leicester_row:
                    data.append(leicester_row)
                
                # Add other matching rows (up to 2 more)
                remaining_slots = min(max_rows, 3) - len(data)
                if remaining_slots > 0 and matching_rows:
                    data.extend(matching_rows[:remaining_slots])
                
                # If we still need more rows to reach 3, add any rows
                remaining_slots = min(max_rows, 3) - len(data)
                if remaining_slots > 0 and all_rows:
                    # Add rows that weren't already included
                    for row in all_rows:
                        if row not in data and len(data) < min(max_rows, 3):
                            data.append(row)
                
                matched_count = len(data)
                
            # For Dataset Selection Mode (include all matching rows)
            else:
                logger.info("Using Dataset Selection Mode logic (all matching rows)")
                
                data = []
                matched_count = 0
                
                for row in all_rows:
                    # Skip empty rows
                    if not row or len(row) != len(headers):
                        continue
                    
                    # Apply mission group filter if needed
                    if mission_group and mission_group_col_idx >= 0:
                        if mission_group_col_idx >= len(row) or mission_group.lower() not in row[mission_group_col_idx].lower():
                            continue
                    
                    # Apply institution filter
                    if institution_names and institution_col_idx >= 0:
                        if institution_col_idx >= len(row):
                            continue
                        
                        row_institution = row[institution_col_idx]
                        found_match = False
                        
                        # Always include University of Leicester
                        if "university of leicester" in row_institution.lower():
                            found_match = True
                        else:
                            # Try exact matching first
                            exact_match_found = False
                            for name in institution_names:
                                # Skip Leicester since we already checked for it
                                if "leicester" in name.lower():
                                    continue
                                    
                                # Try exact match (case insensitive)
                                if name.lower() == row_institution.lower():
                                    exact_match_found = True
                                    found_match = True
                                    break
                            
                            # If no exact matches found, try partial matching
                            if not exact_match_found:
                                for name in institution_names:
                                    # Skip Leicester since we already checked for it
                                    if "leicester" in name.lower():
                                        continue
                                        
                                    # Check if institution name contains the search term
                                    if name.lower() in row_institution.lower():
                                        found_match = True
                                        break
                        
                        if not found_match:
                            continue
                    
                    # Include this row in the output
                    data.append(row)
                    matched_count += 1
            
            # Prepare response
            result = {
                'columns': headers,
                'data': data,
                'matched_rows': matched_count,
                'has_more': is_preview_mode and len(all_rows) > matched_count
            }
            
            return result
            
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def find_relevant_csv_files(file_pattern, keywords=None, requested_years=None):
    """Find relevant CSV files based on keywords and years.
    
    Args:
        file_pattern: Pattern to match CSV files (e.g., 'hesa_*.csv')
        keywords: List of keywords to match in the file content
        requested_years: Optional list of years to filter by
        
    Returns:
        List of matching files with metadata, grouped by title/dataset
    """
    try:
        # Log parameters for debugging
        logger = logging.getLogger(__name__)
        logger.info(f"Looking for CSVs with pattern: {file_pattern}, keywords: {keywords}, years: {requested_years}")
        
        # Check if index file exists
        index_path = os.path.join(CLEANED_FILES_DIR, "hesa_files_index.json")
        if not os.path.exists(index_path):
            logger.warning(f"Index file not found: {index_path}")
            # Fallback to older method if index not available
            return find_relevant_csv_files_fallback(file_pattern, keywords, requested_years)
        
        # Load the index file
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            logger.info(f"Loaded index with {index_data.get('total_files', 0)} files")
        except Exception as e:
            logger.error(f"Error loading index file: {str(e)}")
            # Fallback to older method if there's an error
            return find_relevant_csv_files_fallback(file_pattern, keywords, requested_years)
        
        # Extract the files from the index
        all_files = index_data.get("hesa_files", [])
        if not all_files:
            logger.warning("No files found in the index")
            return []
        
        # Create a map to group files by title
        grouped_files = {}
        
        # Process each file from the index
        for file_info in all_files:
            file_path = file_info.get("file_path")
            title = file_info.get("title", "Unknown title")
            academic_year = file_info.get("academic_year", "Unknown")
            keywords_title = file_info.get("keywords_title", [])
            keywords_columns = file_info.get("keywords_columns", [])
            
            # Normalize academic year for comparison
            year = academic_year.split('/')[0] if '/' in academic_year else academic_year
            
            # Skip if the file doesn't match the file pattern
            if file_pattern and not fnmatch.fnmatch(os.path.basename(file_path), file_pattern):
                continue
            
            # If years are specified and this file doesn't match, skip it
            if requested_years and not any(year.startswith(req_year) for req_year in requested_years):
                continue
            
            # Calculate relevance score
            # Start with base score
            score = 0.5
            match_percentage = 50
            matched_keywords = []
            
            # If keywords are provided, calculate match score
            if keywords:
                # Combine all available keywords
                all_file_keywords = keywords_title + keywords_columns
                # Count matching keywords
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    for file_keyword in all_file_keywords:
                        file_keyword_lower = file_keyword.lower()
                        # Direct match or substring match
                        if keyword_lower == file_keyword_lower or keyword_lower in file_keyword_lower or file_keyword_lower in keyword_lower:
                            score += 0.1
                            matched_keywords.append(keyword)
                            break
                            
                # Adjust percentage based on matches
                if len(keywords) > 0:
                    match_percentage = min(100, int((len(matched_keywords) / len(keywords)) * 100))
                    # Boost score based on match percentage
                    score = min(1.0, 0.5 + (match_percentage / 200))
            
            # Create a match object
            match = {
                'file_path': str(file_path),
                'title': title, 
                'year': year,
                'academic_year': academic_year,
                'score': score,
                'match_percentage': match_percentage,
                'matched_keywords': list(set(matched_keywords))  # Remove duplicates
            }
            
            # Group by title
            if title not in grouped_files:
                grouped_files[title] = {
                    'title': title,
                    'group_id': f"group_{len(grouped_files) + 1}",
                    'score': score,
                    'match_percentage': match_percentage,
                    'matched_keywords': matched_keywords,
                    'years': [academic_year],
                    'files': [match]
                }
            else:
                # Update existing group
                group = grouped_files[title]
                # Add this file
                group['files'].append(match)
                # Add unique academic year
                if academic_year not in group['years']:
                    group['years'].append(academic_year)
                # Update score if this file has a better score
                if score > group['score']:
                    group['score'] = score
                    group['match_percentage'] = match_percentage
                    group['matched_keywords'] = matched_keywords
        
        # Convert the grouped files dictionary to a list and sort by score
        file_groups = list(grouped_files.values())
        file_groups.sort(key=lambda x: (-x['score'], x['title']))
        
        logger.info(f"Found {len(file_groups)} file groups from index")
        return file_groups
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in find_relevant_csv_files: {str(e)}")
        return []

def find_relevant_csv_files_fallback(file_pattern, keywords=None, requested_years=None):
    """Legacy method to find relevant CSV files without using the index."""
    try:
        # Log parameters for debugging
        logger = logging.getLogger(__name__)
        logger.info(f"Using fallback to find CSVs with pattern: {file_pattern}, keywords: {keywords}, years: {requested_years}")
        
        # Get all CSV files from cleaned directory
        csv_dir = CLEANED_FILES_DIR
        file_paths = []
        
        if os.path.exists(csv_dir):
            # Get all CSV files that match the pattern if provided, otherwise get all CSVs
            if file_pattern:
                pattern_path = os.path.join(csv_dir, file_pattern)
                file_paths = glob.glob(pattern_path)
                logger.info(f"Found {len(file_paths)} files matching pattern: {pattern_path}")
            else:
                # If no pattern, get all CSV files
                file_paths = list(Path(csv_dir).glob('*.csv'))
                logger.info(f"Found {len(file_paths)} total CSV files in {csv_dir}")
        else:
            logger.warning(f"CSV directory not found: {csv_dir}")
            return []
        
        # No files found
        if not file_paths:
            logger.warning("No CSV files found matching the pattern. Using all available CSV files.")
            # Fallback to all CSV files
            file_paths = list(Path(csv_dir).glob('*.csv'))
            if not file_paths:
                logger.error("No CSV files found at all in the directory.")
            return []
        
        # Create a map to group files by title
        grouped_files = {}
        
        # Process each file
        for file_path in file_paths:
            # Extract filename without path and extension
            file_name = os.path.basename(str(file_path))
            file_title = file_name.replace('.csv', '').replace('_', ' ').title()
            
            # Try to extract year from filename
            year_match = re.search(r'(\d{4})', file_name)
            year = year_match.group(1) if year_match else 'Unknown'
            academic_year = f"{year}/{str(int(year)+1)[2:4]}" if year != 'Unknown' else 'Unknown'
            
            # If years are specified and this file doesn't match, skip it
            if requested_years and year not in requested_years:
                # Also try to match academic year format
                if academic_year not in requested_years:
                    continue
            
            # Calculate a simple relevance score based on keyword matches
            score = 0.5  # Default base score
            match_percentage = 50  # Default match percentage
            matched_keywords = []
            
            # If we have keywords, calculate matches
            if keywords:
                # Try to find each keyword in the file name
                for keyword in keywords:
                    if keyword.lower() in file_name.lower():
                        score += 0.1
                        matched_keywords.append(keyword)
                
                # Adjust percentage based on matches
                if len(keywords) > 0:
                    match_percentage = min(100, int((len(matched_keywords) / len(keywords)) * 100))
                    # Boost score based on match percentage
                    score = min(1.0, 0.5 + (match_percentage / 200))
            
            # Create a match object
            match = {
                'file_path': str(file_path),
                'file_name': file_name,
                'title': file_title, 
                'year': year,
                'academic_year': academic_year,
                'score': score,
                'match_percentage': match_percentage,
                'matched_keywords': matched_keywords
            }
            
            # Group by title
            if file_title not in grouped_files:
                grouped_files[file_title] = {
                    'title': file_title,
                    'group_id': f"group_{len(grouped_files) + 1}",
                    'score': score,
                    'percentage': match_percentage,
                    'matched_keywords': matched_keywords,
                    'available_years': [academic_year],
                    'files': [match]
                }
            else:
                # Add this file to existing group
                grouped_files[file_title]['files'].append(match)
                # Add unique year
                if academic_year not in grouped_files[file_title]['available_years']:
                    grouped_files[file_title]['available_years'].append(academic_year)
                # Update score if higher
                if score > grouped_files[file_title]['score']:
                    grouped_files[file_title]['score'] = score
                    grouped_files[file_title]['percentage'] = match_percentage
                    grouped_files[file_title]['matched_keywords'] = matched_keywords
        
        # Convert the grouped files dictionary to a list and sort by score
        file_groups = list(grouped_files.values())
        file_groups.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Found {len(file_groups)} file groups using fallback method")
        return file_groups
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in find_relevant_csv_files_fallback: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def dataset_details(request, group_id):
    """
    View a specific dataset with all its details.
    Shows only the selected dataset instead of all matching datasets.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get query parameters from the request
        query = request.GET.get('query', '')
        institution = request.GET.get('institution', '')
        mission_group = request.GET.get('mission_group', '')
        start_year = request.GET.get('start_year', '')
        end_year = request.GET.get('end_year', '')
        
        logger.info(f"Viewing dataset details for group: {group_id}")
        logger.info(f"Query: {query}")
        logger.info(f"Institution: {institution}")
        logger.info(f"Mission Group: {mission_group}")
        logger.info(f"Years: {start_year} - {end_year}")
        
        # Parse the query to extract query information
        query_info = parse_hesa_query(query)
        keywords = query_info.get('keywords', [])
        file_pattern = query_info.get('file_pattern', '')
        filtered_terms = query_info.get('filtered_terms', [])
        removed_words = query_info.get('removed_words', [])
        
        # Get years in academic year format
        years = []
        if start_year:
            try:
                start = int(start_year)
                if end_year:
                    # Range of years
                    end = int(end_year)
                    for year in range(start, end + 1):
                        academic_year = f"{year}/{str(year+1)[2:4]}"
                        years.append(academic_year)
                else:
                    # Single year provided
                    academic_year = f"{start}/{str(start+1)[2:4]}"
                    years.append(academic_year)
            except ValueError:
                logger.warning(f"Invalid year format: start_year={start_year}, end_year={end_year}")
                
        # Find all CSV files
        all_files = list(CLEANED_FILES_DIR.glob('*.csv'))
        
        # Extract parts from the group_id (format varies: group_N or group_N_score)
        parts = group_id.split('_')
        group_number = None
        target_title = None
        
        if len(parts) >= 2:
            try:
                group_number = int(parts[1])  # Extract the group number
                score = parts[2] if len(parts) > 2 else None
                logger.info(f"Looking for dataset with group number {group_number} and score {score}")
                
                # Find the target title by recreating the search
                grouped_matches = find_relevant_csv_files(file_pattern, keywords, years if years else None)
                
                # Try to find the group that matches our group number
                if grouped_matches and 0 < group_number <= len(grouped_matches):
                    target_group = grouped_matches[group_number - 1]
                    target_title = target_group['title']
                    logger.info(f"Found target title from recreated search: {target_title}")
                else:
                    logger.warning(f"Group number {group_number} not found in search results")
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing group ID: {str(e)}")
        
        # If we couldn't find the title through search, try direct file lookups
        if not target_title:
            for file_path in all_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline().strip()
                        if first_line.startswith('#METADATA:'):
                            metadata_str = first_line[len('#METADATA:'):]
                            try:
                                metadata = json.loads(metadata_str)
                                metadata_group_id = metadata.get('group_id', '')
                                
                                # Check if this file belongs to our group
                                if group_id == metadata_group_id or \
                                   group_id in metadata_group_id or \
                                   metadata_group_id.startswith(f"group_{group_number}_"):
                                    target_title = metadata.get('title', '')
                                    logger.info(f"Found target title: {target_title}")
                                    break
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue
            
            # Direct approach for common datasets if we still don't have a title
            if not target_title:
            # These are common dataset titles that we can fall back to
                default_titles = {
                    1: "HE student enrolments by HE provider",
                    2: "UK permanent address HE students by HE provider and permanent address",
                    3: "HE qualifiers by HE provider and level of qualification obtained"
                }
            
            if group_number in default_titles:
                target_title = default_titles[group_number]
                logger.info(f"Using default title for group {group_number}: {target_title}")
        
        if not target_title:
            return render(request, 'dashboard.html', {
                'error': 'Dataset not found',
                'query': query,
                'institution': institution,
                'start_year': start_year,
                'end_year': end_year,
            })
        
        # Find files that match the target title
        csv_files = []
        for file_path in all_files:
            # First try to extract title from metadata
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#METADATA:'):
                        metadata_str = first_line[len('#METADATA:'):]
                        try:
                            metadata = json.loads(metadata_str)
                            file_title = metadata.get('title', '')
                            file_year = metadata.get('academic_year', '')
                            
                            # Match by title
                            if file_title.lower() == target_title.lower() or \
                               target_title.lower() in file_title.lower() or \
                               file_title.lower() in target_title.lower():
                                # Check year filter if provided
                                if not years or file_year in years:
                                    csv_files.append({
                                        'file_path': str(file_path),
                                        'file_name': os.path.basename(file_path),
                                        'title': file_title,
                                        'year': file_year,
                                        'score': parts[2] if len(parts) > 2 else None,
                                        'matched_keywords': metadata.get('keywords_title', []) + metadata.get('keywords_columns', [])
                                    })
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid metadata JSON in {file_path}")
            except Exception as e:
                logger.warning(f"Error reading metadata from {file_path}: {str(e)}")
            
            # If no title from metadata, try from filename
        if not csv_files:
            for file_path in all_files:
                file_name = os.path.basename(file_path)
                derived_title = file_name.replace('.csv', '').replace('_', ' ').title()
                
                # Check if the derived title matches the target title
                if derived_title.lower() == target_title.lower() or \
                   target_title.lower() in derived_title.lower() or \
                   derived_title.lower() in target_title.lower():
                    # Extract year from filename if possible
                    year_match = re.search(r'(20\d{2})', file_name)
                    file_year = year_match.group(1) if year_match else 'Unknown'
                    academic_year = f"{file_year}/{str(int(file_year)+1)[2:4]}" if file_year.isdigit() else 'Unknown'
                    
                    # Check year filter if provided
                    if not years or academic_year in years:
                        csv_files.append({
                            'file_path': str(file_path),
                            'file_name': file_name,
                            'title': derived_title,
                            'year': academic_year,
                            'score': parts[2] if len(parts) > 2 else None,
                            'matched_keywords': []
                        })
        
        if not csv_files:
            logger.warning(f"No files found for title: {target_title}")
            return render(request, 'dashboard.html', {
                'error': 'No files found for the selected dataset',
                'query': query,
                'institution': institution,
                'start_year': start_year,
                'end_year': end_year,
            })
        
        # Generate file previews for each file - WITHOUT row limits for dataset details
        file_previews = []
        for file_match in csv_files:
            file_path = file_match['file_path']
            
            # Call get_csv_preview with None for max_rows to get all rows, include mission group if provided
            preview = get_csv_preview(
                file_path, 
                max_rows=None, 
                institution=institution,
                mission_group=mission_group
            )
            if preview:
                preview['year'] = file_match['year']
                preview['file_name'] = file_match['file_name']
                file_previews.append(preview)
        
        # Create a summary of this single group
        score = parts[2] if len(parts) > 2 else None
        percentage = float(score) * 100 / 1.0 if score else None
        group_summary = {
            'group_id': group_id,
            'title': target_title,
            'score': float(score) if score else None,
            'match_percentage': f"{percentage:.2f}" if percentage else None,
            'matched_keywords': list(set([keyword for file in csv_files for keyword in file.get('matched_keywords', [])])),
            'available_years': sorted(list(set([file['year'] for file in csv_files if file['year']]))),
            'file_count': len(csv_files),
            'file_previews': file_previews
        }
        
        # Return the dataset_details template with the dataset information
        return render(request, 'core/dataset_details.html', {
            'query': query,
            'institution': institution,
            'mission_group': mission_group,
            'start_year': start_year,
            'end_year': end_year,
            'filtered_terms': filtered_terms,
            'removed_words': removed_words,
            'max_preview_rows': None,  # Set to None to indicate no limit
            'dataset': group_summary
        })
        
    except Exception as e:
        logger.error(f"Error in dataset details: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return render(request, 'dashboard.html', {
            'error': f'Error viewing dataset: {str(e)}',
            'query': query,
            'institution': institution,
            'mission_group': mission_group,
            'start_year': start_year,
            'end_year': end_year,
        })

def find_matching_datasets(query, data_request, start_year=None, end_year=None):
    """
    Find datasets that match the AI-extracted query entities.
    First filter by year range using regex, then use Gemini AI for semantic matching.
    Datasets with the same title but different academic years are grouped as a single match.
    
    Args:
        query (str): The original query
        data_request (list): Categories of data requested
        start_year (str): Start year of the query range (if available)
        end_year (str): End year of the query range (if available)
    
    Returns:
        list: Matched datasets sorted by relevance, with same-title datasets grouped together
    """
    # Import necessary modules
    import re
    import json
    import logging
    import os
    from collections import defaultdict
    
    # Get the logger for this module
    logger = logging.getLogger(__name__)
    
    logger.info(f"Finding matching datasets for query: '{query}', data_request: {data_request}, years: {start_year}-{end_year}")
    
    # Load the index file
    index_file_path = os.path.join(settings.BASE_DIR, 'hesa_files_index.json')
    if not os.path.exists(index_file_path):
        logger.error(f"Index file not found at {index_file_path}")
        return []
    
    try:
        with open(index_file_path, 'r') as f:
            index_data = json.load(f)
            datasets = index_data.get('hesa_files', [])
            logger.info(f"Loaded {len(datasets)} datasets from index file")
    except Exception as e:
        logger.error(f"Error loading index file: {str(e)}")
        return []
    
    # Filter datasets by year range if specified (keep this regex filtering for efficiency)
    if start_year or end_year:
        filtered_datasets = []
        for dataset in datasets:
            academic_year = dataset.get('academic_year', '')
            if not academic_year:
                continue
                
            # Extract the first year from academic year (e.g. 2020/21 -> 2020)
            year_match = re.match(r'(\d{4})', academic_year)
            if not year_match:
                continue
                
            dataset_year = year_match.group(1)
            
            # Check if the dataset's year is within the requested range
            if start_year and end_year:
                if int(start_year) <= int(dataset_year) <= int(end_year):
                    filtered_datasets.append(dataset)
            elif start_year:
                if int(start_year) <= int(dataset_year):
                    filtered_datasets.append(dataset)
            elif end_year:
                if int(dataset_year) <= int(end_year):
                    filtered_datasets.append(dataset)
        
        logger.info(f"Filtered down to {len(filtered_datasets)} datasets based on year range")
        datasets = filtered_datasets
    
    # If no datasets after filtering, return empty list
    if not datasets:
        logger.warning("No datasets found after year filtering")
        return []
    
    try:
        # Use Gemini API for semantic matching
        gemini_client = GeminiClient(settings.GEMINI_API_KEY)
        
        # Prepare prompt for Gemini to match datasets based on semantic understanding
        # Note: No longer mentioning years in prompt since datasets are already filtered
        prompt = f"""
        I need to find the most relevant datasets based on a user query. 

        USER QUERY: "{query}"

        EXTRACTED INFORMATION:
        - Data requested: {data_request}

        I have the following datasets (already filtered for relevance):

        {json.dumps([{
            "title": d.get("title", ""),
            "columns": d.get("columns", []),
            "academic_year": d.get("academic_year", ""),
            "reference": d.get("reference", "")
        } for d in datasets[:30]], indent=2)}

        For each dataset, analyze how well it matches the user query semantically. 
        Consider variations in terminology (e.g., "enrollment" vs "enrolment", "registration") 
        and related concepts (e.g., if user asks for "postgraduates", datasets about "graduates" or "masters students" may be relevant).

        Return a JSON array with the following format for each matching dataset:
        [
          {{
            "reference": "filename.csv",
            "score": 0.95,
            "matched_terms": ["term1", "term2", "term3"],
            "description": "Brief explanation of why this dataset matches"
          }}
        ]

        Only include datasets with a score > 0.1. Sort the results by score in descending order.
        """
        
        # Call Gemini API for semantic matching
        logger.info("Calling Gemini API for semantic dataset matching")
        ai_response = gemini_client.get_completion(prompt)
        
        # Extract JSON response from Gemini
        json_match = re.search(r'\[\s*\{.*\}\s*\]', ai_response, re.DOTALL)
        
        if json_match:
            matches_json = json_match.group(0)
            matches = json.loads(matches_json)
            
            # Enhance matches with full dataset information
            individual_matches = []
            for match in matches:
                reference = match.get("reference")
                # Find the complete dataset info
                for dataset in datasets:
                    if dataset.get("reference") == reference:
                        enhanced_match = {
                            "title": dataset.get("title", "Untitled Dataset"),
                            "reference": reference,
                            "academic_year": dataset.get("academic_year", "Unknown"),
                            "columns": dataset.get("columns", []),
                            "score": match.get("score", 0),
                            "match_percentage": int(match.get("score", 0) * 100),
                            "matched_terms": match.get("matched_terms", []),
                            "description": match.get("description", "")
                        }
                        individual_matches.append(enhanced_match)
                        break
            
            # Group matches by title
            title_grouped_matches = defaultdict(list)
            for match in individual_matches:
                title_grouped_matches[match['title']].append(match)
            
            # Create combined matches with datasets of the same title grouped together
            grouped_matches = []
            for title, matches_list in title_grouped_matches.items():
                # Calculate average score
                avg_score = sum(match['score'] for match in matches_list) / len(matches_list)
                avg_percentage = int(avg_score * 100)
                
                # Collect all unique matched terms
                all_matched_terms = set()
                for match in matches_list:
                    if match.get('matched_terms'):
                        all_matched_terms.update(match.get('matched_terms', []))
                
                # Collect all academic years
                academic_years = [match['academic_year'] for match in matches_list]
                academic_years.sort()  # Sort years for better display
                
                # Collect all references
                references = [match['reference'] for match in matches_list]
                
                # Combine descriptions if they're different
                descriptions = []
                for match in matches_list:
                    if match.get('description') and match.get('description') not in descriptions:
                        descriptions.append(match.get('description'))
                
                combined_match = {
                    'title': title,
                    'score': avg_score,
                    'match_percentage': avg_percentage,
                    'academic_years': academic_years,
                    'references': references,
                    'matched_terms': list(all_matched_terms),
                    'descriptions': descriptions,
                    'matches': matches_list,  # Include individual matches for reference
                    'columns': matches_list[0].get('columns', []) if matches_list else []
                }
                
                grouped_matches.append(combined_match)
            
            # Sort grouped matches by score (highest first)
            grouped_matches.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Found {len(individual_matches)} individual matching datasets grouped into {len(grouped_matches)} dataset groups using Gemini AI")
            return grouped_matches
        else:
            logger.error("Failed to parse JSON response from Gemini API")
            # Fallback to regex matching
            return regex_matching_fallback(query, data_request, datasets)
            
    except Exception as e:
        logger.error(f"Error using Gemini API for semantic matching: {str(e)}")
        # Fallback to regex matching
        return regex_matching_fallback(query, data_request, datasets)

@csrf_exempt
@require_http_methods(["POST"])
def process_gemini_query(request):
    """
    Process a query using the Gemini API.
    Extract entities and find matching datasets.
    """
    import json
    import logging
    import os
    
    # Get the logger for this module
    logger = logging.getLogger(__name__)
    
    try:
        # Get the query from the request
        data = json.loads(request.body)
        query = data.get('query', '')
        max_matches = int(data.get('max_matches', 3))  # Convert to integer
        
        if not query:
            return JsonResponse({'error': 'No query provided'}, status=400)
            
        logger.info(f"Processing Gemini query: {query}")
        
        # Use Gemini API to extract entities from query
        try:
            gemini_client = GeminiClient(settings.GEMINI_API_KEY)
            result = gemini_client.analyze_query(query)
            using_mock = False
        except Exception as e:
            logger.error(f"Gemini API error (using local fallback): {str(e)}")
            # Fallback to local analysis
            result = local_analyze_query(query)
            using_mock = True
            
        # Find matching datasets based on the extracted entities
        matching_datasets = find_matching_datasets(
            query, 
            result.get('data_request', []), 
            result.get('start_year'), 
            result.get('end_year')
        )
        
        # Get institutions for CSV preview filtering
        institutions = result.get('institutions', [])
        
        # Always include University of Leicester
        if not any('leicester' in inst.lower() for inst in institutions):
            institutions.append('University of Leicester')
            
        # For preview mode, we want exactly 3 rows per dataset
        preview_max_rows = 3
            
        # Generate CSV previews for each dataset
        for dataset in matching_datasets:
            dataset_previews = []
            
            # Process each file in the dataset (stored in matches)
            if 'matches' in dataset:
                for match in dataset['matches']:
                    reference = match.get('reference', '')
                    if reference:
                        # Construct the full file path
                        file_path = os.path.join(CLEANED_FILES_DIR, reference)
                        
                        # Get CSV preview for this file filtered by the extracted institutions
                        try:
                            # Convert institutions list to comma-separated string
                            institutions_str = ','.join(institutions)
                            
                            # In preview mode, we want exactly 3 rows per dataset
                            preview = get_csv_preview(
                                file_path=file_path,
                                max_rows=preview_max_rows,
                                institution=institutions_str
                            )
                            
                            # Add the preview to the match
                            match['preview'] = preview
                            
                            # Also add to a collection of previews for the dataset
                            if preview:
                                preview_with_metadata = preview.copy()
                                preview_with_metadata['reference'] = reference
                                preview_with_metadata['academic_year'] = match.get('academic_year', '')
                                dataset_previews.append(preview_with_metadata)
                    
                        except Exception as e:
                            logger.error(f"Error generating preview for {reference}: {str(e)}")
                            match['preview'] = {
                                'columns': ['Error'],
                                'data': [[f"Could not generate preview: {str(e)}"]],
                                'error': str(e)
                            }
            
            # Add the collection of previews to the dataset
            dataset['previews'] = dataset_previews
        
        # Limit the number of datasets based on max_matches
        matching_datasets = matching_datasets[:max_matches]
        
        # Format response
        response_data = {
            'query': query,
            'institutions': result.get('institutions', []),
            'original_institutions': result.get('original_institutions', []),
            'has_institution_typos': result.get('has_institution_typos', False),
            'years': result.get('years', []),
            'original_years': result.get('original_years', []),
            'has_year_typos': result.get('has_year_typos', False),
            'start_year': result.get('start_year'),
            'end_year': result.get('end_year'),
            'data_request': result.get('data_request', []),
            'matching_datasets': [match.get('references', [])[0] if match.get('references') else "" for match in matching_datasets],
            'grouped_datasets': matching_datasets,  # Include the new grouped datasets
            'total_matches': len(matching_datasets),
            'using_mock': using_mock
        }
        
        # Check if any requested years are missing from the available datasets
        requested_years = result.get('years', [])
        available_years = set()
        
        # Collect all available years from matching datasets
        for dataset in matching_datasets:
            if 'academic_years' in dataset:
                available_years.update(dataset['academic_years'])
        
        # Find missing years
        missing_years = []
        if requested_years:
            for year in requested_years:
                if year not in available_years:
                    missing_years.append(year)
        
        # Add missing years information to the response data
        response_data['missing_years'] = missing_years
        
        logger.info(f"Gemini query analysis results: {json.dumps(response_data)}")
        
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error processing Gemini query: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

def local_analyze_query(query):
    """
    A simple function to perform basic entity extraction from a query
    when the Gemini API is unavailable.
    """
    import re
    import logging
    from datetime import datetime

    logger = logging.getLogger(__name__)
    logger.info(f"Local analysis for query: {query}")
    
    # Extract institutions
    institutions = []
    
    # Common university names
    uni_patterns = [
        r"University of ([A-Za-z\s]+)",
        r"([A-Za-z\s]+) University"
    ]
    
    for pattern in uni_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            if match.group(0).lower() != "university of" and match.group(0).lower() != "in university":
                institutions.append(match.group(0).strip())
    
    # Also look for standalone city/location names (not part of university patterns)
    # This is to handle cases like "london" without transforming to "University of London"
    
    # Get the original query with university matches removed to find standalone locations
    query_without_unis = query
    for inst in institutions:
        query_without_unis = query_without_unis.replace(inst, "", 1)
    
    # Common UK cities/locations that might be used as education institutions
    city_patterns = [
        r"\b(london)\b",
        r"\b(manchester)\b",
        r"\b(birmingham)\b",
        r"\b(leeds)\b",
        r"\b(edinburgh)\b",
        r"\b(glasgow)\b",
        r"\b(cardiff)\b",
        r"\b(oxford)\b",
        r"\b(cambridge)\b",
    ]
    
    # Check for standalone city names
    for pattern in city_patterns:
        matches = re.finditer(pattern, query_without_unis, re.IGNORECASE)
        for match in matches:
            city_name = match.group(1)
            # If we find a standalone city, add it exactly as typed in the original query
            original_case = query[match.start():match.end()]
            institutions.append(original_case)
    
    # Always include University of Leicester
    if institutions and "University of Leicester" not in institutions:
        institutions.append("University of Leicester")
    
    # Extract years
    years = []
    year_pattern = r"\b(20\d{2})(?:/(\d{2}))?\b"
    matches = re.finditer(year_pattern, query)
    for match in matches:
        if match.group(2):  # academic year format (e.g., 2020/21)
            years.append(f"{match.group(1)}/{match.group(2)}")
        else:  # plain year format (e.g., 2020)
            years.append(match.group(1))
    
    # Extract start and end years
    start_year = None
    end_year = None
    
    # Extract years with context
    query_lower = query.lower()
    
    # Start year patterns
    start_patterns = [
        r'start(?:ing|s|ed)?\s+(?:in|from|at)?\s+(?:the\s+)?(?:year\s+)?(\d{4})',
        r'begin(?:ning|s)?\s+(?:in|from|at)?\s+(?:the\s+)?(?:year\s+)?(\d{4})',
        r'from\s+(?:the\s+)?(?:year\s+)?(\d{4})'
    ]
    
    # End year patterns
    end_patterns = [
        r'end(?:ing|s|ed)?\s+(?:in|at|of)?\s+(?:the\s+)?(?:year\s+)?(\d{4})',
        r'finish(?:ing|es|ed)?\s+(?:in|at)?\s+(?:the\s+)?(?:year\s+)?(\d{4})',
        r'(?:in|at)\s+(?:the\s+)?end\s+of\s+(?:the\s+)?(?:year\s+)?(\d{4})'
    ]
    
    # Process years with "start" context
    for pattern in start_patterns:
        start_match = re.search(pattern, query_lower)
        if start_match:
            year = start_match.group(1)
            start_year = year
            academic_year = f"{year}/{str(int(year)+1)[2:4]}"
            if academic_year not in years:
                years.append(academic_year)
            logger.info(f"Processed starting year {year} as academic year {academic_year}")
    
    # Process years with "end" context
    for pattern in end_patterns:
        end_match = re.search(pattern, query_lower)
        if end_match:
            year = end_match.group(1)
            end_year = year
            previous_year = str(int(year) - 1)
            academic_year = f"{previous_year}/{year[2:4]}"
            if academic_year not in years:
                years.append(academic_year)
            logger.info(f"Processed ending year {year} as academic year {academic_year}")
    
    # Look for year ranges like "2019 to 2022" or "between 2019 and 2022"
    range_pattern = r"(?:between|from)?\s*(20\d{2})\s*(?:to|and|[-])\s*(20\d{2})"
    range_match = re.search(range_pattern, query, re.IGNORECASE)
    if range_match:
        start_year = range_match.group(1)
        end_year = range_match.group(2)
        
        # Handle range logic - treat all as starting years
        years = []  # Clear existing years
        for year in range(int(start_year), int(end_year) + 1):
            academic_year = f"{year}/{str(year+1)[2:4]}"
            years.append(academic_year)
        logger.info(f"Processed year range: {start_year}-{end_year} as academic years: {', '.join(years)}")
    
    # Look for "past X years" pattern
    past_years_pattern = r"past\s+(\d+)\s+years?"
    past_match = re.search(past_years_pattern, query, re.IGNORECASE)
    if past_match:
        num_years = int(past_match.group(1))
        current_year = datetime.now().year
        end_year = str(current_year)
        start_year = str(current_year - num_years)
        
        # Replace years with academic years for the range
        years = []
        for year in range(current_year - num_years, current_year + 1):
            years.append(str(year))
            years.append(f"{year}/{str(year+1)[2:4]}")
    
    # For any plain years without context, treat as starting years
    plain_year_pattern = r'\b(20\d{2})\b'
    years_mentioned = re.findall(plain_year_pattern, query)
    
    # Track which years were already processed with context
    processed_years = set()
    
    # Collect years that were processed with start/end context
    for pattern in start_patterns + end_patterns:
        match = re.search(pattern, query_lower)
        if match:
            processed_years.add(match.group(1))
    
    # Also add years from explicit ranges
    if range_match:
        processed_years.add(range_match.group(1))
        processed_years.add(range_match.group(2))
    
    # Process remaining plain years
    for year in years_mentioned:
        if year not in processed_years:
            # Default: treat as starting year
            academic_year = f"{year}/{str(int(year)+1)[2:4]}"
            if academic_year not in years:
                years.append(academic_year)
                if not start_year:  # Only set if not already set
                    start_year = year
            logger.info(f"Processed plain year {year} as academic year {academic_year}")
    
    # Determine the type of data being requested
    data_request = ["general_data"]  # Default
    
    # Pattern matching for data categories
    data_patterns = {
        "student_enrollment": ["enrollment", "enrolment", "students", "student numbers"],
        "student_demographics": ["demographics", "gender", "ethnicity", "nationality"],
        "graduation_rates": ["graduation", "graduates", "qualifiers", "degrees awarded"],
        "staff_data": ["staff", "faculty", "employees", "lecturers"],
        "research_data": ["research", "publications", "funding"]
    }
    
    for category, keywords in data_patterns.items():
        for keyword in keywords:
            if keyword in query.lower():
                data_request = [category]
                break
                
    # Handle postgraduate specific queries
    if "postgraduate" in query.lower() or "post-graduate" in query.lower() or "masters" in query.lower() or "phd" in query.lower():
        data_request = ["student_enrollment"]  # Categorize as enrollment for postgraduates
    
    # Build and return the result
    result = {
        "institutions": institutions,
        "years": list(set(years)),  # Ensure uniqueness
        "start_year": start_year,
        "end_year": end_year,
        "data_request": data_request
    }
    
    logger.info(f"Local analysis result: {result}")
    return result

def regex_matching_fallback(query, data_request, datasets):
    """
    Fallback method for dataset matching based on regex.
    Use when Gemini API is unavailable.
    Groups datasets with the same title as a single match.
    """
    import re
    import logging
    from collections import defaultdict
    
    logger = logging.getLogger(__name__)
    logger.info("Using regex fallback for dataset matching")
    
    # Case-insensitive matching
    query_lower = query.lower()
    
    # Prepare regex patterns for common education terms
    student_patterns = [r'student', r'pupil', r'learner', r'enrollment', r'enrolment']
    staff_patterns = [r'staff', r'faculty', r'teacher', r'professor', r'lecturer']
    research_patterns = [r'research', r'publication', r'study', r'project', r'academic']
    
    # Postgraduate/undergraduate patterns
    postgrad_patterns = [r'postgraduate', r'post-graduate', r'graduate', r'masters', r'phd', r'doctoral']
    undergrad_patterns = [r'undergraduate', r'under-graduate', r'bachelor']
    
    # Create dictionaries to track matches
    individual_matches = []
    
    # Process each dataset for matches
    for dataset in datasets:
        title = dataset.get('title', '').lower()
        academic_year = dataset.get('academic_year', '')
        reference = dataset.get('reference', '')
        
        # Skip if no title or reference
        if not title or not reference:
            continue
            
        # Check for title match with the query terms
        score = 0
        matched_terms = []
        description_points = []
        
        # Add points for data request match
        requested_data_found = False
        if 'student' in ' '.join(data_request).lower():
            for pattern in student_patterns:
                if re.search(pattern, title):
                    score += 0.2
                    matched_terms.append('student data')
                    description_points.append("Contains student data")
                    requested_data_found = True
                    break
        
        if 'staff' in ' '.join(data_request).lower():
            for pattern in staff_patterns:
                if re.search(pattern, title):
                    score += 0.2
                    matched_terms.append('staff data')
                    description_points.append("Contains staff information")
                    requested_data_found = True
                    break
                    
        if 'research' in ' '.join(data_request).lower():
            for pattern in research_patterns:
                if re.search(pattern, title):
                    score += 0.2
                    matched_terms.append('research data')
                    description_points.append("Contains research information")
                    requested_data_found = True
                    break
        
        # Add points for education level match
        if any(re.search(pattern, query_lower) for pattern in postgrad_patterns):
            if any(re.search(pattern, title) for pattern in postgrad_patterns):
                score += 0.3
                matched_terms.append('postgraduate')
                description_points.append("Specifically includes postgraduate data")
        
        if any(re.search(pattern, query_lower) for pattern in undergrad_patterns):
            if any(re.search(pattern, title) for pattern in undergrad_patterns):
                score += 0.3
                matched_terms.append('undergraduate')
                description_points.append("Specifically includes undergraduate data")
        
        # Add points for title keywords matching query
        title_keywords = re.findall(r'\w+', title)
        query_keywords = re.findall(r'\w+', query_lower)
        
        common_keywords = set(title_keywords) & set(query_keywords)
        if common_keywords:
            score += 0.1 * len(common_keywords)
            matched_terms.extend(list(common_keywords))
            description_points.append(f"Matches query keywords: {', '.join(common_keywords)}")
        
        # Adjust score if data request wasn't found but title seems relevant
        if not requested_data_found and score > 0:
            score *= 0.7  # Reduce score if specific data type not found
        
        # Add bonus for key HESA data types if they appear in title
        key_types = ['enrollment', 'enrolment', 'qualification', 'demographics', 'finance']
        for key_type in key_types:
            if key_type in title:
                score += 0.1
                if key_type not in matched_terms:
                    matched_terms.append(key_type)
        
        # Only include if it has some meaningful score
        if score > 0.1:
            # Create a match object similar to Gemini AI output
            match = {
                'title': dataset.get('title', 'Untitled'),
                'reference': reference,
                'academic_year': academic_year,
                'score': min(score, 0.95),  # Cap at 0.95 to avoid perfect scores
                'match_percentage': int(min(score, 0.95) * 100),
                'matched_terms': matched_terms,
                'description': " ".join(description_points),
                'columns': dataset.get('columns', [])
            }
            
            individual_matches.append(match)
    
    # Group matches by title
    title_grouped_matches = defaultdict(list)
    for match in individual_matches:
        title_grouped_matches[match['title']].append(match)
    
    # Create combined matches with datasets of the same title grouped together
    grouped_matches = []
    for title, matches_list in title_grouped_matches.items():
        # Calculate average score
        avg_score = sum(match['score'] for match in matches_list) / len(matches_list)
        avg_percentage = int(avg_score * 100)
        
        # Collect all unique matched terms
        all_matched_terms = set()
        for match in matches_list:
            if match.get('matched_terms'):
                all_matched_terms.update(match.get('matched_terms', []))
        
        # Collect all academic years
        academic_years = [match['academic_year'] for match in matches_list]
        academic_years.sort()  # Sort years for better display
        
        # Collect all references
        references = [match['reference'] for match in matches_list]
        
        # Combine descriptions if they're different
        descriptions = []
        for match in matches_list:
            if match.get('description') and match.get('description') not in descriptions:
                descriptions.append(match.get('description'))
        
        combined_match = {
            'title': title,
            'score': avg_score,
            'match_percentage': avg_percentage,
            'academic_years': academic_years,
            'references': references,
            'matched_terms': list(all_matched_terms),
            'descriptions': descriptions,
            'matches': matches_list,  # Include individual matches for reference
            'columns': matches_list[0].get('columns', []) if matches_list else []
        }
        
        grouped_matches.append(combined_match)
    
    # Sort by score (highest first)
    grouped_matches.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Found {len(individual_matches)} individual matching datasets grouped into {len(grouped_matches)} dataset groups using regex fallback")
    return grouped_matches

@csrf_exempt
@require_http_methods(["GET", "POST"])
def ai_dataset_details(request):
    """
    View a specific dataset selected from the AI dashboard.
    GET: Displays the dataset details page with data passed as URL parameter
    POST: Handles form submission with dataset data or API endpoint to get dataset details
    """
    import json
    import logging
    import os
    
    # Get the logger for this module
    logger = logging.getLogger(__name__)
    
    # Handle GET request to display the dataset details page
    if request.method == "GET":
        try:
            # Get data from URL parameter
            data_json = request.GET.get('data', '{}')
            dataset_data = json.loads(data_json)
            
            logger.info(f"Displaying AI dataset details for: {dataset_data.get('dataset_title', 'Unknown dataset')}")
            
            # Extract years from the file results
            years = set()
            for result in dataset_data.get('file_results', []):
                if 'academic_year' in result and result['academic_year']:
                    years.add(result['academic_year'])
            
            # Render the dataset details template with the data
            return render(request, 'core/ai_dataset_details.html', {
                'dataset_data': dataset_data,
                'dataset_title': dataset_data.get('dataset_title', 'Dataset Details'),
                'institutions': dataset_data.get('institutions', []),
                'original_institutions': dataset_data.get('original_institutions', []),
                'corrected_query': dataset_data.get('corrected_query', ''),
                'query': dataset_data.get('query', ''),
                'file_results': dataset_data.get('file_results', []),
                'has_data': dataset_data.get('has_data', False),
                'years': sorted(list(years))
            })
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding dataset data JSON: {str(e)}")
            return render(request, 'core/error.html', {
                'error': 'Invalid dataset data format',
                'details': str(e)
            })
        except Exception as e:
            logger.error(f"Error displaying dataset details: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return render(request, 'core/error.html', {
                'error': 'Error displaying dataset details',
                'details': str(e)
            })
    
    # Handle POST request for dataset details
    try:
        # Check if this is a form submission from the dashboard
        dataset_data_json = request.POST.get('dataset_data')
        
        if dataset_data_json:
            # This is a form submission with dataset_data
            try:
                dataset_data = json.loads(dataset_data_json)
                
                # Extract years from the file results
                years = set()
                for result in dataset_data.get('file_results', []):
                    if 'academic_year' in result and result['academic_year']:
                        years.add(result['academic_year'])
                
                # Render the dataset details template with the data
                return render(request, 'core/ai_dataset_details.html', {
                    'dataset_data': dataset_data,
                    'dataset_title': dataset_data.get('dataset_title', 'Dataset Details'),
                    'institutions': dataset_data.get('institutions', []),
                    'original_institutions': dataset_data.get('original_institutions', []),
                    'corrected_query': dataset_data.get('corrected_query', ''),
                    'query': dataset_data.get('query', ''),
                    'file_results': dataset_data.get('file_results', []),
                    'has_data': dataset_data.get('has_data', False),
                    'years': sorted(list(years))
                })
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding POST dataset_data JSON: {str(e)}")
                return render(request, 'core/error.html', {
                    'error': 'Invalid dataset data format',
                    'details': str(e)
                })
            
        # This is a regular API request for dataset details
        # Get the dataset information from the request
        data = json.loads(request.body)
        dataset_title = data.get('dataset_title', '')
        dataset_references = data.get('dataset_references', [])
        institutions = data.get('institutions', [])
        original_institutions = data.get('original_institutions', [])
        query = data.get('query', '')
        corrected_query = data.get('corrected_query', '')
        
        logger.info(f"AI Dataset details request for: {dataset_title}")
        logger.info(f"References: {dataset_references}")
        logger.info(f"Institutions: {institutions}")
        logger.info(f"Original institutions: {original_institutions}")
        
        if not dataset_title or not dataset_references:
            return JsonResponse({'error': 'Dataset information is missing'}, status=400)
        
        # Collect all institutions to search for, including original and corrected versions
        all_institutions = set()
        if institutions:
            all_institutions.update(institutions)
        if original_institutions:
            all_institutions.update(original_institutions)
        
        # If no institutions were provided, include University of Leicester by default
        if not all_institutions:
            all_institutions.add('University of Leicester')
        
        # Convert to list for easier handling
        all_institutions = list(all_institutions)
        logger.info(f"Searching for institutions: {all_institutions}")
        
        # Process each reference to get the complete data
        file_results = []
        years = set()
        
        for reference in dataset_references:
            file_path = os.path.join(CLEANED_FILES_DIR, reference)
            
            # Extract academic year from reference name
            year_match = re.search(r'(20\d{2}).{0,4}(20\d{2}|[0-9]{2})', reference)
            academic_year = None
            if year_match:
                if len(year_match.group(2)) == 2:
                    academic_year = f"{year_match.group(1)}/{year_match.group(2)}"
                else:
                    academic_year = f"{year_match.group(1)}/{year_match.group(2)[2:4]}"
            
            # Add to years set if valid
            if academic_year:
                years.add(academic_year)
                
            try:
                # Get all the rows for the specified institutions using fuzzy matching
                preview = get_csv_preview(
                    file_path=file_path,
                    max_rows=None,  # Get all matching rows, not just a preview
                    institution=','.join(all_institutions)
                )
                
                if preview:
                    preview['academic_year'] = academic_year
                    preview['reference'] = reference
                    file_results.append(preview)
            except Exception as e:
                logger.error(f"Error processing file {reference}: {str(e)}")
                # Add error information to the results
                file_results.append({
                    'academic_year': academic_year,
                    'reference': reference,
                    'error': str(e),
                    'columns': ['Error'],
                    'data': [[f"Could not process file: {str(e)}"]]
                })
        
        # Prepare response with dataset details and file results
        response_data = {
            'dataset_title': dataset_title,
            'institutions': institutions,
            'original_institutions': original_institutions,
            'query': query,
            'corrected_query': corrected_query,
            'file_results': file_results,
            'total_files': len(dataset_references),
            'processed_files': len(file_results),
            'has_data': any(result.get('data') for result in file_results),
            'years': sorted(list(years))
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error in AI dataset details: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return JsonResponse({'error': str(e)}, status=500)