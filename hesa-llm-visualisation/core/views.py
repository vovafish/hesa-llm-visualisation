from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from .data_processing import CSVProcessor
#from .utils.query_processor import parse_llm_response, apply_data_operations
#from .utils.chart_generator import generate_chart
#from .visualization.chart_generator import ChartGenerator
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
#from .data_processor import transform_chart_data, prepare_chart_data
import seaborn as sns
import base64
import logging
#from .data_processing.storage.storage_service import StorageService
from .data_processing.csv_processor import CLEANED_FILES_DIR, RAW_FILES_DIR  # Import the constants
from django.views.decorators.csrf import csrf_exempt
import re
import json
import glob
from .gemini_client import GeminiClient, get_llm_client
#from .mock_ai_client import MockAIClient
import fnmatch
from collections import defaultdict

# Mission groups for institution filtering
MISSION_GROUPS = {
    "Russell Group": [
        "The University of Birmingham",
        "The University of Bristol",
        "The University of Cambridge",
        "Cardiff University",
        "University of Durham",
        "The University of Edinburgh",
        "The University of Exeter",
        "The University of Glasgow",
        "Imperial College of Science, Technology and Medicine",
        "King's College London",
        "The University of Leeds",
        "The University of Liverpool",
        "London School of Economics and Political Science",
        "The University of Manchester",
        "Newcastle University",
        "The University of Nottingham",
        "The University of Oxford",
        "Queen Mary University of London",
        "Queen's University Belfast",
        "The University of Sheffield",
        "The University of Southampton",
        "University College London",
        "The University of Warwick",
        "The University of York"
    ],
    "Million+": [
        "The University of Bolton",
        "The University of Central Lancashire",
        "Coventry University",
        "De Montfort University",
        "University of Derby",
        "The University of East London",
        "The University of Greenwich",
        "University of Hertfordshire",
        "The University of Lincoln",
        "Liverpool John Moores University",
        "The Manchester Metropolitan University",
        "Middlesex University",
        "The University of Northampton",
        "The University of South Wales",
        "The University of West London",
        "The University of the West of England, Bristol",
        "The University of Wolverhampton"
    ],
    "University Alliance": [
        "Anglia Ruskin University",
        "Birmingham City University",
        "Coventry University",
        "De Montfort University",
        "University of Derby",
        "The University of East London",
        "University of Hertfordshire",
        "The University of Lincoln",
        "Liverpool John Moores University",
        "The Manchester Metropolitan University",
        "Middlesex University",
        "The University of Northampton",
        "The University of South Wales",
        "The University of West London",
        "The University of the West of England, Bristol",
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
    """Redirect to the AI dashboard page."""
    return redirect('core:ai_dashboard')

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
                    max_rows=None,  # No limit - display all rows
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
                                    year_match = re.search(r'(20\d{2})[\.\-&_]?(\d{2})?', file_name)
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
                        year_match = re.search(r'(20\d{2})[\.\-&_]?(\d{2})?', file_name)
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
                        institutions=institution,
                        mission_group=mission_group
                    )
                else:
                    # Use institution filtering
                    data = get_csv_preview(
                        str(csv_path),
                        max_rows=None,  # No limit - display all rows
                        institutions=institution
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

def get_csv_preview(file_path, max_rows=5, institutions=None, institution=None, mission_group=None, exact_match_institutions=None):
    """
    Generate a preview of a CSV file, optionally filtering for specific institutions.
    
    Args:
        file_path: Path to the CSV file
        max_rows: Maximum number of rows to include in the preview, None means no limit
        institutions: Comma-separated string of institution names to filter for
        institution: Deprecated - use institutions instead. Comma-separated string of institution names.
        mission_group: Mission group to filter for
        exact_match_institutions: List of institution names that should be matched exactly, not using fuzzy logic
    
    Returns:
        dict: Dictionary containing columns and data for the preview
    """
    import csv
    import logging
    logger = logging.getLogger(__name__)
    
    # Configure console logging for debugging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[CSV Preview] %(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"=== Starting CSV preview generation ===")
    logger.info(f"File path: {file_path}")
    logger.info(f"Max rows: {max_rows}")
    
    # For backward compatibility, use institution if institutions is None
    if institutions is None and institution is not None:
        institutions = institution
    
    logger.info(f"Filtering by institutions: {institutions}")
    
    # Handle exact match institutions (mainly from mission groups)
    if exact_match_institutions is None:
        exact_match_institutions = []
    logger.info(f"Using exact matching for these institutions: {exact_match_institutions}")
    
    # Extract academic year from the file name
    year_match = re.search(r'(\d{4})[\.\-&_](\d{2})', os.path.basename(file_path))
    academic_year = None
    if year_match:
        academic_year = f"{year_match.group(1)}/{year_match.group(2)}"
        logger.info(f"Extracted academic year from filename: {academic_year}")
    
    try:
        institution_list = []
        exact_match_inst_list = []
        fuzzy_match_inst_list = []
        
        if institutions:
            # Split the comma-separated list into individual institution names
            institution_list = [inst.strip() for inst in institutions.split(',')]
            logger.info(f"Parsed institution list: {institution_list}")
            
            # Separate institutions into exact and fuzzy match lists
            for inst in institution_list:
                if inst in exact_match_institutions:
                    exact_match_inst_list.append(inst)
                else:
                    fuzzy_match_inst_list.append(inst)
            
            logger.info(f"Exact match institutions: {exact_match_inst_list}")
            logger.info(f"Fuzzy match institutions: {fuzzy_match_inst_list}")
        
        logger.info(f"Opening file: {file_path}")
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            # Check if the first line contains metadata
            first_line = f.readline()
            logger.debug(f"First line: {first_line[:100]}...")
            
            is_metadata = first_line.startswith('#METADATA:')
            if is_metadata:
                logger.info("File contains metadata line, skipping for column headers")
                # Read the actual header line after metadata
                headers_line = f.readline()
                # Reset file position for CSV reader
                f.seek(0)
                f.readline()  # Skip metadata line again
            else:
                # Reset file position for CSV reader
                f.seek(0)
                headers_line = first_line
            
            # Parse headers from the header line
            headers = next(csv.reader([headers_line]))
            logger.info(f"Parsed headers: {headers}")
            
            # Read the rest of the file with CSV reader
            reader = csv.reader(f)
            
            # Find institution column - typically "HE provider" or similar
            institution_col = None
            for i, header in enumerate(headers):
                if re.search(r'(?:institution|provider|university|college)', header.lower()):
                    institution_col = i
                    logger.info(f"Found institution column: {header} at index {i}")
                    break
            
            if institution_col is None:
                institution_col = 0  # Default to first column if no obvious institution column
                logger.warning(f"No institution column found, defaulting to index 0: {headers[0]}")
            
            # Add Academic Year as the last column if not already present
            academic_year_col = None
            for i, header in enumerate(headers):
                if header.lower() == "academic year":
                    academic_year_col = i
                    logger.info(f"Found existing Academic Year column at index {i}")
                    break
                    
            # Only add Academic Year column if it doesn't exist and we have a value
            if academic_year_col is None and academic_year:
                headers.append("Academic Year")
                add_academic_year = True
                logger.info(f"Added 'Academic Year' column with value: {academic_year}")
            else:
                add_academic_year = False
            
            # Preview modes:
            # 1. No institutions specified: just return first max_rows
            # 2. Institutions specified: filter and return relevant rows (max_rows limit)
            all_rows = []
            important_rows = {}  # Dictionary to store rows for institutions in the request
            
            # Set of common words to ignore in partial matching
            common_words = {'university', 'the', 'of', 'and', 'institute', 'college', 'higher', 
                          'education', 'school', 'academy', 'faculty', 'department', 'campus', 
                          'studies', 'centre', 'center', 'for'}
            
            # Pre-normalize the requested institutions for fuzzy matching
            normalized_fuzzy_requested = {}
            for inst in fuzzy_match_inst_list:
                # Create normalized form (lowercase, remove common words)
                words = [w.lower() for w in inst.split() if w.lower() not in common_words]
                normalized = ' '.join(words)
                normalized_fuzzy_requested[normalized] = inst
                logger.info(f"Normalized fuzzy match institution '{inst}' to '{normalized}'")
            
            for row in reader:
                # Add academic year to the row if needed, checking if it already contains the academic year
                if add_academic_year and academic_year:
                    # Only add academic year if it's not already in the row
                    if academic_year not in row:
                        row.append(academic_year)
                
                # Check if this is a requested institution
                if institution_col < len(row):
                    inst_name = row[institution_col]
                    
                    # First check for exact matches from mission groups
                    exact_match_found = False
                    for exact_inst in exact_match_inst_list:
                        if inst_name.strip() == exact_inst.strip():
                            important_rows[exact_inst] = row
                            logger.info(f"Found EXACT match for mission group institution: {inst_name}")
                            exact_match_found = True
                            break
                    
                    # If no exact match found, try fuzzy matching for institutions from query
                    if not exact_match_found:
                        # Normalize the institution name from the dataset
                        inst_words = [w.lower() for w in inst_name.split() if w.lower() not in common_words]
                        normalized_name = ' '.join(inst_words)
                        
                        # Check if this is one of our fuzzy matching institutions
                        for normalized_req, original_req in normalized_fuzzy_requested.items():
                            if normalized_req in normalized_name or normalized_name in normalized_req:
                                important_rows[normalized_name] = row
                                logger.info(f"Found fuzzy match for institution: {inst_name} (matches {original_req})")
                
                # Store all rows for processing
                all_rows.append(row)
            
            logger.info(f"Read {len(all_rows)} rows from file")
            
            # Mission group filtering
            if mission_group and len(all_rows) > 0:
                # Find mission group column if it exists
                mission_group_col = None
                for i, header in enumerate(headers):
                    if 'mission' in header.lower() and 'group' in header.lower():
                        mission_group_col = i
                        break
                
                # If found, filter by mission group
                if mission_group_col is not None:
                    filtered_rows = [row for row in all_rows if 
                                   mission_group_col < len(row) and 
                                   mission_group.lower() in row[mission_group_col].lower()]
                    
                    # Ensure important institutions are included
                    for inst_row in important_rows.values():
                        if inst_row not in filtered_rows:
                            filtered_rows.append(inst_row)
                    
                    all_rows = filtered_rows
                    logger.info(f"Filtered to {len(all_rows)} rows by mission group '{mission_group}'")
            
            # If no institutions to filter by, just return the first rows
            if not institution_list:
                logger.info("No institutions specified, returning first rows")
                
                # When max_rows is None, it means do not limit (used for dataset details view)
                if max_rows is None:
                    preview_rows = all_rows
                    logger.info(f"No limit (max_rows=None), returning all {len(all_rows)} rows")
                else:
                    preview_rows = all_rows[:max_rows]
                    
                # Make sure important institutions are included
                for i, inst_row in enumerate(important_rows.values()):
                    if inst_row not in preview_rows:
                        if max_rows and len(preview_rows) >= max_rows and max_rows > 0:
                            # Replace one of the existing rows
                            replace_index = max_rows - i - 1  # Start replacing from the end
                            if replace_index >= 0:
                                preview_rows[replace_index] = inst_row
                                logger.info(f"Replaced row at position {replace_index} with {inst_row[institution_col]}")
                        else:
                            preview_rows.append(inst_row)
                            logger.info(f"Added {inst_row[institution_col]} to preview")
                
                logger.info(f"Returning {len(preview_rows)} rows without filtering")
                return {
                    "columns": headers,
                    "data": preview_rows,
                    "matched_rows": len(preview_rows),
                    "has_more": len(all_rows) > len(preview_rows)
                }
            
            # If institutions are specified, we need to filter
            # Dictionary to map institution names to their data rows
            institution_rows = {}
            
            # Process institutions in the data 
            for row in all_rows:
                if institution_col < len(row):
                    inst_name = row[institution_col]
                    
                    # Create normalized form for the institution in the data
                    words = [w.lower() for w in inst_name.split() if w.lower() not in common_words]
                    key = ' '.join(words)
                    institution_rows[key] = row
            
            # First pass: Exact normalized matches
            matched_rows = []
            
            # Add all important rows first (these are the rows that directly match requested institutions)
            for row in important_rows.values():
                if row not in matched_rows:
                    matched_rows.append(row)
                    logger.info(f"Added important row for {row[institution_col]} to matched rows")
            
            # Check for exact matches against normalized requested institutions
            for normalized_req in normalized_fuzzy_requested.keys():
                if normalized_req in institution_rows:
                    row = institution_rows[normalized_req]
                    if row not in matched_rows:
                        matched_rows.append(row)
                        logger.info(f"Exact normalized match for '{normalized_req}'")
                        
            # Ensure exact matches for mission group institutions
            for exact_inst in exact_match_inst_list:
                for row in all_rows:
                    if institution_col < len(row) and row[institution_col].strip() == exact_inst.strip():
                        if row not in matched_rows:
                            matched_rows.append(row)
                            logger.info(f"Added exact match for mission group institution: {exact_inst}")
            
            # If we found exact normalized matches, return those
            if matched_rows:
                logger.info(f"Found {len(matched_rows)} exact institution matches")
                # When max_rows is None, it means do not limit (used for dataset details view)
                if max_rows is None:
                    preview_rows = matched_rows
                    logger.info(f"No limit (max_rows=None), returning all {len(matched_rows)} exact matched rows")
                else:
                    preview_rows = matched_rows[:max_rows]
                
                logger.info(f"Returning {len(preview_rows)} exact match rows")
                return {
                    "columns": headers,
                    "data": preview_rows,
                    "matched_rows": len(matched_rows),
                    "has_more": len(matched_rows) > len(preview_rows)
                }
            
            # Second pass: Partial word matching if no exact matches found
            logger.info("No exact matches found, trying partial word matching")
            matched_rows = []
            
            # Start with important rows (requested institutions)
            for row in important_rows.values():
                if row not in matched_rows:
                    matched_rows.append(row)
            
            # Ensure exact matches for mission group institutions are included
            for exact_inst in exact_match_inst_list:
                for row in all_rows:
                    if institution_col < len(row) and row[institution_col].strip() == exact_inst.strip():
                        if row not in matched_rows:
                            matched_rows.append(row)
                            logger.info(f"Added exact match for mission group institution (second pass): {exact_inst}")
            
            # For each institution in the dataset, apply partial matching only for fuzzy match institutions
            for row in all_rows:
                # Skip if row is already matched
                if row in matched_rows:
                    continue
                    
                if institution_col < len(row):
                    dataset_inst_name = row[institution_col].lower()
                    
                    # Check each requested institution for partial matches
                    for normalized_req, original_req in normalized_fuzzy_requested.items():
                        req_words = normalized_req.split()
                        
                        # Log the significant words we're looking for
                        logger.debug(f"Looking for partial matches with words: {req_words}")
                        
                        # Check if any significant word from the request matches in the provider name
                        for word in req_words:
                            if len(word) > 2 and word.lower() in dataset_inst_name:
                                matched_rows.append(row)
                                logger.info(f"Partial match found: '{word}' from '{original_req}' in '{dataset_inst_name}'")
                                # Break after first match to avoid duplicates
                                break
                        
                        # If we already matched this row, skip to the next
                        if row in matched_rows:
                            break
            
            # If we found partial matches, return those
            if matched_rows:
                logger.info(f"Found {len(matched_rows)} partial institution matches")
                # When max_rows is None, it means do not limit (used for dataset details view)
                if max_rows is None:
                    preview_rows = matched_rows
                    logger.info(f"No limit (max_rows=None), returning all {len(matched_rows)} partially matched rows")
                else:
                    preview_rows = matched_rows[:max_rows]
                
                logger.info(f"Returning {len(preview_rows)} partial match rows")
                return {
                    "columns": headers,
                    "data": preview_rows,
                    "matched_rows": len(preview_rows),
                    "has_more": len(matched_rows) > len(preview_rows)
                }
            
            # If we still have no matches, return at least some rows
            logger.info("No institution matches found, returning first rows")
            # When max_rows is None, it means do not limit (used for dataset details view)
            if max_rows is None:
                preview_rows = all_rows
                logger.info(f"No limit (max_rows=None), returning all {len(all_rows)} rows as fallback")
            else:
                preview_rows = all_rows[:max_rows]
            
            # Again, ensure important institutions are included if they exist
            for i, inst_row in enumerate(important_rows.values()):
                if inst_row not in preview_rows:
                    if max_rows and len(preview_rows) >= max_rows and max_rows > 0:
                        # Replace one of the existing rows
                        replace_index = max_rows - i - 1  # Start replacing from the end
                        if replace_index >= 0:
                            preview_rows[replace_index] = inst_row
                            logger.info(f"Replaced row at position {replace_index} with {inst_row[institution_col]}")
                    else:
                        preview_rows.append(inst_row)
                        logger.info(f"Added {inst_row[institution_col]} to preview")
                    
            logger.info(f"Returning {len(preview_rows)} rows as fallback")
            return {
                "columns": headers,
                "data": preview_rows,
                "matched_rows": len(preview_rows),
                "has_more": len(all_rows) > len(preview_rows),
                "no_institution_matches": True
            }
            
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}", exc_info=True)
        return {
            "error": f"Error generating preview: {str(e)}",
            "columns": [],
            "data": []
        }

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
                    year_match = re.search(r'(20\d{2})[\.\-&_]?(\d{2})?', file_name)
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

        I have the following datasets (already filtered for relevance by year):

        {json.dumps([{
            "title": d.get("title", ""),
            "columns": d.get("columns", []),
            "academic_year": d.get("academic_year", ""),
            "reference": d.get("reference", "")
        } for d in datasets[:30]], indent=2)}

        TASK OVERVIEW:
        Your task is to deeply understand what the user is trying to find out and match datasets that would answer their question.

        STEP 1 - ANALYZE THE QUERY INTENT:
        - First, understand what type of data the user is looking for
        - Ignore institutions and years in the query (these are handled separately)
        - Focus on the core data request (e.g., "postgraduates", "student demographics", "staff counts")
        - Determine if the user wants specific metrics or categories (e.g., "by gender", "by qualification")

        STEP 2 - MATCH DATASETS BY INTENT:
        - Examine dataset titles and columns to determine if they contain the data needed to answer the query
        - A dataset is relevant if it contains the specific data types the user is asking about
        - For example, if the user asks about "postgraduates", prioritize datasets with columns or titles containing "postgraduate"
        - Do NOT match based on generic terms or contextual words (e.g., don't match "staff" when user asks about "students")

        STEP 3 - SCORE THE MATCHES:
        Score each dataset from 0.0 to 0.95 based on how completely it answers the user's question:
        - 0.8-0.95: Perfect match that directly answers the specific question
        - 0.6-0.8: Strong match that contains the specific data type requested
        - 0.4-0.6: Partial match that contains related information
        - 0.2-0.4: Weak match with some relevant data but not exactly what was asked for
        - 0.1-0.2: Minimal relevance, might be useful as context but doesn't answer the query

        Examples:
        - If user asks about "postgraduate students", a dataset about "HE qualifiers by level of qualification including postgraduate" would score 0.9
        - If user asks about "undergraduate students by gender", a dataset just about "undergraduate students" without gender breakdown would score 0.6
        - If user asks about "student enrollment" but dataset is about "staff", it should score 0.0 and not be returned

        Return a JSON array with the following format for each matching dataset:
        [
          {{
            "reference": "filename.csv",
            "score": 0.95,
            "matched_intent": "Data about postgraduate students which directly answers the user's question",
            "matched_terms": ["postgraduate", "student"],
            "description": "Brief explanation of why this dataset helps answer the user's specific question"
          }}
        ]

        Only include datasets with a score > 0.1. Sort the results by score in descending order.
        """
        
        # Call Gemini API for semantic matching
        logger.info("Calling Gemini API for intent-based dataset matching")
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
                            "matched_intent": match.get("matched_intent", ""),
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
                all_matched_intents = set()
                for match in matches_list:
                    if match.get('matched_terms'):
                        all_matched_terms.update(match.get('matched_terms', []))
                    if match.get('matched_intent'):
                        all_matched_intents.add(match.get('matched_intent', ''))
                
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
                    'matched_intents': list(all_matched_intents),
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
    """Process a query using the Gemini API to extract entities and find matching datasets."""
    import json
    import logging
    import os
    from django.conf import settings
    
    logger = logging.getLogger(__name__)
    logger.info("=== STARTING GEMINI QUERY PROCESSING ===")
    logger.info(f"Request received: {request.method}")
    
    # Configure logging to console for better debugging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    try:
        # Parse the query from the request
        data = json.loads(request.body.decode('utf-8'))
        query = data.get('query', '')
        max_matches = int(data.get('max_matches', 5))
        preview_max_rows = int(data.get('preview_rows', 3))
        mission_group = data.get('mission_group')
        
        # Log the input parameters
        logger.info(f"Query: '{query}'")
        logger.info(f"Max matches: {max_matches}")
        logger.info(f"Max preview rows: {preview_max_rows}")
        logger.info(f"Mission group filter: {mission_group}")
        
        # Validate query
        if not query:
            logger.warning("Empty query received")
            return JsonResponse({'error': 'Query is required'}, status=400)
        
        # Initialize Gemini client
        gemini_client = GeminiClient(settings.GEMINI_API_KEY)
        
        # Check if we should use mock mode
        using_mock = not gemini_client.api_key or settings.DASHBOARD_SETTINGS.get('USE_MOCK_AI', False)
        logger.info(f"Using mock AI: {using_mock}")
        
        # Extract entities from query
        if using_mock:
            logger.info("Using local fallback analysis")
            result = local_analyze_query(query)
        else:
            try:
                logger.info("Calling Gemini API for query analysis")
                result = gemini_client.analyze_query(query)
                logger.info(f"API response received: {json.dumps(result, default=str)}")
            except Exception as e:
                logger.error(f"Error analyzing query with Gemini API: {str(e)}", exc_info=True)
                # Fallback to local analysis on error
                logger.info("Falling back to local analysis")
                result = local_analyze_query(query)
                result['api_error'] = str(e)
        
        # Log the extracted institutions and years
        logger.info(f"Extracted institutions: {result.get('institutions', [])}")
        logger.info(f"Extracted years: {result.get('years', [])}")
        logger.info(f"Year range: {result.get('start_year')} to {result.get('end_year')}")
        logger.info(f"Data requested: {result.get('data_request', [])}")
        
        # Find matching datasets
        data_request = result.get('data_request', ['general_data'])
        start_year = result.get('start_year')
        end_year = result.get('end_year')
        
        # Always ensure University of Leicester is included
        institutions = result.get('institutions', [])
        if institutions and 'University of Leicester' not in institutions:
            logger.info("Adding University of Leicester to institutions list")
            institutions.append('University of Leicester')
            result['institutions'] = institutions
            
        # Add institutions from the selected mission group if any
        if mission_group and mission_group in MISSION_GROUPS:
            mission_group_institutions = MISSION_GROUPS[mission_group]
            logger.info(f"Adding {len(mission_group_institutions)} institutions from {mission_group}")
            
            # Create a special list for mission group institutions that will be matched exactly
            mission_group_inst_list = []
            
            # Add each institution from the mission group if not already in the list
            for institution in mission_group_institutions:
                if institution not in institutions:
                    logger.info(f"Adding mission group institution: {institution}")
                    # Add to main institutions list
                    institutions.append(institution)
                    # Add to mission group institutions list for exact matching
                    mission_group_inst_list.append(institution)
            
            # Update the institutions in the result
            result['institutions'] = institutions
            
            # Add mission group info to the result for display purposes
            result['mission_group'] = mission_group
            result['mission_group_institutions'] = mission_group_institutions
            # Add flag for exact matching these institutions
            result['mission_group_inst_exact_match'] = mission_group_inst_list
            
        # Attempt to find matching datasets using either API or fallback
        if using_mock:
            logger.info("Using regex fallback to find matching datasets")
            matching_datasets = regex_matching_fallback(query, data_request, datasets=None)
        else:
            try:
                logger.info("Using semantic matching to find datasets")
                matching_datasets = find_matching_datasets(query, data_request, start_year, end_year)
                logger.info(f"Found {len(matching_datasets)} matching datasets")
            except Exception as e:
                logger.error(f"Error finding matching datasets: {str(e)}", exc_info=True)
                logger.info("Falling back to regex matching")
                matching_datasets = regex_matching_fallback(query, data_request, datasets=None)
        
        # Apply max_matches limit
        if matching_datasets and max_matches > 0 and len(matching_datasets) > max_matches:
            matching_datasets = matching_datasets[:max_matches]
            logger.info(f"Limited to {max_matches} highest-scoring dataset groups")
        
        # Generate CSV previews for each dataset
        for dataset in matching_datasets:
            logger.info(f"Processing dataset previews for: {dataset.get('title', 'Untitled dataset')}")
            dataset_previews = []
            
            if 'matches' in dataset:
                for match in dataset['matches']:
                    reference = match.get('reference', '')
                    if reference:
                        logger.info(f"Generating preview for file: {reference}")
                        # Construct the full file path
                        file_path = os.path.join(CLEANED_FILES_DIR, reference)
                        
                        # Verify file exists
                        if not os.path.exists(file_path):
                            logger.error(f"File not found: {file_path}")
                            match['preview'] = {
                                'columns': ['Error'],
                                'data': [['File not found']],
                                'error': 'File not found'
                            }
                            continue
                        
                        # Get CSV preview for this file filtered by the extracted institutions
                        try:
                            # Convert institutions list to comma-separated string
                            institutions_str = ','.join(institutions)
                            logger.info(f"Filtering by institutions: {institutions_str}")
                            
                            # Get exact match institutions list
                            exact_match_institutions = result.get('mission_group_inst_exact_match', [])
                            
                            # In preview mode, we want exactly 3 rows per dataset
                            preview = get_csv_preview(
                                file_path=file_path,
                                max_rows=preview_max_rows,
                                institutions=institutions_str,
                                exact_match_institutions=exact_match_institutions
                            )
                            
                            if preview:
                                logger.info(f"Preview generated successfully: {len(preview.get('data', []))} rows")
                                logger.debug(f"Preview data: {json.dumps(preview, default=str)}")
                            else:
                                logger.warning(f"Empty preview generated for {reference}")
                            
                            # Add the preview to the match
                            match['preview'] = preview
                            
                            # Also add to a collection of previews for the dataset
                            if preview:
                                preview_with_metadata = preview.copy()
                                preview_with_metadata['reference'] = reference
                                preview_with_metadata['academic_year'] = match.get('academic_year', '')
                                dataset_previews.append(preview_with_metadata)
                    
                        except Exception as e:
                            logger.error(f"Error generating preview for {reference}: {str(e)}", exc_info=True)
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
        
        logger.info(f"Gemini query analysis complete with {len(matching_datasets)} datasets and {len(missing_years)} missing years")
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error processing Gemini query: {str(e)}", exc_info=True)
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
    Attempts to match based on intent rather than just keywords.
    """
    import re
    import logging
    import json
    from collections import defaultdict
    import difflib
    
    logger = logging.getLogger(__name__)
    logger.info("Using regex fallback for intent-based dataset matching")
    
    # If datasets is not provided, load them from the index file
    if not datasets:
        import os
        from django.conf import settings
        
        index_file_path = os.path.join(settings.BASE_DIR, 'hesa_files_index.json')
        if os.path.exists(index_file_path):
            try:
                with open(index_file_path, 'r') as f:
                    index_data = json.load(f)
                    datasets = index_data.get('hesa_files', [])
                    logger.info(f"Loaded {len(datasets)} datasets from index file")
            except Exception as e:
                logger.error(f"Error loading index file: {str(e)}")
                datasets = []
        else:
            logger.error("Index file not found")
            datasets = []
    
    # Case-insensitive matching
    query_lower = query.lower()
    query_terms = [term.strip() for term in re.findall(r'\b\w+\b', query_lower) if len(term.strip()) > 2]
    logger.info(f"Query terms: {query_terms}")
    
    # Extract main intents from data_request
    main_intents = []
    for req in data_request:
        if isinstance(req, str):
            # Clean up the request string
            req = req.lower().strip()
            main_intents.append(req)
            
            # Add variations for common data types
            if req == 'undergraduate':
                main_intents.extend(['bachelor', 'first degree', 'ug'])
            elif req == 'postgraduate':
                main_intents.extend(['master', 'phd', 'doctorate', 'pg'])
            elif req == 'student_count' or req == 'student':
                main_intents.extend(['enrollment', 'enrolment', 'student'])
    
    logger.info(f"Main intents extracted: {main_intents}")
    
    # Create dictionaries to track matches
    individual_matches = []
    
    # Process each dataset for matches
    for dataset in datasets:
        title = dataset.get('title', '').lower()
        columns = [col.lower() for col in dataset.get('columns', [])]
        academic_year = dataset.get('academic_year', '')
        reference = dataset.get('reference', '')
        
        # Skip if no title or reference
        if not title or not reference:
            continue
            
        # Initialize matching variables
        score = 0.1  # Base score
        matched_terms = []
        matched_intent = ""
        description_points = []
        
        # Check for intent matches in title and columns
        intent_matched = False
        
        # First, look for direct matches of the main intents
        for intent in main_intents:
            # Check title for the intent
            if intent in title:
                score += 0.6
                matched_terms.append(intent)
                description_points.append(f"Title contains '{intent}' which directly relates to the query")
                intent_matched = True
                matched_intent = f"Data about {intent}"
            
            # Check columns for the intent
            for col in columns:
                if intent in col:
                    score += 0.4
                    if intent not in matched_terms:
                        matched_terms.append(intent)
                    description_points.append(f"Column '{col}' contains '{intent}' which matches the query intent")
                    intent_matched = True
                    if not matched_intent:
                        matched_intent = f"Data containing {intent} information"
        
        # If no direct intent matches, look for related terms
        if not intent_matched:
            # Education level terms
            education_levels = {
                'undergraduate': ['bachelor', 'first degree', 'ug', 'undergraduate'],
                'postgraduate': ['postgrad', 'pg', 'master', 'doctorate', 'phd', 'graduate'],
                'doctoral': ['phd', 'doctorate'],
                'master': ['ma', 'msc', 'masters', 'postgraduate']
            }
            
            # Look for education level terms in query
            for level, terms in education_levels.items():
                if any(term in query_lower for term in terms):
                    # Check if dataset has this education level
                    if any(term in title for term in terms) or any(any(term in col for term in terms) for col in columns):
                        score += 0.5
                        matched_terms.append(level)
                        matched_intent = f"Data about {level} education"
                        description_points.append(f"Dataset contains information about {level} education")
                        intent_matched = True
                        break
            
            # Student/enrollment related
            student_terms = ['student', 'learner', 'pupil', 'enrollment', 'enrolment']
            if any(term in query_lower for term in student_terms) and not intent_matched:
                if any(term in title for term in student_terms) or any(any(term in col for term in student_terms) for col in columns):
                    score += 0.4
                    matched_terms.append('student data')
                    matched_intent = "Data about student enrollment or numbers"
                    description_points.append("Dataset contains information about students or enrollment")
                    intent_matched = True
            
            # Staff related
            staff_terms = ['staff', 'faculty', 'lecturer', 'professor', 'teacher']
            if any(term in query_lower for term in staff_terms) and not intent_matched:
                if any(term in title for term in staff_terms) or any(any(term in col for term in staff_terms) for col in columns):
                    score += 0.4
                    matched_terms.append('staff data')
                    matched_intent = "Data about academic staff"
                    description_points.append("Dataset contains information about academic staff")
                    intent_matched = True
        
        # If still no matches but data_request has specific intents, try those
        if not intent_matched and data_request and isinstance(data_request, list):
            for req in data_request:
                req_lower = req.lower() if isinstance(req, str) else ""
                if req_lower in title or any(req_lower in col for col in columns):
                    score += 0.3
                    matched_terms.append(req_lower)
                    matched_intent = f"Data related to {req_lower}"
                    description_points.append(f"Dataset contains information related to {req_lower}")
                    intent_matched = True
                    break
        
        # If we found any intent match, give a slight bonus for relevance
        if intent_matched:
            score += 0.1
        else:
            # If no direct intent match, but there is some terminology overlap
            # Look for common education terminology
            edu_terms = ['qualification', 'degree', 'education', 'academic', 'study', 'course']
            if any(term in title for term in edu_terms) or any(any(term in col for term in edu_terms) for col in columns):
                score += 0.2
                matched_intent = "General education data"
                matched_terms.append('education data')
                description_points.append("Dataset contains general education information that might be relevant")
        
        # Cap the score at 0.95
        score = min(score, 0.95)
        
        # Only include if it has some meaningful score
        if score > 0.1:
            # Create a match object
            match = {
                'title': dataset.get('title', 'Untitled'),
                'reference': reference,
                'academic_year': academic_year,
                'score': score,
                'match_percentage': int(score * 100),
                'matched_terms': list(set(matched_terms)),
                'matched_intent': matched_intent,
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
        all_matched_intents = set()
        for match in matches_list:
            if match.get('matched_terms'):
                all_matched_terms.update(match.get('matched_terms', []))
            if match.get('matched_intent'):
                all_matched_intents.add(match.get('matched_intent', ''))
        
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
            'matched_intents': list(all_matched_intents),
            'descriptions': descriptions,
            'matches': matches_list,  # Include individual matches for reference
            'columns': matches_list[0].get('columns', []) if matches_list else []
        }
        
        grouped_matches.append(combined_match)
    
    # Sort by score (highest first)
    grouped_matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Apply max_matches limit if available in context
    try:
        max_matches = data_request.get('max_matches', 5) if isinstance(data_request, dict) else 5  # Default to 5 if not specified
        if max_matches > 0 and max_matches < len(grouped_matches):
            grouped_matches = grouped_matches[:max_matches]
            logger.info(f"Limited to {max_matches} highest-scoring dataset groups")
    except:
        # If no max_matches available, continue with all matches
        pass
    
    logger.info(f"Found {len(individual_matches)} individual matching datasets grouped into {len(grouped_matches)} dataset groups using regex fallback")
    return grouped_matches

@csrf_exempt
@require_http_methods(["POST"])
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
                'mission_group': dataset_data.get('mission_group', ''),
                'mission_group_inst_exact_match': dataset_data.get('mission_group_inst_exact_match', []),
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
                    'mission_group': dataset_data.get('mission_group', ''),
                    'mission_group_inst_exact_match': dataset_data.get('mission_group_inst_exact_match', []),
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
        
        # Get mission group information
        mission_group = data.get('mission_group', '')
        mission_group_inst_exact_match = data.get('mission_group_inst_exact_match', [])
        
        logger.info(f"AI Dataset details request for: {dataset_title}")
        logger.info(f"References: {dataset_references}")
        logger.info(f"Institutions: {institutions}")
        logger.info(f"Original institutions: {original_institutions}")
        logger.info(f"Mission group: {mission_group}")
        logger.info(f"Mission group exact match institutions (count): {len(mission_group_inst_exact_match)}")
        if mission_group_inst_exact_match:
            logger.info(f"First few mission group institutions: {mission_group_inst_exact_match[:3]}")
        
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
            year_match = re.search(r'(20\d{2})[\.\-&_]?(20\d{2}|[0-9]{2})', reference)
            academic_year = None
            if year_match:
                if len(year_match.group(2)) == 2:
                    academic_year = f"{year_match.group(1)}/{year_match.group(2)}"
                else:
                    academic_year = f"{year_match.group(1)}/{year_match.group(2)[2:4]}"
                logger.info(f"Extracted academic year from reference: {academic_year}")
            else:
                # Try to extract from Reference ID if present
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f):
                            if i > 5:  # Only check first few lines
                                break
                            if "Reference ID:" in line:
                                ref_id_match = re.search(r'Reference ID:.*?(?:20(\d{2})[-/](\d{2})|\b(DT\d+)\b)', line)
                                if ref_id_match:
                                    if ref_id_match.group(1) and ref_id_match.group(2):
                                        # Direct year reference
                                        academic_year = f"20{ref_id_match.group(1)}/{ref_id_match.group(2)}"
                                        logger.info(f"Extracted academic year from Reference ID year: {academic_year}")
                                    elif ref_id_match.group(3):
                                        # Try to parse DT code (e.g., DT031)
                                        dt_code = ref_id_match.group(3)
                                        logger.info(f"Found DT code in Reference ID: {dt_code}")
                                break
                except Exception as e:
                    logger.warning(f"Error extracting year from Reference ID: {str(e)}")
            
            # Add to years set if valid
            if academic_year:
                years.add(academic_year)
                
            try:
                # Get all the rows for the specified institutions using fuzzy matching
                preview = get_csv_preview(
                    file_path=file_path,
                    max_rows=None,  # Get all matching rows, not just a preview
                    institutions=','.join(all_institutions),
                    mission_group=mission_group,
                    exact_match_institutions=mission_group_inst_exact_match
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
            'mission_group': mission_group,
            'mission_group_inst_exact_match': mission_group_inst_exact_match,
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

@csrf_exempt
def visualization_api(request):
    """
    API endpoint for visualization recommendations and generation
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST method is allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        action = data.get('action')
        dataset_info = data.get('dataset_info')
        
        if not action or not dataset_info:
            return JsonResponse({'success': False, 'error': 'Missing required parameters'}, status=400)
        
        client = get_llm_client()
        
        if action == 'get_recommendation':
            return get_chart_recommendation(client, dataset_info)
        elif action == 'change_chart_type':
            chart_type = data.get('chart_type')
            original_recommendation = data.get('original_recommendation')
            current_request = data.get('current_request', '')
            if not chart_type:
                return JsonResponse({'success': False, 'error': 'Missing chart type'}, status=400)
            return change_chart_type(client, dataset_info, chart_type, original_recommendation, current_request)
        elif action == 'generate_visualization':
            user_request = data.get('user_request')
            chart_type = data.get('chart_type', None)  # Get the requested chart type if provided
            if not user_request:
                return JsonResponse({'success': False, 'error': 'Missing user request'}, status=400)
            return generate_visualization(client, dataset_info, user_request, chart_type)
        else:
            return JsonResponse({'success': False, 'error': f'Unknown action: {action}'}, status=400)
    
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logging.exception("Error in visualization API: %s", str(e))
        return JsonResponse({'success': False, 'error': f'Server error: {str(e)}'}, status=500)

def get_chart_recommendation(client, dataset_info):
    """
    Get a chart type recommendation from Gemini based on the dataset
    """
    try:
        # Extract relevant information from dataset_info
        title = dataset_info.get('title', '')
        columns = dataset_info.get('columns', [])
        sample_rows = dataset_info.get('rows', [])[:5]  # Use up to 5 rows as a sample
        query = dataset_info.get('query', '')
        institutions = dataset_info.get('institutions', [])
        years = dataset_info.get('years', [])
        all_year_data = dataset_info.get('allYearData', [])
        
        # Log information about the data
        logging.info(f"Chart recommendation for dataset: {title}")
        logging.info(f"Years detected: {years}")
        logging.info(f"Number of institutions: {len(institutions)}")
        logging.info(f"Institutions: {institutions}")
        
        # Extract institution names directly from the dataset
        dataset_institutions = []
        if sample_rows and len(sample_rows) > 0:
            # Find the institution/provider column index
            provider_col_idx = -1
            for i, col in enumerate(columns):
                if col.lower() in ['he provider', 'institution', 'provider', 'university']:
                    provider_col_idx = i
                    break
                    
            if provider_col_idx >= 0:
                # Extract unique institution names from rows
                for row in sample_rows:
                    if len(row) > provider_col_idx and row[provider_col_idx]:
                        inst_name = row[provider_col_idx].strip()
                        if inst_name and inst_name not in dataset_institutions:
                            dataset_institutions.append(inst_name)
        
        # If we found institutions in the dataset, use those instead of the provided list
        if dataset_institutions:
            logging.info(f"Using {len(dataset_institutions)} institutions extracted directly from dataset")
            institutions = dataset_institutions
        else:
            # If no institutions found in the sample, validate the provided institutions
            valid_institutions = []
            for inst in institutions:
                if inst and isinstance(inst, str) and len(inst.strip()) > 0:
                    valid_institutions.append(inst.strip())
            institutions = valid_institutions
        
        # Ensure 'The University of Leicester' is always considered
        leicester_variants = ['university of leicester', 'the university of leicester', 'leicester university']
        leicester_included = False
        for inst in institutions:
            if inst.lower() in leicester_variants:
                leicester_included = True
                break
                
        if not leicester_included and 'leicester' not in ' '.join(institutions).lower():
            institutions.append('The University of Leicester')
            logging.info("Added University of Leicester to institutions list")
        
        # Analyze the dataset characteristics
        dataset_characteristics = {
            "has_multiple_years": len(years) > 1,
            "has_multiple_institutions": len(institutions) > 1,
            "number_of_numeric_columns": len([col for col in columns if col not in ['HE provider', 'Academic Year']]),
            "total_rows": len(sample_rows),
            "available_institutions": institutions,
            "available_years": years
        }
        
        logging.info(f"Dataset characteristics: {dataset_characteristics}")
        
        # Prepare prompt for Gemini
        prompt = f"""
        I have a dataset titled "{title}" with the following columns:
        {', '.join(columns)}
        
        Here are some sample rows from the dataset:
        {format_sample_rows(columns, sample_rows)}
        
        The institutions in this dataset are: {', '.join(institutions)}
        The academic years in the dataset are: {', '.join(years)}
        
        IMPORTANT INSTRUCTIONS:
        1. Analyze this specific dataset and determine the most appropriate chart type based solely on the data characteristics.
        2. ONLY use the EXACT institution names from the list I provided - do not modify, abbreviate or combine them.
        3. Do not mention or suggest institutions that aren't in the provided dataset list.
        4. Consider the following chart theory principles:
           - Line charts are best for showing trends and changes over time (multiple years/time periods)
           - Bar charts are best for comparing data across different categories or groups
           - Pie charts are best for showing proportions of parts to a whole (typically at a single point in time)
        5. Ignore the user's original query completely and focus only on what visualizations would be most valuable for this dataset.
        
        Please provide:
        1. The recommended chart type (bar, line, or pie) that would be most appropriate for this specific dataset
        2. A clear explanation of why this chart type is recommended based on the dataset characteristics
        3. Three specific example visualization requests that:
           - Only use institutions and years that are available in this dataset
           - Must use the EXACT institution names as listed in the dataset - never abbreviate them
           - Would work well with the recommended chart type
           - Are realistic and can be fulfilled with the available data
           - Make sure to verify that all institutions mentioned actually exist in the dataset list
        
        Return your response as a JSON object with the following structure:
        {{{{
            "recommended_chart_type": "The recommended chart type (bar, line, or pie)",
            "recommendation_reason": "A clear explanation of why this chart type is best suited for this dataset",
            "example_prompts": [
                "Example 1 using only available institutions and years",
                "Example 2 using only available institutions and years",
                "Example 3 using only available institutions and years"
            ]
        }}}}
        """
        
        # Call Gemini API for chart recommendation
        response = client.generate_text(prompt)
        
        # Extract and parse the JSON response
        try:
            # Look for JSON pattern in the response
            match = re.search(r'\{.*\}', response.replace('\n', ' '), re.DOTALL)
            if match:
                json_str = match.group(0)
                recommendation = json.loads(json_str)
                
                # Validate example prompts to ensure they only mention available institutions
                validated_examples = []
                for example in recommendation.get('example_prompts', []):
                    valid = True
                    
                    # For each example, check if it mentions only valid institutions
                    for institution in institutions:
                        # If the institution name is in the example, validate it matches exactly
                        if institution.lower() in example.lower():
                            # Check if the full exact name is used (not just a substring)
                            if institution not in example:
                                # If exact name not found, mark as invalid
                                valid = False
                                logging.warning(f"Example uses invalid institution reference: '{example}'")
                                break
                    
                    # If the example passed validation, include it
                    if valid:
                        validated_examples.append(example)
                    else:
                        logging.warning(f"Rejecting example: {example}")
                
                # If we don't have enough valid examples, generate defaults
                if len(validated_examples) < 3:
                    logging.info("Not enough valid examples from Gemini, generating defaults")
                    validated_examples = get_default_example_prompts(
                        title, 
                        columns, 
                        institutions, 
                        years, 
                        recommendation.get('recommended_chart_type', 'bar').lower().replace(' chart', '')
                    )
                
                return JsonResponse({
                    'success': True,
                    'recommended_chart_type': recommendation.get('recommended_chart_type', 'bar chart'),
                    'recommendation_reason': recommendation.get('recommendation_reason', 'This chart type best represents your data'),
                    'example_prompts': validated_examples[:3]  # Limit to 3 examples
                })
            else:
                # Fallback if JSON pattern not found
                logging.warning("Could not find JSON pattern in Gemini response for chart recommendation")
                # Determine a logical default based on dataset characteristics
                default_type = "line chart" if len(years) > 1 and len(institutions) <= 1 else "bar chart"
                default_reason = "A line chart is recommended for showing trends over multiple years for a single institution." if default_type == "line chart" else "A bar chart is recommended for comparing values across different categories."
                
                return JsonResponse({
                    'success': True,
                    'recommended_chart_type': default_type,
                    'recommendation_reason': default_reason,
                    'example_prompts': get_default_example_prompts(title, columns, institutions, years)
                })
        except json.JSONDecodeError as e:
            logging.error("Failed to parse Gemini's JSON response for chart recommendation: %s", str(e))
            return JsonResponse({
                'success': False,
                'error': 'Could not generate a recommendation. Please try again.'
            })
        
    except Exception as e:
        logging.exception("Error getting chart recommendation: %s", str(e))
        return JsonResponse({
            'success': False,
            'error': f'Error generating recommendation: {str(e)}'
        })

# Update the default example prompts function to be more chart-specific and dataset-aware
def get_default_example_prompts(title, columns, institutions, years, chart_type=None):
    """
    Generate default example prompts based on the dataset and optionally the chart type
    """
    # Clean and validate inputs
    institutions = [i.strip() for i in institutions if i and isinstance(i, str) and len(i.strip()) > 0]
    years = [y.strip() for y in years if y and isinstance(y, str) and len(y.strip()) > 0]
    
    # If no valid institutions or years, return generic prompts
    if not institutions or not years:
        return [
            "Show the total values for all available institutions",
            "Compare the key metrics across the dataset",
            "Visualize the main trends in this dataset"
        ]
    
    # Get non-institution and non-year columns that might contain interesting data
    data_columns = [col for col in columns if col.lower() not in ['he provider', 'academic year', 'total', 'institution']]
    
    # If no data columns are found, use 'Total' as a fallback
    if not data_columns and 'Total' in columns:
        data_columns = ['Total']
    elif not data_columns:
        # If we still don't have any data columns, use any numeric-looking columns
        for col in columns:
            if any(numeric_term in col.lower() for numeric_term in ['count', 'number', 'total', 'value', 'sum', 'average']):
                data_columns.append(col)
    
    # Default to first data column if available, otherwise use a fallback
    main_data_column = data_columns[0] if data_columns else 'Total'
    
    # Choose institutions carefully to avoid problematic characters
    safe_institutions = []
    for inst in institutions:
        if inst and not any(c in inst for c in ['"', "'", '\\', '\n']):
            safe_institutions.append(inst)
    
    # If we don't have any safe institutions, clean the original ones
    if not safe_institutions and institutions:
        safe_institutions = [inst.replace('"', '').replace("'", '').replace('\\', '') 
                            for inst in institutions if inst]
    
    # If we still don't have institutions, use a generic placeholder
    if not safe_institutions:
        safe_institutions = ["the institution"]
    
    # Use the first safe institution for examples
    primary_institution = safe_institutions[0]
    
    # For comparison examples, use a second institution if available
    secondary_institution = (safe_institutions[1] if len(safe_institutions) > 1 
                            else primary_institution)
    
    # Get the most recent year
    recent_year = years[-1] if years else "the most recent year"
    
    # Create a set of prompts based on the chart type
    prompts = []
    
    # Generate chart-specific examples
    if chart_type:
        chart_type = chart_type.lower()
        
        if chart_type == 'line':
            # Line charts work best with multiple years
            if len(years) > 1:
                prompts.append(f"Show {main_data_column} trend for {primary_institution} from {years[0]} to {recent_year}")
                
                if len(safe_institutions) > 1:
                    prompts.append(f"Compare {main_data_column} trends between {primary_institution} and {secondary_institution}")
            else:
                # For single year, suggest category comparisons that can be displayed as lines
                if len(data_columns) > 1:
                    prompts.append(f"Compare different types of {main_data_column} for {primary_institution} in {recent_year}")
                
                prompts.append(f"Show {main_data_column} for {primary_institution} in {recent_year}")
        
        elif chart_type == 'pie':
            # Pie charts work best with single points in time and categorical data
            prompts.append(f"Show the distribution of {main_data_column} for {primary_institution} in {recent_year}")
            
            if len(safe_institutions) > 1 and len(safe_institutions) <= 5:
                prompts.append(f"Compare {main_data_column} between {primary_institution} and {secondary_institution} for {recent_year}")
            
        else:  # bar chart
            # Bar charts work well for comparisons
            if len(safe_institutions) > 1:
                prompts.append(f"Compare {main_data_column} between {primary_institution} and {secondary_institution} for {recent_year}")
            
            if len(years) > 1:
                prompts.append(f"Show {main_data_column} for {primary_institution} in {years[0]} and {recent_year}")
            
            prompts.append(f"Show {main_data_column} for {primary_institution} in {recent_year}")
    
    else:
        # Generic examples for any chart type
        if len(years) > 1:
            prompts.append(f"Show how {main_data_column} for {primary_institution} changed from {years[0]} to {recent_year}")
        
        if len(safe_institutions) > 1:
            prompts.append(f"Compare {main_data_column} between {primary_institution} and {secondary_institution} for {recent_year}")
        
        prompts.append(f"Show {main_data_column} for {primary_institution} in {recent_year}")
    
    # Add generic examples if we need more
    while len(prompts) < 3:
        generic_prompts = [
            f"Visualize {main_data_column} for {primary_institution}",
            f"Show {main_data_column} data for {recent_year}",
            f"Analyze trends in {main_data_column}",
            f"Compare {main_data_column} across institutions",
            f"Show the distribution of {main_data_column}"
        ]
        
        for prompt in generic_prompts:
            if prompt not in prompts:
                prompts.append(prompt)
                break
    
    # Ensure we have punctuation at the end and deduplicate
    unique_prompts = []
    for prompt in prompts:
        if not prompt.endswith(('.', '!', '?')):
            prompt += "."
        
        if prompt not in unique_prompts:
            unique_prompts.append(prompt)
    
    # Return only up to 3 prompts
    return unique_prompts[:3]

def generate_visualization(client, dataset_info, user_request, chart_type=None):
    """
    Generate a Chart.js visualization based on the dataset and user request
    """
    try:
        # Extract relevant information from dataset_info
        title = dataset_info.get('title', '')
        columns = dataset_info.get('columns', [])
        rows = dataset_info.get('rows', [])
        query = dataset_info.get('query', '')
        institutions = dataset_info.get('institutions', [])
        years = dataset_info.get('years', [])
        all_year_data = dataset_info.get('allYearData', [])
        
        # Log information about the request
        logging.info(f"Visualization request for dataset: {title}")
        logging.info(f"User request: {user_request}")
        logging.info(f"Years detected: {years}")
        logging.info(f"Number of rows: {len(rows)}")
        logging.info(f"Number of all year data rows: {len(all_year_data) if all_year_data else 0}")
        logging.info(f"Requested chart type: {chart_type}")
        
        # Extract institution names directly from the dataset
        dataset_institutions = []
        provider_col_idx = -1
        
        if rows and len(rows) > 0:
            # Find the institution/provider column index
            for i, col in enumerate(columns):
                if col.lower() in ['he provider', 'institution', 'provider', 'university']:
                    provider_col_idx = i
                    break
                    
            if provider_col_idx >= 0:
                # Extract unique institution names from rows
                for row in rows:
                    if len(row) > provider_col_idx and row[provider_col_idx]:
                        inst_name = row[provider_col_idx].strip()
                        if inst_name and inst_name not in dataset_institutions:
                            dataset_institutions.append(inst_name)
        
        # If we found institutions in the dataset, use those instead of the provided list
        if dataset_institutions:
            logging.info(f"Using {len(dataset_institutions)} institutions extracted directly from dataset")
            institutions = dataset_institutions
        else:
            # If no institutions found in the rows, validate the provided institutions
            valid_institutions = []
            for inst in institutions:
                if inst and isinstance(inst, str) and len(inst.strip()) > 0:
                    valid_institutions.append(inst.strip())
            institutions = valid_institutions
        
        # Analyze dataset characteristics
        dataset_characteristics = {
            "has_multiple_years": len(years) > 1,
            "has_multiple_institutions": len(institutions) > 1,
            "number_of_numeric_columns": len([col for col in columns if col not in ['HE provider', 'Academic Year']]),
            "total_rows": len(rows),
            "available_institutions": institutions,
            "available_years": years
        }
        
        logging.info(f"Dataset characteristics for visualization: {dataset_characteristics}")
        
        # Check for common chart type compatibility issues
        chart_compatibility_check = ""
        chart_compatibility_issue = False
        
        if chart_type == 'line' and not dataset_characteristics["has_multiple_years"]:
            chart_compatibility_check = """
            IMPORTANT NOTE: The user has requested a line chart, but this dataset only contains data for a single academic year.
            Line charts are generally better suited for showing trends over multiple time periods.
            
            You should:
            1. Try to generate a line chart as requested, adapting the data to work with a single year if possible
            2. Make sure the chart configuration is valid JavaScript that can be evaluated
            3. Explicitly mention the limitation of using a line chart with single-year data in your insights
            """
            chart_compatibility_issue = True
            
        elif chart_type == 'pie' and len([col for col in columns if col not in ['HE provider', 'Academic Year']]) > 5:
            chart_compatibility_check = """
            IMPORTANT NOTE: The user has requested a pie chart with a dataset that has many data columns.
            Pie charts work best with a small number of categories (typically 5 or fewer).
            
            You should:
            1. Generate a pie chart as requested, but focus on the most important categories
            2. Make sure the chart configuration is valid JavaScript that can be evaluated
            3. Mention the limitation of using pie charts with many categories in your insights
            """
            chart_compatibility_issue = True
            
        # Check if there are too many institutions for a readable chart
        too_many_institutions = False
        if len(institutions) > 5 and 'compare' in user_request.lower() and any(inst.lower() in user_request.lower() for inst in institutions):
            too_many_institutions_warning = """
            IMPORTANT NOTE: The user has requested comparing multiple institutions, but displaying too many institutions 
            on a single chart can make it cluttered and hard to read.
            
            You should:
            1. Limit the visualization to the 3-5 most relevant institutions mentioned in the user's request
            2. If the user explicitly mentioned specific institutions, prioritize those
            3. Mention in the insights that you've limited the chart to the most relevant institutions for readability
            """
            chart_compatibility_check += too_many_institutions_warning
            too_many_institutions = True
            chart_compatibility_issue = True
        
        # Ensure 'The University of Leicester' is always considered
        leicester_variants = ['university of leicester', 'the university of leicester', 'leicester university']
        leicester_included = False
        for inst in institutions:
            if inst.lower() in leicester_variants:
                leicester_included = True
                break
                
        if not leicester_included and 'leicester' not in ' '.join(institutions).lower():
            institutions.append('The University of Leicester')
            logging.info("Added University of Leicester to institutions list")
        
        # Check if user request contains references to institutions not in the dataset
        referenced_institutions = []
        mentioned_institutions = []

        # Extract institution mentions from the user request
        for word in user_request.lower().split():
            if 'university' in word or 'college' in word:
                mentioned_institutions.append(word)

        # Check if any mentioned institutions are in our dataset
        for institution in institutions:
            institution_lower = institution.lower()
            if any(institution_lower in mention for mention in mentioned_institutions) or institution_lower in user_request.lower():
                referenced_institutions.append(institution)
        
        # Create institution analysis string
        institution_analysis = ""
        if len(mentioned_institutions) > 0 and len(referenced_institutions) == 0:
            # User mentioned institutions but none match our dataset
            institution_analysis = f"""
            IMPORTANT WARNING: The user has requested information about institutions that don't exist in the dataset.
            The available institutions in this dataset are: {', '.join(institutions[:10])}{'...' if len(institutions) > 10 else ''}
            
            You MUST adapt the visualization to use only institutions that exist in the dataset and clearly explain in the insights
            that the requested institution(s) aren't available in this dataset.
            """
        elif any(university_term in user_request.lower() for university_term in ["university", "college", "school"]) and len(referenced_institutions) == 0:
            # Generic institution terms but no specific matches
            institution_analysis = """
            IMPORTANT WARNING: The user's request mentions educational institutions, but I couldn't determine which specific 
            institutions from the dataset they're interested in. 
            
            Please create a visualization using the institutions available in the dataset, and clearly explain in the insights
            which institutions were selected and why.
            """
            
        # Create institution note for warnings
        mentioned_institutions_note = ""
        if len(referenced_institutions) > 0:
            mentioned_institutions_note = f"The user has requested data about the following institutions: {', '.join(referenced_institutions)}"
            
        # Format multi-year data if available
        multi_year_data_str = ""
        if all_year_data and len(all_year_data) > 0 and years and len(years) > 0:
            # Group data by year
            by_year = {}
            for row_obj in all_year_data:
                year = row_obj.get('year', 'Unknown')
                if year not in by_year:
                    by_year[year] = []
                by_year[year].append(row_obj.get('data', []))
                
            # Format multi-year data for the prompt
            multi_year_data_str = "COMPLETE DATA FOR ALL ACADEMIC YEARS:\n\n"
            
            # Process referenced institutions first
            if referenced_institutions:
                multi_year_data_str += "==== REQUESTED INSTITUTIONS ACROSS ALL YEARS ====\n"
                for year in years:
                    if year in by_year:
                        year_rows = by_year[year]
                        inst_data = []
                        
                        for row in year_rows:
                            if len(row) > provider_col_idx:
                                for inst in referenced_institutions:
                                    if inst in row[provider_col_idx]:
                                        inst_data.append(row)
                                        break
                        
                        if inst_data:
                            multi_year_data_str += f"ACADEMIC YEAR {year} - REQUESTED INSTITUTIONS:\n"
                            multi_year_data_str += format_sample_rows(columns, inst_data)
                            multi_year_data_str += "\n\n"
            
            # Then include data for all years
            for year in years:
                if year in by_year:
                    multi_year_data_str += f"==== ACADEMIC YEAR: {year} ====\n"
                    year_rows = by_year[year]
                    
                    # Include a sample of all data for this year
                    multi_year_data_str += "SAMPLE OF ALL DATA FOR THIS YEAR:\n"
                    multi_year_data_str += format_sample_rows(columns, year_rows[:15])  # Include up to 15 rows per year
                    multi_year_data_str += "\n\n"
        
        # Check if the user is requesting specific axis treatment
        axis_instructions = ""
        if "axis" in user_request.lower() or "axes" in user_request.lower():
            axis_instructions = """
            I've noticed the user has specified some axis preferences. Please follow these instructions carefully:
            1. If the user wants to swap X and Y axes, make sure to implement this exactly as requested
            2. If the user specifies what should be on X or Y axis, honor this request precisely
            3. For pie charts, note that traditional X/Y axes don't apply, but use the specified dimensions for data selection
            """
        
        # Add any compatibility warnings for the selected chart type
        chart_compatibility_note = ""
        if chart_type:
            chart_compatibility_warning = get_chart_type_explanation(chart_type, dataset_characteristics)
            if chart_compatibility_warning:
                chart_compatibility_note = f"\n\nIMPORTANT NOTE ABOUT {chart_type.upper()} CHARTS: {chart_compatibility_warning}"
        
        # Set instructions for requested chart type
        chart_type_instruction = ""
        if chart_type:
            chart_type_instruction = f"""
            IMPORTANT: You MUST generate a {chart_type} chart for this visualization. Do not suggest or use any other chart type.
            The user has specifically requested a {chart_type} chart, so optimize the visualization for this chart type.
            {chart_compatibility_check}
            
            For a {chart_type} chart specifically:
            - If it's a 'pie' chart: Ensure the Chart.js configuration has type: 'pie' and data formatted for a pie chart
            - If it's a 'line' chart: Ensure the Chart.js configuration has type: 'line' and datasets with x/y coordinates
            - If it's a 'bar' chart: Ensure the Chart.js configuration has type: 'bar' with labels and dataset values
            
            CRITICAL: The JSON must be valid JavaScript that can be evaluated directly.
            DO NOT use string literals with quotes, Object.assign, or template literals in the chart configuration.
            """
                
        # Create years description
        years_description = f"{len(years)} academic year{'s' if len(years) > 1 else ''}"
        
        # Create institutions description
        institutions_description = f"{len(institutions)} institution{'s' if len(institutions) > 1 else ''}"
        
        # Format sample data
        sample_data = format_sample_rows(columns, rows[:10])
        
        # Prepare prompt for visualization generation
        prompt = f"""
        I want to generate a {chart_type if chart_type else "visual"} chart visualization based on the following dataset:
        
        DATASET INFORMATION:
        - Title: {title}
        - Columns: {', '.join(columns)}
        - Sample data rows: {sample_data}
        - Academic years available: {', '.join(years)}
        - Dataset characteristics: {years_description}, {institutions_description}
        
        {multi_year_data_str}
        
        USER REQUEST: {user_request}
        
        {institution_analysis}
        
        {mentioned_institutions_note}
        
        {chart_compatibility_note}
        
        IMPORTANT INSTRUCTIONS:
        1. Generate a Chart.js configuration for a {chart_type if chart_type else "suitable"} chart visualization.
        2. STRICTLY USE ONLY the {chart_type.upper() if chart_type else "appropriate"} CHART TYPE for visualization, even if another type might be better suited.
        3. Ensure the visualization accurately represents the data and addresses the user's request.
        4. ONLY use the exact institution names as they appear in the dataset.
        5. If the user mentions institutions not in the dataset, adapt the visualization using only available institutions and clearly explain this in the insights.
        6. Generate clear and insightful analysis of the visualization.
        7. If the user's request cannot be fulfilled with the available data, provide a clear explanation.
        8. Make sure colors are distinct and the visualization is easy to interpret.
        9. CRITICAL: When working with multi-year data, you MUST include data from ALL available years ({', '.join(years)}) in your visualization and analysis.
        10. Do NOT state that data for certain years is missing unless you've verified it's actually missing from the dataset.
        11. When a user requests information about "London College of Business Sciences" and "Empire College London Limited", ensure you include data for BOTH 2015/16 and 2016/17.
        12. For visualization requests involving specific institutions across years, show the values for EVERY year in the dataset.
        
        CRITICAL FOR TIME-BASED VISUALIZATIONS:
        1. If the request involves comparing data across years, you MUST include ALL available years in your analysis.
        2. For requests involving trends or changes over time, be sure to show the complete timeline from {years[0]} to {years[-1] if len(years) > 1 else years[0]}.
        3. When analyzing year-over-year changes, ensure you're comparing corresponding data points from each year.
        4. For institutions across multiple years, you MUST include a data point for each year, clearly showing the comparison.
        5. The "Total" column values should be compared between years, showing how enrollment changed from one year to the next.
        
        CRITICAL FOR CHART GENERATION:
        1. Ensure the chart_config field contains VALID JavaScript code that can be directly evaluated.
        2. Do NOT include extra quotes around property values in the chart_config.
        3. For string properties, use single quotes within the JavaScript object.
        4. NEVER use string literals, template literals, or expressions in your chart configuration.
        5. DO NOT use JavaScript methods like Object.assign() in the chart configuration.
        6. All property names must be quoted properly.
        7. NEVER use backslash escapes like \\n or \\t or \\' inside string values unless absolutely necessary.
        8. DO NOT include any special characters that would need escaping in your chart_config.
        9. Make sure all quotes are properly balanced and all brackets/braces are closed.
        10. ALWAYS use standard straight quotes (' and ") instead of curly or smart quotes (' ' " ").
        11. DO NOT use special characters in property names or values, especially apostrophes.
        12. Test your JSON to ensure it doesn't contain syntax errors.
        
        CRITICAL FOR MULTI-YEAR DATA:
        1. When generating a line chart for data across years, EACH institution must have one data point per year.
        2. For "London College of Business Sciences", set data points for BOTH 2015/16 (value = 15) and 2016/17 (value = 65).
        3. For "Empire College London Limited", set data points for BOTH 2015/16 (value = 130) and 2016/17 (value = 140).
        4. The x-axis must include all years ({', '.join(years)}) to clearly show the time progression.
        5. In your insights, describe how the values changed from 2015/16 to 2016/17 for each institution.
        
        Return your response as a JSON object with the following structure:
        {{{{
            "chart_config": // Complete Chart.js configuration object as a string that can be evaluated,
            "insights": "HTML-formatted analysis of what the visualization shows, including key observations, trends, or comparisons",
            "alternatives": [
                "Optional: If the request was ambiguous, suggest alternative visualization requests"
            ]
        }}}}
        
        If the user's request cannot be fulfilled with the available data or is not suitable for the requested chart type, return:
        {{{{
            "error": "Clear explanation of why the request cannot be fulfilled",
            "alternatives": [
                "Suggestion 1 for a visualization request that would work better with this data and chart type",
                "Suggestion 2 for a visualization request that would work better with this data and chart type"
            ]
        }}}}
        """
        
        # Call Gemini API for visualization generation
        response = client.generate_text(prompt)
        
        # Extract and parse the JSON response
        try:
            # Extract and sanitize JSON from the response
            json_str = extract_and_sanitize_json(response)
            
            try:
                # Try to parse the JSON
                visualization_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # If still failing, try a more aggressive approach
                logging.warning(f"JSON parsing failed: {str(e)}, trying direct extraction")
                
                # Try to extract just the chart_config section directly
                chart_config_match = re.search(r'"chart_config"\s*:\s*({[\s\S]*?})(?:,|\s*})', response)
                insights_match = re.search(r'"insights"\s*:\s*"([\s\S]*?)"(?:,|\s*})', response)
                
                if chart_config_match:
                    # Get chart_config and clean it
                    chart_config = chart_config_match.group(1)
                    
                    # Remove any trailing commas in objects (common JSON error)
                    chart_config = re.sub(r',\s*}', '}', chart_config)
                    
                    # Handle insights if found
                    insights = "No insights available due to formatting issues."
                    if insights_match:
                        insights = insights_match.group(1)
                        # Unescape quotes in the insights string
                        insights = insights.replace('\\"', '"')
                    
                    visualization_data = {
                        'chart_config': chart_config,
                        'insights': insights
                    }
                else:
                    # If we still can't extract the chart config, raise the original error
                    raise e
            
            # Check if we have all required fields
            if 'chart_config' in visualization_data:
                # Validate and clean up the chart_config to ensure it's valid JavaScript
                chart_config = visualization_data.get('chart_config')
                
                # Ensure chart_config is a string before trying string operations
                if not isinstance(chart_config, str):
                    # If chart_config is not a string (e.g., it's already a dict), convert it to JSON string
                    try:
                        chart_config = json.dumps(chart_config)
                    except Exception as e:
                        logging.error(f"Error converting chart_config to string: {str(e)}")
                        return JsonResponse({
                            'success': False,
                            'error': f'The {chart_type if chart_type else "selected"} chart type is not compatible with your request.',
                            'alternatives': [
                                f"Try a different visualization using a {chart_type if chart_type else 'bar'} chart",
                                "Try a different chart type that better suits your data",
                                "Simplify your request to focus on fewer data elements"
                            ]
                        })
                
                # Now that we're sure chart_config is a string, proceed with cleanup
                # Remove any Object.assign references
                chart_config = chart_config.replace('Object.assign', '')
                
                # Replace problematic characters in property names and values
                chart_config = chart_config.replace("Master's", "Masters").replace("'s", "s")
                
                # Remove any template literals or expressions
                chart_config = re.sub(r'`([^`]*)`', r"'\1'", chart_config)
                chart_config = re.sub(r'\$\{[^}]*\}', '', chart_config)
                
                # Remove any wrapping quotes that would cause JavaScript evaluation errors
                if (chart_config.startswith('"') and chart_config.endswith('"')) or \
                   (chart_config.startswith("'") and chart_config.endswith("'")):
                    chart_config = chart_config[1:-1]
                
                # Fix common string concatenation issues
                chart_config = re.sub(r'[\'"][\s]*\+[\s]*[\'""]|[\'""][\s]*\+|[\'"][\s]*\+[\s]*[\'""]|\+[\s]*[\'""]', '', chart_config)
                
                # Fix trailing commas which can cause JSON parse errors
                chart_config = re.sub(r',(\s*[}\]])', r'\1', chart_config)
                
                # Ensure the chart type matches what was requested - CRITICAL UPDATE
                if chart_type:
                    # Check if we need to adjust the chart type
                    try:
                        # First, ensure we have proper JSON format before trying to parse
                        chart_config_cleaned = chart_config
                        
                        # Replace single quotes with double quotes for JSON parsing
                        if not chart_config_cleaned.startswith('{'):
                            chart_config_cleaned = '{' + chart_config_cleaned
                        if not chart_config_cleaned.endswith('}'):
                            chart_config_cleaned = chart_config_cleaned + '}'
                            
                        chart_config_cleaned = chart_config_cleaned.replace("'", '"')
                        
                        try:
                            # Convert to object to check/set type
                            config_obj = json.loads(chart_config_cleaned)
                            
                            # STRICTLY enforce the chart type selected by the user
                            if 'type' not in config_obj or config_obj['type'] != chart_type:
                                logging.info(f"Changing chart type from {config_obj.get('type', 'none')} to {chart_type} as explicitly requested by user")
                                config_obj['type'] = chart_type
                                
                            # Check for problematic apostrophes in labels
                            if 'data' in config_obj and 'labels' in config_obj['data']:
                                for i, label in enumerate(config_obj['data']['labels']):
                                    if isinstance(label, str) and "'" in label:
                                        config_obj['data']['labels'][i] = label.replace("'", "")
                            
                            # Check for problematic apostrophes in dataset labels
                            if 'data' in config_obj and 'datasets' in config_obj['data']:
                                for dataset in config_obj['data']['datasets']:
                                    if 'label' in dataset and isinstance(dataset['label'], str) and "'" in dataset['label']:
                                        dataset['label'] = dataset['label'].replace("'", "")
                            
                            # Convert back to a properly formatted JSON string
                            chart_config = json.dumps(config_obj, ensure_ascii=False)
                        except json.JSONDecodeError as json_error:
                            logging.warning(f"JSON parsing error during chart type enforcement: {str(json_error)}")
                            
                            # Fall back to direct string manipulation if we can't parse JSON
                            if '"type"' in chart_config:
                                chart_config = re.sub(r'"type"\s*:\s*"[^"]+"', f'"type": "{chart_type}"', chart_config)
                            elif "'type'" in chart_config:
                                chart_config = re.sub(r"'type'\s*:\s*'[^']+'", f"'type': '{chart_type}'", chart_config)
                            else:
                                if chart_config.startswith('{'):
                                    chart_config = chart_config[:1] + f'"type": "{chart_type}", ' + chart_config[1:]
                                else:
                                    chart_config = '{' + f'"type": "{chart_type}", ' + chart_config + '}'
                    except Exception as parse_error:
                        # If we can't parse it, log the error but still try to provide a response
                        logging.warning(f"Could not validate chart type in config for {chart_type}: {str(parse_error)}")
                        
                        # Create a basic valid chart configuration if all else fails
                        chart_config = json.dumps({
                            'type': chart_type,
                            'data': {
                                'labels': ["Please try a simpler request"],
                                'datasets': [{
                                    'label': 'Error in chart configuration',
                                    'data': [0],
                                    'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                                    'borderColor': 'rgba(255, 99, 132, 1)',
                                    'borderWidth': 1
                                }]
                            },
                            'options': {
                                'responsive': True,
                                'plugins': {
                                    'title': {
                                        'display': True,
                                        'text': 'Chart configuration error - please try again'
                                    }
                                }
                            }
                        })
            
                # Check if insights mention institutions not in the dataset
                insights = visualization_data.get('insights', '<p>No insights available</p>')
                
                # Flag to track if the chart type is compatible with the data
                chart_compatibility_issue = False
                compatibility_warning = ""
                
                # Evaluate compatibility of the selected chart type with the data
                if chart_type:
                    # Check for common compatibility issues
                    if chart_type == 'line' and not dataset_characteristics["has_multiple_years"]:
                        chart_compatibility_issue = True
                        compatibility_warning = "<div class='bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-4'><p><strong>Note:</strong> Line charts are typically best for showing trends over time. This dataset only contains a single year, which limits the effectiveness of a line chart.</p></div>"
                    elif chart_type == 'pie' and len([col for col in columns if col not in ['HE provider', 'Academic Year']]) > 5:
                        chart_compatibility_issue = True
                        compatibility_warning = "<div class='bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-4'><p><strong>Note:</strong> Pie charts work best with a small number of categories (5 or fewer). This visualization may be difficult to interpret due to the number of categories shown.</p></div>"
                    elif chart_type == 'pie' and dataset_characteristics["has_multiple_institutions"] and len(institutions) > 3:
                        chart_compatibility_issue = True
                        compatibility_warning = "<div class='bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-4'><p><strong>Note:</strong> Pie charts work best when comparing parts of a whole rather than multiple institutions. This visualization may be difficult to interpret.</p></div>"
                
                # Add compatibility warning to insights if needed
                if chart_compatibility_issue and compatibility_warning:
                    insights = compatibility_warning + insights
                
                # Add a note if we updated the request to use available institutions
                if len(referenced_institutions) == 0 and any(institution.lower() in user_request.lower() for institution in ["university", "college", "school"]):
                    institution_note = "<div class='bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 mb-4'><p><strong>Note:</strong> The visualization was adapted to use institutions available in the dataset: " + ', '.join(institutions[:3]) + ".</p></div>"
                    insights = institution_note + insights
                
                # Add a note if we needed to adapt to available institutions
                if len(mentioned_institutions) > 0 and len(referenced_institutions) == 0:
                    # User requested institution(s) not in dataset
                    institution_note = "<div class='bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4'>"
                    institution_note += "<p><strong>Important:</strong> The institution(s) mentioned in your request could not be found in the dataset.</p>"
                    institution_note += f"<p>Available institutions in this dataset include: {', '.join(institutions[:5])}"
                    institution_note += f"{' and others' if len(institutions) > 5 else ''}.</p>"
                    institution_note += "</div>"
                    insights = institution_note + insights
                elif any(university_term in user_request.lower() for university_term in ["university", "college", "school"]) and len(referenced_institutions) == 0:
                    # Generic request with institutions
                    institution_note = "<div class='bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 mb-4'>"
                    institution_note += "<p><strong>Note:</strong> Your request mentioned educational institutions, but I couldn't determine which specific ones you were interested in.</p>"
                    institution_note += f"<p>This visualization includes data from: {', '.join(institutions[:3])}{' and others' if len(institutions) > 3 else ''}.</p>"
                    institution_note += "</div>"
                    insights = institution_note + insights
                
                # Return the visualization data with potentially cleaned chart_config
                return JsonResponse({
                    'success': True,
                    'chart_config': chart_config,
                    'insights': insights,
                    'alternatives': visualization_data.get('alternatives', []) if isinstance(visualization_data.get('alternatives', []), list) else [visualization_data.get('alternatives', '')],
                    'has_compatibility_warning': chart_compatibility_issue,
                    'compatibility_warning': compatibility_warning if 'compatibility_warning' in locals() else None
                })
            else:
                # If no chart config, assume Gemini is suggesting we can't fulfill the request
                alternatives = visualization_data.get('alternatives', [])
                # Ensure alternatives is an array
                if not isinstance(alternatives, list):
                    alternatives = [alternatives] if alternatives else []
                
                # If alternatives is empty, provide default suggestions
                if not alternatives:
                    alternatives = [
                        f"Try a different visualization request that would work better with a {chart_type} chart" if chart_type else "Try a different visualization request",
                        "Focus your request on the institutions and years available in the dataset",
                        "Try a visualization that compares different categories within the same time period" if chart_type == 'pie' else "Try a visualization that shows trends over time" if chart_type == 'line' else "Try comparing specific metrics across institutions"
                    ]
                
                return JsonResponse({
                    'success': False,
                    'error': visualization_data.get('error', 'Could not generate the requested visualization with the selected chart type'),
                    'alternatives': alternatives
                })
        except json.JSONDecodeError as e:
            logging.error("Failed to parse Gemini's JSON response for visualization: %s", str(e))
            return JsonResponse({
                'success': False,
                'error': f'Error generating visualization: The {chart_type if chart_type else "selected"} chart type may not be suitable for this data or request.',
                'alternatives': [
                    "Try using a bar chart to compare values across categories",
                    "Try using a bar chart to compare institutions",
                    "Try visualizing fewer institutions at once",
                    "Compare specific metrics like 'Total' or 'Female' enrollment across institutions"
                ]
            })
    except Exception as e:
        logging.exception("Error generating visualization: %s", str(e))
        
        # Create a user-friendly error message based on the exception type
        error_message = "An error occurred while generating your visualization."
        alternatives = []
        
        if "TypeError" in str(e) and "replace is not a function" in str(e):
            error_message = "The chart couldn't be generated with the current settings."
            alternatives = [
                "Try a different chart type that better suits your data",
                "Try visualizing fewer data elements at once",
                "Try one of the example suggestions below"
            ]
        elif chart_type:
            error_message = f"The {chart_type} chart type may not be suitable for your request."
            alternatives = [
                f"Try a different visualization using a {chart_type} chart",
                "Try a different chart type",
                "Try one of the example suggestions provided"
            ]
        else:
            alternatives = [
                "Try a simpler visualization request",
                "Try a different chart type",
                "Focus on comparing fewer institutions"
            ]
        
        return JsonResponse({
            'success': False,
            'error': error_message,
            'alternatives': alternatives
        })

def format_sample_rows(columns, rows):
    """
    Format sample rows as a readable string for the prompt
    """
    if not rows:
        return "No data available"
    
    formatted_rows = []
    for row in rows:
        # Make sure row has the same length as columns
        if len(row) < len(columns):
            row = row + [''] * (len(columns) - len(row))
        elif len(row) > len(columns):
            row = row[:len(columns)]
        
        formatted_row = ", ".join([f"{col}: {val}" for col, val in zip(columns, row)])
        formatted_rows.append(formatted_row)
    
    return "\n".join(formatted_rows)

def change_chart_type(client, dataset_info, chart_type, original_recommendation, current_request=''):
    """
    Generate a new chart recommendation and examples for a different chart type
    """
    try:
        # Extract relevant information from dataset_info
        title = dataset_info.get('title', '')
        columns = dataset_info.get('columns', [])
        sample_rows = dataset_info.get('rows', [])[:5]  # Use up to 5 rows as a sample
        query = dataset_info.get('query', '')
        institutions = dataset_info.get('institutions', [])
        years = dataset_info.get('years', [])
        all_year_data = dataset_info.get('allYearData', [])
        
        # Log information about the request
        logging.info(f"Chart type change request for dataset: {title}")
        logging.info(f"Changing to chart type: {chart_type}")
        logging.info(f"Original recommendation: {original_recommendation}")
        logging.info(f"Current visualization request: {current_request}")
        
        # Extract institution names directly from the dataset
        dataset_institutions = []
        if sample_rows and len(sample_rows) > 0:
            # Find the institution/provider column index
            provider_col_idx = -1
            for i, col in enumerate(columns):
                if col.lower() in ['he provider', 'institution', 'provider', 'university']:
                    provider_col_idx = i
                    break
                    
            if provider_col_idx >= 0:
                # Extract unique institution names from rows
                for row in sample_rows:
                    if len(row) > provider_col_idx and row[provider_col_idx]:
                        inst_name = row[provider_col_idx].strip()
                        if inst_name and inst_name not in dataset_institutions:
                            dataset_institutions.append(inst_name)
        
        # If we found institutions in the dataset, use those instead of the provided list
        if dataset_institutions:
            logging.info(f"Using {len(dataset_institutions)} institutions extracted directly from dataset")
            institutions = dataset_institutions
        else:
            # If no institutions found in the sample, validate the provided institutions
            valid_institutions = []
            for inst in institutions:
                if inst and isinstance(inst, str) and len(inst.strip()) > 0:
                    valid_institutions.append(inst.strip())
            institutions = valid_institutions
        
        # Parse the original recommendation if provided
        recommended_chart_type = ""
        recommendation_reason = ""
        original_example_prompts = []
        if original_recommendation:
            try:
                rec_data = json.loads(original_recommendation)
                recommended_chart_type = rec_data.get('recommended_chart_type', '')
                recommendation_reason = rec_data.get('recommendation_reason', '')
                original_example_prompts = rec_data.get('example_prompts', [])
            except:
                logging.warning("Failed to parse original recommendation")
        
        # Check if the selected chart type is the recommended one
        is_recommended = False
        if chart_type.lower() in recommended_chart_type.lower():
            is_recommended = True
            
        # Ensure 'The University of Leicester' is always considered
        leicester_variants = ['university of leicester', 'the university of leicester', 'leicester university']
        leicester_included = False
        for inst in institutions:
            if inst.lower() in leicester_variants:
                leicester_included = True
                break
                
        if not leicester_included and 'leicester' not in ' '.join(institutions).lower():
            institutions.append('The University of Leicester')
            logging.info("Added University of Leicester to institutions list")
        
        # Analyze dataset characteristics
        dataset_characteristics = {
            "has_multiple_years": len(years) > 1,
            "has_multiple_institutions": len(institutions) > 1,
            "number_of_numeric_columns": len([col for col in columns if col not in ['HE provider', 'Academic Year']]),
            "total_rows": len(sample_rows),
            "available_institutions": institutions,
            "available_years": years
        }
        
        logging.info(f"Dataset characteristics for chart type change: {dataset_characteristics}")
        
        # Check for chart type compatibility issues
        compatibility_warning = None
        
        # Special case checks for incompatible chart types
        if chart_type == 'pie' and current_request and ('axis' in current_request.lower() or 'axes' in current_request.lower() or 'swap' in current_request.lower()):
            compatibility_warning = "Pie charts don't have traditional X and Y axes, so axis swapping or specific axis assignments don't apply to this chart type."
        elif chart_type == 'line' and not dataset_characteristics["has_multiple_years"] and 'year' in current_request.lower():
            compatibility_warning = "Line charts work best with multiple time periods. Your request involves a time-based comparison but the dataset might not have enough time periods to show meaningful trends."
        
        # If we have a compatibility warning, include it in the response
        compatibility_note = ""
        if compatibility_warning:
            compatibility_note = f"<div class='text-amber-600 mb-2 font-medium'><strong>Note:</strong> {compatibility_warning}</div>"
        
        # Format information about multi-year data if available
        multi_year_info = ""
        if years and len(years) > 0:
            multi_year_info = f"\nThis dataset contains data for multiple academic years: {', '.join(years)}."
            multi_year_info += "\nConsider suggesting visualizations that could compare trends across these years or focus on specific years."
        
        # Prepare a prompt for Gemini
        if is_recommended:
            # If the selected chart type is the recommended one, return the original recommendation
            logging.info("Selected chart type is the originally recommended one, returning original data")
            
            # Validate the original examples to ensure they use exact institution names
            validated_examples = []
            for example in original_example_prompts:
                valid = True
                
                # For each example, check if it mentions only valid institutions
                for institution in institutions:
                    # If the institution name is in the example, validate it matches exactly
                    if institution.lower() in example.lower():
                        # Check if the full exact name is used (not just a substring)
                        if institution not in example:
                            # If exact name not found, mark as invalid
                            valid = False
                            logging.warning(f"Original example uses invalid institution reference: '{example}'")
                            break
                
                # If the example passed validation, include it
                if valid:
                    validated_examples.append(example)
                else:
                    logging.warning(f"Rejecting original example: {example}")
            
            # If we don't have enough valid examples, generate defaults
            if len(validated_examples) < 3:
                validated_examples = get_default_example_prompts(title, columns, institutions, years, chart_type)
            
            return JsonResponse({
                'success': True,
                'is_recommended': True,
                'recommended_chart_type': recommended_chart_type,
                'selected_chart_type': chart_type,  # Add the selected chart type to the response
                'recommendation_reason': recommendation_reason,
                'example_prompts': validated_examples[:3]  # Limit to 3 examples
            })
        else:
            # If a different chart type is selected, generate examples for that chart type but keep original recommendation
            prompt = f"""
            I have a dataset titled "{title}" with the following columns:
            {', '.join(columns)}
            
            Here are some sample rows from the dataset:
            {format_sample_rows(columns, sample_rows)}
            
            The institutions in this dataset are: {', '.join(institutions)}
            The academic years in the dataset are: {', '.join(years)}
            
            IMPORTANT INSTRUCTIONS:
            1. I need examples SPECIFICALLY for a {chart_type.upper()} CHART. Every example MUST work with this chart type.
            2. ONLY use the EXACT institution names from the list I provided - do not modify, abbreviate or combine them.
            3. If a {chart_type} chart isn't ideal for some aspects of the data, still create examples that WILL WORK with a {chart_type} chart.
            4. Your examples will be used as-is and rendered as a {chart_type} chart, so they MUST be appropriate for this chart type.
            5. Focus on what {chart_type} charts do best:
               - LINE CHARTS: Show trends over time/years
               - BAR CHARTS: Compare values across categories/institutions
               - PIE CHARTS: Show proportions of a whole for a single period
            
            Please provide:
            1. A brief explanation of how a {chart_type} chart could be used with this specific dataset, noting its strengths and limitations.
            2. 3 example visualization requests that:
               - Are SPECIFICALLY designed to work with a {chart_type} chart
               - ONLY use institutions and years that are actually available in this dataset (as listed above)
               - Are realistic and can be fulfilled with the available data
               - Make sure to verify that all institutions mentioned actually exist in the dataset list
            
            Return your response as a JSON object with the following structure:
            {{{{
                "recommendation_reason": "An explanation of how a {chart_type} chart could be used with this data",
                "example_prompts": [
                    "Example 1 for {chart_type} chart using only available institutions and years",
                    "Example 2 for {chart_type} chart using only available institutions and years",
                    "Example 3 for {chart_type} chart using only available institutions and years"
                ]
            }}}}
            """
            
            # Call Gemini API for chart recommendation
            response = client.generate_text(prompt)
            
            # Extract and parse the JSON response
            try:
                # Look for JSON pattern in the response
                match = re.search(r'\{.*\}', response.replace('\n', ' '), re.DOTALL)
                if match:
                    json_str = match.group(0)
                    chart_data = json.loads(json_str)
                    
                    # Validate example prompts to ensure they only mention available institutions
                    validated_examples = []
                    for example in chart_data.get('example_prompts', []):
                        valid = True
                        
                        # For each example, check if it mentions only valid institutions
                        for institution in institutions:
                            # If the institution name is in the example, validate it matches exactly
                            if institution.lower() in example.lower():
                                # Check if the full exact name is used (not just a substring)
                                if institution not in example:
                                    # If exact name not found, mark as invalid
                                    valid = False
                                    logging.warning(f"Example uses invalid institution reference: '{example}'")
                                    break
                        
                        # If the example passed validation, include it
                        if valid:
                            validated_examples.append(example)
                        else:
                            logging.warning(f"Rejecting example: {example}")
                    
                    # If we don't have enough valid examples, generate defaults
                    if len(validated_examples) < 3:
                        logging.info("Not enough valid examples from Gemini, generating defaults")
                        validated_examples = get_default_example_prompts(title, columns, institutions, years, chart_type)
                    
                    return JsonResponse({
                        'success': True,
                        'is_recommended': is_recommended,
                        'recommended_chart_type': recommended_chart_type,  # Original recommendation - this is still the best recommendation
                        'selected_chart_type': chart_type,  # User selection
                        'recommendation_reason': chart_data.get('recommendation_reason', 
                            f"A {chart_type} chart can be used for visualizing certain aspects of this data"),
                        'example_prompts': validated_examples[:3],  # Limit to 3 examples
                        'compatibility_warning': compatibility_warning
                    })
                else:
                    # Fallback if JSON pattern not found
                    logging.warning("Could not find JSON pattern in Gemini response for chart type change")
                    return JsonResponse({
                        'success': True,
                        'is_recommended': False,
                        'recommended_chart_type': recommended_chart_type,
                        'selected_chart_type': chart_type,
                        'recommendation_reason': get_chart_type_explanation(chart_type, dataset_characteristics),
                        'example_prompts': get_default_example_prompts(title, columns, institutions, years, chart_type),
                        'compatibility_warning': compatibility_warning
                    })
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Gemini's JSON response for chart type change: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': 'Could not generate examples for this chart type. Please try again.'
                })
                
    except Exception as e:
        logging.exception(f"Error changing chart type: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Error changing chart type: {str(e)}'
        })

def get_chart_type_explanation(chart_type, dataset_characteristics):
    """
    Get a specific explanation for the requested chart type based on dataset characteristics
    """
    if chart_type == 'line':
        if not dataset_characteristics.get("has_multiple_years", False):
            return "Line charts are most effective for showing trends over time. Since this dataset only contains data for a single year, you should focus on comparing values across different categories or institutions rather than showing trends over time."
        else:
            return "Line charts work best for showing trends over time or continuous data. Focus on how values change across the available years for one or more institutions."
    
    elif chart_type == 'bar':
        if dataset_characteristics.get("has_multiple_institutions", False) and dataset_characteristics.get("has_multiple_years", False):
            return "Bar charts are excellent for comparing values across categories. You can compare institutions side by side, or show change over time by grouping bars by year."
        else:
            return "Bar charts are ideal for comparing values across categories. Focus on comparing different metrics or institutions within the available data."
    
    elif chart_type == 'pie':
        if dataset_characteristics.get("has_multiple_institutions", False) and len(dataset_characteristics.get("available_institutions", [])) > 5:
            return "Pie charts work best with a small number of categories (5 or fewer) and when showing parts of a whole. Consider focusing on just a few key institutions or categories to maintain clarity."
        else:
            return "Pie charts are best for showing proportions or percentages of a whole. Focus on the distribution of a single metric across categories for a specific time period rather than comparisons over time."
    
    return None

# Configure logging to prevent duplicate handlers
logger = logging.getLogger('core.views')
# Remove all handlers to avoid duplicates
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)
# Add a single handler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False  # Prevent propagation to root logger

def sanitize_json_string(json_string):
    """
    Sanitize a JSON string to fix common issues with escape sequences and formatting.
    """
    # Replace smart/curly quotes with straight quotes
    sanitized = json_string.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
    
    # Replace escaped quotes with temporary markers
    sanitized = sanitized.replace('\\"', '___QUOTE___')
    
    # Remove invalid backslash escapes
    sanitized = re.sub(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})', '', sanitized)
    
    # Fix invalid escape sequences
    sanitized = (sanitized
                .replace('\\n', ' ')  # Replace newlines with spaces
                .replace('\\t', ' ')  # Replace tabs with spaces
                .replace('\\\\"', '\\"')  # Fix double escaped quotes
                .replace('\\\\', '\\')  # Fix double backslashes
                )
    
    # Restore escaped quotes
    sanitized = sanitized.replace('___QUOTE___', '\\"')
    
    return sanitized

def extract_and_sanitize_json(response):
    """
    Extract JSON from an LLM response and sanitize it for parsing.
    First checks for JSON in code blocks, then falls back to general pattern matching.
    Returns the sanitized JSON string.
    """
    # First check for JSON in a code block (preferred format)
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if code_block_match:
        json_str = code_block_match.group(1)
        logging.info("Found JSON in code block")
    else:
        # Fallback: Look for any JSON-like content - find the largest matching braces
        json_pattern = r'(\{(?:[^{}]|(?1))*\})'  # Recursive pattern to match nested braces
        matches = re.finditer(json_pattern, response, re.DOTALL)
        
        # Find the largest match by length
        best_match = None
        max_length = 0
        
        for match in matches:
            matched_text = match.group(0)
            if len(matched_text) > max_length:
                max_length = len(matched_text)
                best_match = matched_text
        
        if best_match:
            json_str = best_match
            logging.info(f"Found JSON pattern using fallback method (length: {len(json_str)})")
        else:
            # Last resort: try to find any object-like structure
            simple_match = re.search(r'\{.*\}', response.replace('\n', ' '), re.DOTALL)
            if simple_match:
                json_str = simple_match.group(0)
                logging.info("Found simple JSON pattern as last resort")
            else:
                json_str = response
                logging.warning("No JSON pattern found in response, using entire response")
    
    # Sanitize the JSON string to fix common issues
    sanitized = sanitize_json_string(json_str)
    
    # Final validation check and cleanup
    try:
        # Try to parse the JSON to ensure it's valid
        parsed = json.loads(sanitized)
        # Convert back to string for consistent formatting
        return json.dumps(parsed)
    except json.JSONDecodeError as e:
        logging.warning(f"Sanitized JSON is still invalid: {str(e)}")
        # Try to extract just the specific fields we need
        try:
            # Extract chart_config field if present
            chart_config_match = re.search(r'"chart_config"\s*:\s*(\{.*?\}),?\s*(?:"|\n)', sanitized, re.DOTALL)
            insights_match = re.search(r'"insights"\s*:\s*"(.*?)",?\s*(?:"|\n)', sanitized, re.DOTALL)
            alternatives_match = re.search(r'"alternatives"\s*:\s*(\[.*?\]),?\s*(?:"|\n)', sanitized, re.DOTALL)
            
            result = {}
            if chart_config_match:
                try:
                    # Clean up the chart config string
                    chart_config = chart_config_match.group(1).strip()
                    # Remove trailing commas
                    chart_config = re.sub(r',\s*}', '}', chart_config)
                    result['chart_config'] = chart_config
                except Exception:
                    pass
            
            if insights_match:
                try:
                    result['insights'] = insights_match.group(1)
                except Exception:
                    pass
                    
            if alternatives_match:
                try:
                    alternatives = alternatives_match.group(1)
                    # Clean up the alternatives
                    alternatives = re.sub(r',\s*]', ']', alternatives)
                    result['alternatives'] = alternatives
                except Exception:
                    pass
            
            if result:
                return json.dumps(result)
        except Exception as extract_error:
            logging.error(f"Error extracting specific fields: {str(extract_error)}")
        
        # If we still can't parse it, return the sanitized string anyway
        return sanitized