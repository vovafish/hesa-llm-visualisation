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
            return JsonResponse({'error': 'No query provided'}, status=400)

        # Log the incoming query
        logger = logging.getLogger(__name__)
        logger.info(f"Processing query: {query}")

        # For testing, return a simple response
        return JsonResponse({
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
        return JsonResponse({
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
            return JsonResponse({'error': 'No data available'}, status=404)
        
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
            return JsonResponse({'error': 'Invalid format'}, status=400)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

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
        
        return JsonResponse({
            'data': {
                'image': f'data:image/png;base64,{graphic}'
            }
        })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

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
            return JsonResponse({'status': 'error', 'error': 'No query provided'}, status=400)
        
        # Parse the query to extract components
        logger.info("Parsing query to extract components")
        query_info = parse_hesa_query(query)
        
        if not query_info:
            logger.warning(f"Failed to parse query: {query}")
            return JsonResponse({
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
            return JsonResponse({
                'status': 'error', 
                'error': f"No CSV files found matching pattern: {query_info['file_pattern']}"
            }, status=404)
        
        logger.info(f"Found {len(file_matches)} matching CSV files: {file_matches}")
        
        # Track all files used and data extracted
        all_file_info = []
        all_data = []
        all_columns = None
        
        # Process each year in the query
        for year in query_info['years']:
            # Find the specific file for the current year
            logger.info(f"Looking for file matching year: {year}")
            target_file = find_file_for_year(file_matches, year)
            
            if not target_file:
                logger.warning(f"No data found for year: {year}")
                continue  # Skip this year but continue with others
            
            logger.info(f"Found file for requested year {year}: {target_file}")
            
            # Get just the filename for display
            file_name = Path(target_file).name
            
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
            result = extract_provider_data(target_file, query_info['he_providers'])
            
            if not result:
                logger.warning(f"No data found for providers: {query_info['he_providers']} in year {year}")
                continue  # Skip this year but continue with others
            
            logger.info(f"Data extracted successfully for year {year}: {len(result['data'])} rows found")
            
            # Store the columns from the first successful result
            if all_columns is None:
                all_columns = result['columns']
            
            # Add year information to each row
            for row in result['data']:
                row['Year'] = year  # Add the year as a new column
                all_data.append(row)
            
            # Store file info
            all_file_info.append({
                'year': year,
                'raw_file': target_file,
                'using_cleaned_file': cleaned_file_path.exists(),
                'cleaned_file_path': str(cleaned_file_path) if cleaned_file_path.exists() else None,
                'file_name': file_name
            })
        
        # Check if we found any data
        if not all_data:
            logger.warning(f"No data found for any year in range: {query_info['years']}")
            return JsonResponse({
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
        return JsonResponse({
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
        return JsonResponse({
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
    """Find CSV files in the raw_files directory that match the given pattern."""
    logger = logging.getLogger(__name__)
    
    # Get path to raw_files directory
    raw_files_dir = Path(settings.BASE_DIR) / 'data' / 'raw_files'
    
    logger.info(f"Searching for CSV files in: {raw_files_dir}")
    
    # Check if directory exists
    if not raw_files_dir.exists():
        logger.error(f"Directory does not exist: {raw_files_dir}")
        return []
    
    # List all CSV files in the directory
    all_csv_files = list(raw_files_dir.glob('*.csv'))
    logger.info(f"Found {len(all_csv_files)} CSV files in directory")
    
    # Extract keywords from file pattern
    pattern_keywords = [keyword.lower() for keyword in file_pattern.lower().split()]
    
    # Find all CSV files that contain most of the pattern keywords
    matching_files = []
    for file_path in all_csv_files:
        filename = file_path.name.lower()
        logger.info(f"Checking file: {filename}")
        
        # Count how many keywords match
        keyword_matches = 0
        for keyword in pattern_keywords:
            # Check for variations to improve matching
            if keyword in filename:
                keyword_matches += 1
            elif keyword == 'enrollment' and 'enrolment' in filename:
                keyword_matches += 1  # Handle UK/US spelling differences
            elif keyword == 'enrollments' and 'enrolments' in filename:
                keyword_matches += 1
            elif keyword == 'term-time' and 'term time' in filename:
                keyword_matches += 1
        
        # If at least half of the keywords match, consider it a match
        match_threshold = max(1, len(pattern_keywords) // 2)
        if keyword_matches >= match_threshold:
            matching_files.append(str(file_path))
            logger.info(f"File matched ({keyword_matches} keywords): {file_path}")
    
    return matching_files

def find_file_for_year(file_matches, year):
    """Find the specific file for the requested year."""
    logger = logging.getLogger(__name__)
    
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
    
    for file_path in file_matches:
        for year_format in academic_year_formats:
            if year_format in file_path:
                logger.info(f"Found matching year format '{year_format}' in {file_path}")
                return file_path
    
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
                # Try reading the cleaned file directly
                df = pd.read_csv(cleaned_file_path)
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
        
        # Log the first few lines to help with debugging
        logger.info(f"First 10 lines of the file:")
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
            header_df = pd.read_csv(file_path, skiprows=header_row, nrows=1)
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
            df = pd.read_csv(file_path, skiprows=data_start_row, names=column_names)
            logger.info(f"Read data with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            # Try with different approach
            try:
                df = pd.read_csv(file_path, skiprows=range(0, data_start_row), names=column_names)
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
                        if isinstance(value, str):
                            value = value.replace(',', '')  # Remove commas
                        dataset['data'].append(float(value))
                        value_found = True
                        break
                    except (ValueError, TypeError):
                        pass
            
            # If no value found for this year, add null for continuity
            if not value_found:
                dataset['data'].append(None)
        
        datasets.append(dataset)
    
    return {
        'labels': years,
        'datasets': datasets
    }

def get_random_color():
    """Generate a random color for chart datasets."""
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'rgb({r}, {g}, {b})'