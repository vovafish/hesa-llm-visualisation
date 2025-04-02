"""
Google Gemini API client for entity extraction and query analysis.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from google import genai

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google's Gemini API for natural language processing."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client with API key."""
        # Try to get API key from environment variable if not provided
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("No Gemini API key provided. API calls will not work.")
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a natural language query using the Gemini API.
        
        Args:
            query: The natural language query string
            
        Returns:
            dict: Dictionary containing extracted entities:
                - institutions: List of institution names
                - years: List of years mentioned
                - start_year: Start year of range (if applicable)
                - end_year: End year of range (if applicable)
                - data_request: List of data categories requested
                - original_institutions: Original institution names with potential typos
                - corrected_institutions: Corrected institution names
                - has_typos: Boolean indicating if typos were detected
        """
        import requests
        import json
        import re
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"Analyzing query with Gemini API: {query}")
        
        if not self.api_key:
            logger.error("No Gemini API key provided")
            raise ValueError("Gemini API key is required")
            
        # Construct the prompt
        system_prompt = """
        You are a helpful assistant that extracts structured information from a user's query about higher education statistics.
        
        Extract the following information and output it in JSON format:
        1. Institutions mentioned (e.g., "University of Oxford", "University of Leicester")
        2. Years mentioned (e.g., "2020", "2020/21")
        3. If a year range is given, identify the start_year and end_year
        4. The type of data being requested (e.g., "student numbers", "enrollment", "graduates")
        
        IMPORTANT: You need to detect and correct ONLY SPELLING typos in institution names and years.
        
        For institutions, follow these strict rules:
        - ONLY consider a term to have a typo if it has clear misspellings (e.g., "Univercity" → "University", "Liecester" → "Leicester")
        - DO NOT consider "Oxford University" vs "University of Oxford" as a typo - these are different ways to refer to the same institution
        - DO NOT mark an institution as having a typo if it's spelled correctly but uses a different naming convention
        - DO NOT transform a city name like "london" into "The University of London"
        - PRESERVE the original terms from the query - if user says "london", keep it as "london" in the institutions list
        - If there are no spelling errors, original_institutions should exactly match institutions, and has_institution_typos should be false
        
        For years:
        - ONLY mark years as having typos if they have clear numerical errors (e.g., "20025" → "2025")
        - If no year typos exist, original_years should exactly match years, and has_year_typos should be false
        
        Be careful about interpreting years in academic context:
        - If the query contains phrases like "starting in [YEAR]" or "beginning in [YEAR]", interpret [YEAR] as the start of an academic year.
        - If the query contains phrases like "end of [YEAR]" or "ending in [YEAR]", interpret [YEAR] as the end of an academic year.
        - For ranges like "2016 to 2017", treat both as starting years of academic years.
        - If there's no clarification, assume a year refers to the starting year of an academic year.
        - For "past X years", calculate the years based on the current year (2025).
        
        Output format:
        {
          "institutions": ["University X", "london"],
          "original_institutions": ["Universcity X", "london"],
          "has_institution_typos": true,
          "years": ["2019/20", "2020/21"],
          "original_years": ["20019", "2020"],
          "has_year_typos": true,
          "start_year": "2019",
          "end_year": "2020",
          "data_request": ["student_enrollment", "graduation_rates"]
        }
        
        If no specific institutions are mentioned, return an empty list for institutions and original_institutions.
        If no typos were detected, original_institutions should match institutions, and has_institution_typos should be false.
        The same applies to years and original_years.
        
        For the data_request field, categorize the request into one of these categories:
        - student_enrollment
        - student_demographics
        - graduation_rates
        - staff_data
        - research_data
        - general_data (default if no specific category is identified)
        """
        
        # The actual user query
        user_prompt = f"Extract information from this query, paying special attention to academic year conventions and potential typos in institution names and years: '{query}'"
        
        try:
            # Gemini API endpoint - Updated to use the most current endpoint
            url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
            
            # Request payload
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": system_prompt},
                            {"text": user_prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 2048
                }
            }
            
            # Headers with API key
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            # Make the request
            response = requests.post(url, json=payload, headers=headers)
            
            # Check for successful response
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                raise Exception(f"Gemini API returned status code {response.status_code}: {response.text}")
                
            # Parse the response
            response_data = response.json()
            
            # Extract text from response
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                if 'content' in response_data['candidates'][0]:
                    response_text = response_data['candidates'][0]['content']['parts'][0]['text']
                else:
                    logger.error("No content in Gemini API response")
                    raise Exception("No content in Gemini API response")
            else:
                logger.error("No candidates in Gemini API response")
                raise Exception("No candidates in Gemini API response")
                
            # Find the JSON response within the text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.error(f"Failed to extract JSON from response: {response_text}")
                raise Exception("Failed to extract JSON from the Gemini API response")
                
            # Parse the JSON data
            result = json.loads(json_match.group(0))
            
            # Default values for missing fields
            result.setdefault('institutions', [])
            result.setdefault('original_institutions', [])
            result.setdefault('has_institution_typos', False)
            result.setdefault('years', [])
            result.setdefault('original_years', [])
            result.setdefault('has_year_typos', False)
            result.setdefault('data_request', ['general_data'])
            
            # Always ensure University of Leicester is included if any institution is mentioned
            if result['institutions'] and 'University of Leicester' not in result['institutions']:
                result['institutions'].append('University of Leicester')
                
            # Fix "in University" issue by filtering out partial names
            result['institutions'] = [
                inst for inst in result['institutions'] 
                if len(inst.split()) > 1  # Must be at least two words
                and inst.lower() != "in university"  # Explicitly exclude "in university"
            ]
                
            # If we don't have typo information but have institutions, assume no typos
            if 'original_institutions' not in result or not result['original_institutions']:
                result['original_institutions'] = result['institutions'].copy()
                result['has_institution_typos'] = False
            
            # If we don't have typo information but have years, assume no typos
            if 'original_years' not in result or not result['original_years']:
                result['original_years'] = result['years'].copy()
                result['has_year_typos'] = False
                
            # Handle special case for "past X years"
            if 'start_year' not in result or 'end_year' not in result:
                # Check if the query contains "past X years"
                past_years_match = re.search(r'past\s+(\d+)\s+years?', query.lower())
                if past_years_match:
                    num_years = int(past_years_match.group(1))
                    # Use current year from datetime
                    current_year = datetime.now().year
                    end_year = current_year
                    start_year = current_year - num_years
                    
                    result['end_year'] = str(end_year)
                    result['start_year'] = str(start_year)
                    
                    # Expand years array to include all years in the range
                    if not result['years']:
                        for year in range(start_year, end_year + 1):
                            # Add both plain year and academic year format
                            result['years'].append(str(year))
                            result['years'].append(f"{year}/{str(year+1)[2:4]}")
            
            # Apply our academic year logic for special cases if Gemini didn't handle it well
            self._process_academic_year_logic(query, result)
            
            logger.info(f"Extracted information: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing query with Gemini API: {str(e)}")
            raise
    
    def _process_academic_year_logic(self, query, result):
        """
        Process academic year logic based on context clues in the query.
        This handles cases like:
        - "starting in 2017" → 2017/18
        - "end of 2017" → 2016/17
        - "2016 to 2017" → 2016/17 - 2017/18 (both as starting years)
        
        Args:
            query: The original query string
            result: The result dictionary to modify
        """
        import re
        import logging
        
        logger = logging.getLogger(__name__)
        query_lower = query.lower()
        
        # Extract years with context
        start_patterns = [
            r'start(?:ing|s|ed)?\s+(?:in|from|at)?\s+(?:the\s+)?(?:year\s+)?(\d{4})',
            r'begin(?:ning|s)?\s+(?:in|from|at)?\s+(?:the\s+)?(?:year\s+)?(\d{4})',
            r'from\s+(?:the\s+)?(?:year\s+)?(\d{4})'
        ]
        
        end_patterns = [
            r'end(?:ing|s|ed)?\s+(?:in|at|of)?\s+(?:the\s+)?(?:year\s+)?(\d{4})',
            r'finish(?:ing|es|ed)?\s+(?:in|at)?\s+(?:the\s+)?(?:year\s+)?(\d{4})',
            r'(?:in|at)\s+(?:the\s+)?end\s+of\s+(?:the\s+)?(?:year\s+)?(\d{4})'
        ]
        
        # Handle range logic for years without context
        if 'start_year' in result and 'end_year' in result and result['start_year'] is not None and result['end_year'] is not None:
            # Range years should be treated as starting years of academic years
            start_year = result['start_year']
            end_year = result['end_year']
            
            try:
                # Clear existing years array to rebuild it with correct academic years
                result['years'] = []
                
                # Add academic years for the range
                for year in range(int(start_year), int(end_year) + 1):
                    academic_year = f"{year}/{str(year+1)[2:4]}"
                    result['years'].append(academic_year)
                
                logger.info(f"Processed year range: {start_year}-{end_year} as academic years: {', '.join(result['years'])}")
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing year range: {e}. start_year={start_year}, end_year={end_year}")
            
            return
        
        # Explicitly defined start years
        for pattern in start_patterns:
            start_match = re.search(pattern, query_lower)
            if start_match:
                year = start_match.group(1)
                try:
                    academic_year = f"{year}/{str(int(year)+1)[2:4]}"
                    
                    if academic_year not in result['years']:
                        result['years'].append(academic_year)
                        result['start_year'] = year
                        logger.info(f"Processed starting year {year} as academic year {academic_year}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing starting year {year}: {e}")
        
        # Explicitly defined end years
        for pattern in end_patterns:
            end_match = re.search(pattern, query_lower)
            if end_match:
                year = end_match.group(1)
                try:
                    previous_year = str(int(year) - 1)
                    academic_year = f"{previous_year}/{year[2:4]}"
                    
                    if academic_year not in result['years']:
                        result['years'].append(academic_year)
                        result['end_year'] = year
                        logger.info(f"Processed ending year {year} as academic year {academic_year}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing ending year {year}: {e}")
        
        # For single years without context, treat as starting years
        plain_year_pattern = r'\b(20\d{2})\b'
        years_mentioned = re.findall(plain_year_pattern, query)
        
        # Skip years that were already processed with context
        for year in years_mentioned:
            # Check if this year was already handled as start/end year
            already_processed = False
            for pattern in start_patterns + end_patterns:
                if re.search(pattern, query_lower) and re.search(pattern, query_lower).group(1) == year:
                    already_processed = True
                    break
            
            if not already_processed:
                try:
                    # No context, treat as starting year
                    academic_year = f"{year}/{str(int(year)+1)[2:4]}"
                    if academic_year not in result['years']:
                        result['years'].append(academic_year)
                        # If no explicit start_year is set, use this as default
                        if 'start_year' not in result or result['start_year'] is None:
                            result['start_year'] = year
                        logger.info(f"Processed plain year {year} as academic year {academic_year}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing plain year {year}: {e}")
                
        # Make sure years are unique
        result['years'] = list(set(result['years']))
    
    def get_completion(self, prompt):
        """
        Get a completion from the Gemini API based on the provided prompt.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            str: The completion text from Gemini
        """
        import requests
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("Sending completion request to Gemini API")
        
        if not self.api_key:
            logger.error("No Gemini API key provided")
            raise ValueError("Gemini API key is required")
        
        try:
            # Updated Gemini API endpoint
            url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
            
            # Request payload
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 4096
                }
            }
            
            # Headers with API key
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            # Make the request
            response = requests.post(url, json=payload, headers=headers)
            
            # Check for successful response
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                raise Exception(f"Gemini API returned status code {response.status_code}: {response.text}")
                
            # Parse the response
            response_data = response.json()
            
            # Extract text from response
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                if 'content' in response_data['candidates'][0]:
                    response_text = response_data['candidates'][0]['content']['parts'][0]['text']
                    return response_text
                else:
                    logger.error("No content in Gemini API response")
                    raise Exception("No content in Gemini API response")
            else:
                logger.error("No candidates in Gemini API response")
                raise Exception("No candidates in Gemini API response")
                
        except Exception as e:
            logger.error(f"Error getting completion from Gemini API: {str(e)}")
            raise
    
    def _fallback_analysis(self, query: str, error: str = None) -> Dict[str, Any]:
        """Basic fallback analysis if the API call fails."""
        logger.warning(f"Using fallback analysis for query. Error: {error}")
        
        current_year = datetime.now().year
        
        # Very simple fallback with explicit inclusion of University of Leicester
        result = {
            "institutions": ["University of Leicester"],
            "years": [],
            "start_year": None,
            "end_year": None,
            "data_request": []
        }
        
        # Basic regex patterns for extracting institutions and years
        if query:
            # Extract institution names
            result["institutions"] = []
            
            # Check for University of X patterns
            uni_patterns = re.finditer(r'university\s+of\s+(\w+)', query.lower())
            for match in uni_patterns:
                uni_name = f"University of {match.group(1).title()}"
                if uni_name not in result["institutions"]:
                    result["institutions"].append(uni_name)
            
            # Check for X University patterns
            uni2_patterns = re.finditer(r'(\w+)\s+university', query.lower())
            for match in uni2_patterns:
                uni_name = f"{match.group(1).title()} University"
                if uni_name not in result["institutions"]:
                    result["institutions"].append(uni_name)
            
            # Ensure Leicester is always included
            leicester_included = False
            for inst in result["institutions"]:
                if "leicester" in inst.lower():
                    leicester_included = True
                    break
            
            if not leicester_included:
                result["institutions"].append("University of Leicester")
            
            # Extract specific years
            year_pattern = r'\b(20\d{2})\b'
            years = re.findall(year_pattern, query)
            if years:
                result["years"] = years
                result["start_year"] = min(years)
                if len(years) > 1:
                    result["end_year"] = max(years)
            
            # Check for year range
            range_match = re.search(r'(\d{4})\s*-\s*(\d{4})', query)
            if range_match:
                start = range_match.group(1)
                end = range_match.group(2)
                result["start_year"] = start
                result["end_year"] = end
                # Create a years list from start to end
                result["years"] = [str(y) for y in range(int(start), int(end)+1)]
            
            # Check for "past X years" pattern
            past_years_match = re.search(r'past\s+(\d+)\s+years?', query.lower())
            if past_years_match:
                num_years = int(past_years_match.group(1))
                result["start_year"] = str(current_year - (num_years - 1))
                result["end_year"] = str(current_year)
                # Create years list
                result["years"] = [str(y) for y in range(current_year - (num_years - 1), current_year + 1)]
            
            # Extract data types requested
            data_terms = []
            
            # Check for common data request terms
            for term in ["student", "enrollment", "enrolment", "postgraduate", "undergraduate", 
                         "accommodation", "demographic", "finance", "staff", "research"]:
                if term in query.lower():
                    data_terms.append(term)
            
            if data_terms:
                result["data_request"] = data_terms
            else:
                result["data_request"] = ["general_data"]
        
        # Add error information if provided
        if error:
            result["api_error"] = error
            
        return result 