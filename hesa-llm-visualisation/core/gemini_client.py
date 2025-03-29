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
        
        If the query mentions "past X years", calculate the years based on the current year (2025).
        
        Output format:
        {
          "institutions": ["University X", "University Y"],
          "years": ["2019/20", "2020/21"],
          "start_year": "2019",
          "end_year": "2020",
          "data_request": ["student_enrollment", "graduation_rates"]
        }
        
        If no specific institutions are mentioned, return an empty list.
        For the data_request field, categorize the request into one of these categories:
        - student_enrollment
        - student_demographics
        - graduation_rates
        - staff_data
        - research_data
        - general_data (default if no specific category is identified)
        """
        
        # The actual user query
        user_prompt = f"Extract information from this query: '{query}'"
        
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
            result.setdefault('years', [])
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
            
            logger.info(f"Extracted information: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing query with Gemini API: {str(e)}")
            raise
    
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