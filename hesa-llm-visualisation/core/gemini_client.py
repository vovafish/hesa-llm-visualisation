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
        Analyze a HESA data query using Gemini to extract important entities.
        
        Args:
            query: The natural language query to analyze
            
        Returns:
            Dictionary containing extracted information (institutions, years, data_request)
        """
        if not self.api_key:
            logger.error("Cannot analyze query: No API key provided")
            return self._fallback_analysis(query, error="API key not configured")
        
        current_year = datetime.now().year
        
        prompt = f"""
        Analyze the following query about HESA (Higher Education Statistics Agency) data and extract the relevant information.
        
        Query: "{query}"
        
        Extract and format the following details:
        1. Institutions: List only the specific institution names mentioned (e.g., "University of Leicester", "University of London")
        2. Years: Extract any years mentioned (individual years or ranges in format YYYY)
        3. Start year and end year:
           - If a specific year like "2017" is mentioned, use it as start_year with no end_year
           - If a year range like "2018-2020" is mentioned, use 2018 as start_year and 2020 as end_year
           - If "past X years" is mentioned, calculate from current year ({current_year}) backwards
             For example, "past 2 years" = start_year: {current_year-2}, end_year: {current_year}
        4. Data request: List specific categories of data requested (e.g., "student", "enrollment", "postgraduates", etc.)
        
        Current year is {current_year}.
        
        VERY IMPORTANT INSTRUCTIONS:
        - Be extremely precise with institution names - extract ONLY proper university names without additional text
        - University of Leicester MUST ALWAYS be included in the institutions list, even if not mentioned in the query
        - NEVER include phrases like "for the past X years" as part of institution names
        - For "past X years", calculate years correctly by subtracting X-1 from current year for start_year
        - If a specific data category is mentioned, include it in data_request as a list of terms
        - List individual data request terms as separate items (e.g., ["student", "enrollment"] not ["student enrollment"])
        - ONLY return the JSON object without ANY additional text
        
        Return your analysis as JSON with the following structure:
        {{
            "institutions": ["University Name 1", "University Name 2"],
            "years": ["2015", "2016"],
            "start_year": "2015",
            "end_year": "2016",
            "data_request": ["student", "enrollment"]
        }}
        """
        
        try:
            logger.info(f"Sending query to Gemini API: {query}")
            
            # Initialize the Gemini client
            client = genai.Client(api_key=self.api_key)
            
            # Send the prompt to Gemini
            response = client.models.generate_content(
                model='gemini-1.5-pro',  # Updated to 1.5 version to fix 404 error
                contents=prompt
            )
            
            # Extract the text content from the response
            if hasattr(response, 'text') and response.text:
                content = response.text
                logger.info(f"Received Gemini response: {content}")
                
                # Parse the JSON from the response
                try:
                    # Find JSON within the response if it's wrapped in text
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        result = json.loads(json_content)
                    else:
                        result = json.loads(content)
                    
                    # Ensure University of Leicester is in the institutions list
                    if "institutions" in result:
                        # Clean up institution names to remove phrases like "For The Past 2 Years"
                        cleaned_institutions = []
                        for inst in result["institutions"]:
                            # Only keep the actual institution name
                            # Extract just the university name part
                            if "university of" in inst.lower():
                                match = re.match(r'(University of \w+)', inst, re.IGNORECASE)
                                if match:
                                    cleaned_institutions.append(match.group(1))
                            elif "university" in inst.lower():
                                match = re.match(r'(\w+ University)', inst, re.IGNORECASE)
                                if match:
                                    cleaned_institutions.append(match.group(1))
                            else:
                                cleaned_institutions.append(inst)
                        
                        # Remove duplicates
                        cleaned_institutions = list(set(cleaned_institutions))
                        
                        # Check if Leicester is included
                        leicester_included = False
                        for inst in cleaned_institutions:
                            if "leicester" in inst.lower():
                                leicester_included = True
                                break
                        
                        if not leicester_included:
                            cleaned_institutions.append("University of Leicester")
                            
                        result["institutions"] = cleaned_institutions
                    
                    # Ensure years and start/end years are properly set
                    if "years" in result and result["years"]:
                        # Keep the years as provided
                        pass
                        
                    # Make sure data_request is a list
                    if "data_request" in result and isinstance(result["data_request"], str):
                        result["data_request"] = [result["data_request"]]
                    
                    logger.info(f"Gemini parsed entities: {result}")
                    return result
                except json.JSONDecodeError as e:
                    error_msg = f"Error parsing JSON from Gemini response: {str(e)}"
                    logger.error(f"{error_msg} - Response: {content}")
                    return self._fallback_analysis(query, error=error_msg)
            else:
                error_msg = "Empty or invalid response from Gemini API"
                logger.error(error_msg)
                return self._fallback_analysis(query, error=error_msg)
                
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            logger.error(error_msg)
            return self._fallback_analysis(query, error=error_msg)
    
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