"""
Mock AI Client for entity extraction.
This is a local implementation that simulates AI responses without using external APIs.
"""

import re
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MockAIClient:
    """A mock AI client that simulates intelligent entity extraction without requiring an API key."""
    
    def __init__(self):
        """Initialize the mock AI client with pattern matching rules."""
        self.institution_patterns = [
            (r'university of (\w+(?:\s\w+)*)', lambda m: f"University of {m.group(1).title()}"),
            (r'(\w+(?:\s\w+)*) university', lambda m: f"{m.group(1).title()} University"),
            (r'(\w+) college', lambda m: f"{m.group(1).title()} College"),
        ]
        
        # Special institutions to look for explicitly
        self.special_institutions = {
            "london": "University of London",
            "leicester": "University of Leicester",
            "oxford": "University of Oxford",
            "cambridge": "University of Cambridge",
        }
        
        # Data request categories and their keywords
        self.data_categories = {
            "enrollment": ["enrollment", "enrolment", "students", "postgraduates", "undergraduates", 
                          "how many", "number of", "student numbers", "studying"],
            "accommodation": ["accommodation", "housing", "term-time", "residence", "living", "halls"],
            "demographics": ["demographics", "age", "gender", "ethnicity", "nationality", "international"],
            "finances": ["finance", "tuition", "fees", "funding", "scholarship", "income", "spending", "budget"],
            "staff": ["staff", "faculty", "professor", "lecturer", "teaching", "researchers"],
            "research": ["research", "publications", "grants", "projects", "papers"],
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to extract entities like institutions, years, and data type.
        
        Args:
            query: The natural language query to analyze
            
        Returns:
            Dictionary containing extracted entities
        """
        logger.info(f"Mock AI analyzing query: {query}")
        
        # Convert query to lowercase for easier matching
        query_lower = query.lower()
        
        # Extract institutions
        institutions = self._extract_institutions(query_lower)
        
        # Extract years
        years = self._extract_years(query_lower)
        
        # Extract start and end years
        start_year, end_year = self._extract_year_bounds(query_lower, years)
        
        # Determine data request categories
        data_request = self._determine_data_categories(query_lower)
        
        # Create response
        response = {
            "institutions": institutions,
            "years": years,
            "start_year": start_year,
            "end_year": end_year,
            "data_request": data_request
        }
        
        logger.info(f"Mock AI extracted entities: {response}")
        return response
    
    def _extract_institutions(self, query: str) -> List[str]:
        """Extract institution names from the query."""
        extracted_institutions = set()
        
        # Check for special institutions first
        for keyword, institution in self.special_institutions.items():
            if keyword in query:
                extracted_institutions.add(institution)
        
        # Apply regex patterns
        for pattern, formatter in self.institution_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                # Get the institution name
                institution = formatter(match)
                
                # Clean up institution name to only include actual institution name
                if "university of" in institution.lower():
                    clean_match = re.match(r'(University of \w+)', institution, re.IGNORECASE)
                    if clean_match:
                        institution = clean_match.group(1)
                elif "university" in institution.lower():
                    clean_match = re.match(r'(\w+ University)', institution, re.IGNORECASE)
                    if clean_match:
                        institution = clean_match.group(1)
                
                # Don't add if it's just a part of "how many" or similar phrases
                if not any(exclude in institution.lower() for exclude in ["how", "what", "when", "where", "why", "many"]):
                    extracted_institutions.add(institution)
        
        # Always include Leicester if not already found
        leicester_found = False
        for inst in extracted_institutions:
            if "leicester" in inst.lower():
                leicester_found = True
                break
                
        if not leicester_found:
            extracted_institutions.add("University of Leicester")
            
        return list(extracted_institutions)
    
    def _extract_years(self, query: str) -> List[str]:
        """Extract years from the query."""
        years = []
        
        # Match 4-digit years
        year_matches = re.finditer(r'(?<!\d)(\d{4})(?!\d)', query)
        for match in year_matches:
            years.append(match.group(1))
        
        # Match academic years (e.g., 2015/16)
        academic_year_matches = re.finditer(r'(\d{4})[/-](\d{2})', query)
        for match in academic_year_matches:
            full_year = match.group(1)
            abbreviated_year = match.group(2)
            
            years.append(full_year)
            
            # Convert abbreviated year to full year
            if len(abbreviated_year) == 2:
                prefix = "20" if int(abbreviated_year) < 50 else "19"
                second_year = prefix + abbreviated_year
                years.append(second_year)
        
        # Match year ranges (e.g., 2015-2019)
        range_matches = re.finditer(r'(\d{4})\s*-\s*(\d{4})', query)
        for match in range_matches:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            
            # Add all years in the range
            for year in range(start_year, end_year + 1):
                years.append(str(year))
                
        # Check for "past X years" pattern
        past_years_match = re.search(r'past\s+(\d+)\s+years?', query)
        if past_years_match:
            num_years = int(past_years_match.group(1))
            current_year = datetime.now().year
            
            # Add years from (current - (num_years-1)) to current
            # For example, "past 2 years" means current year and previous year
            for year in range(current_year - (num_years - 1), current_year + 1):
                years.append(str(year))
        
        # Remove duplicates and sort
        return sorted(list(set(years)))
    
    def _extract_year_bounds(self, query: str, years: List[str]) -> tuple:
        """Extract start and end years from the query."""
        start_year = None
        end_year = None
        
        # If no years, check for special patterns
        if not years:
            # Check for "past X years" pattern
            past_years_match = re.search(r'past\s+(\d+)\s+years?', query)
            if past_years_match:
                num_years = int(past_years_match.group(1))
                current_year = datetime.now().year
                start_year = str(current_year - (num_years - 1))  # Correct calculation for "past X years"
                end_year = str(current_year)
            return start_year, end_year
        
        # Check for explicit range pattern
        range_match = re.search(r'(\d{4})\s*-\s*(\d{4})', query)
        if range_match:
            start_year = range_match.group(1)
            end_year = range_match.group(2)
            return start_year, end_year
            
        # If we have years, use them to determine bounds
        years_as_int = [int(y) for y in years]
        if years_as_int:
            start_year = str(min(years_as_int))
            if len(years_as_int) > 1:
                end_year = str(max(years_as_int))
            # If only one year, only set start_year
        
        return start_year, end_year
    
    def _determine_data_categories(self, query: str) -> List[str]:
        """Determine the data categories being requested."""
        data_terms = []
        
        # Check for individual data terms
        terms_to_check = ["student", "enrollment", "enrolment", "postgraduate", "undergraduate", 
                          "accommodation", "demographic", "finance", "staff", "research"]
                          
        for term in terms_to_check:
            if term in query:
                data_terms.append(term)
        
        # Special case for "postgraduates" 
        if "postgraduate" in query or "postgraduates" in query:
            if "student" not in data_terms:
                data_terms.append("student")
            if "enrollment" not in data_terms and "enrolment" not in data_terms:
                data_terms.append("enrollment")
        
        # If no specific data request found, use general_data
        if not data_terms:
            data_terms = ["general_data"]
            
        return data_terms 