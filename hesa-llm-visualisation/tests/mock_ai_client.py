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
            "enrollment": ["enrollment", "enrolment", "students", "undergraduate", "undergraduates", "postgraduates", 
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
        """Determine data request categories from the query."""
        categories = []
        specific_keywords = []
        
        # Extract specific keywords from the query that should be included directly
        keyword_checks = ["undergraduate", "postgraduate", "international", "domestic", "part-time", "full-time"]
        for keyword in keyword_checks:
            if keyword in query.lower():
                specific_keywords.append(keyword)
        
        # Check for each category
        for category, keywords in self.data_categories.items():
            if any(keyword in query.lower() for keyword in keywords):
                categories.append(category)
        
        # Add generic categories if nothing specific was found
        if not categories:
            categories = ["student", "data"]
            
        # Add the specific keywords to the returned categories
        return categories + specific_keywords
    
    def get_chart_recommendation(self, visualization_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a chart recommendation based on the data.
        
        """
        logger.info("Mock AI generating chart recommendation")
        
        # Get the columns and guess what kind of data we have
        columns = visualization_data.get('columns', [])
        rows = visualization_data.get('rows', [])
        
        # Default to bar chart
        chart_type = "bar"
        reason = "A bar chart is good for comparing values across categories."
        
        # If we have time/year data, suggest a line chart
        if any('year' in col.lower() for col in columns):
            chart_type = "line"
            reason = "A line chart is recommended for showing trends over time."
        
        # If we have percentage data or small number of rows, suggest pie chart
        elif any('percent' in col.lower() for col in columns) and len(rows) <= 7:
            chart_type = "pie"
            reason = "A pie chart works well for showing proportions of a whole with a small number of categories."
        
        # For geographical data, suggest a map
        elif any(col.lower() in ['country', 'region', 'location'] for col in columns):
            chart_type = "map"
            reason = "A map visualization is ideal for geographical data."
        
        # Example prompts based on the chart type
        example_prompts = {
            "bar": [
                "Compare values across categories",
                "Show student numbers by university",
                "Display enrollment by degree type"
            ],
            "line": [
                "Show trends over time",
                "Display changes in enrollment from 2018 to 2022",
                "Visualize growth in student numbers"
            ],
            "pie": [
                "Show composition of student population",
                "Display proportion of international students",
                "Visualize budget allocation"
            ],
            "map": [
                "Show student distribution by region",
                "Visualize geographical spread of universities",
                "Display student origins by country"
            ]
        }
        
        return {
            "recommended_chart_type": chart_type,
            "recommendation_reason": reason,
            "example_prompts": example_prompts.get(chart_type, ["Generic visualization prompt"])
        }
    
    def generate_visualization_config(self, visualization_data: Dict[str, Any], query: str, chart_type: str) -> Dict[str, Any]:
        """
        Generate a visualization configuration for the specified chart type.
      
        """
        logger.info(f"Mock AI generating {chart_type} visualization")
        
        # Extract data
        title = visualization_data.get('title', 'HESA Data Visualization')
        columns = visualization_data.get('columns', [])
        rows = visualization_data.get('rows', [])
        
        if not columns or not rows:
            return {
                "error": "Insufficient data for visualization",
                "chart_config": {
                    "type": "bar",
                    "data": {
                        "labels": [],
                        "datasets": []
                    }
                }
            }
        
        # For simplicity, assume first column is labels, and second column is data
        labels = [row[0] for row in rows if len(row) > 0]
        
        # Initialize chart configuration
        chart_config = {
            "type": chart_type,
            "data": {
                "labels": labels,
                "datasets": []
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": title
                    }
                }
            }
        }
        
        # Generate datasets based on columns
        if len(columns) > 1:
            for i in range(1, min(len(columns), 4)):  # Limit to 3 data columns
                if i < len(columns):
                    data = [float(row[i]) if i < len(row) and row[i] and row[i].replace('.', '').isdigit() else 0 
                           for row in rows]
                    
                    dataset = {
                        "label": columns[i],
                        "data": data,
                        "backgroundColor": [
                            "#4e73df", "#1cc88a", "#36b9cc", "#f6c23e", "#e74a3b", 
                            "#858796", "#5a5c69", "#6610f2", "#fd7e14", "#20c9a6"
                        ][:len(data)]
                    }
                    chart_config["data"]["datasets"].append(dataset)
        
        # Generate insights
        insights = self._generate_insights(chart_type, columns, rows, title)
        
        return {
            "chart_config": chart_config,
            "insights": insights
        }
    
    def _generate_insights(self, chart_type: str, columns: List[str], rows: List[str], title: str) -> str:
        """Generate insights for the visualization."""
        if not rows or not columns:
            return "<p>No insights available for the provided data.</p>"
            
        insights = []
        
        # Find the highest value
        if len(columns) > 1 and all(len(row) > 1 for row in rows):
            max_index = 0
            max_value = 0
            max_column = columns[1] if len(columns) > 1 else "value"
            
            for i, row in enumerate(rows):
                if len(row) > 1:
                    try:
                        value = float(row[1]) if isinstance(row[1], str) and row[1].replace('.', '').isdigit() else 0
                        if value > max_value:
                            max_value = value
                            max_index = i
                    except (ValueError, TypeError):
                        pass
            
            if max_index < len(rows) and len(rows[max_index]) > 0:
                insights.append(f"<p>{rows[max_index][0]} has the highest {max_column} at {max_value}.</p>")
        
        # Add a generic insight about the data
        insights.append(f"<p>This visualization shows {title.lower()}.</p>")
        
        return "".join(insights) 