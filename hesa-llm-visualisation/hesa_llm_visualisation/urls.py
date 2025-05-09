"""
URL configuration for hesa_llm_visualisation project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# Remove admin import since we're not using it
# from django.contrib import admin
from django.urls import path, include
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.generic.base import RedirectView
from django.conf import settings
from django.conf.urls.static import static
import os
import json
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Clean up timing queries file from parent directory if it exists
def cleanup_timing_file():
    try:
        # Check if there's a timing_queries.json file in the parent directory
        parent_timing_file = Path(settings.BASE_DIR).parent / 'timing_queries.json'
        project_timing_file = Path(settings.BASE_DIR) / 'timing_queries.json'
        
        # If file exists in parent directory
        if parent_timing_file.exists():
            logger.warning(f"Found timing file in wrong location: {parent_timing_file}")
            
            # If we also have a file in the project directory, merge the data
            if project_timing_file.exists():
                logger.info("Merging timing data from parent directory to project directory")
                try:
                    # Load data from both files
                    with open(parent_timing_file, 'r') as f:
                        parent_data = json.load(f)
                    
                    with open(project_timing_file, 'r') as f:
                        project_data = json.load(f)
                    
                    # Merge queries
                    if 'queries' in parent_data and 'queries' in project_data:
                        project_data['queries'].extend(parent_data['queries'])
                        
                        # Renumber IDs
                        for i, query in enumerate(project_data['queries']):
                            query['id'] = i + 1
                        
                        # Save merged data
                        with open(project_timing_file, 'w') as f:
                            json.dump(project_data, f, indent=2)
                            
                        logger.info(f"Successfully merged timing data from {len(parent_data['queries'])} queries")
                except Exception as e:
                    logger.error(f"Error merging timing data: {str(e)}")
            else:
                # If no file in project directory, just move the parent file
                logger.info("Moving timing file from parent directory to project root")
                try:
                    with open(parent_timing_file, 'r') as f:
                        parent_data = json.load(f)
                        
                    with open(project_timing_file, 'w') as f:
                        json.dump(parent_data, f, indent=2)
                        
                    logger.info("Successfully moved timing data to project root")
                except Exception as e:
                    logger.error(f"Error moving timing data: {str(e)}")
            
            # Try to delete the parent file
            try:
                os.remove(parent_timing_file)
                logger.info(f"Deleted timing file from wrong location: {parent_timing_file}")
            except Exception as e:
                logger.error(f"Error deleting timing file from wrong location: {str(e)}")
        
        # Create an empty file in the project root if none exists
        if not project_timing_file.exists():
            logger.info("Creating empty timing file in project root")
            try:
                with open(project_timing_file, 'w') as f:
                    json.dump({"queries": []}, f, indent=2)
                logger.info(f"Created new timing file at: {project_timing_file}")
            except Exception as e:
                logger.error(f"Error creating timing file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error during timing file cleanup: {str(e)}")

# Run cleanup on import
cleanup_timing_file()

urlpatterns = [
    path('', include('core.urls')),
    path('favicon.ico', RedirectView.as_view(url=staticfiles_storage.url('favicon.ico'))),
]

# Add static file serving during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
