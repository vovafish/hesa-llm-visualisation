from django.shortcuts import render
from .llm_utils import generate_response
from .data_processing import CSVProcessor
from django.contrib import messages

def query_view(request):
    response = None
    available_datasets = []
    
    # Initialize CSV processor
    processor = CSVProcessor()
    
    # Get list of available datasets
    available_datasets = processor.get_available_datasets()
    
    if request.method == 'POST':
        user_query = request.POST.get('query')
        if user_query:
            response = generate_response(user_query)
    
    return render(request, 'core/query.html', {
        'response': response,
        'available_datasets': available_datasets
    })