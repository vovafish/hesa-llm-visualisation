from django.shortcuts import render
from .llm_utils import generate_response
from .data_processing import CSVProcessor
from .utils.query_processor import parse_llm_response, apply_data_operations
from .utils.chart_generator import generate_chart
from django.contrib import messages

def query_view(request):
    response = None
    chart_data = None
    available_datasets = []
    
    # Initialize CSV processor
    processor = CSVProcessor()
    
    # Get list of available datasets
    available_datasets = processor.get_available_datasets()
    
    if request.method == 'POST':
        try:
            user_query = request.POST.get('query')
            if user_query:
                # Get LLM response
                llm_response = generate_response(user_query)
                
                # Parse response
                operations = parse_llm_response(llm_response)
                
                if 'error' not in operations:
                    # Get and process data
                    df = processor.get_dataset(available_datasets[0])  # Use first available dataset for now
                    processed_data = apply_data_operations(df, operations)
                    
                    # Generate chart
                    chart_data = generate_chart(processed_data, operations.get('chart_type', 'bar'))
                
                response = llm_response
        except Exception as e:
            messages.error(request, f"Error processing query: {str(e)}")
    
    return render(request, 'core/query.html', {
        'response': response,
        'chart_data': chart_data,
        'available_datasets': available_datasets
    })