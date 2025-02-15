from django.shortcuts import render
from .llm_utils import generate_response

def query_view(request):
    response = None
    if request.method == 'POST':
        user_query = request.POST.get('query')
        if user_query:
            response = generate_response(user_query)
    
    return render(request, 'core/query.html', {
        'response': response
    })