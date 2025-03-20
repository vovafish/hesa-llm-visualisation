from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('documentation/', views.documentation, name='documentation'),
    path('process_query/', views.process_query, name='process_query'),
    path('process_hesa_query/', views.process_hesa_query, name='process_hesa_query'),
    path('select_file_source/', views.select_file_source, name='select_file_source'),
    path('download/<str:format>/', views.download_data, name='download_data'),
    path('test-charts/', views.test_charts, name='test_charts'),
    path('query-builder/', views.query_builder, name='query_builder'),
    path('api/chart/<str:chart_type>/', views.get_chart_data, name='get_chart_data'),
    path('api/process-hesa-query/', views.process_hesa_query, name='api_process_hesa_query'),
]
