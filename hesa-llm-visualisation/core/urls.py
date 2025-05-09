from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    # Make AI dashboard the default landing page
    path('', views.ai_dashboard, name='home'),
    # Keep old dashboard path for backward compatibility but redirect to AI dashboard
    #path('dashboard/', views.dashboard, name='dashboard'),
    path('documentation/', views.documentation, name='documentation'),
    path('process_query/', views.process_query, name='process_query'),
    path('process_hesa_query/', views.process_hesa_query, name='process_hesa_query'),
    path('select_file_source/', views.select_file_source, name='select_file_source'),
    path('dataset_details/<str:group_id>/', views.dataset_details, name='dataset_details'),
    path('download/<str:format>/', views.download_data, name='download_data'),
    path('api/chart/<str:chart_type>/', views.get_chart_data, name='get_chart_data'),
    path('api/process-hesa-query/', views.process_hesa_query, name='api_process_hesa_query'),
    path('api/select-file-source/', views.select_file_source, name='api_select_file_source'),
    # Keep AI dashboard at its original URL too for consistency
    path('ai-dashboard/', views.ai_dashboard, name='ai_dashboard'),
    path('process_gemini_query/', views.process_gemini_query, name='process_gemini_query'),
    path('ai_dataset_details/', views.ai_dataset_details, name='ai_dataset_details'),
    path('api/select-file-source/<str:file_id>/', views.select_file_source, name='select_file_source'),
    path('ai_dataset_details/<int:dataset_id>/', views.ai_dataset_details, name='ai_dataset_details'),
    path('visualization_api/', views.visualization_api, name='visualization_api'),
    path('save_feedback/', views.save_feedback, name='save_feedback'),
    path('upload-data/', views.upload_data_view, name='upload_data'),
    path('process-upload/', views.process_upload, name='process_upload'),
    path('record_query_timing/', views.record_query_timing, name='record_query_timing'),
]
