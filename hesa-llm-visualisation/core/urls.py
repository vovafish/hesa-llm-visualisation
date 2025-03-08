from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('documentation/', views.documentation, name='documentation'),
    path('process_query/', views.process_query, name='process_query'),
    path('download/<str:format>/', views.download_data, name='download_data'),
    path('test-charts/', views.test_charts, name='test_charts'),
    path('api/chart/<str:chart_type>/', views.get_chart_data, name='get_chart_data'),
]
