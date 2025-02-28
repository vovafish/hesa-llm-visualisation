from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('documentation/', views.documentation, name='documentation'),
    path('process_query/', views.process_query, name='process_query'),
    path('download/<str:format>/', views.download_data, name='download_data'),
]
