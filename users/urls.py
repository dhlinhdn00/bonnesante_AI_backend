from django.urls import path
from rest_framework.schemas import get_schema_view
from django.views.generic import TemplateView
import torch
from . import views

urlpatterns = [
    path('login/', views.login_api),
    path('user/', views.get_user_data),
    path('register/', views.register_api),
    path('model/', views.execute_model),
    path('history/', views.get_user_results),
    path('swagger-ui/', TemplateView.as_view(
        template_name='swagger-ui.html',
        extra_context={'schema_url': 'api_schema'}
    ), name='swagger-ui'),
    path('api_schema', get_schema_view(title='API Schema',
         description='Guided API Schema'), name='api_schema'),
    path('syspath/', views.get_sys_path),

    # path('deploy/',views.deploy)
]
