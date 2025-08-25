# apps/ai_service/urls.py
from django.urls import path
from .views import ai_detect

urlpatterns = [
    path("api/ai/detect/", ai_detect, name="ai-detect"),
]
