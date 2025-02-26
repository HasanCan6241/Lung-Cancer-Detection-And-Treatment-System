# core/urls.py

from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .views import ChatbotView, ChatMessageView

urlpatterns = [
    # Ana sayfalar
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('faq/', views.faq, name='faq'),
    path('contact/', views.contact, name='contact'),

    # Kullanıcı yönetimi
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),

    # Kullanıcı paneli
    path('dashboard/', views.dashboard, name='dashboard'),
    path('predictions/', views.prediction_history, name='prediction_history'),
    path('tahmin/<int:prediction_id>/', views.prediction_detail, name='prediction_detail'),
    path('tahmin/<int:prediction_id>/indir/', views.download_prediction, name='download_prediction'),
    path("export_predictions/", views.export_predictions, name="export_predictions"),
    path("analysis_dashboard/", views.analysis_dashboard, name="analysis_dashboard"),
    path('treatment-recommendation/<int:prediction_id>/', views.treatment_recommendation, name='treatment_recommendation'),
    path('get-recommendation/<int:prediction_id>/', views.get_recommendation, name='get_recommendation'),
    path('chat/', ChatbotView.as_view(), name='chatbot'),
    path('chat/message/', ChatMessageView.as_view(), name='chat_message'),


]