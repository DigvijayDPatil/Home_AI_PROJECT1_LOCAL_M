from django.urls import path
from . import views

urlpatterns = [
    path('', views.redesign_home, name='redesign_home'),
]