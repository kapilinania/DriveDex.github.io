from django.urls import path
from .views import index, result, bankfd

urlpatterns = [
    path('', bankfd, name='index'),
    path('result', result, name='result')
]
