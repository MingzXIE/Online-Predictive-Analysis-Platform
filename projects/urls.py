from django.urls import path
from . import views
from django.conf.urls import handler500

handler500 = views.page_error
app_name = 'projects'
urlpatterns = [
    path('', views.project_list, name="list"),
    path('initial/', views.create_initial, name="initial"),
    path('classification_create/', views.classification_create, name="classification_create"),
    path('regression_create/', views.regression_create, name="regression_create"),
    path('classification_process/<int:project_id>/', views.classification_process, name="classification_process"),
    path('regression_process/<int:project_id>/', views.regression_process, name="regression_process"),
    path('classification_train/<int:project_id>/', views.classification_train, name="classification_train"),
    path('regression_train/<int:project_id>/', views.regression_train, name="regression_train"),
    path('classification_predict/<int:project_id>/', views.classification_predict, name="classification_predict"),
    path('regression_predict/<int:project_id>/', views.regression_predict, name="regression_predict"),
    path('classification_result/<int:project_id>/', views.classification_result, name="classification_result"),
    path('regression_result/<int:project_id>/', views.regression_result, name="regression_result"),
    path('project_comment/<int:project_id>/', views.project_addcomment, name="comment"),
    path('individual/<slug:user_name>/', views.individual_list, name="individual_list"),
    path('<int:project_id>/', views.project_detail, name="detail"),


]
