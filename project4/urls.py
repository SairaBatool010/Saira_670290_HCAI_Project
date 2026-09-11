from django.urls import path

from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("report/", views.download_report, name="download_report"),
    path("start/", views.start_study, name="start_study"),
    path("consent/", views.consent, name="consent"),
    path("instructions/", views.instructions, name="instructions"),
    path("task/", views.task, name="task"),
    path("complete/", views.complete, name="complete"),
]
