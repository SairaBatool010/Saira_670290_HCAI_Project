# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Saira Batool", "matriculation": "670290"},
       ]
    
    projects = [
        #{"name": "Home", "url_name": "home:index"},
        #{"name": "Home 2", "url_name": "home:index"},
        {"name": "Project 1: Supervised Learning Interface", "url_name": "project1:index"},
        {"name": "Project 2: Explainability", "url_name": "project2:index"},
        {"name": "Project 3: Active Learning for Learning-to-Defer", "url_name": "project3:index"},
        {"name": "Project 4: Preference Elicitation", "url_name": "project4:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request)) 