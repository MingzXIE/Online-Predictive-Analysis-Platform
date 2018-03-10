from django.shortcuts import render, redirect
from .models import Project
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from . import forms
from django.http import HttpResponseRedirect
from django.core.files import File
from . import knn
import os


def project_list(request):
    projects = Project.objects.all().order_by('date')
    return render(request, 'projects/project_list.html',{'projects':projects})


def project_detail(request, project_id):
    project = Project.objects.get(pk=project_id)
    return render(request, 'projects/project_detail.html',{'project':project})


@login_required(login_url="accounts:login")
def create_initial(request):
    return render(request, 'projects/project_initial.html')


@login_required(login_url="accounts:login")
def classification_create(request):
    if request.method == 'POST':
        form=forms.CreateProject(request.POST, request.FILES)
        if form.is_valid():
            instance=form.save(commit=False)
            instance.creator=request.user
            instance.save()
            project_id = instance.id
            return redirect(reverse('projects:classification_process', args=[project_id]))
    else:
        form = forms.CreateProject()
    return render(request, 'projects/classification_create.html', { 'form':form })


@login_required(login_url="accounts:login")
def regression_create(request):
    if request.method == 'POST':
        form=forms.CreateProject(request.POST, request.FILES)
        if form.is_valid():
            instance=form.save(commit=False)
            instance.creator=request.user
            instance.save()
            project_id = instance.id
            return redirect(reverse('projects:regression_process', args=[project_id]))
    else:
        form = forms.CreateProject()
    return render(request, 'projects/regression_create.html', { 'form':form })


@login_required(login_url="accounts:login")
def classification_process(request, project_id):
    project = Project.objects.get(pk=project_id)
    project.predict_type = "classification"
    project.save()
    return render(request, 'projects/classification_process.html',{ 'project':project })


@login_required(login_url="accounts:login")
def regression_process(request, project_id):
    project = Project.objects.get(pk=project_id)
    project.predict_type = "regression"
    project.save()
    return render(request, 'projects/regression_process.html',{ 'project':project })


@login_required(login_url="accounts:login")
def classification_train(request, project_id):
    project = Project.objects.get(pk=project_id)
    training_file = project.labeled_data.name
    training_path = os.path.join('media/',training_file)
    attribute_str = project.attribute_num
    attribute_num = int(attribute_str)
    project.error_rate = knn.knn_test(training_path,attribute_num, 0.1)
    project.save()
    return render(request, 'projects/classification_train.html',{'project':project})


@login_required(login_url="accounts:login")
def regression_train(request, project_id):
    project = Project.objects.get(pk=project_id)
    training_file = project.labeled_data.name
    training_path = os.path.join('media/',training_file)
    project.error_rate = knn.knn_test(training_path, 3, 3, 0.05)
    project.save()
    return render(request, 'projects/regression_train.html',{'project':project})


@login_required(login_url="accounts:login")
def classification_predict(request, project_id):
    project = Project.objects.get(pk=project_id)
    if request.method == 'POST':
        project.unlabeled_data = request.POST['unlabeled_data']
        project.save()
        return redirect(reverse('projects:classification_result', args=[project_id]))
    else:
        return render(request, 'projects/classification_predict.html',{'project':project})


# @login_required(login_url="accounts:login")
# def regression_predict(request, project_id):
#     project = Project.objects.get(pk=project_id)
#
#     return render(request, 'projects/regression_predict.html',{'project':project})


@login_required(login_url="accounts:login")
def classification_result(request, project_id):
    project = Project.objects.get(pk=project_id)
    training_file = project.labeled_data.name
    training_path = os.path.join('media/',training_file)
    attribute_str = project.attribute_num
    attribute_num = int(attribute_str)
    array_to_predict = project.unlabeled_data
    project.training_result = knn.knn_predict(array_to_predict,training_path,attribute_num)
    project.save()
    return render(request, 'projects/classification_result.html',{'project':project})

@login_required(login_url="accounts:login")
def project_addcomment(request, project_id):
    project = Project.objects.get(pk=project_id)
    if request.method == 'POST':
        project.comment = request.POST['comment']
        project.save()
        return redirect(reverse('projects:list'))
    else:
        return render(request,'projects/project_comment.html',{'project':project})
