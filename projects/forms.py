from django import forms
from . import models


class CreateProject(forms.ModelForm):
    class Meta:
        model = models.Project
        fields = ['title', 'description', 'labeled_data']


class ClassificationPredict(forms.ModelForm):
    class Meta:
        model = models.Project
        fields = ['unlabeled_data']
