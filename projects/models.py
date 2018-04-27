from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Project(models.Model):
    title = models.CharField(max_length = 100)
    description = models.TextField(blank=True)
    predict_type = models.CharField(max_length=100,blank=True)
    creator = models.ForeignKey(User, default=None, on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True) #automatically added when created.
    labeled_data = models.FileField(upload_to='labeled_data')
    attribute_num = models.IntegerField(null=True)
    accurate_list = models.TextField(blank=True)
    accurate_rate = models.CharField(blank=True, max_length = 50)
    mean_squared_error = models.CharField(blank=True, max_length = 50)
    algorithm_selected = models.CharField(blank=True, max_length = 50)
    unlabeled_data = models.CharField(max_length=500)
    predict_result = models.TextField(blank=True)
    comment = models.TextField(blank=True)



    def __str__(self):
        return self.title
