# Generated by Django 2.0.2 on 2018-03-06 23:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('projects', '0008_project_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='attribute_num',
            field=models.CharField(default=3, max_length=50),
        ),
    ]
