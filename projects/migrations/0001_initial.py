# Generated by Django 2.0.2 on 2018-03-02 17:08

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100)),
                ('description', models.TextField(blank=True)),
                ('labeled_data', models.TextField()),
                ('algorithm_selected', models.CharField(blank=True, max_length=5)),
                ('error_date', models.CharField(blank=True, max_length=10)),
                ('training_result', models.TextField(blank=True)),
                ('unlabeled_data', models.TextField(blank=True)),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('command', models.TextField(blank=True)),
                ('creator', models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]