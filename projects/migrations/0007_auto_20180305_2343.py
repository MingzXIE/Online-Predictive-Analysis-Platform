# Generated by Django 2.0.2 on 2018-03-05 23:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('projects', '0006_auto_20180305_2249'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='labeled_data',
            field=models.FileField(upload_to='labeled_data'),
        ),
        migrations.AlterField(
            model_name='project',
            name='unlabeled_data',
            field=models.FileField(blank=True, upload_to='unlabeled_data'),
        ),
    ]
