# Generated by Django 4.2.6 on 2023-11-26 04:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='video',
            name='video_file',
            field=models.FileField(upload_to='videos/TESTAPP/testapp/'),
        ),
    ]
