from django.db import models
from django.contrib.auth.models import User


class Result(models.Model):
    owner = models.ForeignKey(to=User, on_delete=models.CASCADE)
    heartrate = models.IntegerField()
    ecg = models.JSONField()
    time = models.DateTimeField()


class Video(models.Model):
    owner = models.ForeignKey(to=User, on_delete=models.CASCADE)
    video_file = models.FileField(upload_to='videos/TESTAPP/testapp/')
