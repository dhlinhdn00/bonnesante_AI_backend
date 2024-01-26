from django.contrib.auth.models import User
from rest_framework import serializers, validators
from .models import Result, Video


class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['email', 'username', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User(
            email=validated_data['email'],
            username=validated_data['username']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user


class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = ['owner', 'video_file']
        # fields = ['user_id', 'video_file']


class ResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result
        fields = ('owner', 'ecg', 'time')
