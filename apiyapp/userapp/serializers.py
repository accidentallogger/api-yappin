from rest_framework import serializers
from .models import User, Apparel

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'name', 'email','password', 'gender','Age', 'bio']

class ApparelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Apparel
        fields = ['id', 'ownership', 'color', 'material', 'upper_lower', 'type', 'image']
