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
from rest_framework import serializers
from .models import Combination, Apparel

class CombinationSerializer(serializers.ModelSerializer):
    apparels = serializers.PrimaryKeyRelatedField(queryset=Apparel.objects.all(), many=True)

    class Meta:
        model = Combination
        fields = '__all__'
