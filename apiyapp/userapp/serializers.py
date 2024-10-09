from rest_framework import serializers
from .models import User, Apparel, Combination

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'name', 'email', 'gender', 'Age', 'bio','password']


    def create(self, validated_data):
        user = User(
            name=validated_data['name'],
            email=validated_data['email'],
            gender=validated_data['gender'],
            Age=validated_data['Age'],
            bio=validated_data['bio'],
            password=validated_data['password']
        )
        user.save()
        return user

    def update(self, instance, validated_data):
        instance.name = validated_data.get('name', instance.name)
        instance.email = validated_data.get('email', instance.email)
        instance.gender = validated_data.get('gender', instance.gender)
        instance.Age = validated_data.get('Age', instance.age)
        instance.bio = validated_data.get('bio', instance.bio)

        password = validated_data.get('password', None)
        if password:
            instance.set_password(password)  # Hash the password if updated
        instance.save()
        return instance



class ApparelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Apparel
        fields = ['ownership', 'color','material', 'upper_lower', 'occasion', 'image']

    def create(self, validated_data):
        # Directly create the apparel instance without base64 logic
        return Apparel.objects.create(**validated_data)
class CombinationSerializer(serializers.ModelSerializer):
    apparels = ApparelSerializer(many=True)  # Display full details of related apparels

    class Meta:
        model = Combination
        fields = '__all__'
