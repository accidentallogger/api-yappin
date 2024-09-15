from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User, Apparel
from .serializers import UserSerializer, ApparelSerializer
import base64


class CreateUserView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "User created successfully!"}, status=status.HTTP_201_CREATED)
        # Log errors for debugging
        print(serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class CheckEmailExistsView(APIView):
    def get(self, request, email):
        exists = User.objects.filter(email=email).exists()
        return Response(exists)

class CheckNameExistsView(APIView):
    def get(self, request, name):
        exists = User.objects.filter(name=name).exists()
        return Response(exists)

class GetPasswordByEmailView(APIView):
    def get(self, request, email):
        try:
            user = User.objects.get(email=email)
            return Response(user.password)
        except User.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class GetPasswordByNameAndEmailView(APIView):
    def get(self, request, name, email):
        try:
            user = User.objects.get(name=name, email=email)
            return Response(user.password)
        except User.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class GetUserIdByEmailView(APIView):
    def get(self, request, email):
        try:
            user = User.objects.get(email=email)
            return Response(user.id)
        except User.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class UpdateUserView(APIView):
    def put(self, request):
        try:
            user = User.objects.get(id=request.data['id'])
            serializer = UserSerializer(user, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class GetUserByIdView(APIView):
    def get(self, request, id):
        try:
            user = User.objects.get(id=id)
            serializer = UserSerializer(user)
            return Response(serializer.data)
        except User.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class GetUserByEmailView(APIView):
    def get(self, request, email):
        try:
            user = User.objects.get(email=email)
            serializer = UserSerializer(user)
            return Response(serializer.data)
        except User.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class GetGenderByEmailView(APIView):
    def get(self, request, email):
        try:
            user = User.objects.get(email=email)
            return Response(user.gender)
        except User.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class UpdateUserProfileView(APIView):
    def put(self, request):
        try:
            user = User.objects.get(id=request.data['id'])
            serializer = UserSerializer(user, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(True)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response(False)

# Apparel-related views

class CreateApparelTableView(APIView):
    def post(self, request):
        return Response(status=status.HTTP_201_CREATED)

class FetchImgView(APIView):
    def get(self, request, upperLower):
        try:
            apparel = Apparel.objects.get(upper_lower=upperLower)
            return Response(apparel.image)
        except Apparel.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class CreateApparelView(APIView):
    def post(self, request):
        serializer = ApparelSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ApparelByTypeAndEmailView(APIView):
    serializer_class = ApparelSerializer

    def get_queryset(self):
        email = self.kwargs['email']
        category = self.kwargs['category']
        return Apparel.objects.filter(email=email, category=category)

class GetAllApparelsByEmail(APIView):
    
    def get(self, request, email, format=None):
        try:
            # Find the user by email
            user = User.objects.get(emailaddress=email)
            
            # Get all apparel items belonging to the user
            apparels = Apparel.objects.filter(ownership=user)

            # Convert images to base64 strings
            apparel_list = []
            for apparel in apparels:
                apparel_image_data = apparel.image.read()  # Read the image file
                encoded_image = base64.b64encode(apparel_image_data).decode('utf-8')  # Convert to base64 string
                apparel_list.append(encoded_image)  # Add encoded image to the list

            return Response(apparel_list, status=status.HTTP_200_OK)
        
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
