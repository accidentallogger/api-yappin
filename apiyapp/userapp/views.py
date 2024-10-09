from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User, Apparel
from .serializers import UserSerializer, ApparelSerializer
import base64
from django.http import JsonResponse
from apiyapp import settings 
from .models import Apparel
from .models import User, Combination, Apparel
from django.core.files.base import ContentFile
from .recommendationsystem import *
import random
import os
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
            return Response(str(user.id))
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


class CreateApparelView(APIView):
    def post(self, request):
        # Get the base64 image data from the request
        image_base64 = request.data.get('image')

        # Prepare the data for the serializer
        apparel_data = {
            'ownership': request.data.get('ownership'),
            'upper_lower': request.data.get('upper_lower'),
            'occasion': request.data.get('occasion'),
            'color':request.data.get('color'),
            'material':request.data.get('material'),
        }

        # Decode the base64 image and save it as a file
        if image_base64:
            format, imgstr = image_base64.split(';base64,')  # Split the string to get the data
            ext = format.split('/')[-1]  # Get file extension (e.g., 'png', 'jpg')
            image_data = base64.b64decode(imgstr)  # Decode the base64 string
            image_file = ContentFile(image_data, name=f"uploaded_image.{ext}")

            # Add the image file to the apparel data
            apparel_data['image'] = image_file

        # Create the serializer with the prepared data
        serializer = ApparelSerializer(data=apparel_data)
        
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "Apparel entry created successfully!"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ApparelByTypeAndEmailView(APIView):
    serializer_class = ApparelSerializer

    def get(self, request, email, upper_lower):
        try:
            # Fetch apparel items based on email and upper_lower
            apparels = Apparel.objects.filter(ownership__email=email, upper_lower=upper_lower)
            
            # Convert apparel images to base64 strings
            apparel_list = []
            for apparel in apparels:
                apparel_image_data = apparel.image.read()  # Read the image file
                encoded_image = base64.b64encode(apparel_image_data).decode('utf-8')  # Convert to base64 string
                apparel_list.append({
                    'id': apparel.id,  # Include apparel ID or any other relevant info
                    'image': encoded_image,
                })

            return Response(apparel_list, status=status.HTTP_200_OK)
        
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetAllApparelsByEmail(APIView):
    
    def get(self, request, email, format=None):
        try:
            # Find the user by email
            user = User.objects.get(email=email)
            
            # Get all apparel items belonging to the user
            apparels = Apparel.objects.filter(ownership=user)

            # Convert images to base64 strings
            apparel_list = []
            for apparel in apparels:
                apparel_image_data = apparel.image.read()  # Read the image file
                encoded_image = base64.b64encode(apparel_image_data).decode('utf-8')  # Convert to base64 string
                apparel_list.append({
                    'id': apparel.id,  # Include apparel ID or any other relevant info
                    'image': encoded_image,
                })

            return Response(apparel_list, status=status.HTTP_200_OK)
        
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




class PostCombinationView(APIView):
    def post(self, request, *args, **kwargs):
        ownership = request.data.get('ownership')
        occasion = request.data.get('occasion')
        image_base64 = request.data.get('image')  # Assuming image is base64-encoded
        # apparels = request.data.getlist('apparels')  # Array of apparel IDs

        try:
            # Fetch the user
            user = User.objects.get(id=ownership)

            # Decode the base64 image
            if image_base64:
                format, imgstr = image_base64.split(';base64,')  # Split the image data
                ext = format.split('/')[-1]  # Extract file extension (e.g., 'png', 'jpg')
                image_data = base64.b64decode(imgstr)  # Decode the base64 image string

                # Create a ContentFile to handle the decoded image
                image_file = ContentFile(image_data, name=f"uploaded_image.{ext}")

            else:
                image_file = None  # No image provided

            # Create the combination object
            combination = Combination(ownership=user, occasion=occasion, image=image_file)
            combination.save()

            # Add apparels to the combination (if you plan to use this in the future)
            # for apparel_id in apparels:
            #    apparel = Apparel.objects.get(id=apparel_id)
            #    combination.apparels.add(apparel)

            return Response({"message": "Combination posted successfully!"}, status=status.HTTP_201_CREATED)

        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Apparel.DoesNotExist:
            return Response({"error": "Apparel not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
class GetLastCombinationsView(APIView):
    def post(self, request, *args, **kwargs):
        ownership_id = request.data.get('ownership_id')  # Get ownership_id from the POST request body

        if not ownership_id or not str(ownership_id).isdigit():
            return Response({"error": "Invalid ownership ID"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(id=ownership_id)

            # Get the last 5 combinations (excluding the requesting user's posts)
            combinations = Combination.objects.exclude(ownership=user).order_by('-date_posted')[:5]

            # Serialize the combinations
            serialized_combinations = []
            for combination in combinations:
                combination_data = {
                    "id": combination.id,
                    "owner": combination.ownership.name,
                    "occasion": combination.occasion,
                    "date_posted": combination.date_posted,
                }
                # Serialize the image
                if combination.image:
                    with combination.image.open('rb') as img_file:
                        combination_data['image'] = base64.b64encode(img_file.read()).decode('utf-8')
                else:
                    combination_data['image'] = None

                serialized_combinations.append(combination_data)

            return Response(serialized_combinations, status=status.HTTP_200_OK)

        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)


import os
import base64
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Apparel

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Base64 decoding function
def decode_base64_image(data):
    image_data = base64.b64decode(data)
    return Image.open(BytesIO(image_data)).convert('RGB')

# Model definition (ResNet-18 based outfit recommender)
class SimpleOutfitRecommender(nn.Module):
    def __init__(self):
        super(SimpleOutfitRecommender, self).__init__()
        
        # Feature extractor (ResNet-18)
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),  # 512 features for each item
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, shirt, pants):
        shirt_features = self.feature_extractor(shirt).view(-1, 512)
        pants_features = self.feature_extractor(pants).view(-1, 512)
        combined_features = torch.cat((shirt_features, pants_features), dim=1)
        return self.classifier(combined_features)

# View for outfit recommendation
class RecommendOutfitView(APIView):
    def post(self, request):
        user_id = request.data.get('user_id')
        occasion = request.data.get('occasion')

        # Fetch upper and lower apparels for the user
        shirts = Apparel.objects.filter(ownership=user_id, upper_lower='upper', occasion=occasion)
        pants = Apparel.objects.filter(ownership=user_id, upper_lower='lower', occasion=occasion)

        # Check if any shirts or pants are found
        if not shirts.exists() or not pants.exists():
            return Response({'error': 'No shirts or pants found for the given criteria'}, status=status.HTTP_404_NOT_FOUND)

        # Create directory for saving images
        base_dir = os.path.join(settings.MEDIA_ROOT, 'recommendations', str(user_id), occasion)
        os.makedirs(os.path.join(base_dir, 'upper'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'lower'), exist_ok=True)

        # Save the filtered images to their respective folders
        for shirt in shirts:
            shirt_image = decode_base64_image(shirt.image.read())
            shirt_image.save(os.path.join(base_dir, 'upper', f"{shirt.id}.jpg"))

        for pant in pants:
            pant_image = decode_base64_image(pant.image.read())
            pant_image.save(os.path.join(base_dir, 'lower', f"{pant.id}.jpg"))

        # Convert images to lists for model input
        shirt_pool = [decode_base64_image(shirt.image.read()) for shirt in shirts]
        pant_pool = [decode_base64_image(pant.image.read()) for pant in pants]

        # Load the model
        model = SimpleOutfitRecommender()
        model.eval()  # Set the model to evaluation mode

        # Function to recommend an outfit based on the model
        def recommend_outfit(model, shirt_pool, pant_pool, num_combinations=10):
            best_outfit = None
            best_score = -1

            for _ in range(num_combinations):
                shirt = random.choice(shirt_pool)
                pants = random.choice(pant_pool)

                shirt_tensor = preprocess_image(shirt)
                pants_tensor = preprocess_image(pants)

                with torch.no_grad():
                    score = model(shirt_tensor, pants_tensor).item()

                if score > best_score:
                    best_score = score
                    best_outfit = (shirt, pants)

            return best_outfit, best_score

        # Apply the model to get the best recommendation
        best_outfit, best_score = recommend_outfit(model, shirt_pool, pant_pool, num_combinations=10)

        # Convert the recommended images back to base64
        def image_to_base64(image):
            buffered = BytesIO()
            image.save(buffered, format="jpg")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        shirt_image_base64 = image_to_base64(best_outfit[0])
        pant_image_base64 = image_to_base64(best_outfit[1])

        # Return the best recommendation with the highest score
        return Response({
            'shirt': shirt_image_base64,
            'pants': pant_image_base64,
            'score': best_score
        }, status=status.HTTP_200_OK)
