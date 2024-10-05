from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User, Apparel
from .serializers import UserSerializer, ApparelSerializer
import base64
from django.http import JsonResponse
from django.views import View
from .models import Apparel
from .models import User, Combination, Apparel
from django.core.files.base import ContentFile
from .recommendationsystem import *
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
        return str(Apparel.objects.filter(email=email, category=category))

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






# View for outfit recommendation
class RecommendOutfitView(View):
    def post(self, request):
        user_id = request.POST.get('user_id')
        gender = request.POST.get('gender')
        occasion = request.POST.get('occasion')
        
        # Fetch upper and lower apparels for the user
        shirts = Apparel.objects.filter(ownership=user_id, upper_lower='upper', type=occasion, gender=gender)
        pants = Apparel.objects.filter(ownership=user_id, upper_lower='lower', type=occasion, gender=gender)
        
        # Convert images to lists
        shirt_pool = [decode_base64_image(shirt.image) for shirt in shirts]
        pant_pool = [decode_base64_image(pant.image) for pant in pants]

        # If no valid apparels found, return an error response
        if not shirt_pool or not pant_pool:
            return JsonResponse({'error': 'No shirts or pants found for the given criteria'}, status=404)

        # Load the model
        model = SimpleOutfitRecommender()

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

        # Apply the model multiple times to get the best recommendation
        best_outfit, best_score = recommend_outfit(model, shirt_pool, pant_pool, num_combinations=10)

        # Convert the recommended images back to base64
        def image_to_base64(image):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        shirt_image_base64 = image_to_base64(best_outfit[0])
        pant_image_base64 = image_to_base64(best_outfit[1])

        # Return the best recommendation with the highest score
        return JsonResponse({
            'shirt': shirt_image_base64,
            'pants': pant_image_base64,
            'score': best_score
        })