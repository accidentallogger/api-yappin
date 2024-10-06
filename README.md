
---
#API-YAPPIN
# Outfit Recommendation API

The **Outfit Recommendation API** is designed to handle user and apparel data management, as well as provide outfit recommendations based on the user’s preferences such as occasion and gender. The API includes user registration, apparel management, and outfit combination functionalities.

## Features

- User registration and profile management
- Apparel creation and retrieval
- Outfit recommendations based on gender and occasion
- Storing and retrieving combinations of outfits
- Image-based apparel fetching

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd outfit-recommendation-api
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**:
   - Run migrations to set up the required tables:
     ```bash
     python manage.py migrate
     ```
   - Create a superuser for admin access:
     ```bash
     python manage.py createsuperuser
     ```

5. **Run the server**:
   ```bash
   python manage.py runserver
   ```

## API Endpoints

### User Management

1. **Create User**  
   **URL**: `/users/`  
   **Method**: `POST`  
   **Description**: Register a new user.  
   **Request Parameters**:
   - `name` (string): The name of the user.
   - `emailaddress` (string): The email of the user.
   - `password` (string): The user's password.  
   **Example Request**:
   ```json
   {
     "name": "John Doe",
     "emailaddress": "john@example.com",
     "password": "password123"
   }
   ```  
   **Response**:
   - `success`: Boolean indicating whether the registration was successful.
   - `message`: Confirmation or error message.

2. **Check if Email Exists**  
   **URL**: `/users/email/<str:email>/`  
   **Method**: `GET`  
   **Description**: Check if an email is already registered.  
   **Example Response**:
   ```json
   {
     "exists": true
   }
   ```

3. **Check if Name Exists**  
   **URL**: `/users/name/<str:name>/`  
   **Method**: `GET`  
   **Description**: Check if a name is already registered.  
   **Example Response**:
   ```json
   {
     "exists": true
   }
   ```

4. **Get Password by Email**  
   **URL**: `/users/password/email/<str:email>/`  
   **Method**: `GET`  
   **Description**: Retrieve the password associated with an email.

5. **Get Password by Name and Email**  
   **URL**: `/users/password/name/<str:name>/email/<str:email>/`  
   **Method**: `GET`  
   **Description**: Retrieve the password by providing both the user's name and email.

6. **Get User ID by Email**  
   **URL**: `/users/id/<str:email>/`  
   **Method**: `GET`  
   **Description**: Retrieve the user ID associated with an email.

7. **Update User Info**  
   **URL**: `/users/update/`  
   **Method**: `PUT`  
   **Description**: Update user information.

8. **Get User by ID**  
   **URL**: `/users/<int:id>/`  
   **Method**: `GET`  
   **Description**: Retrieve a user by their user ID.

9. **Get User by Email**  
   **URL**: `/users/email/<str:email>/`  
   **Method**: `GET`  
   **Description**: Retrieve a user by their email address.

10. **Get Gender by Email**  
    **URL**: `/users/gender/<str:email>/`  
    **Method**: `GET`  
    **Description**: Retrieve the gender associated with an email.

11. **Update User Profile**  
    **URL**: `/users/profile/update/`  
    **Method**: `PUT`  
    **Description**: Update user profile details such as name, email, etc.

### Apparel Management

1. **Create Apparel Item**  
   **URL**: `/apparel/`  
   **Method**: `POST`  
   **Description**: Add a new apparel item to the database.  
   **Request Parameters**:
   - `ownership` (string): The ID of the user who owns the apparel.
   - `color` (string): The color of the apparel.
   - `material` (string): The material of the apparel.
   - `upper_lower` (string): Type of apparel (upper for shirts, lower for pants).
   - `type` (string): The type of apparel (formal, casual, etc.).
   - `gender` (string): The gender the apparel is for.
   - `image` (base64 string): The image of the apparel.  
   **Example Request**:
   ```json
   {
     "ownership": "123",
     "color": "blue",
     "material": "cotton",
     "upper_lower": "upper",
     "type": "formal",
     "gender": "male",
     "image": "<base64_encoded_image>"
   }
   ```

2. **Fetch Apparel Images by Upper/Lower**  
   **URL**: `/apparel/<str:upperLower>/`  
   **Method**: `GET`  
   **Description**: Retrieve images of apparel items based on whether they are upper or lower apparel.

3. **Get All Apparel by User Email**  
   **URL**: `/apparel/<str:email>/`  
   **Method**: `GET`  
   **Description**: Retrieve all apparel items associated with a specific email.

4. **Get Apparel by Type and Email**  
   **URL**: `/apparel/<str:email>/<str:category>/`  
   **Method**: `GET`  
   **Description**: Retrieve apparel items filtered by type (formal, casual, etc.) and user email.

### Outfit Recommendation

1. **Post Outfit Combination**  
   **URL**: `/combinations/post/`  
   **Method**: `POST`  
   **Description**: Post a new combination of apparel items (upper and lower).

2. **Get Last Combinations**  
   **URL**: `/combinations/last/`  
   **Method**: `GET`  
   **Description**: Retrieve the last outfit combinations used by the user.

3. **Recommend an Outfit**  
   **URL**: `/outfitRecommendation/`  
   **Method**: `POST`  
   **Description**: Recommend an outfit (upper and lower apparel) based on the user's gender and occasion (formal, casual, traditional).  
   **Request Parameters**:
   - `user_id` (string): The user ID.
   - `gender` (string): The gender of the user.
   - `occasion` (string): The type of occasion (e.g., formal, casual, traditional).

   **Example Request**:
   ```json
   {
     "user_id": "123",
     "gender": "male",
     "occasion": "formal"
   }
   ```

   **Example Response**:
   ```json
   {
     "upper": {
       "apparel_id": "456",
       "color": "blue",
       "type": "shirt",
       "material": "cotton"
     },
     "lower": {
       "apparel_id": "789",
       "color": "black",
       "type": "pants",
       "material": "wool"
     }
   }
   ```

## Future Scope

- **Clothing Detection and Classification on the API side**: Future enhancements could include adding machine learning models for the detection and classification of apparel from uploaded images. This could allow users to upload images of clothing and have the system automatically classify them into categories like formal, casual, or traditional, and further distinguish between male and female apparel.

- **Personalized Recommendations**: Implementing more personalized outfit recommendations based on weather, events, and preferences learned over time from user behavior and outfit combinations.

- **Outfit Scoring System**: Introduce a rating or scoring system where users can rate their outfit combinations, and the system can use this data to provide better recommendations in the future.

---
