from django.urls import path
from .views import (
    CreateUserView, CheckEmailExistsView, CheckNameExistsView, 
    GetPasswordByEmailView, GetPasswordByNameAndEmailView, GetUserIdByEmailView, 
    UpdateUserView, GetUserByIdView, GetUserByEmailView, GetGenderByEmailView, 
    UpdateUserProfileView,  CreateApparelView,
    GetAllApparelsByEmail, ApparelByTypeAndEmailView, PostCombinationView, 
    GetLastCombinationsView, RecommendOutfitView
)

urlpatterns = [
    path('users/', CreateUserView.as_view(), name='create_user'),
    path('users/email/check/<str:email>/', CheckEmailExistsView.as_view(), name='check_email'),
    path('users/name/check/<str:name>/', CheckNameExistsView.as_view(), name='check_name'),
    path('users/password/email/<str:email>/', GetPasswordByEmailView.as_view(), name='get_password_email'),
    path('users/password/name/<str:name>/email/<str:email>/', GetPasswordByNameAndEmailView.as_view(), name='get_password_name_email'),
    path('users/id/email/<str:email>/', GetUserIdByEmailView.as_view(), name='get_user_id_email'),
    path('users/update/', UpdateUserView.as_view(), name='update_user'),
    path('users/<int:id>/', GetUserByIdView.as_view(), name='get_user_by_id'),
    path('users/details/email/<str:email>/', GetUserByEmailView.as_view(), name='get_user_by_email'),
    path('users/gender/email/<str:email>/', GetGenderByEmailView.as_view(), name='get_gender_email'),
    path('users/profile/update/', UpdateUserProfileView.as_view(), name='update_user_profile'),



    path('apparel/create', CreateApparelView.as_view(), name='create_apparel'),
    path('apparel/user/<str:email>/', GetAllApparelsByEmail.as_view(), name='get_allapparels_by_email'),
    path('apparel/user/<str:email>/<str:upper_lower>/', ApparelByTypeAndEmailView.as_view(), name='apparels_by_type_and_email'),

    path('combinations/post/', PostCombinationView.as_view(), name='post_combination'),
    path('combinations/last/', GetLastCombinationsView.as_view(), name='get_last_combinations'),
    
    path('outfit/recommendation/', RecommendOutfitView.as_view(), name='recommend_outfit'),  # API endpoint for outfit recommendation
]
